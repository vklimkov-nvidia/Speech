# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Text-conditioned CTC utilities for EasyMagpie token timing."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class CTCTokenSpan:
    """A transcript token's half-open span on the acoustic-frame timeline."""

    token_id: int
    start_frame: int
    end_frame: int
    score: float


class TextConditionedCTCHead(nn.Module):
    """Score acoustic states against blank and tokens in the supplied transcript.

    A per-utterance vocabulary avoids a large ``hidden_size x text_vocab_size``
    classifier. Acoustic states are projected into the model's text-embedding
    space and compared by cosine similarity with the unique transcript-token
    embeddings. Local class zero is reserved for the learned CTC blank.
    """

    def __init__(
        self,
        hidden_size: int,
        text_embedding_size: int,
        bottleneck_size: int = 256,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        if bottleneck_size <= 0:
            raise ValueError(f"bottleneck_size must be positive, got {bottleneck_size}")
        if temperature <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, bottleneck_size),
            nn.SiLU(),
            nn.Linear(bottleneck_size, text_embedding_size),
        )
        self.blank = nn.Linear(hidden_size, 1)
        self.temperature = temperature

    @staticmethod
    def candidates(target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return unique token IDs, local CTC targets, and first-occurrence indices."""
        if target_ids.ndim != 1:
            raise ValueError(f"target_ids must be one-dimensional, got shape {tuple(target_ids.shape)}")
        unique_ids: list[int] = []
        first_indices: list[int] = []
        local_by_id: dict[int, int] = {}
        local_targets: list[int] = []
        for index, token_id_value in enumerate(target_ids.detach().cpu().tolist()):
            token_id = int(token_id_value)
            if token_id not in local_by_id:
                local_by_id[token_id] = len(unique_ids) + 1
                unique_ids.append(token_id)
                first_indices.append(index)
            local_targets.append(local_by_id[token_id])
        return (
            torch.tensor(unique_ids, device=target_ids.device, dtype=torch.long),
            torch.tensor(local_targets, device=target_ids.device, dtype=torch.long),
            torch.tensor(first_indices, device=target_ids.device, dtype=torch.long),
        )

    def emissions(
        self,
        hidden_states: torch.Tensor,
        target_ids: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute ``[frames, blank + unique transcript tokens]`` logits."""
        if hidden_states.ndim != 2:
            raise ValueError(f"hidden_states must have shape [frames, hidden], got {tuple(hidden_states.shape)}")
        if target_embeddings.ndim != 2 or target_embeddings.shape[0] != target_ids.numel():
            raise ValueError("target_embeddings must have one row per target token")
        _, local_targets, first_indices = self.candidates(target_ids)
        head_input = hidden_states.to(self.blank.weight.dtype)
        projected = F.normalize(self.projection(head_input).float(), dim=-1)
        candidate_embeddings = F.normalize(target_embeddings[first_indices].detach().float(), dim=-1)
        token_logits = projected @ candidate_embeddings.transpose(0, 1)
        token_logits = token_logits / self.temperature
        blank_logits = self.blank(head_input).float()
        return torch.cat([blank_logits, token_logits], dim=-1), local_targets

    def loss(
        self,
        hidden_states: torch.Tensor,
        input_lengths: torch.Tensor,
        target_ids: torch.Tensor,
        target_lengths: torch.Tensor,
        target_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean CTC loss, skipping sequences too short for a valid path."""
        losses: list[torch.Tensor] = []
        for row in range(hidden_states.shape[0]):
            input_length = int(input_lengths[row])
            target_length = int(target_lengths[row])
            if input_length <= 0 or target_length <= 0:
                continue
            frame_states = hidden_states[row, :input_length]
            target = target_ids[row, :target_length]
            embeddings = target_embeddings[row, :target_length]
            logits, local_targets = self.emissions(frame_states, target, embeddings)
            repeated = int((local_targets[1:] == local_targets[:-1]).sum())
            if input_length < local_targets.numel() + repeated:
                continue
            losses.append(
                F.ctc_loss(
                    logits.log_softmax(-1).unsqueeze(1),
                    local_targets,
                    input_lengths=torch.tensor([input_length], device=hidden_states.device),
                    target_lengths=torch.tensor([target_length], device=hidden_states.device),
                    blank=0,
                    reduction="mean",
                    zero_infinity=True,
                )
            )
        if not losses:
            zero = hidden_states.sum() * 0.0
            for parameter in self.parameters():
                zero = zero + parameter.sum() * 0.0
            return zero
        return torch.stack(losses).mean()


def forced_align_ctc(
    log_probs: torch.Tensor, local_targets: torch.Tensor, token_ids: torch.Tensor
) -> list[CTCTokenSpan]:
    """Viterbi-align a known token sequence to CTC frame log probabilities."""
    if log_probs.ndim != 2:
        raise ValueError("log_probs must have shape [frames, classes]")
    frame_count = log_probs.shape[0]
    target_count = local_targets.numel()
    if target_count == 0:
        return []
    if token_ids.numel() != target_count:
        raise ValueError("token_ids and local_targets must have the same length")
    if frame_count == 0:
        raise ValueError("CTC alignment requires at least one frame")
    if int(local_targets.max()) >= log_probs.shape[1]:
        raise ValueError("local target index exceeds the CTC class count")
    repeated = int((local_targets[1:] == local_targets[:-1]).sum())
    minimum_frames = target_count + repeated
    if frame_count < minimum_frames:
        raise ValueError(
            f"CTC alignment needs at least {minimum_frames} frames for {target_count} tokens, received {frame_count}"
        )

    device = log_probs.device
    states = torch.zeros(target_count * 2 + 1, dtype=torch.long, device=device)
    states[1::2] = local_targets
    state_count = states.numel()
    scores = log_probs.new_full((frame_count, state_count), -torch.inf)
    backpointers = torch.full((frame_count, state_count), -1, dtype=torch.long, device=device)
    scores[0, 0] = log_probs[0, 0]
    scores[0, 1] = log_probs[0, states[1]]

    for frame in range(1, frame_count):
        for state in range(state_count):
            predecessors = [state]
            if state > 0:
                predecessors.append(state - 1)
            if state > 1 and states[state] != 0 and states[state] != states[state - 2]:
                predecessors.append(state - 2)
            predecessor_tensor = torch.tensor(predecessors, device=device)
            predecessor_scores = scores[frame - 1, predecessor_tensor]
            best_index = int(predecessor_scores.argmax())
            previous = predecessors[best_index]
            scores[frame, state] = predecessor_scores[best_index] + log_probs[frame, states[state]]
            backpointers[frame, state] = previous

    final_candidates = torch.tensor([state_count - 1, state_count - 2], device=device)
    final_state = int(final_candidates[scores[-1, final_candidates].argmax()])
    path = [final_state]
    for frame in range(frame_count - 1, 0, -1):
        final_state = int(backpointers[frame, final_state])
        path.append(final_state)
    path.reverse()

    spans: list[CTCTokenSpan] = []
    for target_index in range(target_count):
        token_state = target_index * 2 + 1
        frames = [frame for frame, state in enumerate(path) if state == token_state]
        if not frames:
            continue
        class_id = int(local_targets[target_index])
        score = float(log_probs[frames, class_id].exp().mean().detach().cpu())
        spans.append(
            CTCTokenSpan(
                token_id=int(token_ids[target_index]),
                start_frame=frames[0],
                end_frame=frames[-1] + 1,
                score=score,
            )
        )
    return spans
