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
"""Nemotron-H backbone with grouped codebook prediction in its tail."""
from __future__ import annotations

from typing import Any, Optional

import torch
from torch import nn
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding
from vllm.model_executor.models.nemotron_h import (
    ALL_DECODER_LAYER_TYPES,
    NemotronHAttentionDecoderLayer,
    NemotronHMambaDecoderLayer,
    NemotronHMLPDecoderLayer,
    NemotronHModel,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory, make_layers
from vllm.sequence import IntermediateTensors

from easymagpie_vllm_omni.config import EasyMagpieOmniArch

_DEFAULT_TOP_K = 80
_MIN_SAMPLING_TEMPERATURE = 1e-4


class _AttentionFFNCodebookBlock(nn.Module):
    """One logical codebook block composed of attention then a dense FFN."""

    def __init__(self, *, vllm_config: VllmConfig, layer_idx: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        common = {
            "config": config,
            "layer_idx": layer_idx,
            "model_config": vllm_config.model_config,
            "cache_config": vllm_config.cache_config,
            "quant_config": vllm_config.quant_config,
            "parallel_config": vllm_config.parallel_config,
        }
        self.attention = NemotronHAttentionDecoderLayer(prefix=f"{prefix}.attention", **common)
        self.ffn = NemotronHMLPDecoderLayer(prefix=f"{prefix}.ffn", **common)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.attention(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        return self.ffn(hidden_states=hidden_states, residual=residual)


class _MambaFFNCodebookBlock(nn.Module):
    """One logical codebook block composed of Mamba then a dense FFN."""

    def __init__(self, *, vllm_config: VllmConfig, layer_idx: int, prefix: str) -> None:
        super().__init__()
        config = vllm_config.model_config.hf_config
        common = {
            "config": config,
            "layer_idx": layer_idx,
            "model_config": vllm_config.model_config,
            "cache_config": vllm_config.cache_config,
            "quant_config": vllm_config.quant_config,
            "parallel_config": vllm_config.parallel_config,
        }
        self.mamba = NemotronHMambaDecoderLayer(prefix=f"{prefix}.mamba", **common)
        self.ffn = NemotronHMLPDecoderLayer(prefix=f"{prefix}.ffn", **common)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_states, residual = self.mamba(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        return self.ffn(hidden_states=hidden_states, residual=residual)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": 0,
        "inputs_embeds": 0,
        "code_prediction_mask": 0,
        "gumbel_noise": 0,
    }
)
class EasyMagpieBackboneCodebookModel(NemotronHModel):
    """Nemotron-H whose final blocks predict groups of codebooks.

    The first ``backbone_codebook_start_layer`` entries use the ordinary
    Nemotron-H pattern. Each tail group runs its configured logical blocks and
    then predicts one or more codebooks in parallel. Their sampled embeddings
    are averaged and added to the live residual stream before the next group,
    so it receives both the preceding hidden state and explicit acoustic-token
    connections.
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        nn.Module.__init__(self)
        arch = EasyMagpieOmniArch.from_hf_config(vllm_config.model_config.hf_config)
        if not arch.uses_backbone_codebook_layers:
            raise ValueError("EasyMagpieBackboneCodebookModel requires codebook_prediction_mode='backbone'")
        if vllm_config.parallel_config.pipeline_parallel_size != 1:
            raise ValueError("backbone codebook prediction currently requires pipeline_parallel_size=1")

        config = vllm_config.model_config.hf_config
        self.config = config
        self.arch = arch
        self.vocab_size = config.vocab_size
        self.codebook_start_layer = arch.backbone_codebook_start_layer
        self.num_codebooks = arch.num_stacked_codebooks
        self.codebook_layers_per_group = arch.backbone_codebook_layers_per_group
        self.codebooks_per_group = arch.backbone_codebooks_per_group
        self.num_tokens_per_codebook = arch.num_all_tokens_per_codebook
        self.audio_embedding_dim = arch.audio_embedding_dim
        self.embedding_dim = arch.embedding_dim
        self.temperature = 0.7
        self.top_k = _DEFAULT_TOP_K
        self._sample_top_k = min(self.top_k, self.num_tokens_per_codebook)

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, config.hidden_size)
        self.has_moe = "E" in config.hybrid_override_pattern

        def get_layer(prefix: str) -> nn.Module:
            layer_idx = int(prefix.rsplit(".", 1)[1])
            if layer_idx < self.codebook_start_layer:
                layer_class = ALL_DECODER_LAYER_TYPES[config.hybrid_override_pattern[layer_idx]]
                return layer_class(
                    config=config,
                    layer_idx=layer_idx,
                    model_config=vllm_config.model_config,
                    cache_config=vllm_config.cache_config,
                    quant_config=vllm_config.quant_config,
                    parallel_config=vllm_config.parallel_config,
                    prefix=prefix,
                )
            if arch.backbone_codebook_layer_type == "attention_ffn":
                return _AttentionFFNCodebookBlock(
                    vllm_config=vllm_config,
                    layer_idx=layer_idx,
                    prefix=prefix,
                )
            if arch.backbone_codebook_layer_type == "mamba_ffn":
                return _MambaFFNCodebookBlock(
                    vllm_config=vllm_config,
                    layer_idx=layer_idx,
                    prefix=prefix,
                )
            return NemotronHAttentionDecoderLayer(
                config=config,
                layer_idx=layer_idx,
                model_config=vllm_config.model_config,
                cache_config=vllm_config.cache_config,
                quant_config=vllm_config.quant_config,
                parallel_config=vllm_config.parallel_config,
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            len(config.hybrid_override_pattern),
            get_layer,
            prefix=f"{prefix}.layers",
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        self.norm_f = RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)

        self.audio_embeddings = nn.ModuleList(
            [nn.Embedding(self.num_tokens_per_codebook, self.audio_embedding_dim) for _ in range(self.num_codebooks)]
        )
        if self.audio_embedding_dim != self.embedding_dim:
            self.audio_in_projection = nn.Linear(self.audio_embedding_dim, self.embedding_dim)
        else:
            self.audio_in_projection = nn.Identity()
        self.codebook_output_norms = nn.ModuleList(
            [RMSNorm(self.embedding_dim, eps=config.layer_norm_epsilon) for _ in range(self.num_codebooks)]
        )
        self.codebook_output_projections = nn.ModuleList(
            [nn.Linear(self.embedding_dim, self.num_tokens_per_codebook) for _ in range(self.num_codebooks)]
        )

        forbidden = torch.zeros(self.num_tokens_per_codebook, dtype=torch.bool)
        forbidden[arch.codebook_size :] = True
        forbidden[arch.audio_eos_id] = False
        self.register_buffer("forbidden_mask", forbidden, persistent=False)

        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self._gumbel_buf = torch.zeros(
            max_num_tokens,
            self.num_codebooks,
            self._sample_top_k,
            dtype=torch.float32,
        )
        self._temperature_buf = torch.zeros(1, dtype=torch.float32)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def embed_audio_frame(self, codes: torch.Tensor) -> torch.Tensor:
        """Average a frame's per-codebook embeddings for the next decode step."""
        acc = self.audio_embeddings[0](codes[:, 0])
        for codebook_idx in range(1, self.num_codebooks):
            acc = acc + self.audio_embeddings[codebook_idx](codes[:, codebook_idx])
        return self.audio_in_projection(acc / self.num_codebooks)

    @torch.no_grad()
    def prepare_sampling(self, num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw fresh sampling noise into fixed-address buffers."""
        noise = self._gumbel_buf[:num_tokens]
        noise.uniform_(1e-20, 1.0 - 1e-20)
        noise.log_().neg_().log_().neg_()
        self._temperature_buf.fill_(max(float(self.temperature), _MIN_SAMPLING_TEMPERATURE))
        return noise, self._temperature_buf

    def _sample_codebook(
        self,
        codebook_idx: int,
        hidden_states: torch.Tensor,
        gumbel_noise: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.codebook_output_projections[codebook_idx](hidden_states)
        logits = logits.masked_fill(self.forbidden_mask, float("-inf")) / temperature
        vals, idxs = torch.topk(logits, self._sample_top_k, dim=-1)
        picked = (vals + gumbel_noise).argmax(dim=-1, keepdim=True)
        return idxs.gather(-1, picked).squeeze(-1)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        code_prediction_mask: Optional[torch.Tensor] = None,
        gumbel_noise: Optional[torch.Tensor] = None,
        temperature: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the temporal prefix, then predict codebooks through the tail."""
        if intermediate_tensors is not None:
            raise ValueError("backbone codebook prediction does not support pipeline intermediate tensors")
        if code_prediction_mask is None or gumbel_noise is None or temperature is None:
            raise ValueError("backbone codebook prediction requires mask, Gumbel noise, and temperature")

        hidden_states = inputs_embeds if inputs_embeds is not None else self.embed_input_ids(input_ids)
        residual = None
        codes: list[torch.Tensor] = []

        for layer_idx, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if layer_idx < self.codebook_start_layer:
                continue

            tail_layer_idx = layer_idx - self.codebook_start_layer
            if (tail_layer_idx + 1) % self.codebook_layers_per_group != 0:
                continue

            group_idx = tail_layer_idx // self.codebook_layers_per_group
            first_codebook_idx = group_idx * self.codebooks_per_group
            next_codebook_idx = first_codebook_idx + self.codebooks_per_group
            prediction_state = hidden_states if residual is None else hidden_states + residual
            group_codes: list[torch.Tensor] = []
            for codebook_idx in range(first_codebook_idx, next_codebook_idx):
                normalized_state = self.codebook_output_norms[codebook_idx](prediction_state)
                code = self._sample_codebook(
                    codebook_idx,
                    normalized_state,
                    gumbel_noise[:, codebook_idx, :],
                    temperature,
                )
                group_codes.append(code)
                codes.append(code)

            if next_codebook_idx < self.num_codebooks:
                feedback = self.audio_embeddings[first_codebook_idx](group_codes[0])
                for group_offset in range(1, self.codebooks_per_group):
                    codebook_idx = first_codebook_idx + group_offset
                    feedback = feedback + self.audio_embeddings[codebook_idx](group_codes[group_offset])
                feedback = self.audio_in_projection(feedback / self.codebooks_per_group)
                feedback = feedback * code_prediction_mask.unsqueeze(-1).to(feedback.dtype)
                hidden_states = hidden_states + feedback

        hidden_states, _ = self.norm_f(hidden_states, residual)
        return hidden_states, torch.stack(codes, dim=1)
