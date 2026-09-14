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
"""EasyMagpieTTS architecture configuration for vLLM-Omni."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Each audio codebook appends BOS, EOS, context, mask, and reserved tokens.
NUM_SPECIAL_AUDIO_TOKENS: int = 8

# Offsets within the trailing special-token block.
SPECIAL_AUDIO_BOS: int = 0
SPECIAL_AUDIO_EOS: int = 1
SPECIAL_AUDIO_CONTEXT_BOS: int = 2
SPECIAL_AUDIO_CONTEXT_EOS: int = 3
SPECIAL_AUDIO_MASK: int = 4


@dataclass
class EasyMagpieOmniArch:
    """Static architecture description for an EasyMagpieTTS checkpoint."""

    hidden_dim: int = 1536
    embedding_dim: int = 1536
    audio_embedding_dim: int = 1536

    num_audio_codebooks: int = 8
    codebook_size: int = 1024
    frame_stacking_factor: int = 2

    phoneme_stacking_factor: int = 1
    phoneme_vocab_size: int = 2051

    # Text EOS is normally the second-to-last text-vocabulary row. Multiturn
    # checkpoints append an interruption token after CFG_UNK, so converters pin
    # the actual ID explicitly instead of deriving it from the final table size.
    text_eos_id: int | None = None
    use_multiturn_dataset: bool = False

    # The text/phoneme/audio streams are temporally offset: at decode step ``k``
    # the text channel consumes ``text_tokens[k]``, the phoneme channel starts at
    # ``k == streaming_phonemes_delay`` (seeded with phoneme BOS), and the audio
    # channel starts at ``k == streaming_speech_delay`` (seeded with audio BOS).
    # Both default to 0 (lock-step), which reproduces a non-delayed / "full" mode.
    streaming_phonemes_delay: int = 0
    streaming_speech_delay: int = 0

    # Phoneme special-token ids (into the per-stack ``phoneme_embeddings`` table)
    # and the confidence→UNK replacement threshold. ``None`` falls back to the
    # IPABPETokenizer convention (bos/eos/unk = vocab-3/-2/-1).
    phoneme_bos_id: int | None = None
    phoneme_eos_id: int | None = None
    phoneme_unk_id: int | None = None
    phoneme_confidence_unk_threshold: float = 0.0

    # Number of task embeddings; zero disables task conditioning.
    num_task_embeddings: int = 0

    # How each backbone hidden state is expanded into the stacked audio
    # codebooks. ``autoregressive`` uses the intra-frame local transformer;
    # ``parallel`` projects the backbone state to every codebook at once;
    # ``backbone`` assigns one trailing backbone block to each codebook.
    codebook_prediction_mode: str = "autoregressive"
    local_transformer_n_layers: int = 3
    local_transformer_n_heads: int = 12
    local_transformer_hidden_dim: int = 1536

    # ``backbone`` mode keeps this many ordinary temporal layers before the
    # per-codebook tail. Each tail entry is one logical backbone block, with the
    # selected internal mixer composition.
    backbone_codebook_start_layer: int = 16
    backbone_codebook_layer_type: str = "attention_ffn"

    # Optional checkpoint-specific special-token ids.
    forced_audio_bos_id: int | None = None
    forced_audio_eos_id: int | None = None
    forced_mask_token_id: int | None = None

    extra: dict[str, Any] = field(default_factory=dict)

    def validate(self, *, text_vocab_size: int | None = None) -> None:
        """Reject architecture variants the current vLLM implementation cannot serve."""

        if self.codebook_prediction_mode not in {"autoregressive", "parallel", "backbone"}:
            raise ValueError(
                "codebook_prediction_mode must be 'autoregressive', 'parallel', or 'backbone', got "
                f"{self.codebook_prediction_mode!r}"
            )
        if self.uses_backbone_codebook_layers:
            if self.backbone_codebook_start_layer <= 0:
                raise ValueError(
                    "backbone_codebook_start_layer must be positive in backbone mode, got "
                    f"{self.backbone_codebook_start_layer}"
                )
            if self.backbone_codebook_layer_type not in {"attention_ffn", "mamba_ffn", "attention"}:
                raise ValueError(
                    "backbone_codebook_layer_type must be 'attention_ffn', 'mamba_ffn', or 'attention', got "
                    f"{self.backbone_codebook_layer_type!r}"
                )

        positive_fields = (
            "hidden_dim",
            "embedding_dim",
            "audio_embedding_dim",
            "num_audio_codebooks",
            "codebook_size",
            "frame_stacking_factor",
        )
        if self.uses_local_transformer:
            positive_fields += (
                "local_transformer_n_layers",
                "local_transformer_n_heads",
                "local_transformer_hidden_dim",
            )
        for name in positive_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        if self.hidden_dim != self.embedding_dim:
            raise ValueError(
                "hidden_dim must equal embedding_dim because the vLLM backbone currently consumes text/audio "
                "embeddings without an input projection. Add the corresponding projection to support unequal widths; "
                f"got hidden_dim={self.hidden_dim}, embedding_dim={self.embedding_dim}."
            )
        if self.uses_local_transformer and self.local_transformer_hidden_dim % self.local_transformer_n_heads != 0:
            raise ValueError(
                "local_transformer_hidden_dim must be divisible by local_transformer_n_heads for the current "
                "attention implementation; extend EasyMagpieCodePredictor to support other head layouts. Got "
                f"{self.local_transformer_hidden_dim} and {self.local_transformer_n_heads}."
            )

        phonemes_enabled = self.phoneme_vocab_size > 0 and self.phoneme_stacking_factor > 0
        if phonemes_enabled != (self.phoneme_vocab_size > 0 or self.phoneme_stacking_factor > 0):
            raise ValueError(
                "phoneme_vocab_size and phoneme_stacking_factor must be both enabled or both disabled; "
                f"got {self.phoneme_vocab_size} and {self.phoneme_stacking_factor}."
            )
        if self.phoneme_vocab_size < 0 or self.phoneme_stacking_factor < 0:
            raise ValueError("phoneme_vocab_size and phoneme_stacking_factor cannot be negative")
        if phonemes_enabled:
            if self.phoneme_vocab_size < 3:
                raise ValueError("phoneme_vocab_size must include at least BOS, EOS, and UNK tokens")
            phoneme_ids = {
                "phoneme_bos_id": self.resolved_phoneme_bos_id,
                "phoneme_eos_id": self.resolved_phoneme_eos_id,
                "phoneme_unk_id": self.resolved_phoneme_unk_id,
            }
            for name, token_id in phoneme_ids.items():
                if not 0 <= token_id < self.phoneme_vocab_size:
                    raise ValueError(f"{name}={token_id} must be in [0, {self.phoneme_vocab_size})")
            if len(set(phoneme_ids.values())) != len(phoneme_ids):
                raise ValueError("phoneme BOS, EOS, and UNK token ids must be distinct")
        if not 0.0 <= self.phoneme_confidence_unk_threshold <= 1.0:
            raise ValueError("phoneme_confidence_unk_threshold must be in [0, 1]")

        if self.streaming_phonemes_delay < 0 or self.streaming_speech_delay < 0:
            raise ValueError("streaming delays cannot be negative")
        if (self.streaming_phonemes_delay or self.streaming_speech_delay) and (
            self.streaming_speech_delay <= self.streaming_phonemes_delay
        ):
            raise ValueError(
                "streaming_speech_delay must be greater than streaming_phonemes_delay for delayed streaming; "
                "extend the prefill/decode scheduling before using other delay layouts. Got "
                f"{self.streaming_speech_delay} and {self.streaming_phonemes_delay}."
            )

        audio_special_ids = {
            "forced_audio_bos_id" if self.forced_audio_bos_id is not None else "audio_bos_id": self.audio_bos_id,
            "forced_audio_eos_id" if self.forced_audio_eos_id is not None else "audio_eos_id": self.audio_eos_id,
            "forced_mask_token_id" if self.forced_mask_token_id is not None else "mask_token_id": self.mask_token_id,
        }
        special_start = self.codebook_size
        special_end = self.num_all_tokens_per_codebook
        for name, token_id in audio_special_ids.items():
            if not special_start <= token_id < special_end:
                raise ValueError(
                    f"{name}={token_id} must be in the special-token range [{special_start}, {special_end})"
                )
        if len(set(audio_special_ids.values())) != len(audio_special_ids):
            raise ValueError("audio BOS, EOS, and MASK token ids must be distinct")

        if self.num_task_embeddings < 0:
            raise ValueError("num_task_embeddings cannot be negative")
        if text_vocab_size is not None:
            if text_vocab_size <= 0:
                raise ValueError(f"text_vocab_size must be positive, got {text_vocab_size}")
            text_eos_id = self.resolved_text_eos_id(text_vocab_size)
            if not 0 <= text_eos_id < text_vocab_size:
                raise ValueError(f"text_eos_id={text_eos_id} must be in [0, {text_vocab_size})")

    @property
    def num_stacked_codebooks(self) -> int:
        """Number of independent codebooks the model predicts per frame (``C * S``)."""
        return self.num_audio_codebooks * self.frame_stacking_factor

    @property
    def uses_local_transformer(self) -> bool:
        """Whether codebooks are predicted by the autoregressive local transformer."""
        return self.codebook_prediction_mode == "autoregressive"

    @property
    def uses_backbone_codebook_layers(self) -> bool:
        """Whether trailing backbone blocks predict codebooks sequentially."""
        return self.codebook_prediction_mode == "backbone"

    def validate_backbone_codebook_layout(self, hf_config: Any) -> None:
        """Validate the logical layer count and cache topology for backbone mode."""
        if not self.uses_backbone_codebook_layers:
            return

        expected_layers = self.backbone_codebook_start_layer + self.num_stacked_codebooks
        num_hidden_layers = int(getattr(hf_config, "num_hidden_layers", 0) or 0)
        pattern = str(getattr(hf_config, "hybrid_override_pattern", "") or "")
        if num_hidden_layers != expected_layers or len(pattern) != expected_layers:
            raise ValueError(
                "backbone codebook prediction requires one logical tail block per stacked codebook: "
                f"expected num_hidden_layers=len(hybrid_override_pattern)={expected_layers}, got "
                f"{num_hidden_layers} and {len(pattern)}"
            )

        expected_tail_symbol = "M" if self.backbone_codebook_layer_type == "mamba_ffn" else "*"
        tail = pattern[self.backbone_codebook_start_layer :]
        if tail != expected_tail_symbol * self.num_stacked_codebooks:
            raise ValueError(
                f"backbone_codebook_layer_type={self.backbone_codebook_layer_type!r} requires a tail of "
                f"{self.num_stacked_codebooks} {expected_tail_symbol!r} symbols, got {tail!r}"
            )

    @property
    def text_prefill_num(self) -> int:
        """Text-led decode positions that can be folded into causal prefill.

        Positions before ``streaming_phonemes_delay`` have no phoneme input.
        The next position is also deterministic because it receives phoneme
        BOS, so it can be prefetched as long as speech starts later.
        """
        if self.phoneme_vocab_size <= 0 or self.phoneme_stacking_factor <= 0:
            return 0
        if self.streaming_speech_delay <= 0:
            return 0
        if self.streaming_speech_delay <= self.streaming_phonemes_delay:
            raise ValueError(
                "streaming_speech_delay must be greater than streaming_phonemes_delay for text-led prefill"
            )
        return self.streaming_phonemes_delay + 1

    @property
    def num_all_tokens_per_codebook(self) -> int:
        """Per-codebook vocabulary size including the trailing special tokens."""
        return self.codebook_size + NUM_SPECIAL_AUDIO_TOKENS

    @property
    def audio_bos_id(self) -> int:
        """Embedding-table id of the audio BOS token."""
        if self.forced_audio_bos_id is not None:
            return self.forced_audio_bos_id
        return self.codebook_size + SPECIAL_AUDIO_BOS

    @property
    def audio_eos_id(self) -> int:
        """Embedding-table id of the audio EOS token."""
        if self.forced_audio_eos_id is not None:
            return self.forced_audio_eos_id
        return self.codebook_size + SPECIAL_AUDIO_EOS

    @property
    def mask_token_id(self) -> int:
        """Embedding-table id of the MaskGit MASK token."""
        if self.forced_mask_token_id is not None:
            return self.forced_mask_token_id
        return self.codebook_size + SPECIAL_AUDIO_MASK

    @property
    def resolved_phoneme_bos_id(self) -> int:
        """Phoneme BOS id, falling back to the IPABPETokenizer convention (vocab-3)."""
        return self.phoneme_bos_id if self.phoneme_bos_id is not None else self.phoneme_vocab_size - 3

    @property
    def resolved_phoneme_eos_id(self) -> int:
        """Phoneme EOS id, falling back to the IPABPETokenizer convention (vocab-2)."""
        return self.phoneme_eos_id if self.phoneme_eos_id is not None else self.phoneme_vocab_size - 2

    @property
    def resolved_phoneme_unk_id(self) -> int:
        """Phoneme UNK id, falling back to the IPABPETokenizer convention (vocab-1)."""
        return self.phoneme_unk_id if self.phoneme_unk_id is not None else self.phoneme_vocab_size - 1

    def resolved_text_eos_id(self, text_vocab_size: int) -> int:
        """Text EOS id, preserving the legacy second-to-last-row convention."""
        return self.text_eos_id if self.text_eos_id is not None else text_vocab_size - 2

    @classmethod
    def from_hf_config(cls, hf_config: Any) -> "EasyMagpieOmniArch":
        """Build an arch description from a vLLM ``hf_config``.

        Attributes present on ``hf_config`` override the defaults; unknown
        attributes are ignored.
        """
        defaults = cls()
        kwargs: dict[str, Any] = {}
        for f in (
            "hidden_dim",
            "embedding_dim",
            "audio_embedding_dim",
            "num_audio_codebooks",
            "codebook_size",
            "frame_stacking_factor",
            "phoneme_stacking_factor",
            "phoneme_vocab_size",
            "text_eos_id",
            "use_multiturn_dataset",
            "streaming_phonemes_delay",
            "streaming_speech_delay",
            "phoneme_bos_id",
            "phoneme_eos_id",
            "phoneme_unk_id",
            "phoneme_confidence_unk_threshold",
            "num_task_embeddings",
            "codebook_prediction_mode",
            "local_transformer_n_layers",
            "local_transformer_n_heads",
            "local_transformer_hidden_dim",
            "backbone_codebook_start_layer",
            "backbone_codebook_layer_type",
            "forced_audio_bos_id",
            "forced_audio_eos_id",
            "forced_mask_token_id",
        ):
            if hasattr(hf_config, f):
                kwargs[f] = getattr(hf_config, f)
        # ``hidden_size`` is the canonical HF name for the backbone width.
        if "hidden_dim" not in kwargs and hasattr(hf_config, "hidden_size"):
            kwargs["hidden_dim"] = hf_config.hidden_size
            kwargs.setdefault("embedding_dim", hf_config.hidden_size)
        merged = {**defaults.__dict__, **kwargs}
        merged.pop("extra", None)
        arch = cls(**merged)
        arch.validate(text_vocab_size=getattr(hf_config, "text_vocab_size", None))
        arch.validate_backbone_codebook_layout(hf_config)
        return arch


EASYMAGPIE_SMALLMAMBA = EasyMagpieOmniArch()
