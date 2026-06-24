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
"""Inference-only EasyMagpieTTS model for vLLM-Omni.

EasyMagpieTTS is a decoder-only streaming TTS model. A Nemotron-H hybrid
(Mamba2 + attention + MoE) text-LM backbone consumes a per-frame additive input
embedding (text + phoneme + audio) and emits a per-frame hidden state. A small
autoregressive *local transformer* then samples all ``C * S`` stacked audio
codebooks for that frame (see :mod:`easymagpie_vllm_omni.local_transformer`).

This module wires that architecture into vLLM-Omni's
``preprocess`` / ``forward`` / ``compute_logits`` / ``make_omni_output`` /
``postprocess`` contract:

* **Backbone** — vLLM's
  :class:`~vllm.model_executor.models.nemotron_h.NemotronHModel` is reused
  wholesale (hybrid Mamba2 state + KV cache + paged attention). Every step feeds
  the backbone via ``inputs_embeds``; its own ``embed_tokens`` table is never
  consumed. Because the backbone is a hybrid-Mamba model, the class implements
  vLLM's :class:`HasInnerState` / :class:`IsHybrid` /
  :class:`SupportsMambaPrefixCaching` contracts (mamba-state helpers are
  delegated to :class:`NemotronHForCausalLM`), and a SiLU shared-experts fix is
  applied at construction (see :mod:`easymagpie_vllm_omni.backbone_patches`).
* **Local transformer** — :class:`EasyMagpieCodePredictor`, a
  CUDA-graph-capturable implementation that runs as a single compiled graph.
* **compute_logits** — returns trivial logits so vLLM's sampler always picks
  index 0; the real audio output is the codes tensor surfaced through
  :meth:`make_omni_output` under the ``"audio_codes"`` key.

Text is embedded via a precomputed per-subword lookup table baked at
checkpoint-conversion time, so the char-aware subword encoder is never run
inside the engine.

Per-request I/O (via ``additional_information``):

* ``speaker_embedding`` (prefill only) — ``(T_audio, embedding_dim)``
  speaker-encoded context-audio embedding. ``preprocess`` assembles the full
  prefill context embedding itself as
  ``[task_embedding | speaker_embedding | context_text_embedded]``, so the
  caller only does the speaker-encoder math and passes plain context text (the
  model tokenizes + embeds it and prepends the per-mode service token).
* ``context_text`` (prefill only, optional) — plain conditioning string (e.g.
  ``"[EN]"``); tokenized in-model with the checkpoint's text tokenizer and
  embedded through the baked per-subword ``text_embedding`` table.
* ``task_mode_id`` (prefill only, optional) — int selecting the per-mode task
  ("service token") embedding row; defaults to ``0``. Ignored for single-mode
  checkpoints (no ``task_embedding`` table).

  The caller passes ``prompt_token_ids = [0] * T_ctx``, where ``T_ctx`` is the
  assembled context length (``[task?] + T_audio + len(tokenize(context_text))``).
* ``text`` (prefill only) — the plain target sentence to synthesize. This is the
  caller's text input: the model tokenizes it in-model at prefill with the
  checkpoint's text tokenizer (HF special tokens disabled, trailing text-EOS id
  appended), so callers never tokenize themselves. The resulting subword ids are
  consumed one per decode step (step ``k`` consumes id ``k``, embedded through
  the precomputed per-subword ``text_embedding`` table); once exhausted the text
  channel is masked off. (Internal: the tokenized ids are stashed as
  ``text_tokens`` in the per-request info dict between prefill and decode.)
* ``text_token`` (decode only, **streaming-text mode**) — when the caller omits
  ``text`` at prefill, the request runs in streaming-text mode: the caller pushes
  one subword id per decode step via ``additional_information`` under
  ``text_token`` (a single int / 1-element tensor), embedded through the same
  baked ``text_embedding`` table. This is the per-step counterpart to the whole
  ``text`` string and is driven by vLLM-Omni's streaming-input API (an async
  generator of ``StreamingInput`` chunks passed as the prompt, with
  ``async_chunk=True``). Push the text-EOS id as the last real token; on any step
  with no id (``text_token`` absent or ``< 0``, e.g. the sentinel ``-1``) the text
  channel is masked off so the caller can keep pumping decode steps while the
  audio tail finishes. Caller tokenization mirrors :meth:`_encode_text_stream`
  (``tokenizer.encode(text, add_special_tokens=False) + [text_eos_id]``).
* ``temperature`` / ``top_k`` (prefill only, optional) — audio sampling params
  for the local transformer. vLLM's ``SamplingParams.temperature`` drives only
  the dummy backbone token sampler, so the *audio* temperature/top-k are passed
  here and applied to the code predictor (defaults: ``0.7`` / ``80``).

Streaming delays: the text, phoneme and audio streams are temporally offset by
the checkpoint's ``streaming_phonemes_delay`` / ``streaming_speech_delay`` (baked
into ``config.json`` by the converter from the default inference mode). The text
stream runs from decode step 0; the phoneme stream opens at step
``phonemes_delay`` (seeded with phoneme BOS) and the audio stream at step
``speech_delay`` (seeded with audio BOS). The leading ``speech_delay`` decoded
frames are warm-up only and must be dropped by the caller. Delays of 0/0
reproduce a lock-step / non-delayed model.
"""
from __future__ import annotations

import bisect
import hashlib
import inspect
import os
import re
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn
from vllm import envs
from vllm.compilation.backends import set_model_tag
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import HasInnerState, IsHybrid

try:
    from vllm.model_executor.models.interfaces import SupportsMambaPrefixCaching
except ImportError:

    class SupportsMambaPrefixCaching:  # type: ignore[no-redef]
        """Compatibility marker for vLLM builds that do not expose the mixin."""

        pass
from vllm.model_executor.models.nemotron_h import NemotronHForCausalLM, NemotronHModel
try:
    from vllm.model_executor.models.mamba_cache import MambaCacheManager, MambaCacheParams

    _HAS_VLLM_MAMBA_CACHE_MANAGER = True
except ImportError:
    _HAS_VLLM_MAMBA_CACHE_MANAGER = False

    class MambaCacheParams:  # type: ignore[no-redef]
        """Small compatibility container for vLLM builds without mamba_cache."""

        def __init__(self, conv_state: torch.Tensor, ssm_state: torch.Tensor, state_indices_tensor: torch.Tensor):
            self.conv_state = conv_state
            self.ssm_state = ssm_state
            self.conv_states = conv_state
            self.ssm_states = ssm_state
            self.state_indices_tensor = state_indices_tensor

    class MambaCacheManager:  # type: ignore[no-redef]
        pass
from vllm.model_executor.models.utils import maybe_prefix
from vllm.sequence import IntermediateTensors
try:
    from vllm.utils import LayerBlockType
except ImportError:

    class LayerBlockType:  # type: ignore[no-redef]
        mamba = "mamba"

from vllm_omni.model_executor.models.output_templates import OmniOutput

from easymagpie_vllm_omni.backbone_patches import (
    patch_mamba_streaming_decode,
    patch_moe_routed_scale,
    patch_nemotron_h_moe_layer,
    patch_silu_shared_experts,
)
from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.local_transformer import EasyMagpieCodePredictor

logger = init_logger(__name__)

_MOE_EXPERT_WEIGHT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.mixer\.experts\.(?P<expert>\d+)\."
    r"(?P<projection>up_proj|down_proj)\.weight$"
)
_QKV_WEIGHT_RE = re.compile(
    r"^layers\.(?P<layer>\d+)\.mixer\.(?P<projection>q_proj|k_proj|v_proj)\.weight$"
)

# Placeholder token id stuffed into the per-step ``input_ids`` returned by
# ``preprocess`` — the model never consumes ``input_ids`` (decode behaviour is
# driven by the per-token buffers), and ``compute_logits`` returns
# argmax-at-0 dummy logits, so this only needs to be a valid id.
_DUMMY_TOKEN_ID = 0

# Context text used when the request omits ``context_text``
_DEFAULT_CONTEXT_TEXT = "[EN]"


def _apply_optional_nemotron_h_weight_mapper(
    weights: Iterable[tuple[str, torch.Tensor]],
) -> Iterable[tuple[str, torch.Tensor]]:
    mapper = getattr(NemotronHForCausalLM, "hf_to_vllm_mapper", None)
    apply = getattr(mapper, "apply", None)
    if apply is None:
        return weights
    return apply(weights)


# This class is not wrapped in ``@support_torch_compile``: the Nemotron-H
# backbone and :class:`EasyMagpieCodePredictor` each manage their own
# ``torch.compile`` / CUDA-graph capture internally, so the outer ``forward``
# runs eagerly and dispatches into the two self-compiled subgraphs.
class EasyMagpieTTSForConditionalGeneration(
    nn.Module,
    HasInnerState,
    IsHybrid,
    SupportsMambaPrefixCaching,
):
    """EasyMagpieTTS talker for vLLM-Omni.

    See the module docstring for the per-step flow and the per-request I/O
    contract. The class exposes the omni hooks (``has_preprocess`` /
    ``has_postprocess`` / ``have_multimodal_outputs``) consumed by the
    ``OmniGPUModelRunner``.
    """

    # Hybrid-Mamba bookkeeping (delegated to vLLM's NemotronH causal-LM). vLLM
    # expects these as class attributes.
    get_mamba_state_dtype_from_config = NemotronHForCausalLM.get_mamba_state_dtype_from_config
    get_mamba_state_shape_from_config = NemotronHForCausalLM.get_mamba_state_shape_from_config
    _get_mamba_state_copy_func = getattr(NemotronHForCausalLM, "get_mamba_state_copy_func", None)
    if _get_mamba_state_copy_func is not None:
        get_mamba_state_copy_func = _get_mamba_state_copy_func
    del _get_mamba_state_copy_func

    # Omni runner hooks.
    has_preprocess: bool = True
    has_postprocess: bool = True
    have_multimodal_outputs: bool = True

    # Keep small per-step tensors GPU-resident across steps (no D2H/H2D).
    gpu_resident_buffer_keys: set[str] = {
        "last_audio_codes",
        "last_phoneme_token",
        "last_hidden",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        hf_config = vllm_config.model_config.hf_config
        self.hf_config = hf_config
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.arch = EasyMagpieOmniArch.from_hf_config(hf_config)
        self.model_path = vllm_config.model_config.model

        arch = self.arch
        self.hidden_dim = arch.hidden_dim
        self.embedding_dim = arch.embedding_dim
        self.num_codebooks = arch.num_stacked_codebooks

        # ── Backbone (reused vLLM Nemotron-H LM; fed via inputs_embeds) ──
        patch_nemotron_h_moe_layer()
        self.backbone = NemotronHModel(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "backbone"),
        )
        self._backbone_accepts_mamba_cache_params = self._detect_backbone_mamba_cache_param()
        self.mamba_cache: Optional[MambaCacheManager] = None
        # The checkpoint was trained with mlp_hidden_act=silu but vLLM's
        # NemotronHMLP hard-codes ReLU² in shared_experts. Restore SiLU (no-op
        # when the backbone has no MoE layers).
        patch_silu_shared_experts(self.backbone)
        # vLLM's FusedMoE defers routed_scaling_factor to the decoder layer in
        # FP16, but NemotronH's decoder layer never compensates, so the MoE
        # output is under-scaled by routed_scaling_factor. Restore it (no-op in
        # fp32/bf16 and when there are no MoE layers).
        patch_moe_routed_scale(self.backbone)
        # The streaming-input path keeps extending the prompt, so vLLM's Mamba2
        # metadata builder would classify every single-token decode step as a
        # prefill — breaking the FULL decode cudagraph (stale
        # state_indices_tensor_d). Force single-token extends to classify as
        # decodes so FULL/FULL_DECODE_ONLY cudagraphs read the right Mamba slot.
        patch_mamba_streaming_decode()

        # ── Local transformer (its own compile group / CUDA graph) ──────
        with set_model_tag("local_transformer"):
            self.code_predictor = EasyMagpieCodePredictor(
                vllm_config=vllm_config,
                prefix=maybe_prefix(prefix, "code_predictor"),
            )

        # ── Text + phoneme embedding heads ──────────────────────────────
        # Precomputed per-subword text embedding (one row per subword id), baked
        # at conversion time and fed additively on every decode step.
        text_vocab_size = int(getattr(hf_config, "text_vocab_size", getattr(hf_config, "vocab_size", 0)))
        self.text_embedding = nn.Embedding(text_vocab_size, self.embedding_dim)
        # PyTorch can use a distinct text-embedding path for context text on
        # legacy checkpoints. Most current checkpoints use the target-text table
        # for context too, so keep that memory-neutral unless config.json
        # explicitly requests a separate baked context table.
        if bool(getattr(hf_config, "context_text_embedding_distinct", False)):
            self.context_text_embedding = nn.Embedding(text_vocab_size, self.embedding_dim)
        else:
            self.context_text_embedding = self.text_embedding

        # Text-stream EOS id — the last-but-one row of the text vocab, matching
        # the reference ``EasyMagpieTTSInferenceModel.eos_id = num_tokens - 2``.
        # Appended to the in-model-tokenized target text stream (see
        # :meth:`_encode_text_stream`).
        self.text_eos_id = text_vocab_size - 2

        # Task ("service token") embedding — a single learned per-mode row
        # prepended to the prefill context for multi-mode checkpoints. Built only
        # when the checkpoint carries one; otherwise ``None``.
        self.num_task_embeddings = int(arch.num_task_embeddings)
        if self.num_task_embeddings > 0:
            self.task_embedding = nn.Embedding(self.num_task_embeddings, self.embedding_dim)
        else:
            self.task_embedding = None

        # Context-text tokenizer, loaded lazily from the model directory. It
        # turns the per-request ``context_text`` string (e.g. ``"[EN]"``) into the
        # subword ids that the baked ``text_embedding`` table consumes — so the
        # caller passes plain text, never pre-tokenized ids.
        self._text_tokenizer: Any = None

        # ── Streaming delays (text leads phoneme by ``phonemes_delay`` and audio
        # by ``speech_delay`` decode steps; 0/0 == lock-step). ──
        self.phonemes_delay = int(getattr(arch, "streaming_phonemes_delay", 0) or 0)
        self.speech_delay = int(getattr(arch, "streaming_speech_delay", 0) or 0)

        # Phoneme channel (optional — only built when the checkpoint has one).
        self.has_phoneme = arch.phoneme_vocab_size > 0 and arch.phoneme_stacking_factor > 0
        if self.has_phoneme:
            self.phoneme_embeddings = nn.ModuleList(
                [nn.Embedding(arch.phoneme_vocab_size, self.embedding_dim) for _ in range(arch.phoneme_stacking_factor)]
            )
            self.phoneme_final_proj = nn.Linear(
                self.hidden_dim, arch.phoneme_vocab_size * arch.phoneme_stacking_factor
            )
            # Phoneme special-token ids + confidence→UNK replacement threshold.
            self.phoneme_bos_id = int(arch.resolved_phoneme_bos_id)
            self.phoneme_eos_id = int(arch.resolved_phoneme_eos_id)
            self.phoneme_unk_id = int(arch.resolved_phoneme_unk_id)
            self.phoneme_confidence_unk_threshold = float(arch.phoneme_confidence_unk_threshold)

        # ── Persistent, address-stable scratch buffers ─────────────────
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        dtype = vllm_config.model_config.dtype
        # Combined per-token input embedding fed into the backbone.
        self._combined_embeddings = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        self._debug_combined_input_norm = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._debug_combined_input_vector = torch.zeros(max_num_tokens, self.embedding_dim, dtype=dtype)
        self._debug_text_emb_norm = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._debug_phoneme_emb_norm = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._debug_audio_emb_norm = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._debug_positions = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_input_ids = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_decode_dispatch_index = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_decode_dispatch_count = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_decode_offset = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_mamba_num_prefills = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_mamba_num_decodes = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_mamba_state_indices = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_mamba_cache_state_indices = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_mamba_exec_state_indices = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_attn_num_actual_tokens = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_attn_max_query_len = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_attn_max_seq_len = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_attn_seq_lens = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_attn_slot_mapping = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_audio_feedback_missing = torch.zeros(max_num_tokens, dtype=torch.long)
        self._debug_audio_input_code0 = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_phoneme_input_valid = torch.full((max_num_tokens,), -1, dtype=torch.long)
        self._debug_phoneme_input_token0 = torch.full((max_num_tokens,), -1, dtype=torch.long)
        # Per-token decode inputs assembled by ``preprocess``.
        self._dec_text_tokens = torch.zeros(max_num_tokens, dtype=torch.long)
        self._dec_text_mask = torch.zeros(max_num_tokens, dtype=torch.long)
        self._dec_audio_codes = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.long)
        self._dec_audio_valid = torch.zeros(max_num_tokens, dtype=torch.long)
        if self.has_phoneme:
            self._dec_phoneme_tokens = torch.zeros(
                max_num_tokens, arch.phoneme_stacking_factor, dtype=torch.long
            )
            self._dec_phoneme_valid = torch.zeros(max_num_tokens, dtype=torch.long)

        self._out_codes = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.long)
        self._out_code_logprobs = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.float32)
        self._out_code_sampling_logprobs = torch.zeros(max_num_tokens, self.num_codebooks, dtype=torch.float32)
        self._out_frame_logprobs = torch.zeros(max_num_tokens, dtype=torch.float32)
        self._debug_lt_top_ids = torch.full((max_num_tokens, self.num_codebooks, 5), -1, dtype=torch.long)
        self._debug_lt_top_values = torch.zeros(max_num_tokens, self.num_codebooks, 5, dtype=torch.float32)
        self._debug_outputs_enabled = False
        self._last_output_row_indices: Optional[torch.Tensor] = None

        # ── Audio-EOS → engine stop ─────────────────────────────────────
        # The model signals end-of-speech inside the audio codebooks.
        # To make vLLM terminate the request at the EOS frame,
        # we flags decode positions with ``audio_eos_id`` emit designated ``stop_token_id``
        # in ``compute_logits``.
        # Callers must pass ``SamplingParams(stop_token_ids=[stop_id])`` with
        # ``stop_id = audio_eos_stop_token_id(hf_config)``.
        self.audio_eos_id = int(arch.audio_eos_id)
        self._stop_token_id = self.audio_eos_stop_token_id(hf_config)
        # flags frames in which ``_out_codes`` contain ``audio_eos_id``
        self._token_stop = torch.zeros(max_num_tokens, dtype=torch.bool)
        # slice of ``token_stop`` based on ``logit_idx`` that can be used in
        # ``compute_logits``
        self._sample_stop = torch.zeros(max_num_tokens, dtype=torch.bool)
        self.min_content_audio_frames_before_eos = max(
            0,
            int(os.environ.get("EASYMAGPIE_MIN_CONTENT_AUDIO_FRAMES_BEFORE_EOS", "0") or 0),
        )

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def audio_eos_stop_token_id(hf_config: Any) -> int:
        """Backbone token id this model emits when audio EOS is reached.

        Audio end-of-speech lives in the codebooks, not the backbone token
        stream, so the dummy backbone vocab is repurposed as a 2-way stop
        signal: index ``0`` == "continue", the last index == "stop". Callers
        must pass ``SamplingParams(stop_token_ids=[this])``
        """
        return max(1, int(getattr(hf_config, "vocab_size", 2)) - 1)

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compatibility shim — unused at runtime (everything goes via inputs_embeds)."""
        return self.text_embedding(input_ids)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.get_input_embeddings(input_ids)

    def _embed_phoneme(self, phoneme_tokens: torch.Tensor) -> torch.Tensor:
        """Average the per-stack phoneme embeddings (``[num_tokens, S] -> [num_tokens, dim]``)."""
        acc = self.phoneme_embeddings[0](phoneme_tokens[:, 0])
        for s in range(1, len(self.phoneme_embeddings)):
            acc = acc + self.phoneme_embeddings[s](phoneme_tokens[:, s])
        return acc / len(self.phoneme_embeddings)

    # ------------------------------------------------------------------
    # Decode-token dispatch (which positions need the local transformer)
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_backbone_mamba_cache_param() -> bool:
        try:
            return "mamba_cache_params" in inspect.signature(NemotronHModel.forward).parameters
        except Exception:
            return False

    def _ensure_v0_mamba_cache(self) -> Optional[MambaCacheManager]:
        """Create the vLLM v0 Mamba cache lazily, matching NemotronHForCausalLM."""
        if not _HAS_VLLM_MAMBA_CACHE_MANAGER:
            return None
        if getattr(self, "mamba_cache", None) is not None:
            return self.mamba_cache
        if not hasattr(self, "vllm_config") or not hasattr(self, "model_config"):
            return None

        num_mamba_layers = self.model_config.get_num_layers_by_block_type(
            self.vllm_config.parallel_config,
            LayerBlockType.mamba,
        )
        mamba_state_shape = self.get_mamba_state_shape_from_config(
            self.vllm_config,
            use_v1=False,
        )
        mamba_state_dtype = self.get_mamba_state_dtype_from_config(self.vllm_config)
        self.mamba_cache = MambaCacheManager(
            self.vllm_config,
            num_mamba_layers,
            *mamba_state_shape,
            *mamba_state_dtype,
        )
        return self.mamba_cache

    @staticmethod
    def _metadata_token_count(attn_metadata: Any) -> Optional[int]:
        if attn_metadata is None:
            return None
        metas = list(attn_metadata.values()) if isinstance(attn_metadata, dict) else [attn_metadata]
        for metadata in metas:
            num_prefills = getattr(metadata, "num_prefills", None)
            num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
            if num_prefills is not None and num_decode_tokens is not None:
                return int(num_prefills or 0) + int(num_decode_tokens or 0)

            num_prefill_tokens = getattr(metadata, "num_prefill_tokens", None)
            if num_prefill_tokens is not None and num_decode_tokens is not None:
                return int(num_prefill_tokens or 0) + int(num_decode_tokens or 0)

            query_start_loc = getattr(metadata, "query_start_loc", None)
            if query_start_loc is not None:
                try:
                    if int(query_start_loc.numel()) > 0:
                        return int(query_start_loc[-1].item())
                except Exception:
                    pass
        return None

    @staticmethod
    def _mamba_cache_batch_size_from_metadata(attn_metadata: Any) -> Optional[int]:
        for metadata in EasyMagpieTTSForConditionalGeneration._iter_metadata(attn_metadata):
            state_indices = getattr(metadata, "state_indices_tensor", None)
            if isinstance(state_indices, torch.Tensor) and state_indices.numel() > 0:
                return int(state_indices.numel())
            block_table = getattr(metadata, "block_table_tensor", None)
            if isinstance(block_table, torch.Tensor) and block_table.numel() > 0:
                return int(block_table.shape[0])
            num_decodes = getattr(metadata, "num_decode_tokens", None)
            num_prefills = getattr(metadata, "num_prefills", None)
            if num_decodes is not None and num_prefills is not None:
                return int(num_decodes or 0) + int(num_prefills or 0)
        return None

    def _profile_mamba_cache_params_from_forward_context(
        self,
        batch_size: Optional[int] = None,
    ) -> Optional[MambaCacheParams]:
        mamba_cache = getattr(self, "mamba_cache", None)
        if mamba_cache is None:
            return None
        if batch_size is None:
            try:
                batch_size = self._metadata_token_count(get_forward_context().attn_metadata)
            except Exception:
                batch_size = None
        if batch_size is None:
            return None

        cache_tensors, state_indices_tensor = mamba_cache.get_seqlen_agnostic_capture_inputs(int(batch_size))
        return MambaCacheParams(
            cache_tensors[0],
            cache_tensors[1],
            state_indices_tensor,
        )

    def _current_mamba_cache_params(
        self,
        *,
        mamba_cache_params: Any = None,
        easymagpie_mamba_cache_batch_size: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        if mamba_cache_params is not None:
            return mamba_cache_params

        if kwargs.get("request_ids_to_seq_ids") is not None and kwargs.get("finished_requests_ids") is not None:
            mamba_cache = self._ensure_v0_mamba_cache()
            current_run_tensors = getattr(mamba_cache, "current_run_tensors", None) if mamba_cache is not None else None
            if current_run_tensors is not None:
                try:
                    return current_run_tensors(**kwargs)
                except Exception:
                    logger.exception("Failed to build EasyMagpie Mamba cache params from request ids")

        if easymagpie_mamba_cache_batch_size is not None:
            return self._profile_mamba_cache_params_from_forward_context(easymagpie_mamba_cache_batch_size)

        try:
            use_v1 = bool(getattr(envs, "VLLM_USE_V1", False))
        except Exception:
            use_v1 = False
        if use_v1:
            return self._profile_mamba_cache_params_from_forward_context(easymagpie_mamba_cache_batch_size)

        mamba_cache = self._ensure_v0_mamba_cache()
        if mamba_cache is None:
            return self._profile_mamba_cache_params_from_forward_context(easymagpie_mamba_cache_batch_size)
        current_run_tensors = getattr(mamba_cache, "current_run_tensors", None)
        if current_run_tensors is None:
            return self._profile_mamba_cache_params_from_forward_context(easymagpie_mamba_cache_batch_size)
        return current_run_tensors(**kwargs)

    @staticmethod
    def _select_query_layout(attn_metadata):
        """Return ``(max_query_len, query_start_loc)`` from heterogeneous metadata.

        The Nemotron-H backbone is hybrid, so ``attn_metadata`` is a per-layer
        dict mixing two metadata types:

        * **attention** layers carry standard metadata that exposes the
          batch-level ``max_query_len`` + ``query_start_loc`` (e.g.
          ``TritonAttentionMetadata``);
        * **Mamba2** layers carry ``Mamba2AttentionMetadata``, which has *no*
          ``max_query_len`` and splits the query layout into ``query_start_loc_p``
          / ``query_start_loc_d`` instead.

        Both are built from the same batch query layout, so we prefer any
        attention-layer metadata. As a fallback for a (hypothetical) attention-free
        backbone, we infer a decode-only batch from the Mamba2 ``num_prefills``
        counter. Returns ``(None, None)`` when the layout can't be determined.
        """
        metas = list(attn_metadata.values()) if isinstance(attn_metadata, dict) else [attn_metadata]

        # Preferred: an attention layer exposes the unified query layout.
        for m in metas:
            mql = getattr(m, "max_query_len", None)
            qsl = getattr(m, "query_start_loc", None)
            if mql is not None and qsl is not None:
                return int(mql), qsl

        # Fallback: Mamba2-only backbone. We can at least detect a decode-only
        # batch (every request contributes a single token) from the counters.
        for m in metas:
            if hasattr(m, "num_prefills") and hasattr(m, "num_decodes"):
                if int(getattr(m, "num_prefills", 0)) == 0:
                    return 1, None  # decode-only -> caller runs the LT everywhere
                break
        return None, None

    def _get_decode_idxs(self):
        """Return ``(decode_token_indices, num_requests)`` for code-predictor dispatch.

        * ``(None, 0)`` → run the local transformer on every token (profile /
          dummy run with no ``attn_metadata``, or a decode-only batch where
          ``max_query_len == 1``), so the captured CUDA graph covers every
          ``cudagraph_capture_sizes`` value.
        * ``(indices, num_requests)`` → run only on the listed decode positions
          (mixed prefill+decode batch). ``indices`` is padded to the next
          captured graph size; ``num_requests`` is the unpadded count.
        """
        ctx = get_forward_context()
        attn_metadata = ctx.attn_metadata
        if attn_metadata is None:
            return None, 0

        metas = list(attn_metadata.values()) if isinstance(attn_metadata, dict) else [attn_metadata]
        for metadata in metas:
            num_prefills = getattr(metadata, "num_prefills", None)
            num_decode_tokens = getattr(metadata, "num_decode_tokens", None)
            num_decodes = getattr(metadata, "num_decodes", None)
            if num_prefills is None or (num_decode_tokens is None and num_decodes is None):
                continue
            decode_count = num_decode_tokens if num_decode_tokens is not None else num_decodes
            if int(num_prefills or 0) > 0 and int(decode_count or 0) == 0:
                return torch.empty(0, dtype=torch.long, device=self._combined_embeddings.device), 0

        max_query_len, start_loc = self._select_query_layout(attn_metadata)

        # Decode-only batch (or layout unavailable) -> run the LT on every token.
        if max_query_len is None or max_query_len == 1 or start_loc is None:
            return None, 0

        tokens_per_req = start_loc[1:] - start_loc[:-1]
        is_decode = tokens_per_req == 1
        decode_token_indices = start_loc[:-1][is_decode]

        num_requests = decode_token_indices.shape[0]
        padded_num_requests = num_requests
        if self.vllm_config.compilation_config.cudagraph_mode != CUDAGraphMode.NONE:
            sizes = self.vllm_config.compilation_config.cudagraph_capture_sizes
            idx = bisect.bisect_left(sizes, num_requests)
            if idx < len(sizes):
                padded_num_requests = sizes[idx]
        if padded_num_requests != num_requests:
            decode_token_indices = torch.nn.functional.pad(
                decode_token_indices, (0, padded_num_requests - num_requests)
            )
        return decode_token_indices, num_requests

    @staticmethod
    def _iter_metadata(value: Any) -> Iterable[Any]:
        if isinstance(value, dict):
            for child in value.values():
                yield from EasyMagpieTTSForConditionalGeneration._iter_metadata(child)
        elif value is not None:
            yield value

    @staticmethod
    def _metadata_int(metadata: Any, *names: str) -> Optional[int]:
        for name in names:
            value = getattr(metadata, name, None)
            if value is None:
                continue
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                return int(value.detach().reshape(-1)[0].item())
            try:
                return int(value)
            except Exception:
                continue
        return None

    @staticmethod
    def _metadata_tensor(metadata: Any, *names: str) -> Optional[torch.Tensor]:
        for name in names:
            value = getattr(metadata, name, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                return value.detach().reshape(-1).to(dtype=torch.long)
        return None

    @staticmethod
    def _metadata_list_tensor(metadata: Any, *names: str, device: torch.device) -> Optional[torch.Tensor]:
        tensor = EasyMagpieTTSForConditionalGeneration._metadata_tensor(metadata, *names)
        if tensor is not None:
            return tensor.to(device=device)
        for name in names:
            value = getattr(metadata, name, None)
            if value is None:
                continue
            try:
                items = list(value)
            except TypeError:
                continue
            if items:
                return torch.tensor([int(item) for item in items], dtype=torch.long, device=device)
        return None

    def _record_state_indices_debug(
        self,
        target: torch.Tensor,
        state_indices: torch.Tensor,
        *,
        num_tokens: int,
        decode_idx: Optional[torch.Tensor] = None,
        num_req: int = 0,
    ) -> None:
        device = target.device
        state_indices = state_indices.detach().reshape(-1).to(device=device, dtype=torch.long)
        if decode_idx is not None and num_req > 0 and state_indices.numel() >= num_req:
            valid = decode_idx[:num_req].detach().to(device=device, dtype=torch.long).reshape(-1)
            target[valid].copy_(state_indices[:num_req])
        else:
            n = min(num_tokens, int(state_indices.numel()))
            target[:n].copy_(state_indices[:n])

    def _record_mamba_metadata_debug(
        self,
        num_tokens: int,
        *,
        decode_idx: Optional[torch.Tensor] = None,
        num_req: int = 0,
        mamba_cache_params: Any = None,
    ) -> None:
        try:
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            attn_metadata = None

        metas = list(self._iter_metadata(attn_metadata))
        mamba_metas = [
            meta for meta in metas if getattr(meta, "state_indices_tensor", None) is not None
        ]
        counter_metas = mamba_metas or metas
        rows = slice(0, num_tokens)

        for metadata in counter_metas:
            num_prefills = self._metadata_int(metadata, "num_prefills", "num_prefill_tokens")
            if num_prefills is not None:
                self._debug_mamba_num_prefills[rows].fill_(num_prefills)
                break
        for metadata in counter_metas:
            num_decodes = self._metadata_int(metadata, "num_decodes", "num_decode_tokens")
            if num_decodes is not None:
                self._debug_mamba_num_decodes[rows].fill_(num_decodes)
                break

        cache_state_indices = None
        for name in ("state_indices_tensor_d", "state_indices_tensor", "state_indices_tensor_p"):
            value = getattr(mamba_cache_params, name, None)
            if isinstance(value, torch.Tensor) and value.numel() > 0:
                cache_state_indices = value
                break
        if cache_state_indices is not None:
            self._record_state_indices_debug(
                self._debug_mamba_cache_state_indices,
                cache_state_indices,
                num_tokens=num_tokens,
                decode_idx=decode_idx,
                num_req=num_req,
            )

        exec_state_indices = None
        for metadata in mamba_metas:
            exec_state_indices = self._metadata_tensor(metadata, "state_indices_tensor")
            if exec_state_indices is not None:
                break
        if exec_state_indices is None:
            for metadata in metas:
                block_table = getattr(metadata, "block_table_tensor", None)
                if isinstance(block_table, torch.Tensor) and block_table.numel() > 0:
                    exec_state_indices = block_table[:, 0].detach().reshape(-1).to(dtype=torch.long)
                    break

        if exec_state_indices is not None:
            self._record_state_indices_debug(
                self._debug_mamba_exec_state_indices,
                exec_state_indices,
                num_tokens=num_tokens,
                decode_idx=decode_idx,
                num_req=num_req,
            )
            self._record_state_indices_debug(
                self._debug_mamba_state_indices,
                exec_state_indices,
                num_tokens=num_tokens,
                decode_idx=decode_idx,
                num_req=num_req,
            )
        elif cache_state_indices is not None:
            self._record_state_indices_debug(
                self._debug_mamba_state_indices,
                cache_state_indices,
                num_tokens=num_tokens,
                decode_idx=decode_idx,
                num_req=num_req,
            )

    def _record_debug_forward_metadata(
        self,
        *,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        decode_idx: Optional[torch.Tensor],
        num_req: int,
        mamba_cache_params: Any,
    ) -> None:
        if not self._debug_outputs_enabled:
            return

        num_tokens = int(input_ids.shape[0])
        rows = slice(0, num_tokens)
        device = self._debug_positions.device
        self._debug_positions[rows].copy_(positions.detach().to(device=device, dtype=torch.long).reshape(-1)[:num_tokens])
        self._debug_input_ids[rows].copy_(input_ids.detach().to(device=device, dtype=torch.long).reshape(-1)[:num_tokens])
        self._debug_decode_dispatch_count[rows].fill_(int(num_req))
        if decode_idx is None:
            self._debug_decode_dispatch_index[rows].copy_(torch.arange(num_tokens, dtype=torch.long, device=device))
        elif num_req > 0:
            valid = decode_idx[:num_req].detach().to(device=device, dtype=torch.long).reshape(-1)
            self._debug_decode_dispatch_index[valid].copy_(valid)

        try:
            attn_metadata = get_forward_context().attn_metadata
        except Exception:
            attn_metadata = None
        metas = list(self._iter_metadata(attn_metadata))
        self._record_mamba_metadata_debug(
            num_tokens,
            decode_idx=decode_idx,
            num_req=num_req,
            mamba_cache_params=mamba_cache_params,
        )
        for metadata in metas:
            num_actual = self._metadata_int(metadata, "num_actual_tokens")
            if num_actual is not None:
                self._debug_attn_num_actual_tokens[rows].fill_(num_actual)
                break
        for metadata in metas:
            max_query_len = self._metadata_int(metadata, "max_query_len")
            if max_query_len is not None:
                self._debug_attn_max_query_len[rows].fill_(max_query_len)
                break
        for metadata in metas:
            max_seq_len = self._metadata_int(metadata, "max_seq_len", "max_seq_len_q")
            if max_seq_len is not None:
                self._debug_attn_max_seq_len[rows].fill_(max_seq_len)
                break
        for metadata in metas:
            seq_lens = self._metadata_list_tensor(metadata, "seq_lens", "seq_lens_tensor", device=device)
            if seq_lens is not None:
                n = min(num_tokens, int(seq_lens.numel()))
                self._debug_attn_seq_lens[:n].copy_(seq_lens[:n])
                break
        for metadata in metas:
            slot_mapping = self._metadata_tensor(metadata, "slot_mapping")
            if slot_mapping is not None:
                slot_mapping = slot_mapping.to(device=device)
                n = min(num_tokens, int(slot_mapping.numel()))
                self._debug_attn_slot_mapping[:n].copy_(slot_mapping[:n])
                break

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Assemble the per-token embedding, run the backbone, then the codes.

        ``inputs_embeds`` carries the prefill embedding span produced by
        :meth:`preprocess` (zeros at decode positions). For decode positions we
        assemble ``text_emb + phoneme_emb + audio_emb`` in-place from the
        per-token buffers, run the backbone, then sample the codebooks with the
        local transformer (skipping prefill positions).
        """
        num_tokens = input_ids.shape[0]
        combined = self._combined_embeddings[:num_tokens]
        if inputs_embeds is not None:
            combined.copy_(inputs_embeds)
        else:
            combined.zero_()

        # Reset per-token stop flags for this step (so prefill / warm-up rows stay
        # "continue"); decode positions get set below by :meth:`_flag_audio_eos`.
        self._token_stop[:num_tokens].zero_()
        self._out_code_logprobs[:num_tokens].zero_()
        self._out_code_sampling_logprobs[:num_tokens].zero_()
        self._out_frame_logprobs[:num_tokens].zero_()
        logits_index = kwargs.get("logits_index")
        if isinstance(logits_index, torch.Tensor) and logits_index.numel() > 0:
            self._last_output_row_indices = logits_index.detach().reshape(-1).to(dtype=torch.long)
        else:
            self._last_output_row_indices = None

        decode_idx, num_req = self._get_decode_idxs()

        if decode_idx is None:
            # Profile / dummy run or decode-only batch: assemble decode
            # embeddings everywhere so the captured graph sees the full path.
            self._assemble_decode_embeddings(combined, slice(0, num_tokens))
        elif num_req > 0:
            valid = decode_idx[:num_req]
            self._assemble_decode_embeddings(combined, valid)

        if self._debug_outputs_enabled:
            debug_rows = slice(0, num_tokens)
            self._debug_combined_input_vector[debug_rows].copy_(combined.detach())
            self._debug_combined_input_norm[debug_rows] = combined.detach().float().norm(dim=-1)
        self.code_predictor.debug_collect_logits = bool(self._debug_outputs_enabled)

        backbone_kwargs = {
            "input_ids": input_ids,
            "positions": positions,
            "intermediate_tensors": intermediate_tensors,
            "inputs_embeds": combined,
        }
        mamba_cache_params = None
        if self._backbone_accepts_mamba_cache_params:
            mamba_cache_params = self._current_mamba_cache_params(**kwargs)
            backbone_kwargs["mamba_cache_params"] = mamba_cache_params

        self._record_debug_forward_metadata(
            input_ids=input_ids,
            positions=positions,
            decode_idx=decode_idx,
            num_req=int(num_req),
            mamba_cache_params=mamba_cache_params,
        )

        hidden_states = self.backbone(**backbone_kwargs)

        # Sample codes (local transformer) only where needed.
        if decode_idx is None:
            codes, code_logprobs, sampling_logprobs = self.code_predictor.generate_codes_with_logprobs(hidden_states)
            self._out_codes[:num_tokens].copy_(codes)
            self._out_code_logprobs[:num_tokens].copy_(code_logprobs)
            self._out_code_sampling_logprobs[:num_tokens].copy_(sampling_logprobs)
            self._out_frame_logprobs[:num_tokens].copy_(code_logprobs.sum(dim=-1))
            if self._debug_outputs_enabled:
                self._debug_lt_top_ids[:num_tokens].copy_(self.code_predictor._debug_top_ids[:num_tokens])
                self._debug_lt_top_values[:num_tokens].copy_(self.code_predictor._debug_top_values[:num_tokens])
            self._flag_audio_eos(codes, slice(0, num_tokens))
            if self.has_phoneme:
                self._predict_phonemes(hidden_states, slice(0, num_tokens))
        elif num_req > 0:
            ctx = get_forward_context()
            orig_bd = ctx.batch_descriptor
            ctx.batch_descriptor = BatchDescriptor(num_tokens=decode_idx.shape[0])
            codes, code_logprobs, sampling_logprobs = self.code_predictor.generate_codes_with_logprobs(
                hidden_states[decode_idx]
            )
            ctx.batch_descriptor = orig_bd
            valid = decode_idx[:num_req]
            self._out_codes[valid] = codes[:num_req]
            self._out_code_logprobs[valid] = code_logprobs[:num_req]
            self._out_code_sampling_logprobs[valid] = sampling_logprobs[:num_req]
            self._out_frame_logprobs[valid] = code_logprobs[:num_req].sum(dim=-1)
            if self._debug_outputs_enabled:
                self._debug_lt_top_ids[valid] = self.code_predictor._debug_top_ids[:num_req]
                self._debug_lt_top_values[valid] = self.code_predictor._debug_top_values[:num_req]
            self._flag_audio_eos(codes[:num_req], valid)
            if self.has_phoneme:
                self._predict_phonemes(hidden_states, valid)

        # Re-index _token_stop into _sample_stop.
        # this only happens for mixed/prefill, since for capture logits_index is None,
        # so during decode-only the branch for logits_index is None will be executed.        
        if logits_index is not None:
            self._sample_stop[:logits_index.shape[0]] = self._token_stop[logits_index]
        else:
            self._sample_stop[:num_tokens].copy_(self._token_stop[:num_tokens])

        return hidden_states

    def _flag_audio_eos(self, codes: torch.Tensor, idx) -> None:
        """Flag decode positions whose newly sampled frame ends speech.
        Checks codes for eos and assigns token_stop[idx]

        Note: this uses the *sampled* codes. NeMo also checks armax(logits) == eos_idx,
        i.e. checks if EOS is emited without sampling. Skip for now.
        """
        eos = (codes == self.audio_eos_id).any(dim=1) & (self._dec_audio_valid[idx] == 1)
        min_content_frames = int(getattr(self, "min_content_audio_frames_before_eos", 0) or 0)
        if min_content_frames > 0:
            decode_offset = self._debug_decode_offset[idx].to(device=eos.device)
            previous_content_frames = decode_offset - int(self.speech_delay)
            eos = eos & (previous_content_frames >= min_content_frames)
        self._token_stop[idx] = eos

    def _assemble_decode_embeddings(self, combined: torch.Tensor, idx) -> None:
        """Add ``text + phoneme + audio`` embeddings into ``combined`` at ``idx``."""
        # Mixed prefill/decode batches carry real prefill rows in ``inputs_embeds``.
        # Decode rows must start empty; otherwise a stale context/prefill vector
        # is added to the streaming text/phoneme/audio components. Build a fresh
        # tensor and assign it back because ``idx`` is often a LongTensor; in
        # that case ``combined[idx].zero_()`` would only clear an advanced-index
        # copy.
        assembled = torch.zeros_like(combined[idx])

        # Audio: previous-frame codes (gated by validity).
        audio_codes = self._dec_audio_codes[idx]
        audio_emb = self.code_predictor.embed_audio_frame(audio_codes)
        audio_emb = audio_emb * self._dec_audio_valid[idx].unsqueeze(-1).to(audio_emb.dtype)
        self._debug_audio_emb_norm[idx] = audio_emb.float().norm(dim=-1)
        assembled += audio_emb

        # Text: current subword token (gated by validity).
        text_emb = self.text_embedding(self._dec_text_tokens[idx])
        text_emb = text_emb * self._dec_text_mask[idx].unsqueeze(-1).to(text_emb.dtype)
        self._debug_text_emb_norm[idx] = text_emb.float().norm(dim=-1)
        assembled += text_emb

        # Phoneme: previous predicted phoneme (gated by validity).
        if self.has_phoneme:
            phon_emb = self._embed_phoneme(self._dec_phoneme_tokens[idx])
            phon_emb = phon_emb * self._dec_phoneme_valid[idx].unsqueeze(-1).to(phon_emb.dtype)
            self._debug_phoneme_emb_norm[idx] = phon_emb.float().norm(dim=-1)
            assembled += phon_emb
        else:
            self._debug_phoneme_emb_norm[idx].zero_()
        combined[idx] = assembled

    @torch.no_grad()
    def _predict_phonemes(self, hidden_states: torch.Tensor, idx) -> None:
        """Argmax the phoneme head (with confidence→UNK replacement) and stash it.

        The UNK replacement mirrors the reference: when the max phoneme
        probability of any stacked channel falls below
        ``phoneme_confidence_unk_threshold`` (and the step is not an EOS step),
        the whole step is replaced with the UNK id to curb error propagation.

        This is done here — not in ``preprocess``/``postprocess`` — because this
        is the only place the phoneme logits exist (preprocess has no logits, and
        postprocess only sees the argmax id). It uses only elementwise ops +
        ``torch.where`` (no ``.item()`` / host sync), so it stays CUDA-graph safe.
        """
        # Run in the model dtype (don't force fp32): ``phoneme_final_proj`` weights
        # follow ``model_config.dtype`` (e.g. bf16), and argmax is dtype-insensitive,
        # so an fp32 upcast here would mismatch the weight dtype in ``F.linear``.
        logits = self.phoneme_final_proj(hidden_states[idx])
        s = self.arch.phoneme_stacking_factor
        logits = logits.view(-1, s, self.arch.phoneme_vocab_size)
        preds = logits.argmax(dim=-1).long()  # (n, S)

        if self.phoneme_confidence_unk_threshold > 0.0:
            max_probs = torch.softmax(logits.float(), dim=-1).amax(dim=-1)  # (n, S)
            underconfident = (max_probs < self.phoneme_confidence_unk_threshold).any(dim=1, keepdim=True)
            eos_step = (preds == self.phoneme_eos_id).any(dim=1, keepdim=True)
            replace = underconfident & (~eos_step)
            preds = torch.where(replace, torch.full_like(preds, self.phoneme_unk_id), preds)

        self._dec_phoneme_tokens[idx] = preds
        self._dec_phoneme_valid[idx] = 1

    # ------------------------------------------------------------------
    # compute_logits — dummy (real output is the codes tensor)
    # ------------------------------------------------------------------

    def compute_logits(self, hidden_states, sampling_metadata: Any = None) -> Optional[torch.Tensor]:
        f"""Dummy backbone logits, repurposed as a 2-way continue/stop signal.
        ``_sample_stop`` indicates which frames contain EOS. We set logits,
        based on that: logits[sample_stop == True, stop_token_id] = 30 or -30 otherwise.
        SamplingParams should set stop_token_id as EOS token though.
        """
        if isinstance(hidden_states, OmniOutput):
            hidden_states = hidden_states.text_hidden_states
        if hidden_states is None:
            return None
        batch_size = hidden_states.shape[0]
        logits = hidden_states.new_zeros(batch_size, int(self.hf_config.vocab_size))
        if self._stop_token_id < logits.shape[1]:
            stop_rows = self._sample_stop[:batch_size]
            logits[:, self._stop_token_id] = torch.where(
                stop_rows,
                logits.new_full((), 30.0),
                logits.new_full((), -30.0),
            )
        return logits

    # ------------------------------------------------------------------
    # multimodal output plumbing
    # ------------------------------------------------------------------

    def make_omni_output(self, model_outputs, **_: Any) -> OmniOutput:
        """Surface the sampled codes (``BT x num_codebooks``) under ``audio_codes``."""
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        num_tokens = int(hidden.shape[0])
        row_indices = self._row_indices_for_hidden(
            hidden,
            int(self._out_codes.shape[0]),
            self._out_codes.device,
            explicit_row_indices=getattr(self, "_last_output_row_indices", None),
        )
        audio_codes = self._out_codes.index_select(0, row_indices).clone()
        audio_code_logprobs = self._out_code_logprobs.index_select(0, row_indices).clone()
        audio_code_sampling_logprobs = self._out_code_sampling_logprobs.index_select(0, row_indices).clone()
        audio_frame_logprobs = self._out_frame_logprobs.index_select(0, row_indices).clone()
        multimodal_outputs: dict[str, torch.Tensor] = {
            "audio_codes": audio_codes,
            "audio_codes_feedback": audio_codes,
            "audio_code_logprobs": audio_code_logprobs,
            "audio_code_sampling_logprobs": audio_code_sampling_logprobs,
            "audio_frame_logprobs": audio_frame_logprobs,
        }
        if self.has_phoneme:
            multimodal_outputs["phoneme_tokens_feedback"] = self._dec_phoneme_tokens.index_select(
                0,
                row_indices.to(self._dec_phoneme_tokens.device),
            ).clone()
        if self._debug_outputs_enabled:
            combined = self._combined_embeddings.index_select(
                0,
                row_indices.to(self._combined_embeddings.device),
            ).clone()
            hidden_debug = hidden[:num_tokens].clone()
            multimodal_outputs.update(
                {
                    "debug_text_mask": self._dec_text_mask.index_select(
                        0,
                        row_indices.to(self._dec_text_mask.device),
                    ).clone(),
                    "debug_text_tokens": self._dec_text_tokens.index_select(
                        0,
                        row_indices.to(self._dec_text_tokens.device),
                    ).clone(),
                    "debug_audio_valid": self._dec_audio_valid.index_select(
                        0,
                        row_indices.to(self._dec_audio_valid.device),
                    ).clone(),
                    "debug_text_emb_norm": self._debug_text_emb_norm.index_select(
                        0,
                        row_indices.to(self._debug_text_emb_norm.device),
                    ).clone(),
                    "debug_phoneme_emb_norm": self._debug_phoneme_emb_norm.index_select(
                        0,
                        row_indices.to(self._debug_phoneme_emb_norm.device),
                    ).clone(),
                    "debug_audio_emb_norm": self._debug_audio_emb_norm.index_select(
                        0,
                        row_indices.to(self._debug_audio_emb_norm.device),
                    ).clone(),
                    "debug_positions": self._debug_positions.index_select(
                        0,
                        row_indices.to(self._debug_positions.device),
                    ).clone(),
                    "debug_input_ids": self._debug_input_ids.index_select(
                        0,
                        row_indices.to(self._debug_input_ids.device),
                    ).clone(),
                    "debug_decode_dispatch_index": self._debug_decode_dispatch_index.index_select(
                        0,
                        row_indices.to(self._debug_decode_dispatch_index.device),
                    ).clone(),
                    "debug_decode_dispatch_count": self._debug_decode_dispatch_count.index_select(
                        0,
                        row_indices.to(self._debug_decode_dispatch_count.device),
                    ).clone(),
                    "debug_combined_input_norm": self._debug_combined_input_norm.index_select(
                        0,
                        row_indices.to(self._debug_combined_input_norm.device),
                    ).clone(),
                    "debug_combined_input_vector": self._debug_combined_input_vector.index_select(
                        0,
                        row_indices.to(self._debug_combined_input_vector.device),
                    ).clone(),
                    "debug_combined_norm": combined.float().norm(dim=-1),
                    "debug_combined_vector": combined,
                    "debug_hidden_norm": hidden_debug.float().norm(dim=-1),
                    "debug_hidden_vector": hidden_debug,
                    "debug_audio_input_code0": self._debug_audio_input_code0.index_select(
                        0,
                        row_indices.to(self._debug_audio_input_code0.device),
                    ).clone(),
                    "debug_audio_output_code0": audio_codes[:, 0].clone(),
                    "debug_decode_offset": self._debug_decode_offset.index_select(
                        0,
                        row_indices.to(self._debug_decode_offset.device),
                    ).clone(),
                    "debug_audio_feedback_missing": self._debug_audio_feedback_missing.index_select(
                        0,
                        row_indices.to(self._debug_audio_feedback_missing.device),
                    ).clone(),
                    "debug_phoneme_input_valid": self._debug_phoneme_input_valid.index_select(
                        0,
                        row_indices.to(self._debug_phoneme_input_valid.device),
                    ).clone(),
                    "debug_phoneme_input_token0": self._debug_phoneme_input_token0.index_select(
                        0,
                        row_indices.to(self._debug_phoneme_input_token0.device),
                    ).clone(),
                    "debug_mamba_num_prefills": self._debug_mamba_num_prefills.index_select(
                        0,
                        row_indices.to(self._debug_mamba_num_prefills.device),
                    ).clone(),
                    "debug_mamba_num_decodes": self._debug_mamba_num_decodes.index_select(
                        0,
                        row_indices.to(self._debug_mamba_num_decodes.device),
                    ).clone(),
                    "debug_mamba_state_indices": self._debug_mamba_state_indices.index_select(
                        0,
                        row_indices.to(self._debug_mamba_state_indices.device),
                    ).clone(),
                    "debug_mamba_cache_state_indices": self._debug_mamba_cache_state_indices.index_select(
                        0,
                        row_indices.to(self._debug_mamba_cache_state_indices.device),
                    ).clone(),
                    "debug_mamba_exec_state_indices": self._debug_mamba_exec_state_indices.index_select(
                        0,
                        row_indices.to(self._debug_mamba_exec_state_indices.device),
                    ).clone(),
                    "debug_attn_num_actual_tokens": self._debug_attn_num_actual_tokens.index_select(
                        0,
                        row_indices.to(self._debug_attn_num_actual_tokens.device),
                    ).clone(),
                    "debug_attn_max_query_len": self._debug_attn_max_query_len.index_select(
                        0,
                        row_indices.to(self._debug_attn_max_query_len.device),
                    ).clone(),
                    "debug_attn_max_seq_len": self._debug_attn_max_seq_len.index_select(
                        0,
                        row_indices.to(self._debug_attn_max_seq_len.device),
                    ).clone(),
                    "debug_attn_seq_lens": self._debug_attn_seq_lens.index_select(
                        0,
                        row_indices.to(self._debug_attn_seq_lens.device),
                    ).clone(),
                    "debug_attn_slot_mapping": self._debug_attn_slot_mapping.index_select(
                        0,
                        row_indices.to(self._debug_attn_slot_mapping.device),
                    ).clone(),
                    "debug_lt_top_ids": self._debug_lt_top_ids.index_select(
                        0,
                        row_indices.to(self._debug_lt_top_ids.device),
                    ).clone(),
                    "debug_lt_top_values": self._debug_lt_top_values.index_select(
                        0,
                        row_indices.to(self._debug_lt_top_values.device),
                    ).clone(),
                }
            )
            if self.has_phoneme:
                phoneme_indices = row_indices.to(self._dec_phoneme_valid.device)
                multimodal_outputs.update(
                    {
                        "debug_phoneme_valid": self._dec_phoneme_valid.index_select(0, phoneme_indices).clone(),
                        "debug_phoneme_input_token0": self._dec_phoneme_tokens.index_select(
                            0,
                            row_indices.to(self._dec_phoneme_tokens.device),
                        )[:, 0].clone(),
                    }
                )
        multimodal_outputs = {
            name: tensor.contiguous() if isinstance(tensor, torch.Tensor) else tensor
            for name, tensor in multimodal_outputs.items()
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    @staticmethod
    def _row_indices_for_hidden(
        hidden_states: torch.Tensor,
        row_count: int,
        device: torch.device,
        *,
        explicit_row_indices: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return full-batch row indices matching a request-local hidden slice."""
        num_rows = int(hidden_states.shape[0])
        if num_rows <= 0 or row_count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if isinstance(explicit_row_indices, torch.Tensor) and int(explicit_row_indices.numel()) == num_rows:
            return explicit_row_indices.detach().reshape(-1).to(device=device, dtype=torch.long)

        stride0 = hidden_states.stride(0) or 1
        start = int(hidden_states.storage_offset()) // int(stride0)
        if 0 <= start and start + num_rows <= row_count:
            return torch.arange(start, start + num_rows, dtype=torch.long, device=device)

        return torch.arange(0, min(num_rows, row_count), dtype=torch.long, device=device)

    @staticmethod
    def _last_request_row_index(
        hidden_states: torch.Tensor,
        row_tensor: torch.Tensor,
        explicit_row_indices: Optional[torch.Tensor] = None,
    ) -> int:
        """Map a request slice to either request-local or full-batch row coordinates."""
        local_last = int(hidden_states.shape[0]) - 1
        if int(row_tensor.shape[0]) == int(hidden_states.shape[0]):
            return local_last

        if (
            isinstance(explicit_row_indices, torch.Tensor)
            and int(explicit_row_indices.numel()) == int(hidden_states.shape[0])
        ):
            explicit_last = int(explicit_row_indices.detach().reshape(-1)[local_last].item())
            if 0 <= explicit_last < int(row_tensor.shape[0]):
                return explicit_last

        stride0 = hidden_states.stride(0) or 1
        storage_last = hidden_states.storage_offset() // stride0 + local_last
        if 0 <= storage_last < int(row_tensor.shape[0]):
            return int(storage_last)

        return min(max(local_last, 0), int(row_tensor.shape[0]) - 1)

    # ------------------------------------------------------------------
    # preprocess / postprocess
    # ------------------------------------------------------------------

    @staticmethod
    def _first_str(value: Any) -> str:
        """Return the first element of a list-wrapped scalar, or the scalar itself, as a string."""
        if isinstance(value, list):
            return str(value[0]) if value else ""
        if value is None:
            return ""
        return str(value)

    @staticmethod
    def _coerce_opt_int(value: Any) -> Optional[int]:
        """Best-effort extract a single int from a scalar / list / tensor / str.

        Used to read a per-step streamed ``text_token`` out of the request's
        ``additional_information`` (which may wrap the id as a list, a 1-element
        tensor, or a string depending on how the caller / transport packed it).
        Returns ``None`` when no usable integer is present.
        """
        if value is None:
            return None
        if isinstance(value, bool):  # bool is an int subclass — handle explicitly.
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, torch.Tensor):
            return int(value.reshape(-1)[0].item()) if value.numel() > 0 else None
        if isinstance(value, (list, tuple)):
            return EasyMagpieTTSForConditionalGeneration._coerce_opt_int(value[0]) if value else None
        if isinstance(value, str):
            try:
                return int(value.strip())
            except ValueError:
                return None
        return None

    @staticmethod
    def _coerce_int_list(value: Any) -> Optional[list[int]]:
        """Best-effort normalize a caller-provided 1-D token list."""
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
        if isinstance(value, (list, tuple)):
            out: list[int] = []
            for item in value:
                parsed = EasyMagpieTTSForConditionalGeneration._coerce_opt_int(item)
                if parsed is not None:
                    out.append(int(parsed))
            return out
        parsed = EasyMagpieTTSForConditionalGeneration._coerce_opt_int(value)
        return [int(parsed)] if parsed is not None else None

    @staticmethod
    def _coerce_int_rows(value: Any, row_width: int) -> Optional[list[list[int]]]:
        """Best-effort normalize caller-provided 2-D token rows."""
        if value is None:
            return None
        width = max(1, int(row_width))
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu().long()
            if tensor.ndim <= 1:
                return [EasyMagpieTTSForConditionalGeneration._coerce_int_list(tensor) or []]
            return [[int(item) for item in row[:width].reshape(-1).tolist()] for row in tensor.reshape(-1, tensor.shape[-1])]
        if isinstance(value, (list, tuple)):
            if not value:
                return []
            first = value[0]
            if isinstance(first, (list, tuple, torch.Tensor)):
                rows: list[list[int]] = []
                for row in value:
                    parsed = EasyMagpieTTSForConditionalGeneration._coerce_int_list(row)
                    rows.append((parsed or [])[:width])
                return rows
            parsed = EasyMagpieTTSForConditionalGeneration._coerce_int_list(value)
            return [(parsed or [])[:width]]
        parsed = EasyMagpieTTSForConditionalGeneration._coerce_int_list(value)
        return [(parsed or [])[:width]] if parsed is not None else None

    def _clear_runtime_rows(self, start: int, end: int) -> None:
        """Clear mutable per-row state before reusing prefill/decode slots."""
        row = slice(max(0, int(start)), max(0, int(end)))
        zero_names = (
            "_dec_text_tokens",
            "_dec_text_mask",
            "_dec_audio_codes",
            "_dec_audio_valid",
            "_dec_phoneme_tokens",
            "_dec_phoneme_valid",
            "_out_codes",
            "_out_code_logprobs",
            "_out_code_sampling_logprobs",
            "_out_frame_logprobs",
            "_debug_lt_top_ids",
            "_debug_lt_top_values",
            "_token_stop",
            "_sample_stop",
            "_debug_combined_input_norm",
            "_debug_combined_input_vector",
            "_debug_text_emb_norm",
            "_debug_phoneme_emb_norm",
            "_debug_audio_emb_norm",
            "_debug_combined_pre_norm",
            "_debug_hidden_norm",
            "_debug_backbone_last_layer_norm",
            "_debug_backbone_last_residual_norm",
            "_debug_final_norm_input_norm",
            "_debug_final_norm_residual_norm",
            "_debug_final_norm_output_norm",
            "_debug_attn_qkv_norm",
            "_debug_attn_core_norm",
            "_debug_attn_output_norm",
            "_debug_audio_feedback_missing",
        )
        minus_one_names = (
            "_debug_positions",
            "_debug_input_ids",
            "_debug_decode_dispatch_index",
            "_debug_decode_dispatch_count",
            "_debug_backbone_first_bad_layer",
            "_debug_backbone_first_bad_residual_layer",
            "_debug_mamba_num_prefills",
            "_debug_mamba_num_decodes",
            "_debug_mamba_state_indices",
            "_debug_mamba_cache_state_indices",
            "_debug_mamba_exec_state_indices",
            "_debug_attn_num_actual_tokens",
            "_debug_attn_max_query_len",
            "_debug_attn_max_seq_len",
            "_debug_attn_seq_lens",
            "_debug_attn_slot_mapping",
            "_debug_decode_offset",
            "_debug_audio_input_code0",
            "_debug_phoneme_input_valid",
            "_debug_phoneme_input_token0",
        )
        for name in zero_names:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                value[row].zero_()
        for name in minus_one_names:
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                value[row].fill_(-1)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: Optional[torch.Tensor],
        *,
        start: int = 0,
        end: int = 0,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build per-request ``(input_ids, inputs_embeds)`` for this step.

        Prefill (``span_len > 1``): assemble the full context embedding
        (``[task_embedding | speaker_embedding | context_text_embedded]`` from
        the per-request inputs; see :meth:`_build_prefill_embeds`), slice this
        chunk out of it, and return it;
        ``input_ids`` are placeholders. Decode (``span_len == 1``): write the per-token decode
        inputs (previous codes, current text token, previous phoneme) into the
        model buffers at ``start`` and return a zero embedding that
        :meth:`forward` accumulates into.
        """
        nested = info_dict.get("additional_information")
        if isinstance(nested, dict):
            merged = {k: v for k, v in info_dict.items() if k != "additional_information"}
            for k, v in nested.items():
                merged.setdefault(k, v)
            info_dict = merged

        device = input_ids.device
        span_len = int(input_ids.shape[0])
        if span_len <= 0:
            base = input_embeds if input_embeds is not None else self.embed_input_ids(input_ids)
            return input_ids, base, {}

        if span_len > 1:
            return self._preprocess_prefill(input_ids, span_len, device, info_dict)

        start = self._batch_slot_offset(input_ids, start)
        return self._preprocess_decode(input_ids, start, device, info_dict)

    @staticmethod
    def _batch_slot_offset(input_ids_view: torch.Tensor, fallback: int) -> int:
        """Recover a request's batch-row offset from its 1-D ``input_ids`` view.
        The runner passes ``input_ids = input_ids_buffer[s:e]``
        """
        if input_ids_view.dim() == 1 and input_ids_view.is_contiguous():
            offset = int(input_ids_view.storage_offset())
            if offset > 0:
                return offset
        return int(fallback)

    def _preprocess_prefill(
        self,
        input_ids: torch.Tensor,
        span_len: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        # Forward the audio (local-transformer) sampling params from the request.
        # vLLM's ``SamplingParams.temperature`` drives only the dummy backbone
        # token sampler, so the real audio temperature/top-k are passed via
        # ``additional_information`` and applied to the code predictor here (once,
        # at prefill — they are scalars that persist across decode steps).
        self._maybe_set_lt_sampling_params(info_dict)
        self._debug_outputs_enabled = bool(info_dict.get("debug_outputs", False))

        prefill_embeds = self._build_prefill_embeds(device, info_dict)

        offset = int(info_dict.get("prefill_offset", 0) or 0)
        total = int(prefill_embeds.shape[0])
        s = max(0, min(offset, total))
        e = max(0, min(offset + span_len, total))
        take = prefill_embeds[s:e]
        if int(take.shape[0]) < span_len:
            pad_n = span_len - int(take.shape[0])
            pad_rows = (
                take[-1:].expand(pad_n, -1)
                if take.shape[0] > 0
                else prefill_embeds.new_zeros(pad_n, prefill_embeds.shape[-1])
            )
            take = torch.cat([take, pad_rows], dim=0)

        row_start = self._batch_slot_offset(input_ids, 0)
        self._clear_runtime_rows(row_start, row_start + span_len)

        info_update = {
            "prefill_offset": offset + span_len,
            "decode_offset": 0,
        }
        # Tokenize the caller's ``text`` in-model and stash the subword ids in the
        # per-request info dict (alongside the offsets) so each decode step
        # consumes one id from it without the caller ever running the tokenizer
        # (see :meth:`_preprocess_decode`). When the caller passes ``text`` whole
        # at prefill we bake the ``text_tokens`` list here; an already-present
        # ``text_tokens`` list is left untouched. When *neither* ``text`` nor
        # ``text_tokens`` is provided the request runs in **streaming-text mode**:
        # no list is baked, and :meth:`_preprocess_decode` instead reads one
        # subword id per step from the streamed ``additional_information.text_token``.
        text_tokens = self._coerce_int_list(info_dict.get("text_tokens"))
        if text_tokens is not None:
            info_update["text_tokens"] = text_tokens
        else:
            text = self._first_str(info_dict.get("text"))
            if text:
                info_update["text_tokens"] = self._encode_text_stream(text)
        input_ids_out = torch.full_like(input_ids, _DUMMY_TOKEN_ID)
        return input_ids_out, take, info_update

    def _build_prefill_embeds(
        self,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> torch.Tensor:
        """Assemble the full ``(T_ctx, embedding_dim)`` prefill context embedding::

            [task_embedding | speaker_embedding | context_text_embedded]

        from the per-request inputs:

        * ``speaker_embedding`` — the speaker-encoded context-audio embedding,
          required as a 2-D ``(T_audio, embedding_dim)`` tensor.
        * ``context_text`` — a plain string (e.g. ``"[EN]"``); tokenized in-model
          (see :meth:`_encode_context_text`) and embedded through the baked
          per-subword ``text_embedding`` table.
        * ``task_mode_id`` — selects the per-mode task ("service token")
          embedding row; prepended only when the checkpoint has a task table.

        Returns the full context embedding; the per-chunk slicing/padding is done
        by :meth:`_preprocess_prefill`.
        """
        dtype = self._combined_embeddings.dtype

        speaker_embedding = info_dict.get("speaker_embedding")
        assert isinstance(speaker_embedding, torch.Tensor) and speaker_embedding.ndim == 2, (
            "EasyMagpieTTS preprocess expects additional_information.speaker_embedding to be a 2-D "
            "(T_audio, embedding_dim) tensor (the speaker-encoded context audio); "
            f"got {type(speaker_embedding).__name__}"
            + (f" with ndim={speaker_embedding.ndim}" if isinstance(speaker_embedding, torch.Tensor) else "")
        )

        parts: list[torch.Tensor] = []

        # Task / "service token" embedding (prepended), when present.
        if self.task_embedding is not None:
            task_mode_id = int(info_dict.get("task_mode_id", 0) or 0)
            task_mode_id = max(0, min(task_mode_id, self.num_task_embeddings - 1))
            task_row = self.task_embedding(torch.tensor([task_mode_id], device=device, dtype=torch.long))
            parts.append(task_row.to(dtype))

        # Speaker-encoded context audio.
        parts.append(speaker_embedding.to(device=device, dtype=dtype))

        # Context text: tokenized in-model and embedded through the baked table.
        context_text = self._first_str(info_dict.get("context_text")) or _DEFAULT_CONTEXT_TEXT
        ctx_ids = self._encode_context_text(context_text, device)
        if ctx_ids.numel() > 0:
            parts.append(self.context_text_embedding(ctx_ids).to(dtype))

        return torch.cat(parts, dim=0)

    def _maybe_set_lt_sampling_params(self, info_dict: dict[str, Any]) -> None:
        """Apply per-request audio sampling params to the local transformer.

        Reads ``temperature`` / ``top_k`` (alias ``topk``) from the request's
        ``additional_information`` and stores them on the code predictor. Absent
        keys leave the existing defaults untouched.
        """
        temperature = info_dict.get("temperature")
        if temperature is not None:
            self.code_predictor.temperature = float(self._first_str(temperature) or 0.0)
        top_k = info_dict.get("top_k", info_dict.get("topk"))
        if top_k is not None:
            self.code_predictor.top_k = int(float(self._first_str(top_k) or 0))

    def _get_text_tokenizer(self):
        """Lazily load the context-text tokenizer from the model directory.

        The converted checkpoint ships a HuggingFace ``AutoTokenizer`` (the
        model's text-conditioning tokenizer) alongside its weights, so we load it
        on first use from ``model_path``.
        """
        if self._text_tokenizer is None:
            import json

            from transformers import AutoTokenizer
            from transformers import PreTrainedTokenizerFast

            try:
                self._text_tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            except ValueError as exc:
                if "TokenizersBackend" not in str(exc):
                    raise
                tokenizer_config = json.loads((Path(self.model_path) / "tokenizer_config.json").read_text())
                self._text_tokenizer = PreTrainedTokenizerFast(
                    tokenizer_file=str(Path(self.model_path) / "tokenizer.json"),
                    bos_token=tokenizer_config.get("bos_token"),
                    eos_token=tokenizer_config.get("eos_token"),
                    unk_token=tokenizer_config.get("unk_token"),
                    clean_up_tokenization_spaces=bool(tokenizer_config.get("clean_up_tokenization_spaces", False)),
                    model_max_length=int(tokenizer_config.get("model_max_length", int(1e30))),
                )
        return self._text_tokenizer

    def _encode_context_text(self, context_text: str, device: torch.device) -> torch.Tensor:
        """Tokenize ``context_text`` to subword ids.

        The text-conditioning tokenizer sits at offset 0 in the model's
        tokenizer aggregate, so its raw ids index the baked ``text_embedding``
        table directly.
        """
        tok = self._get_text_tokenizer()
        ids = tok.encode(context_text)
        return torch.tensor(ids, device=device, dtype=torch.long)

    def _encode_text_stream(self, text: str) -> list[int]:
        """Tokenize the target ``text`` into the streaming subword-id list.

        Mirrors the reference ``tokenizer.encode(transcript) + [eos_id]``: HF
        special tokens are disabled so the raw ids index the baked
        ``text_embedding`` table directly, and the trailing text-EOS id closes
        the stream. One id is consumed per decode step (see
        :meth:`_preprocess_decode`); once exhausted the text channel is masked
        off.
        """
        tok = self._get_text_tokenizer()
        ids = tok.encode(text, add_special_tokens=False)
        return list(ids) + [self.text_eos_id]

    @staticmethod
    def estimate_prompt_len(
        speaker_embedding: torch.Tensor,
        *,
        tokenize: Callable[[str], Iterable[int]],
        context_text: str = _DEFAULT_CONTEXT_TEXT,
        has_task_embedding: bool = False,
    ) -> int:
        """Length-only mirror of :meth:`_build_prefill_embeds`.

        The engine assembles the prefill context as
        ``[task_embedding? | speaker_embedding | context_text_embedded]``, so the
        caller must pass ``prompt_token_ids = [0] * estimate_prompt_len(...)`` for
        the placeholder length to match the assembled embedding length (otherwise
        vLLM pads / truncates and quality drops).

        Args:
            speaker_embedding: ``(T_audio, embedding_dim)`` speaker-encoded
                context-audio embedding (only its length is used).
            tokenize: callable turning ``context_text`` into its subword ids
                (e.g. ``lambda t: tokenizer.encode(t)``) — must match the
                tokenizer the engine loads from ``model_path``.
            context_text: conditioning string (default ``"[NO TEXT CONTEXT]"``).
            has_task_embedding: whether the checkpoint prepends a task /
                "service token" embedding (``num_task_embeddings > 0``).
        """
        t_audio = int(speaker_embedding.shape[0])
        ctx_len = len(list(tokenize(context_text or _DEFAULT_CONTEXT_TEXT)))
        task_len = 1 if has_task_embedding else 0
        return task_len + t_audio + ctx_len

    def _preprocess_decode(
        self,
        input_ids: torch.Tensor,
        start: int,
        device: torch.device,
        info_dict: dict[str, Any],
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        decode_offset = int(info_dict.get("decode_offset", 0) or 0)
        info_update: dict[str, Any] = {"decode_offset": decode_offset + 1}
        self._clear_runtime_rows(start, start + 1)
        debug_decode_offset = getattr(self, "_debug_decode_offset", None)
        if isinstance(debug_decode_offset, torch.Tensor):
            debug_decode_offset[start] = decode_offset

        # ── Text channel ── (delay 0: one subword per step from step 0). The text
        # stream leads the phoneme/audio streams by their respective delays. Two
        # mutually exclusive input modes are supported:
        #
        # * **Whole-text (non-streaming)** — the caller passed ``text`` whole at
        #   prefill; it was tokenized in-model and stashed as the ``text_tokens``
        #   list (see :meth:`_preprocess_prefill`). Step k consumes
        #   ``text_tokens[k]`` (the list ends with the text-EOS id); once the
        #   stream is exhausted the channel is masked off (adds nothing) rather
        #   than repeating the last token.
        # * **Streamed** — the caller did *not* pass ``text`` at prefill and
        #   instead pushes one subword id per decode step via
        #   ``additional_information`` under ``text_token`` (a single int / 1-elem
        #   tensor; close the stream by pushing the text-EOS id as the last real
        #   token). The model embeds that step's id and masks the channel off on
        #   any step that carries no id (``text_token`` absent or ``< 0``), so the
        #   caller can keep pumping decode steps after the text ends while the
        #   audio tail finishes. Because each streamed chunk overwrites the
        #   previous ``text_token`` in the per-request buffer, every step gets a
        #   fresh value (or the caller's sentinel ``-1`` to mask).
        text_tokens = info_dict.get("text_tokens")
        if isinstance(text_tokens, list):
            if decode_offset < len(text_tokens):
                self._dec_text_tokens[start] = int(text_tokens[decode_offset])
                self._dec_text_mask[start] = 1
            else:
                self._dec_text_tokens[start] = 0
                self._dec_text_mask[start] = 0
        else:
            streamed_id = self._coerce_opt_int(info_dict.get("text_token"))
            if streamed_id is not None and streamed_id >= 0:
                self._dec_text_tokens[start] = streamed_id
                self._dec_text_mask[start] = 1
            else:
                self._dec_text_tokens[start] = 0
                self._dec_text_mask[start] = 0

        # ── Phoneme channel ── opens at decode step == ``phonemes_delay`` (seeded
        # with phoneme BOS), then feeds back the previous step's prediction, and
        # closes one step after the model emits the phoneme EOS (sticky flag).
        if self.has_phoneme:
            phoneme_ended = bool(info_dict.get("phoneme_ended", False))
            feed_eos = False
            gt_phoneme_rows = self._coerce_int_rows(
                info_dict.get("gt_phoneme_tokens"),
                int(self.arch.phoneme_stacking_factor),
            )
            gt_phoneme_index = decode_offset - self.phonemes_delay
            if phoneme_ended or decode_offset < self.phonemes_delay:
                self._dec_phoneme_tokens[start].zero_()
                self._dec_phoneme_valid[start] = 0
            elif gt_phoneme_rows is not None:
                if 0 <= gt_phoneme_index < len(gt_phoneme_rows):
                    row = torch.tensor(
                        gt_phoneme_rows[gt_phoneme_index],
                        device=device,
                        dtype=torch.long,
                    ).reshape(-1)[: self.arch.phoneme_stacking_factor]
                    self._dec_phoneme_tokens[start].zero_()
                    self._dec_phoneme_tokens[start, : row.shape[0]].copy_(row)
                    self._dec_phoneme_valid[start] = 1
                    feed_eos = bool((row == self.phoneme_eos_id).any())
                else:
                    self._dec_phoneme_tokens[start].zero_()
                    self._dec_phoneme_valid[start] = 0
            elif decode_offset == self.phonemes_delay:
                self._dec_phoneme_tokens[start].fill_(self.phoneme_bos_id)
                self._dec_phoneme_valid[start] = 1
            else:
                last_phon = info_dict.get("last_phoneme_token")
                if isinstance(last_phon, torch.Tensor) and last_phon.numel() > 0:
                    p = last_phon.to(device=device, dtype=torch.long).reshape(-1)[: self.arch.phoneme_stacking_factor]
                    self._dec_phoneme_tokens[start, : p.shape[0]].copy_(p)
                    self._dec_phoneme_valid[start] = 1
                    feed_eos = bool((p == self.phoneme_eos_id).any())
                else:
                    self._dec_phoneme_tokens[start].zero_()
                    self._dec_phoneme_valid[start] = 0
            if phoneme_ended or feed_eos:
                info_update["phoneme_ended"] = True
            debug_phoneme_valid = getattr(self, "_debug_phoneme_input_valid", None)
            if isinstance(debug_phoneme_valid, torch.Tensor):
                debug_phoneme_valid[start] = int(self._dec_phoneme_valid[start].item())
            debug_phoneme_token = getattr(self, "_debug_phoneme_input_token0", None)
            if isinstance(debug_phoneme_token, torch.Tensor):
                debug_phoneme_token[start] = (
                    int(self._dec_phoneme_tokens[start, 0].item())
                    if int(self._dec_phoneme_valid[start].item()) and self._dec_phoneme_tokens.ndim == 2
                    else -1
                )

        # ── Audio channel ── opens at decode step == ``speech_delay`` (seeded with
        # audio BOS), then feeds back the previous frame's codes. For the leading
        # ``speech_delay`` steps the channel is masked off (only text/phoneme
        # condition the backbone); the local transformer still runs for CUDA-graph
        # stability but its codes for those frames are discarded by the caller and
        # never fed back here.
        if decode_offset < self.speech_delay:
            self._dec_audio_codes[start].zero_()
            self._dec_audio_valid[start] = 0
        elif decode_offset == self.speech_delay:
            self._dec_audio_codes[start].fill_(self.arch.audio_bos_id)
            self._dec_audio_valid[start] = 1
        else:
            last_codes = info_dict.get("last_audio_codes")
            missing_feedback = True
            if isinstance(last_codes, torch.Tensor) and last_codes.numel() > 0:
                c = last_codes.to(device=device, dtype=torch.long).reshape(-1)[: self.num_codebooks]
                self._dec_audio_codes[start, : c.shape[0]].copy_(c)
                self._dec_audio_valid[start] = 1
                missing_feedback = False
            else:
                # Fallback (should not happen once audio has started): seed BOS.
                self._dec_audio_codes[start].fill_(self.arch.audio_bos_id)
                self._dec_audio_valid[start] = 1
            debug_missing = getattr(self, "_debug_audio_feedback_missing", None)
            if isinstance(debug_missing, torch.Tensor):
                debug_missing[start] = int(missing_feedback)
        debug_audio_input = getattr(self, "_debug_audio_input_code0", None)
        if isinstance(debug_audio_input, torch.Tensor):
            debug_audio_input[start] = (
                int(self._dec_audio_codes[start, 0].item())
                if int(self._dec_audio_valid[start].item())
                else -1
            )

        inputs_embeds_out = torch.zeros((1, self.embedding_dim), device=device, dtype=self._combined_embeddings.dtype)
        return input_ids, inputs_embeds_out, info_update

    def postprocess(self, hidden_states: torch.Tensor, multimodal_outputs: Optional[dict[str, Any]] = None, **_: Any):
        """Stash the last frame's codes (and phoneme) for the next decode step."""
        if hidden_states.numel() == 0:
            return {}
        out: dict[str, Any] = {}
        multimodal_outputs = multimodal_outputs or {}
        audio_codes = multimodal_outputs.get("audio_codes_feedback")
        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = multimodal_outputs.get("audio_codes")
        if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() > 0:
            last = self._last_request_row_index(
                hidden_states,
                audio_codes,
                getattr(self, "_last_output_row_indices", None),
            )
            out["last_audio_codes"] = audio_codes[last : last + 1].detach()
        if self.has_phoneme:
            phoneme_tokens = multimodal_outputs.get("phoneme_tokens_feedback")
            if isinstance(phoneme_tokens, torch.Tensor) and phoneme_tokens.numel() > 0:
                last = self._last_request_row_index(
                    hidden_states,
                    phoneme_tokens,
                    getattr(self, "_last_output_row_indices", None),
                )
                out["last_phoneme_token"] = phoneme_tokens[last : last + 1].detach().clone()
            else:
                last = self._last_request_row_index(
                    hidden_states,
                    self._dec_phoneme_tokens,
                    getattr(self, "_last_output_row_indices", None),
                )
                out["last_phoneme_token"] = self._dec_phoneme_tokens[last : last + 1].detach().clone()
        return out

    # ------------------------------------------------------------------
    # weight loading
    # ------------------------------------------------------------------

    def _reset_runtime_state_after_weight_load(self) -> None:
        """Clear mutable generation state after initial load or live refit."""

        self.mamba_cache = None
        self._debug_outputs_enabled = False
        self._last_output_row_indices = None
        for name in (
            "_combined_embeddings",
            "_dec_text_tokens",
            "_dec_text_mask",
            "_dec_audio_codes",
            "_dec_audio_valid",
            "_out_codes",
            "_out_code_logprobs",
            "_out_code_sampling_logprobs",
            "_out_frame_logprobs",
            "_token_stop",
            "_sample_stop",
            "_debug_text_emb_norm",
            "_debug_phoneme_emb_norm",
            "_debug_audio_emb_norm",
            "_dec_phoneme_tokens",
            "_dec_phoneme_valid",
        ):
            value = getattr(self, name, None)
            if isinstance(value, torch.Tensor):
                value.zero_()

    # Checkpoint prefixes (EasyMagpieTTS state dict) → in-model paths.
    # ``decoder.*`` is fed to the vLLM backbone loader separately (it understands
    # HF Nemotron-H naming + Mamba/MoE packing). The TTS submodules are copied
    # manually.
    _TTS_PREFIX_MAP = {
        "local_transformer.": "code_predictor.local_transformer.",
        "local_transformer_in_projection.": "code_predictor.local_transformer_in_projection.",
        "local_transformer_audio_out_projection.": "code_predictor.local_transformer_audio_out_projection.",
        "local_transformer_out_projections.": "code_predictor.local_transformer_out_projections.",
        "audio_embeddings.": "code_predictor.audio_embeddings.",
        "audio_in_projection.": "code_predictor.audio_in_projection.",
        "phoneme_embeddings.": "phoneme_embeddings.",
        "phoneme_final_proj.": "phoneme_final_proj.",
        "text_embedding.": "text_embedding.",
        "context_text_embedding.": "context_text_embedding.",
        "task_embedding.": "task_embedding.",
    }

    def _remap_tts_key(self, name: str) -> Optional[str]:
        """Map a raw checkpoint key to its in-model parameter path (or ``None``)."""
        for src, dst in self._TTS_PREFIX_MAP.items():
            if name.startswith(src):
                return dst + name[len(src) :]
        return None

    @staticmethod
    def _embedding_table_is_degenerate(table: torch.Tensor) -> bool:
        table_f = table.detach().float()
        return (
            table_f.numel() == 0
            or (not bool(torch.isfinite(table_f).all().item()))
            or float(table_f.abs().max().item()) == 0.0
        )

    @staticmethod
    def _sample_refit_indices(numel: int, max_values: int = 8) -> list[int]:
        if numel <= 0 or max_values <= 0:
            return []
        if numel <= max_values:
            return list(range(numel))
        candidates = [0, 1, numel // 4, numel // 2, (3 * numel) // 4, numel - 2, numel - 1]
        indices: list[int] = []
        for idx in candidates:
            idx = max(0, min(numel - 1, int(idx)))
            if idx not in indices:
                indices.append(idx)
            if len(indices) >= max_values:
                break
        return indices

    @staticmethod
    def _refit_skip_unchanged_enabled() -> bool:
        raw = os.getenv("EASYMAGPIE_SKIP_UNCHANGED_REFIT_PAYLOAD", "0").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    def _refit_default_float_dtype(self) -> torch.dtype:
        for param in self.parameters():
            if param.is_floating_point():
                return param.dtype
        return torch.float32

    def _refit_source_for_fingerprint(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        own_params: dict[str, torch.Tensor],
        backbone_params: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        source = tensor.detach()
        if name.startswith("decoder."):
            mapped_weights = list(_apply_optional_nemotron_h_weight_mapper([(name[len("decoder.") :], source)]))
            if mapped_weights:
                mapped_name, mapped_source = mapped_weights[0]
            else:
                mapped_name, mapped_source = name[len("decoder.") :], source
            target_name, source, _ = self._backbone_refit_target_name_and_tensor(
                str(mapped_name),
                mapped_source,
                input_name=name,
            )
            target = backbone_params.get(target_name)
        else:
            mapped_name = self._remap_tts_key(name)
            target = own_params.get(mapped_name) if mapped_name is not None else None
            source = tensor.detach()

        if source.is_floating_point():
            dtype = target.dtype if isinstance(target, torch.Tensor) else self._refit_default_float_dtype()
            if source.dtype != dtype:
                source = source.to(dtype=dtype)
        return source

    @staticmethod
    def _shape_equal(left: torch.Tensor, right: torch.Tensor) -> bool:
        return tuple(left.shape) == tuple(right.shape)

    @staticmethod
    def _candidate_if_shape_matches(candidate: torch.Tensor, reference: torch.Tensor) -> list[torch.Tensor]:
        if EasyMagpieTTSForConditionalGeneration._shape_equal(candidate, reference):
            return [candidate.detach()]
        if candidate.ndim == 2 and EasyMagpieTTSForConditionalGeneration._shape_equal(candidate.t(), reference):
            return [candidate.t().contiguous().detach()]
        return []

    def _packed_qkv_resident_sources_for_fingerprint(
        self,
        mapped_name: str,
        source: torch.Tensor,
        *,
        backbone_params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        match = _QKV_WEIGHT_RE.match(mapped_name)
        if match is None:
            return []
        layer = match.group("layer")
        projection = match.group("projection")
        target = backbone_params.get(f"layers.{layer}.mixer.qkv_proj.weight")
        if not isinstance(target, torch.Tensor) or target.ndim != 2 or source.ndim != 2:
            return []

        total_rows = int(target.shape[0])
        q_rows = int(getattr(self, "hidden_dim", 0) or 0)
        if q_rows <= 0 or total_rows <= q_rows:
            q_rows = int(source.shape[0]) if projection == "q_proj" else 0
        if q_rows <= 0:
            return []
        kv_rows = (total_rows - q_rows) // 2
        if kv_rows <= 0 or q_rows + 2 * kv_rows != total_rows:
            return []

        if projection == "q_proj":
            candidate = target[:q_rows, :]
        elif projection == "k_proj":
            candidate = target[q_rows : q_rows + kv_rows, :]
        else:
            candidate = target[q_rows + kv_rows : q_rows + 2 * kv_rows, :]
        return self._candidate_if_shape_matches(candidate, source)

    def _packed_moe_resident_sources_for_fingerprint(
        self,
        mapped_name: str,
        source: torch.Tensor,
        *,
        backbone_params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        match = _MOE_EXPERT_WEIGHT_RE.match(mapped_name)
        if match is None:
            return []
        layer = match.group("layer")
        expert_idx = int(match.group("expert"))
        projection = match.group("projection")
        target_key = "w13_weight" if projection == "up_proj" else "w2_weight"
        target = backbone_params.get(f"layers.{layer}.mixer.experts.{target_key}")
        if not isinstance(target, torch.Tensor) or target.ndim < 3:
            return []
        if expert_idx < 0 or expert_idx >= int(target.shape[0]):
            return []

        slab = target[expert_idx]
        candidates = self._candidate_if_shape_matches(slab, source)
        if projection == "up_proj" and slab.ndim == source.ndim and slab.shape[0] == source.shape[0] * 2:
            first = slab[: source.shape[0]]
            second = slab[source.shape[0] : source.shape[0] * 2]
            candidates.extend(self._candidate_if_shape_matches(first, source))
            candidates.extend(self._candidate_if_shape_matches(second, source))
        return candidates

    def _resident_sources_for_fingerprint(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        own_params: dict[str, torch.Tensor],
        backbone_params: dict[str, torch.Tensor],
    ) -> list[torch.Tensor]:
        source = self._refit_source_for_fingerprint(
            name,
            tensor,
            own_params=own_params,
            backbone_params=backbone_params,
        )
        if name.startswith("decoder."):
            mapped_name = name[len("decoder.") :]
            exact_target_name, _, _ = self._backbone_refit_target_name_and_tensor(
                mapped_name,
                tensor.detach(),
                input_name=name,
            )
            target = backbone_params.get(exact_target_name)
            candidates: list[torch.Tensor] = []
            if isinstance(target, torch.Tensor):
                candidates.extend(self._candidate_if_shape_matches(target, source))
            candidates.extend(
                self._packed_qkv_resident_sources_for_fingerprint(
                    mapped_name,
                    source,
                    backbone_params=backbone_params,
                )
            )
            candidates.extend(
                self._packed_moe_resident_sources_for_fingerprint(
                    mapped_name,
                    source,
                    backbone_params=backbone_params,
                )
            )
            return candidates

        mapped = self._remap_tts_key(name)
        if mapped is None:
            return []
        target = own_params.get(mapped)
        if not isinstance(target, torch.Tensor):
            return []
        return self._candidate_if_shape_matches(target, source)

    def _refit_loaded_target_names_for_input(
        self,
        name: str,
        tensor: torch.Tensor,
        *,
        own_params: dict[str, torch.Tensor],
        backbone_params: dict[str, torch.Tensor],
    ) -> set[str]:
        if name.startswith("decoder."):
            mapped_name = name[len("decoder.") :]
            target_names: set[str] = set()
            exact_target_name, _, _ = self._backbone_refit_target_name_and_tensor(
                mapped_name,
                tensor.detach(),
                input_name=name,
            )
            if exact_target_name in backbone_params:
                target_names.add(f"backbone.{exact_target_name}")
            qkv_match = _QKV_WEIGHT_RE.match(mapped_name)
            if qkv_match is not None:
                qkv_name = f"layers.{qkv_match.group('layer')}.mixer.qkv_proj.weight"
                if qkv_name in backbone_params:
                    target_names.add(f"backbone.{qkv_name}")
            moe_match = _MOE_EXPERT_WEIGHT_RE.match(mapped_name)
            if moe_match is not None:
                target_key = "w13_weight" if moe_match.group("projection") == "up_proj" else "w2_weight"
                moe_name = f"layers.{moe_match.group('layer')}.mixer.experts.{target_key}"
                if moe_name in backbone_params:
                    target_names.add(f"backbone.{moe_name}")
            return target_names

        mapped = self._remap_tts_key(name)
        if mapped is not None and mapped in own_params:
            return {mapped}
        return set()

    @staticmethod
    @torch.no_grad()
    def _refit_source_fingerprint(tensor: torch.Tensor) -> str:
        """Return a content hash for deciding whether a refit tensor is unchanged."""

        source = tensor.detach().contiguous().cpu()
        as_bytes = source.view(torch.uint8).numpy()
        digest = hashlib.sha256()
        digest.update(str(source.dtype).encode("utf-8"))
        digest.update(b"|")
        digest.update(",".join(str(int(dim)) for dim in source.shape).encode("utf-8"))
        digest.update(b"|")
        digest.update(memoryview(as_bytes))
        return digest.hexdigest()

    @classmethod
    @torch.no_grad()
    def _sample_refit_copy_check(
        cls,
        *,
        input_name: str,
        target_name: str,
        source: torch.Tensor,
        target: torch.Tensor,
        group: str,
        note: str | None = None,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "input_name": input_name,
            "target_name": target_name,
            "group": group,
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "source_dtype": str(source.dtype).replace("torch.", ""),
            "target_dtype": str(target.dtype).replace("torch.", ""),
            "checked": False,
            "matched": False,
        }
        if note:
            result["note"] = note
        if tuple(source.shape) != tuple(target.shape):
            result["reason"] = "shape_mismatch"
            return result

        numel = int(target.numel())
        indices = cls._sample_refit_indices(numel)
        result["numel"] = numel
        result["sample_indices"] = indices
        if not indices:
            result["checked"] = True
            result["matched"] = True
            result["max_sample_abs_diff"] = 0.0
            return result

        src_flat = source.detach().reshape(-1)
        tgt_flat = target.detach().reshape(-1)
        src_idx = torch.tensor(indices, device=src_flat.device, dtype=torch.long)
        tgt_idx = torch.tensor(indices, device=tgt_flat.device, dtype=torch.long)
        src_sample = src_flat.index_select(0, src_idx).to(device=tgt_flat.device, dtype=target.dtype)
        tgt_sample = tgt_flat.index_select(0, tgt_idx)
        if target.is_floating_point():
            diff = (tgt_sample.float() - src_sample.float()).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            if target.dtype in (torch.float16, torch.bfloat16):
                abs_tol = 5e-3
                rel_tol = 5e-3
            else:
                abs_tol = 1e-5
                rel_tol = 1e-5
            matched = bool(torch.allclose(tgt_sample.float(), src_sample.float(), atol=abs_tol, rtol=rel_tol))
            result["sample_abs_tol"] = abs_tol
            result["sample_rel_tol"] = rel_tol
        else:
            mismatch = tgt_sample.ne(src_sample)
            max_diff = float(mismatch.to(torch.float32).max().item()) if mismatch.numel() else 0.0
            matched = not bool(mismatch.any().item())
        result.update(
            {
                "checked": True,
                "matched": matched,
                "max_sample_abs_diff": max_diff,
            }
        )
        if not matched:
            result["source_sample"] = [float(x) for x in src_sample.detach().float().cpu().tolist()]
            result["target_sample"] = [float(x) for x in tgt_sample.detach().float().cpu().tolist()]
        return result

    @staticmethod
    def _backbone_refit_target_name_and_tensor(
        name: str,
        tensor: torch.Tensor,
        *,
        input_name: str | None = None,
    ) -> tuple[str, torch.Tensor, str | None]:
        target_name = name
        source = tensor
        if "embeddings" in target_name:
            target_name = target_name.replace("embeddings", "embed_tokens")
        original_name = input_name or name
        if "A_log" in target_name or "A_log" in original_name:
            target_name = target_name.replace("A_log", "A")
            # Nemotron-H stores the continuous-time SSM state matrix as
            # ``A = -exp(A_log)`` in the vLLM module.  Match the loader's value
            # transformation when verifying resident refit tensors.
            source = -torch.exp(source.to(torch.float32))
        if "D" in target_name:
            source = source.to(torch.float32)
        if "dt_bias" in target_name:
            source = source.to(torch.float32)
        if any(proj in target_name for proj in ("q_proj", "k_proj", "v_proj")):
            weight_name = next(proj for proj in ("q_proj", "k_proj", "v_proj") if proj in target_name)
            return target_name.replace(weight_name, "qkv_proj"), source, f"packed_{weight_name}"
        return target_name, source, None

    @staticmethod
    def _summarize_refit_copy_checks(checks: list[dict[str, Any]]) -> dict[str, Any]:
        checked = [item for item in checks if bool(item.get("checked"))]
        failed = [item for item in checked if not bool(item.get("matched"))]
        skipped = [item for item in checks if not bool(item.get("checked"))]
        max_diff = 0.0
        worst: dict[str, Any] | None = None
        for item in checked:
            diff = float(item.get("max_sample_abs_diff", 0.0) or 0.0)
            if diff >= max_diff:
                max_diff = diff
                worst = item

        by_group: dict[str, dict[str, Any]] = {}
        for item in checks:
            group = str(item.get("group", "unknown"))
            bucket = by_group.setdefault(group, {"num_total": 0, "num_checked": 0, "num_matched": 0, "num_failed": 0})
            bucket["num_total"] += 1
            if bool(item.get("checked")):
                bucket["num_checked"] += 1
                if bool(item.get("matched")):
                    bucket["num_matched"] += 1
                else:
                    bucket["num_failed"] += 1

        return {
            "ok": not failed,
            "num_total": len(checks),
            "num_checked": len(checked),
            "num_matched": len(checked) - len(failed),
            "num_failed": len(failed),
            "num_skipped": len(skipped),
            "max_sample_abs_diff": max_diff,
            "worst": worst,
            "failed_head": failed[:8],
            "skipped_head": skipped[:8],
            "by_group": by_group,
        }

    def _repair_context_text_embedding_if_needed(self, input_names: set[str]) -> bool:
        """Fallback for older artifacts without a valid context-text table."""

        context_embedding = getattr(self, "context_text_embedding", None)
        text_embedding = getattr(self, "text_embedding", None)
        context_weight = getattr(context_embedding, "weight", None)
        text_weight = getattr(text_embedding, "weight", None)
        if not isinstance(context_weight, torch.Tensor) or not isinstance(text_weight, torch.Tensor):
            return False
        if context_weight is text_weight:
            return False
        if tuple(context_weight.shape) != tuple(text_weight.shape):
            return False
        context_name = "context_text_embedding.weight"
        if context_name in input_names and not self._embedding_table_is_degenerate(context_weight):
            return False
        with torch.no_grad():
            context_weight.copy_(text_weight.to(dtype=context_weight.dtype, device=context_weight.device))
        return True

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load backbone (Nemotron-H) + TTS submodule weights from a converted checkpoint.

        The converted checkpoint carries the backbone under ``decoder.*`` (HF
        Nemotron-H names) and the TTS submodules at top level
        (``audio_embeddings.*``, ``local_transformer.*``, ``phoneme_*``,
        ``text_embedding.*``, projection heads). Backbone weights are routed to
        :meth:`NemotronHModel.load_weights` (which handles HF naming + Mamba/MoE
        packing); TTS weights are copied directly by name.
        """
        materialized_weights = list(weights)
        input_names = [str(name) for name, _ in materialized_weights]
        own_params = dict(self.named_parameters())
        backbone_params = dict(self.backbone.named_parameters())
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        backbone_input_names: list[str] = []
        forwarded_backbone_input_names: list[str] = []
        skipped_unchanged_backbone: list[tuple[str, str, torch.Tensor]] = []
        tts_input_names: list[str] = []
        skipped_unmapped: list[str] = []
        skipped_optional: list[str] = []
        skipped_unchanged: list[str] = []
        skipped_unchanged_by_source_fingerprint: list[str] = []
        skipped_unchanged_by_resident_fingerprint: list[str] = []
        fingerprint_cache_misses: list[str] = []
        missing_targets: list[str] = []
        tts_copy_check_items: list[tuple[str, str, torch.Tensor, torch.Tensor]] = []
        refit_rpc_active = bool(getattr(self, "_easymagpie_refit_rpc_active", False))
        skip_unchanged_enabled = self._refit_skip_unchanged_enabled() and refit_rpc_active
        previous_fingerprints = getattr(self, "_easymagpie_last_refit_source_fingerprints", {})
        if not isinstance(previous_fingerprints, dict):
            previous_fingerprints = {}
        current_fingerprints: dict[str, str] = {}
        if skip_unchanged_enabled:
            current_fingerprints = {
                str(name): self._refit_source_fingerprint(
                    self._refit_source_for_fingerprint(
                        str(name),
                        tensor,
                        own_params=own_params,
                        backbone_params=backbone_params,
                    )
                )
                for name, tensor in materialized_weights
            }

        for name, tensor in materialized_weights:
            source_fingerprint_match = (
                skip_unchanged_enabled
                and previous_fingerprints.get(str(name)) == current_fingerprints.get(str(name))
            )
            resident_fingerprint_match = False
            if skip_unchanged_enabled and not source_fingerprint_match:
                current_fingerprint = current_fingerprints.get(str(name))
                if current_fingerprint is not None:
                    for resident_source in self._resident_sources_for_fingerprint(
                        str(name),
                        tensor,
                        own_params=own_params,
                        backbone_params=backbone_params,
                    ):
                        if self._refit_source_fingerprint(resident_source) == current_fingerprint:
                            resident_fingerprint_match = True
                            break
                if resident_fingerprint_match:
                    skipped_unchanged_by_resident_fingerprint.append(str(name))
                else:
                    fingerprint_cache_misses.append(str(name))
            elif source_fingerprint_match:
                skipped_unchanged_by_source_fingerprint.append(str(name))
            unchanged = bool(source_fingerprint_match or resident_fingerprint_match)
            if name.startswith("decoder."):
                backbone_input_names.append(name)
                if unchanged:
                    skipped_unchanged.append(name)
                    skipped_unchanged_backbone.append((name, name[len("decoder.") :], tensor))
                    loaded.update(
                        self._refit_loaded_target_names_for_input(
                            str(name),
                            tensor,
                            own_params=own_params,
                            backbone_params=backbone_params,
                        )
                    )
                else:
                    backbone_weights.append((name[len("decoder.") :], tensor))
                    forwarded_backbone_input_names.append(name)
                continue
            mapped = self._remap_tts_key(name)
            if mapped is None:
                # Unrelated checkpoint section (codec, speaker encoder, CAS, etc.).
                skipped_unmapped.append(name)
                continue
            if mapped.startswith("task_embedding.") and self.task_embedding is None:
                # Single-mode model: checkpoint may still ship an (unused) table.
                skipped_optional.append(name)
                continue
            target = own_params.get(mapped)
            if target is None:
                if mapped == "context_text_embedding.weight" and self.context_text_embedding is self.text_embedding:
                    skipped_optional.append(name)
                    continue
                logger.warning("EasyMagpieTTS: no parameter for checkpoint key %s -> %s", name, mapped)
                missing_targets.append(name)
                continue
            if target.shape != tensor.shape:
                raise RuntimeError(
                    f"EasyMagpieTTS weight shape mismatch at {mapped!r}: "
                    f"ckpt {tuple(tensor.shape)} vs model {tuple(target.shape)}"
                )
            if unchanged:
                skipped_unchanged.append(name)
                loaded.update(
                    self._refit_loaded_target_names_for_input(
                        str(name),
                        tensor,
                        own_params=own_params,
                        backbone_params=backbone_params,
                    )
                )
            else:
                with torch.no_grad():
                    target.data.copy_(tensor.to(target.dtype))
            tts_input_names.append(name)
            loaded.add(mapped)
            tts_copy_check_items.append((name, mapped, tensor, target))

        context_text_embedding_repaired = self._repair_context_text_embedding_if_needed(set(input_names))
        if context_text_embedding_repaired and "context_text_embedding.weight" in own_params:
            loaded.add("context_text_embedding.weight")

        # ``NemotronHModel.load_weights`` (the inner model) does *not* apply the
        # HF->vLLM renaming that lives on the ``NemotronHForCausalLM`` wrapper, so
        # raw HF names such as ``embeddings.weight`` / ``...mixer.A_log`` would not
        # match the inner param names (``embed_tokens.weight`` / ``...mixer.A``).
        # Apply that mapper here so the converted checkpoint can keep stock HF
        # Nemotron-H names. The wrapper's ``backbone -> model`` prefix rule is a
        # no-op here because we already stripped the ``decoder.`` prefix.
        backbone_weights = list(_apply_optional_nemotron_h_weight_mapper(backbone_weights))
        if backbone_weights:
            backbone_loaded = self.backbone.load_weights(backbone_weights)
        else:
            backbone_loaded = set()
        loaded |= {f"backbone.{n}" for n in backbone_loaded}
        if skipped_unchanged:
            previous_loaded_targets = getattr(self, "_easymagpie_last_refit_loaded_targets", set())
            if isinstance(previous_loaded_targets, set):
                loaded |= {str(name) for name in previous_loaded_targets}
        copy_checks: list[dict[str, Any]] = []
        for input_name, mapped, tensor, target in tts_copy_check_items:
            copy_checks.append(
                self._sample_refit_copy_check(
                    input_name=input_name,
                    target_name=mapped,
                    source=tensor,
                    target=target,
                    group="tts",
                )
            )
        for original_name, (mapped_name, tensor) in zip(forwarded_backbone_input_names, backbone_weights):
            target_name, source, skip_reason = self._backbone_refit_target_name_and_tensor(
                mapped_name,
                tensor,
                input_name=original_name,
            )
            target = backbone_params.get(target_name)
            if target is None:
                copy_checks.append(
                    {
                        "input_name": original_name,
                        "target_name": target_name,
                        "group": "backbone",
                        "checked": False,
                        "matched": False,
                        "reason": "target_not_found",
                    }
                )
                continue
            if skip_reason is not None:
                copy_checks.append(
                    {
                        "input_name": original_name,
                        "target_name": target_name,
                        "group": "backbone",
                        "checked": False,
                        "matched": False,
                        "reason": skip_reason,
                    }
                )
                continue
            copy_checks.append(
                self._sample_refit_copy_check(
                    input_name=original_name,
                    target_name=f"backbone.{target_name}",
                    source=source,
                    target=target,
                    group="backbone",
                )
            )
        skipped_backbone_weights = list(
            _apply_optional_nemotron_h_weight_mapper(
                [(mapped_name, tensor) for _, mapped_name, tensor in skipped_unchanged_backbone]
            )
        )
        for original_name, (mapped_name, tensor) in zip(
            [name for name, _, _ in skipped_unchanged_backbone],
            skipped_backbone_weights,
        ):
            target_name, source, skip_reason = self._backbone_refit_target_name_and_tensor(
                mapped_name,
                tensor,
                input_name=original_name,
            )
            target = backbone_params.get(target_name)
            if target is None:
                copy_checks.append(
                    {
                        "input_name": original_name,
                        "target_name": target_name,
                        "group": "backbone",
                        "checked": False,
                        "matched": False,
                        "reason": "skipped_unchanged_target_not_found",
                    }
                )
                continue
            if skip_reason is not None:
                copy_checks.append(
                    {
                        "input_name": original_name,
                        "target_name": target_name,
                        "group": "backbone",
                        "checked": False,
                        "matched": False,
                        "reason": f"skipped_unchanged_{skip_reason}",
                    }
                )
                continue
            copy_checks.append(
                self._sample_refit_copy_check(
                    input_name=original_name,
                    target_name=f"backbone.{target_name}",
                    source=source,
                    target=target,
                    group="backbone",
                    note="skipped_unchanged",
                )
            )
        refit_copy_check = self._summarize_refit_copy_checks(copy_checks)

        missing_model_targets = sorted(str(name) for name in own_params if str(name) not in loaded)
        allow_missing_text_tables = bool(getattr(self, "_easymagpie_allow_missing_text_tables_refit", False))
        allowed_missing_model_targets = set()
        if allow_missing_text_tables:
            allowed_missing_model_targets.add("text_embedding.weight")
            allowed_missing_model_targets.add("context_text_embedding.weight")
        blocking_missing_model_targets = [
            name for name in missing_model_targets if name not in allowed_missing_model_targets
        ]
        loaded_names = sorted(str(name) for name in loaded)
        num_backbone_loaded_targets = sum(1 for name in loaded_names if name.startswith("backbone."))
        context_text_embedding_aliased = self.context_text_embedding is self.text_embedding
        fingerprint_cache_size_before = len(previous_fingerprints)
        fingerprint_cache_size_after = (
            len({**previous_fingerprints, **current_fingerprints}) if skip_unchanged_enabled else 0
        )
        self._last_easy_magpie_load_weights_summary = {
            "ok": (
                not skipped_unmapped
                and not missing_targets
                and not blocking_missing_model_targets
                and bool(refit_copy_check.get("ok", False))
            ),
            "num_input_tensors": len(materialized_weights),
            "num_routed_input_tensors": len(backbone_input_names) + len(tts_input_names),
            "num_backbone_input_tensors": len(backbone_input_names),
            "num_tts_input_tensors": len(tts_input_names),
            "num_loaded_targets": len(loaded_names),
            "num_backbone_loaded_targets": num_backbone_loaded_targets,
            "num_tts_loaded_targets": len(loaded_names) - num_backbone_loaded_targets,
            "num_actual_backbone_loader_targets": len(backbone_loaded),
            "refit_rpc_active": bool(refit_rpc_active),
            "skip_unchanged_enabled": bool(skip_unchanged_enabled),
            "fingerprint_cache_size_before": fingerprint_cache_size_before,
            "fingerprint_cache_size_after": fingerprint_cache_size_after,
            "num_source_fingerprint_matches": len(skipped_unchanged_by_source_fingerprint),
            "num_resident_fingerprint_matches": len(skipped_unchanged_by_resident_fingerprint),
            "num_fingerprint_cache_misses": len(fingerprint_cache_misses),
            "num_skipped_unchanged": len(skipped_unchanged),
            "skipped_unchanged_head": skipped_unchanged[:32],
            "skipped_unchanged_tail": skipped_unchanged[-32:],
            "skipped_unchanged_by_source_fingerprint_head": skipped_unchanged_by_source_fingerprint[:32],
            "skipped_unchanged_by_source_fingerprint_tail": skipped_unchanged_by_source_fingerprint[-32:],
            "skipped_unchanged_by_resident_fingerprint_head": skipped_unchanged_by_resident_fingerprint[:32],
            "skipped_unchanged_by_resident_fingerprint_tail": skipped_unchanged_by_resident_fingerprint[-32:],
            "fingerprint_cache_miss_head": fingerprint_cache_misses[:32],
            "fingerprint_cache_miss_tail": fingerprint_cache_misses[-32:],
            "num_backbone_weights_forwarded_to_loader": len(backbone_weights),
            "input_head": input_names[:16],
            "input_tail": input_names[-16:],
            "loaded_head": loaded_names[:16],
            "loaded_tail": loaded_names[-16:],
            "skipped_unmapped": skipped_unmapped[:32],
            "skipped_optional": skipped_optional[:32],
            "missing_targets": missing_targets[:32],
            "missing_model_targets": missing_model_targets[:32],
            "blocking_missing_model_targets": blocking_missing_model_targets[:32],
            "allowed_missing_model_targets": sorted(allowed_missing_model_targets)[:32],
            "num_skipped_unmapped": len(skipped_unmapped),
            "num_skipped_optional": len(skipped_optional),
            "num_missing_targets": len(missing_targets),
            "num_missing_model_targets": len(missing_model_targets),
            "num_blocking_missing_model_targets": len(blocking_missing_model_targets),
            "allow_missing_text_tables": bool(allow_missing_text_tables),
            "context_text_embedding_repaired": bool(context_text_embedding_repaired),
            "context_text_embedding_aliased": bool(context_text_embedding_aliased),
            "refit_copy_check": refit_copy_check,
        }

        # Derived runtime state.
        self.code_predictor.init_forbidden_mask()
        self._reset_runtime_state_after_weight_load()
        if skip_unchanged_enabled and bool(self._last_easy_magpie_load_weights_summary.get("ok", False)):
            cached = dict(previous_fingerprints)
            cached.update(current_fingerprints)
            self._easymagpie_last_refit_source_fingerprints = cached
            self._easymagpie_last_refit_loaded_targets = set(loaded_names)

        logger.info("Loaded %d weights for EasyMagpieTTSForConditionalGeneration", len(loaded))
        return loaded
