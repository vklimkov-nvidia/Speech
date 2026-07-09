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
"""Stage-1 Code2Wav model for the EasyMagpieTTS vLLM-Omni pipeline.

This is the second stage of the two-stage EasyMagpie pipeline (see
:mod:`easymagpie_vllm_omni.pipeline`): it turns the acoustic codes predicted by
the Stage-0 talker (:class:`easymagpie_vllm_omni.easymagpie.EasyMagpieTTSForConditionalGeneration`)
into a waveform, entirely in-engine — no external Triton / TRT codec service.

The codec math mirrors ``scripts/export_codec_decoder_onnx.py``'s
``CodecDecoderWrapper``:

    stacked model codes (B, T, C*S)
        -> clamp special tokens to [0, codebook_size-1]
        -> unstack (B, C*S, T) -> (B, C, T*S)
        -> FSQ index convert (model regrouped space -> codec native space)
        -> AudioCodecModel.decode -> waveform

The heavy pieces (the NeMo ``AudioCodecModel`` and the
``VectorQuantizerIndexConverter``) are loaded from artifacts the converter
(:mod:`scripts.easy_magpietts_convert_to_vllm`) bundles into the model
directory, so this stage is self-contained. The static per-frame decode graph is
optionally captured with a CUDA graph for streaming latency
(:class:`easymagpie_vllm_omni.cuda_graph_codec_wrapper.CUDAGraphCodecDecoder`).

Analogous to ``qwen3_tts/qwen3_tts_code2wav.py`` but adapted to the NeMo codec.
"""
from __future__ import annotations

import math
import os
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger

from vllm_omni.model_executor.models.output_templates import OmniOutput

from easymagpie_vllm_omni.config import EasyMagpieOmniArch
from easymagpie_vllm_omni.cuda_graph_codec_wrapper import CUDAGraphCodecDecoder

logger = init_logger(__name__)


def _patch_codec_for_cudagraph(module: nn.Module) -> int:
    """Make the NeMo codec's causal convs safe to capture in a CUDA graph.

    ``CausalConv1dNorm`` derives its pad amounts from CUDA int64 buffers and
    passes those tensors to ``F.pad``, which materializes them via ``.item()`` --
    a device->host sync that is illegal during CUDA-graph capture. The pad amount
    only depends on the (static) input length, so we rebind each such module's
    ``forward`` to compute it on the host from the shape and constant conv
    geometry. Numerically identical to the original forward; returns the number of
    modules patched (mutates this codec instance only).
    """
    from nemo.collections.common.parts.utils import mask_sequence_tensor

    def _make_forward(mod: nn.Module):
        # Resolve the constant conv geometry to Python ints once, at patch time
        # (these are CUDA int64 buffers, so int(...) syncs -- fine here, not in forward).
        kernel_size = int(mod.kernel_size)
        stride = int(mod.stride)
        padding_total = int(mod.padding_total)

        def forward(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
            length = int(inputs.shape[-1])
            n_frames = math.ceil((length - kernel_size + padding_total) / stride)
            ideal_length = n_frames * stride + kernel_size - padding_total
            extra_padding = ideal_length - length
            hidden_states = mod._pad1d(inputs, (padding_total, extra_padding), mode=mod.extra_pad_mode)
            hidden_states = mod.conv(hidden_states)
            hidden_states = mod.activation(hidden_states)
            hidden_states = mask_sequence_tensor(hidden_states, input_len)
            return hidden_states

        return forward

    patched = 0
    for m in module.modules():
        if type(m).__name__ == "CausalConv1dNorm":
            m.forward = _make_forward(m)
            patched += 1
    return patched


def _codec_ids_from_payload_or_input(
    input_ids: torch.Tensor,
    runtime_info: dict[str, Any] | None,
) -> torch.Tensor:
    """Prefer connector-delivered codec ids over placeholder token ids.

    In non-async full-payload mode the scheduler only allocates placeholder
    tokens; the real codebook-major flat codec stream is delivered through the
    worker connector as ``codes.audio``.
    """
    if isinstance(runtime_info, dict):
        codes = runtime_info.get("codes")
        if isinstance(codes, dict):
            audio = codes.get("audio")
            if isinstance(audio, torch.Tensor) and audio.numel() > 0:
                return audio.reshape(-1).to(device=input_ids.device, dtype=torch.long)
            if isinstance(audio, (list, tuple)) and audio:
                return torch.as_tensor(audio, device=input_ids.device, dtype=torch.long).reshape(-1)
    return input_ids.reshape(-1).to(dtype=torch.long)


class _EasyMagpieCodecDecoder(nn.Module):
    """Static decode graph: stacked model codes ``(B, T, Q)`` -> waveform ``(B, L)``.

    Reproduces ``scripts/export_codec_decoder_onnx.py``'s ``CodecDecoderWrapper``
    so the in-engine decode matches the exported/served reference bit-for-bit.
    """

    def __init__(
        self,
        codec_model: nn.Module,
        converter: nn.Module | None,
        stacking: int,
        clamp_max: int | None,
    ) -> None:
        super().__init__()
        self.codec_model = codec_model
        self.converter = converter
        self.stacking = int(stacking)
        self.clamp_max = clamp_max

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        # (B, T, Q) -> codec wants (B, Q, T)
        tokens = audio_codes.transpose(1, 2).contiguous()
        bsz = tokens.shape[0]

        if self.stacking > 1:
            # Unstack (B, C*S, T) -> (B, C, T*S): inverse of EasyMagpie stack_codes.
            cs, t = tokens.shape[1], tokens.shape[2]
            c = cs // self.stacking
            tokens = tokens.view(bsz, c, self.stacking, t).permute(0, 1, 3, 2).reshape(bsz, c, t * self.stacking)

        if self.clamp_max is not None:
            # Drop special tokens (audio bos/eos/mask live above the codebook).
            tokens = tokens.clamp(0, self.clamp_max)

        tokens = tokens.contiguous()
        frames = tokens.shape[2]
        tokens_len = torch.full((bsz,), frames, dtype=torch.long, device=tokens.device)

        if self.converter is not None:
            tokens = self.converter.convert_new_to_original(audio_tokens=tokens, audio_lens=tokens_len)

        audio, _ = self.codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        if audio.dim() == 3:
            # (B, 1, L) -> (B, L)
            audio = audio.squeeze(1)
        return audio


class EasyMagpieCode2Wav(nn.Module):
    """Stage-1 code2wav model for EasyMagpieTTS (GenerationModelRunner).

    Consumes the codebook-major flat codec stream produced by the talker and
    decodes it to a waveform via the bundled NeMo codec.
    """

    input_modalities = "audio"

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.model_path = vllm_config.model_config.model
        self.hf_config = vllm_config.model_config.hf_config
        self.arch = EasyMagpieOmniArch.from_hf_config(self.hf_config)

        self.have_multimodal_outputs = True
        self.has_preprocess = False
        self.has_postprocess = False
        self.enable_update_additional_information = True
        self.requires_raw_input_tokens = True

        # Q = C * S stacked codebooks (the talker autoregresses over these).
        self._num_quantizers = int(self.arch.num_stacked_codebooks)
        self._stacking = int(self.arch.frame_stacking_factor)
        self._codebook_size = int(self.arch.codebook_size)

        # Built in load_weights() (needs NeMo, only available in the Stage-1 worker).
        self.decode_module: _EasyMagpieCodecDecoder | None = None
        self._graph: CUDAGraphCodecDecoder | None = None
        self._samples_per_frame = 0
        self._output_sample_rate = int(getattr(self.hf_config, "codec_output_sample_rate", 22050))
        self._logged_codec_stats = False
        # Fixed codec chunk (frames). When > 0 every decode is right-padded to this
        # many frames so the codec always runs a single shape (one CUDA graph).
        # Resolved from connector config in load_weights().
        self._fixed_chunk_frames = 0

    # ------------------------------------------------------------------
    # vLLM runner shims
    # ------------------------------------------------------------------
    def embed_input_ids(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        if input_ids.numel() == 0:
            return torch.empty((0, 1), device=input_ids.device, dtype=torch.float32)
        return torch.zeros((input_ids.shape[0], 1), device=input_ids.device, dtype=torch.float32)

    def compute_logits(self, hidden_states: Any, sampling_metadata: Any = None) -> None:
        return None

    def _split_request_ids(self, ids: torch.Tensor, seq_token_counts: list[int] | None = None) -> list[torch.Tensor]:
        """Split concatenated input_ids into per-request segments."""
        if seq_token_counts is not None and len(seq_token_counts) > 1:
            boundaries = [0]
            for count in seq_token_counts:
                boundaries.append(boundaries[-1] + count)
            n = ids.numel()
            return [ids[boundaries[i] : min(boundaries[i + 1], n)] for i in range(len(seq_token_counts))]
        if is_forward_context_available():
            slices = get_forward_context().ubatch_slices
            if slices is not None and len(slices) > 1 and not any(hasattr(s, "token_slice") for s in slices):
                boundaries = [0]
                for s in slices:
                    boundaries.append(boundaries[-1] + s)
                return [ids[boundaries[i] : boundaries[i + 1]] for i in range(len(boundaries) - 1)]
        return [ids]

    @staticmethod
    def _meta_int(value: Any) -> int:
        if isinstance(value, list):
            value = value[0] if value else 0
        if isinstance(value, torch.Tensor):
            value = value.reshape(-1)[0].item() if value.numel() > 0 else 0
        return int(value or 0)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: Any = None,
        inputs_embeds: torch.Tensor | None = None,
        runtime_additional_information: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> OmniOutput:
        q = self._num_quantizers
        spf = self._samples_per_frame
        sr_tensor = torch.tensor(int(self._output_sample_rate), dtype=torch.int32)
        empty = torch.zeros((0,), dtype=torch.float32)

        if input_ids is None or input_ids.numel() == 0:
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": [empty], "sr": [sr_tensor]},
            )

        runtime_infos = runtime_additional_information or []
        ids = input_ids.reshape(-1).to(dtype=torch.long)
        request_ids_list = self._split_request_ids(ids, kwargs.get("seq_token_counts"))
        num_req = len(request_ids_list)

        left_context = [0] * num_req
        for i, info in enumerate(runtime_infos):
            if i >= num_req or not isinstance(info, dict):
                continue
            meta = info.get("meta", {})
            if isinstance(meta, dict) and "left_context_size" in meta:
                left_context[i] = self._meta_int(meta["left_context_size"])

        audios: list[torch.Tensor] = [empty] * num_req
        srs = [sr_tensor] * num_req

        # Phase 1: parse + right-pad each request up to the fixed chunk size so the
        # codec always decodes a captured shape. Padding replicates the last real
        # frame on the right; the codec is causal so it cannot change real frames'
        # audio, and the extra audio is sliced off in phase 3.
        # Each job is (request_index, codes (F, q), padded_frame_count, pad_frames).
        jobs: list[tuple[int, torch.Tensor, int, int]] = []
        for i, req_ids in enumerate(request_ids_list):
            runtime_info = runtime_infos[i] if i < len(runtime_infos) else None
            flat = _codec_ids_from_payload_or_input(req_ids, runtime_info)
            n = int(flat.numel())
            if n == 0 or n % q != 0:
                if n > 0:
                    logger.warning(
                        "EasyMagpie Code2Wav: flat codec length %d not divisible by num_quantizers %d; skipping.",
                        n,
                        q,
                    )
                continue
            frames = n // q
            # codebook-major flat [q*F] -> (q, F) -> (F, q)
            codes_fq = flat.reshape(q, frames).transpose(0, 1).contiguous()

            pad_frames = 0
            fixed = self._fixed_chunk_frames
            if fixed and frames < fixed:
                pad_frames = fixed - frames
                pad = codes_fq[-1:, :].expand(pad_frames, -1)
                codes_fq = torch.cat([codes_fq, pad], dim=0).contiguous()

            if not self._logged_codec_stats:
                self._logged_codec_stats = True
                logger.info(
                    "EasyMagpie Code2Wav codec: frames=%d (+%d pad -> %d) q=%d range=[%d,%d]",
                    frames,
                    pad_frames,
                    frames + pad_frames,
                    q,
                    int(codes_fq.min().item()),
                    int(codes_fq.max().item()),
                )
            jobs.append((i, codes_fq, int(codes_fq.shape[0]), pad_frames))

        # Phase 2: batch-decode. Group requests by (padded) frame count, stack each
        # group into a single (B, F, q) decode chunked to the largest captured batch
        # size; the wrapper pads the batch up to the nearest captured bucket.
        wavs: dict[int, torch.Tensor] = {}
        if jobs:
            max_batch = 1
            if self._graph is not None and self._graph.capture_batch_sizes:
                max_batch = max(self._graph.capture_batch_sizes)
            groups: dict[int, list[tuple[int, torch.Tensor, int, int]]] = defaultdict(list)
            for job in jobs:
                groups[job[2]].append(job)
            for group in groups.values():
                for k in range(0, len(group), max_batch):
                    chunk = group[k : k + max_batch]
                    batch = torch.stack([c for (_, c, _, _) in chunk], dim=0)  # (b, F, q)
                    out = self._decode_codes(batch)  # (b, L)
                    for row, (idx, _, _, _) in enumerate(chunk):
                        wavs[idx] = out[row]

        # Phase 3: per-request trim -- drop the audio generated from the right
        # padding (keep only real frames), then drop the left-context overlap.
        for i, _codes, _pf, pad_frames in jobs:
            wav = wavs.get(i)
            if wav is None:
                continue
            if pad_frames > 0 and spf > 0:
                keep = max(0, wav.shape[0] - pad_frames * spf)
                wav = wav[:keep]
            start = max(0, left_context[i] * spf)
            if start >= wav.shape[0]:
                continue
            wav = wav[start:]
            if wav.shape[0] > 0:
                audios[i] = (wav if wav.dtype == torch.float32 else wav.to(torch.float32)).reshape(-1)

        return OmniOutput(
            text_hidden_states=None,
            multimodal_outputs={"model_outputs": audios, "sr": srs},
        )

    def _decode_codes(self, codes_bfq: torch.Tensor) -> torch.Tensor:
        assert self.decode_module is not None, "EasyMagpieCode2Wav.load_weights was not called"
        if self._graph is not None:
            return self._graph.decode(codes_bfq)
        return self.decode_module(codes_bfq)

    def make_omni_output(self, model_outputs: torch.Tensor | OmniOutput | tuple, **_: Any) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        if isinstance(model_outputs, tuple) and len(model_outputs) == len(OmniOutput._fields):
            return OmniOutput(*model_outputs)
        if isinstance(model_outputs, tuple) and len(model_outputs) == 2:
            audio_tensor, sr = model_outputs
            return OmniOutput(
                text_hidden_states=None,
                multimodal_outputs={"model_outputs": audio_tensor, "sr": sr},
            )
        raise TypeError(
            "EasyMagpieCode2Wav expected OmniOutput, an OmniOutput tuple, or (audio, sr); "
            f"got {type(model_outputs)}"
        )

    # ------------------------------------------------------------------
    # weight / codec loading
    # ------------------------------------------------------------------
    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # The talker weights iterator carries no Code2Wav parameters; drain it so
        # callers don't hang on an unconsumed generator.
        for _ in weights:
            pass

        device = self.vllm_config.device_config.device
        codec_model, converter, stacking, clamp_max, sample_rate = self._build_codec(device)

        self.decode_module = _EasyMagpieCodecDecoder(
            codec_model=codec_model,
            converter=converter,
            stacking=stacking,
            clamp_max=clamp_max,
        ).to(device=device).eval()
        self._output_sample_rate = int(sample_rate)
        self._samples_per_frame = self._probe_samples_per_frame(device)
        logger.info(
            "EasyMagpie Code2Wav ready: stacking=%d clamp_max=%s samples_per_frame=%d sr=%d convert=%s",
            stacking,
            clamp_max,
            self._samples_per_frame,
            self._output_sample_rate,
            converter is not None,
        )

        # Resolve the fixed codec chunk size (frames). Prefer an explicit
        # ``codec_fixed_chunk_frames``; otherwise derive it from the streaming
        # window = ``codec_left_context_frames`` (reused overlap) + ``codec_chunk_frames``
        # (new-frame hop). With this set every decode is right-padded to this size.
        extra_cfg = self._connector_extra()
        hop = int(extra_cfg.get("codec_chunk_frames") or 0)
        left = int(extra_cfg.get("codec_left_context_frames") or 0)
        fixed = int(extra_cfg.get("codec_fixed_chunk_frames") or 0) or (hop + left)
        self._fixed_chunk_frames = fixed if fixed > 0 else 0
        if self._fixed_chunk_frames:
            logger.info(
                "EasyMagpie Code2Wav: fixed codec chunk = %d frames (left_context=%d + hop=%d); "
                "short windows are right-padded to this size and the padded audio is sliced off.",
                self._fixed_chunk_frames,
                left,
                hop,
            )

        # Make the codec's causal convs capture-safe (host-int pad amounts) so the
        # decode CUDA graph can be captured without a device sync inside F.pad.
        n_patched = _patch_codec_for_cudagraph(self.decode_module)
        logger.info("EasyMagpie Code2Wav: patched %d CausalConv1dNorm modules for CUDA-graph capture.", n_patched)

        self._maybe_enable_cudagraph(device)
        # The codec (decode_module.*) params are loaded from the bundled .nemo in
        # _build_codec, not from the talker safetensors iterator. Report every
        # module parameter as "loaded" so vLLM's DefaultModelLoader weight-track
        # check (track_weights_loading) doesn't flag decode_module.* as
        # uninitialized-from-checkpoint.
        return {name for name, _ in self.named_parameters()}

    def _build_codec(self, device: torch.device):
        """Restore the bundled NeMo codec + FSQ index converter."""
        from nemo.collections.tts.models import AudioCodecModel

        codec_path = self._resolve_codec_path()
        codec_cfg = AudioCodecModel.restore_from(codec_path, return_config=True)
        if "use_scl_loss" in codec_cfg:
            codec_cfg.use_scl_loss = False
        codec = AudioCodecModel.restore_from(codec_path, strict=False, override_config_path=codec_cfg)
        if hasattr(codec, "discriminator"):
            del codec.discriminator
        codec = codec.to(device).eval().float()
        codec.freeze()
        if hasattr(codec, "audio_decoder") and hasattr(codec.audio_decoder, "remove_weight_norm"):
            codec.audio_decoder.remove_weight_norm()

        converter = self._build_converter(codec, device)
        clamp_max = self._codebook_size - 1
        sample_rate = int(getattr(codec, "output_sample_rate", getattr(codec, "sample_rate", 22050)))
        return codec, converter, self._stacking, clamp_max, sample_rate

    def _build_converter(self, codec: nn.Module, device: torch.device) -> nn.Module | None:
        """Build the model->codec FSQ index converter from the bundled VQ config.

        Returns ``None`` when the model and codec already share the same FSQ
        grouping (no remap needed).
        """
        vq_cfg_path = self._resolve_optional_artifact(
            getattr(self.hf_config, "codec_vq_config", None), default_name="codec/vector_quantizer.yaml"
        )
        if vq_cfg_path is None:
            logger.info("EasyMagpie Code2Wav: no bundled vector_quantizer config; using codec native decode.")
            return None

        from hydra.utils import instantiate
        from omegaconf import OmegaConf

        from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter

        vq_cfg = OmegaConf.load(vq_cfg_path)
        vq_new = instantiate(vq_cfg).to(device).eval()
        if int(vq_new.num_codebooks) == int(codec.vector_quantizer.num_codebooks):
            return None
        return (
            VectorQuantizerIndexConverter(
                vector_quantizer_original=codec.vector_quantizer,
                vector_quantizer_new=vq_new,
            )
            .to(device)
            .eval()
        )

    def _resolve_codec_path(self) -> str:
        path = getattr(self.hf_config, "codec_model_path", None)
        resolved = self._resolve_optional_artifact(path, default_name="codec/codec.nemo")
        if resolved is None:
            raise FileNotFoundError(
                "EasyMagpieCode2Wav could not locate the codec .nemo. Re-run "
                "scripts/easy_magpietts_convert_to_vllm.py (which bundles the codec under "
                "<model>/codec/), or set `codec_model_path` in the model's config.json."
            )
        return resolved

    def _resolve_optional_artifact(self, path: str | None, *, default_name: str) -> str | None:
        """Resolve an artifact path: absolute, then relative to the model dir, then default."""
        candidates: list[str] = []
        if path:
            candidates.append(path)
            candidates.append(os.path.join(self.model_path, path))
        candidates.append(os.path.join(self.model_path, default_name))
        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand
        return None

    @torch.no_grad()
    def _probe_samples_per_frame(self, device: torch.device) -> int:
        """Measure waveform samples produced per model frame (robust to codec type)."""
        assert self.decode_module is not None
        q = self._num_quantizers
        try:
            a = self.decode_module(torch.zeros(1, 4, q, dtype=torch.long, device=device))
            b = self.decode_module(torch.zeros(1, 8, q, dtype=torch.long, device=device))
            per_frame = (int(b.shape[-1]) - int(a.shape[-1])) // 4
            if per_frame > 0:
                return per_frame
            return int(a.shape[-1]) // 4
        except Exception:
            logger.warning("EasyMagpie Code2Wav: samples-per-frame probe failed; trimming disabled.", exc_info=True)
            return 0

    def _maybe_enable_cudagraph(self, device: torch.device) -> None:
        if device.type != "cuda":
            logger.info("EasyMagpie Code2Wav CUDA Graph disabled (cpu).")
            return
        if self._samples_per_frame <= 0:
            return

        extra_cfg = self._connector_extra()

        # The codec decode graph is decoupled from vLLM's enforce_eager: the codec
        # decode is the expensive part of Stage 1, so it can be replayed even when
        # the (trivial) vLLM model runs eager. Disable via `codec_cudagraph: false`.
        if not bool(extra_cfg.get("codec_cudagraph", True)):
            logger.info("EasyMagpie Code2Wav codec CUDA Graph disabled via connector config.")
            return
        capture_frames = self._int_list(extra_cfg.get("decode_cudagraph_capture_frames"))
        if not capture_frames:
            # With a fixed chunk every decode is exactly that many frames, so a
            # single capture size suffices; otherwise fall back to a bucket set.
            if self._fixed_chunk_frames:
                capture_frames = [self._fixed_chunk_frames]
            else:
                capture_frames = self._default_capture_frames(extra_cfg)
        capture_batch_sizes = self._int_list(extra_cfg.get("decode_cudagraph_batch_sizes")) or [1]

        self._graph = CUDAGraphCodecDecoder(
            self.decode_module,
            num_stacked_codebooks=self._num_quantizers,
            samples_per_frame=self._samples_per_frame,
            capture_frames=capture_frames,
            capture_batch_sizes=capture_batch_sizes,
        )
        # Capture lazily on the first real decode. Stage 1 (codec) has no KV cache,
        # so vLLM never pins a graph pool for it; capturing during load_weights lands
        # the graph buffers on memory that vLLM's later profile/kv-cache/warmup frees
        # and reuses -> corrupted replay. Deferring capture runs it in the stable
        # serving context.
        try:
            self._graph.arm_lazy_warmup(device)
        except Exception:
            logger.warning("EasyMagpie Code2Wav CUDA Graph warmup failed; falling back to eager.", exc_info=True)
            self._graph = None

    def _connector_extra(self) -> dict[str, Any]:
        model_cfg = getattr(self.vllm_config, "model_config", None)
        connector_cfg = getattr(model_cfg, "stage_connector_config", None)
        if isinstance(connector_cfg, dict):
            extra = connector_cfg.get("extra", connector_cfg)
        else:
            extra = getattr(connector_cfg, "extra", None)
        return extra if isinstance(extra, dict) else {}

    @staticmethod
    def _default_capture_frames(extra_cfg: dict[str, Any]) -> list[int]:
        """Streaming chunk + left-context buckets, plus small power-of-two buckets."""
        sizes: set[int] = set()
        chunk = int(extra_cfg.get("codec_chunk_frames") or 0)
        left = int(extra_cfg.get("codec_left_context_frames") or 0)
        if chunk > 0:
            sizes.add(chunk)
            if left > 0:
                sizes.add(chunk + left)
        for p2 in (8, 16, 32, 64, 128, 256):
            sizes.add(p2)
        return sorted(sizes)

    @staticmethod
    def _int_list(value: Any) -> list[int]:
        if value is None:
            return []
        if isinstance(value, str):
            items = [v.strip() for v in value.split(",") if v.strip()]
        elif isinstance(value, int):
            items = [value]
        else:
            try:
                items = list(value)
            except TypeError:
                return []
        out: list[int] = []
        for it in items:
            try:
                iv = int(it)
            except (TypeError, ValueError):
                continue
            if iv > 0:
                out.append(iv)
        return sorted(set(out))
