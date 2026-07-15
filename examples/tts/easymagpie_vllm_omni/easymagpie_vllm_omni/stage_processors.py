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
"""Transfer stacked acoustic codes from the talker to Code2Wav.

Stage 0 emits ``[frames, codebooks]`` codes. Stage 1 consumes the corresponding
codebook-major flat stream.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
from vllm.logger import init_logger
from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayload, OmniPayloadStruct

logger = init_logger(__name__)

# Base codebook size, excluding control tokens.
_CODEBOOK_SIZE = 1024
_NUM_QUANTIZERS_DEFAULT = 16


def _empty_finished_payload() -> dict[str, Any]:
    """Release Stage 1 when no usable codec frames were produced."""
    return {
        "codes": {"audio": torch.zeros(0, dtype=torch.long)},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def _filter_audio_codes(audio_codes: torch.Tensor) -> torch.Tensor:
    """Drop all-zero (padding/warm-up), negative, and special-token frames.

    Special audio tokens (bos/eos/mask) live at ``codebook_size + offset`` — any
    frame containing one is an out-of-band control frame, not audio, so it is
    removed here (the decoder additionally clamps as a safety net).
    """
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0 or audio_codes.ndim != 2:
        return audio_codes
    valid_mask = (
        (audio_codes >= 0).all(dim=1) & audio_codes.any(dim=1) & (audio_codes.max(dim=1).values < _CODEBOOK_SIZE)
    )
    return audio_codes[valid_mask]


def _flatten_codebook_major(audio_codes: torch.Tensor) -> torch.Tensor:
    """``[F, Q]`` -> codebook-major flat ``[Q*F]`` (long, cpu, contiguous)."""
    return audio_codes.transpose(0, 1).to(device="cpu", dtype=torch.long).reshape(-1).contiguous()


def talker2code2wav(
    source_outputs: list[Any],
    prompt: Any = None,
    _requires_multimodal_data: bool = False,
) -> list[Any]:
    """Non-async orchestrator path: collect all talker codes, decode at once."""
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list[OmniTokensPrompt] = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if isinstance(output.multimodal_output, dict) else {}
        audio = _extract_audio_codes(mm)
        if audio is None:
            code2wav_inputs.append(_empty_prompt())
            continue
        audio = _filter_audio_codes(audio.to(torch.long))
        token_ids = getattr(output, "cumulative_token_ids", []) or []
        seq_len = max(len(token_ids) - 1, 0)
        if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
            audio = audio[-seq_len:]
        if audio.numel() == 0:
            code2wav_inputs.append(_empty_prompt())
            continue
        codec_codes = _flatten_codebook_major(audio).tolist()
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=codec_codes,
                multi_modal_data=None,
                mm_processor_kwargs=None,
                additional_information=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_token_only(
    source_outputs: list,
    prompt=None,
    _requires_multimodal_data: bool = False,
) -> list:
    """Sync-side length-only placeholder for the non-async-chunk Stage-1 input.

    Sized to ``Q * num_audio_frames``; the real codec ids ship via the worker
    connector payload built by :func:`talker2code2wav_full_payload`.
    """
    from vllm_omni.inputs.data import OmniTokensPrompt

    code2wav_inputs: list = []
    for talker_output in source_outputs:
        if not talker_output.finished:
            continue
        output = talker_output.outputs[0]
        mm = output.multimodal_output if isinstance(getattr(output, "multimodal_output", None), dict) else {}
        audio = _extract_audio_codes(mm)
        token_ids = getattr(output, "cumulative_token_ids", []) or []
        seq_len = max(len(token_ids) - 1, 0)

        if isinstance(audio, torch.Tensor) and audio.numel() > 0:
            audio = _filter_audio_codes(audio.to(torch.long))
            if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
                audio = audio[-seq_len:]
            num_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
            num_quantizers = int(audio.shape[1]) if audio.ndim == 2 and audio.shape[1] > 0 else _NUM_QUANTIZERS_DEFAULT
        else:
            num_frames = 0
            num_quantizers = _NUM_QUANTIZERS_DEFAULT

        prompt_len = num_quantizers * num_frames
        code2wav_inputs.append(
            OmniTokensPrompt(
                prompt_token_ids=[0] * prompt_len,
                additional_information=None,
                multi_modal_data=None,
                mm_processor_kwargs=None,
            )
        )
    return code2wav_inputs


def talker2code2wav_full_payload(transfer_manager, multimodal_output, request, is_finished: bool = False):
    """Send accumulated codec frames through the worker connector.

    ``is_finished`` is part of the transfer-adapter callback contract.
    """
    pooling_output = multimodal_output
    del transfer_manager, is_finished
    rid = getattr(request, "request_id", "?")
    if not isinstance(pooling_output, dict):
        logger.warning(
            "easymagpie.talker2code2wav_full_payload: pooling_output is %s (not dict) for req=%s",
            type(pooling_output).__name__,
            rid,
        )
        return _empty_finished_payload()

    audio = pooling_output.get("codes.audio")
    if audio is None:
        codes_nested = pooling_output.get("codes")
        if isinstance(codes_nested, dict):
            audio = codes_nested.get("audio")
    if not isinstance(audio, torch.Tensor) or audio.numel() == 0:
        logger.warning(
            "easymagpie.talker2code2wav_full_payload: missing/empty codes.audio (keys=%s) for req=%s",
            list(pooling_output.keys()),
            rid,
        )
        return _empty_finished_payload()

    raw_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
    audio = _filter_audio_codes(audio.to(torch.long))
    kept_frames = int(audio.shape[0]) if audio.ndim == 2 else 0
    logger.info(
        "easymagpie.talker2code2wav_full_payload: req=%s accumulated %d frames "
        "(%d after filtering control/padding).",
        rid,
        raw_frames,
        kept_frames,
    )
    if audio.numel() == 0:
        return _empty_finished_payload()

    output_token_ids = list(getattr(request, "output_token_ids", None) or [])
    seq_len = max(len(output_token_ids) - 1, 0)
    if seq_len > 0 and audio.ndim == 2 and int(audio.shape[0]) > seq_len:
        audio = audio[-seq_len:]

    codec_codes = _flatten_codebook_major(audio)
    return {
        "codes": {"audio": codec_codes},
        "meta": {"finished": torch.tensor(True, dtype=torch.bool)},
    }


def _extract_last_frame(multimodal_output: OmniPayload | dict[str, Any]) -> torch.Tensor | None:
    audio_codes = _extract_audio_codes(multimodal_output)
    if not isinstance(audio_codes, torch.Tensor) or audio_codes.numel() == 0:
        return None
    if audio_codes.ndim == 2:
        frame = audio_codes[-1]
        if frame.numel() == 0 or not bool(frame.any().item()):
            return None
        if int(frame.max().item()) >= _CODEBOOK_SIZE:
            # Control frame (audio eos/mask) — not audio.
            return None
        return frame.to(torch.long).reshape(-1)
    if audio_codes.ndim == 1:
        return audio_codes.to(torch.long).reshape(-1)
    raise ValueError(f"Invalid audio_codes shape for EasyMagpie async_chunk: {tuple(audio_codes.shape)}")


def _resolve_speech_delay(transfer_manager: Any) -> int:
    """Return and cache the number of non-audio frames before speech starts."""
    cached = getattr(transfer_manager, "_easymagpie_speech_delay", None)
    if cached is not None:
        return cached

    # Transfer managers expose model configuration through either interface.
    model_config = getattr(transfer_manager, "config", None)
    if getattr(model_config, "hf_config", None) is None:
        getter = getattr(transfer_manager, "_get_model_config", None)
        if callable(getter):
            try:
                model_config = getter()
            except Exception:
                pass

    hf_config = getattr(model_config, "hf_config", None)
    try:
        delay = int(getattr(hf_config, "streaming_speech_delay", 0) or 0)
    except Exception:
        delay = 0
    transfer_manager._easymagpie_speech_delay = delay
    logger.info("easymagpie: resolved streaming_speech_delay=%d (leading warm-up frames dropped)", delay)
    return delay


def _is_warmup_frame(request: Any, transfer_manager: Any) -> bool:
    """Check whether the newest frame precedes the configured speech start."""
    speech_delay = _resolve_speech_delay(transfer_manager)
    if speech_delay <= 0:
        return False
    n_emitted = len(getattr(request, "output_token_ids", None) or [])
    return n_emitted <= speech_delay


def talker2code2wav_async_chunk(
    transfer_manager: Any,
    multimodal_output: OmniPayload | dict[str, Any] | None,
    request: Any,
    is_finished: bool = False,
) -> OmniPayloadStruct | None:
    """Emit a codec window with bounded left context.

    ``multimodal_output`` must retain this name because the transfer adapter
    passes it by keyword.
    """
    request_id = request.external_req_id
    finished = bool(is_finished or request.is_finished())

    if isinstance(multimodal_output, Mapping):
        frame = _extract_last_frame(multimodal_output)
        if frame is not None and not _is_warmup_frame(request, transfer_manager):
            transfer_manager.code_prompt_token_ids[request_id].append(frame.cpu().tolist())
    elif not finished:
        return None

    connector = getattr(transfer_manager, "connector", None)
    raw_cfg = getattr(connector, "config", {}) or {}
    cfg = raw_cfg.get("extra", raw_cfg) if isinstance(raw_cfg, dict) else {}
    chunk_size = int(cfg.get("codec_chunk_frames", 25))
    left_context_size_config = int(cfg.get("codec_left_context_frames", 0))
    initial_chunk_size = int(cfg.get("initial_codec_chunk_frames") or 0)

    if chunk_size <= 0 or left_context_size_config < 0 or initial_chunk_size < 0:
        raise ValueError(
            f"Invalid EasyMagpie codec chunk config: codec_chunk_frames={chunk_size}, "
            f"codec_left_context_frames={left_context_size_config}, "
            f"initial_codec_chunk_frames={initial_chunk_size}"
        )
    if initial_chunk_size > chunk_size:
        initial_chunk_size = chunk_size

    length = len(transfer_manager.code_prompt_token_ids[request_id])
    if length <= 0:
        if finished:
            return OmniPayloadStruct(
                codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                meta=MetaStruct(finished=torch.tensor(True, dtype=torch.bool)),
            )
        return None

    use_first_chunk = 0 < initial_chunk_size < chunk_size
    if use_first_chunk and length <= initial_chunk_size:
        if not finished and length < initial_chunk_size:
            return None
        context_length = length if finished and length < initial_chunk_size else initial_chunk_size
    else:
        initial_coverage = initial_chunk_size if use_first_chunk else 0
        adjusted = length - initial_coverage
        if not finished and adjusted % chunk_size != 0:
            return None
        chunk_length = adjusted % chunk_size
        context_length = chunk_length if chunk_length != 0 else chunk_size

    end_index = min(length, left_context_size_config + context_length)
    left_context_size = max(0, end_index - context_length)
    window_frames = transfer_manager.code_prompt_token_ids[request_id][-end_index:]

    num_quantizers = len(window_frames[0])
    num_frames = len(window_frames)
    code_predictor_codes = torch.tensor(
        [window_frames[f][q] for q in range(num_quantizers) for f in range(num_frames)],
        dtype=torch.long,
    )

    return OmniPayloadStruct(
        codes=CodesStruct(audio=code_predictor_codes),
        meta=MetaStruct(
            left_context_size=left_context_size,
            finished=torch.tensor(finished, dtype=torch.bool),
        ),
    )


def _extract_audio_codes(mm: Mapping | dict[str, Any] | None) -> torch.Tensor | None:
    """Read the talker acoustic codes, preferring nested ``codes.audio`` then
    the single-stage ``audio_codes`` key."""
    if not isinstance(mm, Mapping):
        return None
    codes = mm.get("codes")
    if isinstance(codes, Mapping):
        audio = codes.get("audio")
        if isinstance(audio, torch.Tensor):
            return audio
    audio = mm.get("audio_codes")
    if isinstance(audio, torch.Tensor):
        return audio
    return None


def _empty_prompt():
    from vllm_omni.inputs.data import OmniTokensPrompt

    return OmniTokensPrompt(
        prompt_token_ids=[0],
        multi_modal_data=None,
        mm_processor_kwargs=None,
        additional_information=None,
    )
