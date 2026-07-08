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
"""Chunked (streaming) decoding of EasyMagpieTTS codec codes to audio.

Background
----------
EasyMagpieTTS produces discrete codec frames at **25 fps** (one frame every
40 ms). The codec is a NeMo ``AudioCodecModel`` (the *25fps spectral codec with
bandwidth extension*) whose decode path is:

    codec tokens  ->  GroupFiniteScalarQuantizer.decode (per-frame, stateless)
                  ->  ResNetDecoder (fully-convolutional)
                  ->  waveform  (length == n_frames * output_samples_per_frame)

Concrete numbers for ``25fps_spectral_codec_with_bandwidth_extension.nemo``:
  * sample_rate (in) = 16000, output_sample_rate = 22050
  * samples_per_frame (in) = 640 -> 16000 / 640 = 25 fps (40 ms / frame)
  * output samples per frame = 640 * 22050 / 16000 = 882 (decode emits 22050 Hz)
  * quantizer = GroupFiniteScalarQuantizer: 5 codebooks, 4^8 = 65536 entries each
  * decoder = ResNetDecoder with ``is_causal=True``  ==>  **CAUSAL**

The dequantizer (FSQ) is a pure per-frame lookup, so it has no temporal
receptive field. All temporal context comes from the convolutional decoder.

Causality: NO lookahead needed (R = 0)
--------------------------------------
This codec's ``ResNetDecoder`` is **causal**: ``CausalConv1dNorm`` left-pads only
and ``CausalConvTranspose1dNorm`` trims the right (future) side. Every output
sample depends only on the current and *past* input frames -- never future ones.
So ``right_context`` (lookahead) must be **0**; adding lookahead would only cost
latency for zero quality gain.

(If you ever point this at a *non-causal* codec -- e.g. the default
``HiFiGANDecoder`` with reflect padding -- you would need ``right_context`` >=
the decoder's future receptive field. ``StreamingDecodeConfig.for_codec``
auto-detects a ``Causal*`` decoder and picks ``R=0`` vs a non-zero default.)

Why we still keep a LEFT context (overlap-save)
-----------------------------------------------
Even causal, the decoder has a non-trivial *past* receptive field: decoding an
isolated chunk would left-pad (replicate) where real history exists, corrupting
the chunk's leading samples. The fix is **overlap-save** (a.k.a.
"decode-with-left-margin, keep the tail"), *not* overlap-add:

  * Decode ``[L left | N body]`` frames, then discard the audio samples that
    belong to the ``L`` left-context frames. The kept body samples are
    numerically (near) identical to a full-sequence decode -- provided ``L``
    covers the decoder's past receptive field.
  * Because the kept regions of adjacent chunks are exactly the non-overlapping
    body frames, you simply **concatenate** them. NO cross-fade / no weighted
    overlap-add / no window function.

Chunk size and context (recommended defaults)
----------------------------------------------
  * ``chunk_frames`` (N): the number of *new* frames emitted per chunk. Pick this
    from your latency budget: at 25 fps, N frames == N * 40 ms of audio. N=25
    gives 1.0 s chunks; N=12 gives ~0.5 s chunks.
  * ``right_context`` (R): **0** for this causal codec (no lookahead, no latency).
  * ``left_context`` (L): past frames decoded then trimmed; must cover the
    decoder's *past* receptive field. For this ResNetDecoder the frame-domain
    past RF is dominated by the layers at (or near) 25 fps:
      - pre_conv (k=3) + first ResidualBlockV2 (2x k=3) at frame rate  -> ~6 frames
      - 6 hidden ResidualBlockV2 (2x k=3) at 2x frame rate             -> ~12 frames
      - layers after the 9x/7x/7x upsamplers contribute < 1 frame
    Total ~= 18-20 frames (~0.8 s). ``left_context = 24`` (~1 s) is a safe
    default. Left context is *past* audio, so it adds NO latency -- only a little
    recompute (L / (L + N) extra decoded frames per chunk). Verify a tight value
    empirically with ``measure_left_receptive_field`` below.

This module implements exact overlap-save. ``crossfade_concat`` is only a
fallback if you ever truncate ``L`` below the receptive field (or run a
non-causal codec with ``R=0``).

This module is decoder-agnostic: pass any callable
``decode_fn(codes_BCT, codes_len_B) -> (audio_BT, audio_len_B)`` -- e.g.
``model._codec_helper.codes_to_audio`` (NeMo torch) or a TensorRT engine
wrapper. See ``make_nemo_decode_fn`` for the NeMo case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import torch

# A decode function maps (codes [B, C, T], codes_len [B]) -> (audio [B, T_audio], audio_len [B]).
DecodeFn = Callable[[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]

# The NeMo codec helper (`_prepare_codes_for_decode`) zero-pads windows shorter than
# this many frames before decoding. Used only by the spf-inference fallback.
_CODEC_MIN_DECODE_FRAMES = 4


def make_nemo_decode_fn(model) -> DecodeFn:
    """Build a ``DecodeFn`` from a loaded EasyMagpieTTSInferenceModel.

    Handles the optional codec index-converter and frame unstacking exactly like
    ``streaming_finalize`` does, so chunk decodes match the one-shot result.
    """

    @torch.inference_mode()
    def _decode(codes: torch.Tensor, codes_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        codes, codes_len = model._prepare_codes_for_decode(codes, codes_len)
        audio, audio_len, _ = model._codec_helper.codes_to_audio(codes, codes_len)
        return audio, audio_len

    return _decode


def _decoder_is_causal(model) -> bool:
    """True if the codec's audio decoder is causal (no future receptive field)."""
    decoder = getattr(getattr(model, "_codec_model", model), "audio_decoder", None)
    if decoder is None:
        return False
    # Explicit flag on ResNetDecoder; otherwise infer from the conv class name.
    if bool(getattr(decoder, "is_causal", False)):
        return True
    return type(decoder).__name__.startswith("Causal")


def output_samples_per_frame(model) -> int:
    """Output waveform samples produced per *model* code frame.

    A model frame unstacks to ``frame_stacking_factor`` codec frames, and each codec
    frame decodes to ``samples_per_frame * output_sample_rate / sample_rate`` output
    samples. Computing this directly (instead of inferring from a decode) avoids the
    trap where the codec helper zero-pads short windows up to its ``min_len`` and
    distorts a naive ``audio_len / n_frames`` estimate.
    """
    codec = getattr(model, "_codec_model", model)
    samples_per_codec_frame = int(round(codec.samples_per_frame * codec.output_sample_rate / codec.sample_rate))
    stacking = int(getattr(model, "frame_stacking_factor", 1))
    return stacking * samples_per_codec_frame


@dataclass
class StreamingDecodeConfig:
    """Overlap-save chunked decoding configuration.

    Attributes:
        chunk_frames: Number of new (body) frames emitted per chunk (N).
        left_context: Trimmed left-margin frames (L). Must cover the decoder's
            *past* receptive field (~20 frames for this codec's ResNetDecoder).
        right_context: Trimmed right-margin / lookahead frames (R). MUST be 0 for
            the causal ResNetDecoder; only >0 for a non-causal decoder.
        samples_per_frame: Output samples produced per codec frame. If None it is
            inferred from the first decode (audio_len / n_frames).
    """

    chunk_frames: int = 25
    left_context: int = 24
    right_context: int = 0
    samples_per_frame: Optional[int] = None

    @classmethod
    def for_codec(cls, model, chunk_frames: int = 25, left_context: int = 24) -> "StreamingDecodeConfig":
        """Build a config that matches the codec: lookahead from causality, exact spf.

        ``right_context = 0`` for a causal decoder (e.g. this codec's causal
        ``ResNetDecoder``); a small non-zero lookahead otherwise. ``samples_per_frame``
        is computed exactly (see :func:`output_samples_per_frame`) so the streaming
        decoder never has to infer it from a possibly-padded short window.
        """
        right_context = 0 if _decoder_is_causal(model) else 5
        return cls(
            chunk_frames=chunk_frames,
            left_context=left_context,
            right_context=right_context,
            samples_per_frame=output_samples_per_frame(model),
        )


class StreamingCodecDecoder:
    """Stateful overlap-save decoder for a single stream (batch size 1).

    Feed codec frames as they are generated with :meth:`push`; it returns audio
    samples as soon as a full body chunk (plus its right context) is available.
    Call :meth:`flush` once generation ends to drain the tail.
    """

    def __init__(self, decode_fn: DecodeFn, config: StreamingDecodeConfig):
        self.decode_fn = decode_fn
        self.cfg = config
        # All frames received so far, shape (1, C, T). Kept on the codes' device.
        self._frames: Optional[torch.Tensor] = None
        # Index (in frames) of the next body frame that still needs to be emitted.
        self._emitted_frames = 0
        self._spf = config.samples_per_frame

    @property
    def _num_frames(self) -> int:
        return 0 if self._frames is None else self._frames.size(-1)

    def _append(self, codes: torch.Tensor) -> None:
        # codes: (C, T_new) or (1, C, T_new)
        if codes.dim() == 2:
            codes = codes.unsqueeze(0)
        if self._frames is None:
            self._frames = codes
        else:
            self._frames = torch.cat([self._frames, codes], dim=-1)

    def _decode_window(self, start: int, end: int) -> torch.Tensor:
        """Decode frames [start, end) plus their context, return only the body audio."""
        L, R = self.cfg.left_context, self.cfg.right_context
        win_start = max(0, start - L)
        win_end = min(self._num_frames, end + R)
        left_pad = start - win_start  # actual left-context frames included
        window = self._frames[:, :, win_start:win_end].contiguous()
        win_len = torch.tensor([window.size(-1)], dtype=torch.long, device=window.device)

        audio, audio_len = self.decode_fn(window, win_len)  # (1, T_audio)
        valid = int(audio_len[0].item())
        audio = audio[:, :valid]

        spf = self._spf
        if spf is None:
            # Infer once. NOTE: the NeMo codec helper zero-pads windows shorter than
            # ``_CODEC_MIN_DECODE_FRAMES`` up to that length, so a short first window
            # decodes to MORE audio than ``window_frames * spf``. Divide by the padded
            # length to recover the true spf. (Prefer setting ``samples_per_frame``
            # explicitly via ``StreamingDecodeConfig.for_codec`` to skip this entirely.)
            decoded_frames = max(window.size(-1), _CODEC_MIN_DECODE_FRAMES)
            spf = valid // decoded_frames
            self._spf = spf

        body_frames = end - start
        body_start = left_pad * spf
        body_end = body_start + body_frames * spf
        body_end = min(body_end, audio.size(-1))
        return audio[:, body_start:body_end]

    def push(self, codes: torch.Tensor) -> torch.Tensor:
        """Add newly generated frames and emit any audio that is now ready.

        Args:
            codes: New codec frames, shape ``(C, T_new)`` or ``(1, C, T_new)``.

        Returns:
            Audio samples ready to play, shape ``(1, T_emit)`` (possibly empty).
        """
        self._append(codes)
        N, R = self.cfg.chunk_frames, self.cfg.right_context
        out_chunks: List[torch.Tensor] = []
        # Emit while a whole body chunk AND its right lookahead are buffered.
        while self._emitted_frames + N + R <= self._num_frames:
            start = self._emitted_frames
            end = start + N
            out_chunks.append(self._decode_window(start, end))
            self._emitted_frames = end
        if not out_chunks:
            device = self._frames.device
            return torch.zeros(1, 0, device=device)
        return torch.cat(out_chunks, dim=-1)

    def flush(self) -> torch.Tensor:
        """Drain remaining frames after the stream ends (uses whatever right context exists)."""
        out_chunks: List[torch.Tensor] = []
        N = self.cfg.chunk_frames
        while self._emitted_frames < self._num_frames:
            start = self._emitted_frames
            end = min(start + N, self._num_frames)
            out_chunks.append(self._decode_window(start, end))
            self._emitted_frames = end
        if not out_chunks:
            device = "cpu" if self._frames is None else self._frames.device
            return torch.zeros(1, 0, device=device)
        return torch.cat(out_chunks, dim=-1)


def chunked_decode(
    codes: torch.Tensor,
    codes_len: torch.Tensor,
    decode_fn: DecodeFn,
    config: StreamingDecodeConfig,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One-shot overlap-save decode of a complete code sequence (batch supported).

    Useful to validate that chunked decoding matches a full one-shot decode, and
    to bound peak memory for very long utterances. Each batch item is decoded
    independently with the same overlap-save schedule.

    Args:
        codes: ``(B, C, T)`` codec codes.
        codes_len: ``(B,)`` valid frame counts.
        decode_fn: See module docstring.
        config: Overlap-save configuration.

    Returns:
        ``(audio [B, T_audio], audio_len [B])``.
    """
    B = codes.size(0)
    device = codes.device
    per_item_audio: List[torch.Tensor] = []
    audio_lens = torch.zeros(B, dtype=torch.long, device=device)
    for b in range(B):
        n = int(codes_len[b].item())
        dec = StreamingCodecDecoder(decode_fn, config)
        emitted = dec.push(codes[b : b + 1, :, :n])
        tail = dec.flush()
        audio = torch.cat([emitted, tail], dim=-1)  # (1, T_audio)
        per_item_audio.append(audio)
        audio_lens[b] = audio.size(-1)

    max_len = int(audio_lens.max().item()) if B > 0 else 0
    out = torch.zeros(B, max_len, device=device, dtype=per_item_audio[0].dtype if per_item_audio else torch.float32)
    for b, audio in enumerate(per_item_audio):
        out[b, : audio.size(-1)] = audio[0]
    return out, audio_lens


def crossfade_concat(prev_tail: torch.Tensor, next_head: torch.Tensor, overlap: int) -> torch.Tensor:
    """Linear-equal-power cross-fade two overlapping audio chunks (R=0 fallback).

    Only needed when running with ``right_context = 0`` (zero lookahead) and you
    decoded adjacent chunks with an ``overlap``-sample overlap. This is the single
    place overlap-*add* is used; prefer overlap-save (trim) when latency allows.

    Args:
        prev_tail: Tail of the previous chunk, shape ``(1, >= overlap)``.
        next_head: Head of the next chunk, shape ``(1, >= overlap)``.
        overlap: Number of overlapping samples to blend.

    Returns:
        Blended overlap region, shape ``(1, overlap)``.
    """
    device = prev_tail.device
    t = torch.linspace(0, 1, overlap, device=device).unsqueeze(0)
    fade_out = torch.cos(t * torch.pi / 2)
    fade_in = torch.sin(t * torch.pi / 2)
    return prev_tail[:, -overlap:] * fade_out + next_head[:, :overlap] * fade_in


@torch.inference_mode()
def measure_left_receptive_field(decode_fn: DecodeFn, num_codebooks: int, device="cuda", max_frames: int = 64) -> int:
    """Empirically measure the decoder's *past* receptive field, in input frames.

    Decode a random sequence twice -- once intact, once with the *first* frame
    perturbed -- and find how far into the future the output difference persists.
    The number of trailing frames that are byte-identical tells you how many
    leading frames are influenced by frame 0; ``max_frames - that`` is the past
    receptive field. Use the returned value (plus a margin) as ``left_context``.

    Args:
        decode_fn: Same signature used everywhere else in this module.
        num_codebooks: C (e.g. 5 for this codec).
        device: Device to run on.
        max_frames: Sequence length to probe; must exceed the true RF.

    Returns:
        Estimated past receptive field in frames.
    """
    g = torch.Generator(device="cpu").manual_seed(0)
    codes = torch.randint(0, 2, (1, num_codebooks, max_frames), generator=g).to(device)
    codes_perturbed = codes.clone()
    # Flip the first frame to something different.
    codes_perturbed[:, :, 0] = 1 - codes_perturbed[:, :, 0]

    lens = torch.tensor([max_frames], dtype=torch.long, device=device)
    a0, _ = decode_fn(codes, lens)
    a1, _ = decode_fn(codes_perturbed, lens)
    diff = (a0 - a1).abs().squeeze(0)  # (T_audio,)

    spf = diff.numel() // max_frames
    # Per-frame max abs difference; the RF is the last frame still affected by frame 0.
    per_frame = diff[: spf * max_frames].reshape(max_frames, spf).amax(dim=-1)
    affected = (per_frame > 1e-6).nonzero().flatten()
    rf = int(affected.max().item()) + 1 if affected.numel() else 0
    return rf
