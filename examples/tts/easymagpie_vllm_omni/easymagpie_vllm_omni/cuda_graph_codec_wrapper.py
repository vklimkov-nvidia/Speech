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
"""CUDA-Graph wrapper for the EasyMagpieTTS codec decoder.

The EasyMagpie codec decode path (clamp specials -> unstack -> FSQ index
convert -> ``AudioCodecModel.decode``) is a static, data-independent graph for a
fixed frame count, so it is a good CUDA-graph capture target: replaying a
captured graph removes the per-launch kernel overhead that dominates the small
per-chunk decodes issued in streaming (async-chunk) mode.

This mirrors ``qwen3_tts/cuda_graph_decoder_wrapper.py`` but is specialized to
the EasyMagpie decode module (see :class:`easymagpie_vllm_omni.code2wav`):

* The captured module consumes **stacked model codes** of shape
  ``(batch, frames, num_stacked_codebooks)`` (int64) and returns a waveform of
  shape ``(batch, frames * samples_per_frame)`` (fp32).
* Capture happens per ``(batch_size, frames)`` bucket; at decode time the actual
  frame count is padded up to the nearest captured bucket and the replayed
  output is trimmed back to ``actual_frames * samples_per_frame``.
* Any shape without a captured bucket (e.g. an unusually long non-streaming
  sequence) falls back to eager decode, so correctness never depends on a graph
  being present.
"""
from __future__ import annotations

import bisect
import time
from collections.abc import Sequence

import torch
from torch.cuda import CUDAGraph
from vllm.logger import init_logger

logger = init_logger(__name__)


def _dedup_sorted(values: Sequence[int]) -> list[int]:
    return sorted({int(v) for v in values if int(v) > 0})


class CUDAGraphCodecDecoder:
    """Capture + replay the EasyMagpie codec decode module for fixed shapes.

    Args:
        decode_module: an ``nn.Module`` whose ``forward(codes)`` takes an int64
            ``(batch, frames, num_stacked_codebooks)`` tensor and returns a fp32
            ``(batch, frames * samples_per_frame)`` waveform.
        num_stacked_codebooks: the ``Q`` (``= C * S``) code channel count.
        samples_per_frame: waveform samples produced per **model** frame (one
            model frame unstacks to ``frame_stacking_factor`` codec frames).
        capture_frames: frame-count buckets to capture graphs for.
        capture_batch_sizes: batch sizes to capture (default ``[1]``).
        enabled: master switch; when ``False`` every call decodes eagerly.
    """

    def __init__(
        self,
        decode_module: torch.nn.Module,
        *,
        num_stacked_codebooks: int,
        samples_per_frame: int,
        capture_frames: Sequence[int] | None = None,
        capture_batch_sizes: Sequence[int] | None = None,
        enabled: bool = True,
    ) -> None:
        self.decode_module = decode_module
        self.num_stacked_codebooks = int(num_stacked_codebooks)
        self.samples_per_frame = int(samples_per_frame)
        self.capture_frames = _dedup_sorted(capture_frames or [])
        self.capture_batch_sizes = _dedup_sorted(capture_batch_sizes or [1]) or [1]
        self.enabled = enabled

        self.graphs: dict[tuple[int, int], CUDAGraph] = {}
        self.static_inputs: dict[tuple[int, int], torch.Tensor] = {}
        self.static_outputs: dict[tuple[int, int], torch.Tensor] = {}
        self._warmed_up = False
        self._device: torch.device | None = None

    # ------------------------------------------------------------------
    # capture / warmup
    # ------------------------------------------------------------------
    def warmup(self, device: torch.device) -> None:
        if device.type != "cuda" or not self.enabled or self._warmed_up:
            return
        if not self.capture_frames:
            logger.info("EasyMagpie Code2Wav CUDA Graph: no capture frame buckets configured; skipping.")
            self._warmed_up = True
            return

        self._device = device
        self.decode_module.eval()
        shapes = sorted({(bs, f) for bs in self.capture_batch_sizes for f in self.capture_frames})

        logger.info(
            "EasyMagpie Code2Wav CUDA Graph warmup for %d shapes: batch_sizes=%s frame_buckets=%s",
            len(shapes),
            self.capture_batch_sizes,
            self.capture_frames,
        )
        start_s = time.perf_counter()

        # Eager warmup so lazy allocations happen before capture.
        for bs, frames in shapes:
            dummy = torch.zeros(bs, frames, self.num_stacked_codebooks, dtype=torch.long, device=device)
            with torch.no_grad():
                _ = self.decode_module(dummy)
        torch.cuda.synchronize(device)

        for bs, frames in shapes:
            try:
                self._capture(bs, frames, device)
            except Exception:
                logger.warning("  Failed to capture Code2Wav graph batch=%d frames=%d", bs, frames, exc_info=True)

        self._warmed_up = True
        logger.info(
            "EasyMagpie Code2Wav CUDA Graph warmup complete: %d/%d captured in %.1f ms",
            len(self.graphs),
            len(shapes),
            (time.perf_counter() - start_s) * 1000.0,
        )

    def _capture(self, batch_size: int, frames: int, device: torch.device) -> None:
        from vllm.platforms import current_platform

        key = (batch_size, frames)
        static_input = torch.zeros(batch_size, frames, self.num_stacked_codebooks, dtype=torch.long, device=device)
        with torch.no_grad():
            _ = self.decode_module(static_input)
        torch.cuda.synchronize(device)

        graph = CUDAGraph()
        with torch.no_grad():
            with torch.cuda.graph(graph, pool=current_platform.get_global_graph_pool()):
                static_output = self.decode_module(static_input)

        self.graphs[key] = graph
        self.static_inputs[key] = static_input
        self.static_outputs[key] = static_output
        logger.info("  Captured Code2Wav graph batch=%d frames=%d", batch_size, frames)

    # ------------------------------------------------------------------
    # replay
    # ------------------------------------------------------------------
    def _padded_frames(self, actual_frames: int) -> int | None:
        idx = bisect.bisect_left(self.capture_frames, actual_frames)
        if idx < len(self.capture_frames):
            return self.capture_frames[idx]
        return None

    @torch.no_grad()
    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode ``codes`` of shape ``(B, frames, Q)`` -> ``(B, frames*spf)``.

        Uses a captured graph when a matching ``(batch, padded_frames)`` bucket
        exists, otherwise falls back to eager decode.
        """
        if codes.dim() != 3:
            raise ValueError(f"EasyMagpie Code2Wav expects (B, frames, Q) codes, got {tuple(codes.shape)}")
        batch_size = int(codes.shape[0])
        actual_frames = int(codes.shape[1])

        if actual_frames == 0:
            return codes.new_zeros((batch_size, 0), dtype=torch.float32)

        # Inner replay is illegal while an outer stream capture is active
        # (e.g. vLLM full-cudagraph warmup on the Stage-1 runner).
        if (
            not self.enabled
            or not self._warmed_up
            or torch.cuda.is_current_stream_capturing()
        ):
            return self.decode_module(codes)

        padded = self._padded_frames(actual_frames)
        key = (batch_size, padded) if padded is not None else None
        if key is None or key not in self.graphs:
            return self.decode_module(codes)

        static_input = self.static_inputs[key]
        if actual_frames == padded:
            static_input.copy_(codes)
        else:
            static_input.zero_()
            static_input[:, :actual_frames, :] = codes
        self.graphs[key].replay()

        out = self.static_outputs[key]
        keep = actual_frames * self.samples_per_frame
        keep = min(keep, int(out.shape[-1]))
        return out[..., :keep].clone()
