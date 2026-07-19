# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Small direct runner for listening to the native vLLM codec in notebooks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import torch
from easymagpie_codec_vllm.model import EasyMagpieCodecForConditionalGeneration
from easymagpie_codec_vllm.packed import CODEC_STATE_ELEMENTS, CodecStateLayer
from safetensors.torch import load_file
from vllm.config import DeviceConfig, ModelConfig, VllmConfig, set_current_vllm_config
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata
from vllm_plugin_easymagpie_codec import register


@dataclass
class DecodeResult:
    """Waveform and timing information returned by ``StatefulCodecRunner``."""

    audio: torch.Tensor
    chunks: list[torch.Tensor]
    chunk_sizes: list[int]
    elapsed_ms: list[float]


def make_chunk_plan(
    total_frames: int,
    *,
    startup: tuple[int, ...] = (1, 1, 2),
    steady: int = 6,
) -> list[int]:
    """Build a low-TTFA then throughput-oriented chunk plan."""
    if total_frames < 0:
        raise ValueError("total_frames must be nonnegative")
    if steady <= 0 or any(size <= 0 for size in startup):
        raise ValueError("all chunk sizes must be positive")

    remaining = total_frames
    plan: list[int] = []
    for requested in startup:
        if remaining == 0:
            break
        size = min(requested, remaining)
        plan.append(size)
        remaining -= size
    while remaining:
        size = min(steady, remaining)
        plan.append(size)
        remaining -= size
    return plan


def normalize_acoustic_tokens(codes: torch.Tensor, *, num_stacked_codebooks: int = 16) -> torch.Tensor:
    """Normalize a predictor result to contiguous time-major ``[T, C*S]``."""
    codes = torch.as_tensor(codes)
    if codes.dim() == 3 and codes.shape[0] == 1:
        codes = codes[0]
    if codes.dim() != 2:
        raise ValueError(f"expected [T, {num_stacked_codebooks}] acoustic codes, got {tuple(codes.shape)}")
    if codes.shape[-1] != num_stacked_codebooks:
        if codes.shape[0] == num_stacked_codebooks:
            codes = codes.transpose(0, 1)
        else:
            raise ValueError(f"expected {num_stacked_codebooks} stacked codebooks, got {tuple(codes.shape)}")
    return codes.to(device="cpu", dtype=torch.long).contiguous()


def load_acoustic_tokens(path: str | Path, *, num_stacked_codebooks: int = 16) -> torch.Tensor:
    """Load ``audio_codes`` saved by ``easymagpie_vllm_omni/benchmark_model.py``."""
    path = Path(path)
    if path.suffix == ".pt":
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(payload, dict):
            if "audio_codes" not in payload:
                raise KeyError(f"{path} has no 'audio_codes' entry")
            payload = payload["audio_codes"]
    elif path.suffix == ".npy":
        import numpy as np

        payload = torch.from_numpy(np.load(path))
    else:
        raise ValueError(f"unsupported token file {path}; expected .pt or .npy")
    return normalize_acoustic_tokens(payload, num_stacked_codebooks=num_stacked_codebooks)


def _prefill_metadata(
    frames: int,
    *,
    has_initial: bool,
    total_frames: int,
    device: torch.device,
) -> Mamba1AttentionMetadata:
    metadata = Mamba1AttentionMetadata(
        num_prefills=1,
        num_prefill_tokens=frames,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=1,
        has_initial_states_p=torch.tensor([has_initial], device=device),
        query_start_loc_p=torch.tensor([0, frames], dtype=torch.int32, device=device),
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.tensor([0], dtype=torch.int32, device=device),
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([total_frames], dtype=torch.int32, device=device),
    )
    metadata.codec_max_query_len = frames
    metadata.codec_uniform = True
    return metadata


def _decode_metadata(*, total_frames: int, device: torch.device) -> Mamba1AttentionMetadata:
    metadata = Mamba1AttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_reqs=1,
        has_initial_states_p=None,
        query_start_loc_p=None,
        num_computed_tokens_p=None,
        state_indices_tensor_p=None,
        state_indices_tensor_d=torch.tensor([[0]], dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.tensor([total_frames], dtype=torch.int32, device=device),
    )
    metadata.codec_uniform = True
    return metadata


class StatefulCodecRunner:
    """Load the registered vLLM model and drive one persistent cache page.

    This is a direct model runner for experimentation, not a replacement for
    vLLM-Omni scheduling. It uses the same model adapter, metadata, state pages,
    weight loader, and packed CUDA kernels as the eventual generation stage.
    """

    def __init__(
        self,
        checkpoint_dir: str | Path,
        *,
        device: str | torch.device = "cuda",
        dtype: str = "float32",
        max_model_len: int = 4096,
    ) -> None:
        checkpoint_dir = Path(checkpoint_dir).resolve()
        if not (checkpoint_dir / "config.json").is_file():
            raise FileNotFoundError(f"missing {checkpoint_dir / 'config.json'}")
        if not (checkpoint_dir / "model.safetensors").is_file():
            raise FileNotFoundError(f"missing {checkpoint_dir / 'model.safetensors'}")

        register()
        self.device = torch.device(device)
        model_config = ModelConfig(
            model=str(checkpoint_dir),
            skip_tokenizer_init=True,
            dtype=dtype,
            max_model_len=max_model_len,
        )
        self.vllm_config = VllmConfig(
            model_config=model_config,
            device_config=DeviceConfig(self.device),
        )
        with set_current_vllm_config(self.vllm_config):
            self.model = EasyMagpieCodecForConditionalGeneration(vllm_config=self.vllm_config).eval()
        weights = load_file(checkpoint_dir / "model.safetensors", device="cpu")
        self.model.load_weights(weights.items())
        self.model.to(device=self.device, dtype=model_config.dtype)

        self.state_layers = [layer for layer in self.model.modules() if isinstance(layer, CodecStateLayer)]
        if not self.state_layers:
            raise RuntimeError("the codec model registered no cache-owning layers")
        for layer in self.state_layers:
            layer.kv_cache = [
                torch.zeros(
                    (1, CODEC_STATE_ELEMENTS),
                    dtype=layer.dtype,
                    device=self.device,
                )
            ]

    @property
    def sample_rate(self) -> int:
        return self.model.config.output_sample_rate

    @property
    def samples_per_frame(self) -> int:
        return self.model.config.samples_per_frame

    @property
    def num_stacked_codebooks(self) -> int:
        return self.model.config.num_stacked_codebooks

    def reset(self) -> None:
        """Clear every convolution/deconvolution history page."""
        for layer in self.state_layers:
            layer.kv_cache[0].zero_()

    def warmup(self, chunk_sizes: tuple[int, ...] = (1, 1, 2, 6)) -> None:
        """Compile both one-frame decode and variable prefill CUDA paths."""
        total = sum(chunk_sizes)
        codes = torch.zeros((total, self.num_stacked_codebooks), dtype=torch.long)
        self.decode(codes, list(chunk_sizes), reset=True)
        self.reset()

    @torch.inference_mode()
    def decode(
        self,
        codes: torch.Tensor,
        chunk_sizes: list[int] | tuple[int, ...],
        *,
        reset: bool = True,
    ) -> DecodeResult:
        """Decode one predictor sequence, preserving state between chunks."""
        codes = normalize_acoustic_tokens(codes, num_stacked_codebooks=self.num_stacked_codebooks)
        chunk_sizes = list(chunk_sizes)
        if any(size <= 0 for size in chunk_sizes) or sum(chunk_sizes) != codes.shape[0]:
            raise ValueError(f"chunk sizes {chunk_sizes} must be positive and sum to {codes.shape[0]}")
        if codes.numel() and (codes.min().item() < 0 or codes.max().item() >= self.model.config.codebook_size):
            raise ValueError(f"codec indices must be in [0, {self.model.config.codebook_size})")
        if reset:
            self.reset()

        chunks: list[torch.Tensor] = []
        elapsed_ms: list[float] = []
        offset = 0
        total_frames = 0
        for chunk_index, size in enumerate(chunk_sizes):
            total_frames += size
            if chunk_index > 0 and size == 1:
                metadata = _decode_metadata(total_frames=total_frames, device=self.device)
            else:
                metadata = _prefill_metadata(
                    size,
                    has_initial=chunk_index > 0,
                    total_frames=total_frames,
                    device=self.device,
                )
            layer_metadata = {layer.prefix: metadata for layer in self.state_layers}
            input_ids = torch.zeros((size,), dtype=torch.long, device=self.device)
            chunk_codes = codes[offset : offset + size].to(self.device)

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            started = time.perf_counter()
            with set_forward_context(layer_metadata, self.vllm_config, num_tokens=size):
                output = self.model(input_ids=input_ids, codec_codes=chunk_codes)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            elapsed_ms.append((time.perf_counter() - started) * 1000.0)

            audio = output.multimodal_outputs["model_outputs"][0].detach().cpu()
            expected_samples = size * self.samples_per_frame
            if audio.numel() != expected_samples:
                raise RuntimeError(f"expected {expected_samples} audio samples, got {audio.numel()}")
            chunks.append(audio)
            offset += size

        audio = torch.cat(chunks) if chunks else torch.empty((0,), dtype=torch.float32)
        return DecodeResult(audio=audio, chunks=chunks, chunk_sizes=chunk_sizes, elapsed_ms=elapsed_ms)
