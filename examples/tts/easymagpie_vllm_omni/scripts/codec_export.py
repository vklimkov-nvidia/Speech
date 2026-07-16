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
"""Utilities for exporting the NeMo EasyMagpie codec decoder with ``torch.export``.

This module is used only in the NeMo conversion environment. The resulting
``ExportedProgram`` contains the original NeMo codec implementation and weights,
so the vLLM serving environment needs PyTorch but does not need NeMo.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn


class CodecDecoderExportWrapper(nn.Module):
    """Stacked EasyMagpie codes ``(B, T, Q)`` to a fp32 waveform ``(B, L)``."""

    def __init__(
        self,
        codec_model: nn.Module,
        converter: nn.Module | None,
        stacking: int,
        clamp_max: int,
    ) -> None:
        super().__init__()
        self.codec_model = codec_model
        self.converter = converter
        self.stacking = int(stacking)
        self.clamp_max = int(clamp_max)

    def forward(self, audio_codes: torch.Tensor) -> torch.Tensor:
        tokens = audio_codes.transpose(1, 2).contiguous()
        batch = tokens.shape[0]

        if self.stacking > 1:
            stacked_codebooks, frames = tokens.shape[1], tokens.shape[2]
            codebooks = stacked_codebooks // self.stacking
            tokens = (
                tokens.view(batch, codebooks, self.stacking, frames)
                .permute(0, 1, 3, 2)
                .reshape(batch, codebooks, frames * self.stacking)
            )

        tokens = tokens.clamp(0, self.clamp_max).contiguous()
        token_frames = tokens.shape[2]
        tokens_len = torch.full((batch,), token_frames, dtype=torch.long, device=tokens.device)
        if self.converter is not None:
            tokens = self.converter.convert_new_to_original(audio_tokens=tokens, audio_lens=tokens_len)

        audio, _ = self.codec_model.decode(tokens=tokens, tokens_len=tokens_len)
        if audio.dim() == 3:
            audio = audio.squeeze(1)
        return audio.float()


def _remove_weight_norm_recursive(module: nn.Module) -> int:
    """Fold modern weight-norm parametrizations into ordinary parameters."""
    import torch.nn.utils.parametrize as parametrize

    folded = 0
    for child in module.modules():
        if parametrize.is_parametrized(child, "weight"):
            parametrize.remove_parametrizations(child, "weight", leave_parametrized=True)
            folded += 1
    return folded


def _patch_codec_for_export(module: nn.Module) -> int:
    """Replace CUDA-synchronizing causal-conv padding with static host arithmetic."""
    from nemo.collections.common.parts.utils import mask_sequence_tensor

    def _make_forward(mod: nn.Module):
        kernel_size = int(mod.kernel_size)
        stride = int(mod.stride)
        padding_total = int(mod.padding_total)

        def forward(inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
            length = inputs.shape[-1]
            n_frames = math.ceil((length - kernel_size + padding_total) / stride)
            ideal_length = n_frames * stride + kernel_size - padding_total
            extra_padding = ideal_length - length
            hidden_states = mod._pad1d(inputs, (padding_total, extra_padding), mode=mod.extra_pad_mode)
            hidden_states = mod.conv(hidden_states)
            hidden_states = mod.activation(hidden_states)
            return mask_sequence_tensor(hidden_states, input_len)

        return forward

    patched = 0
    for child in module.modules():
        if type(child).__name__ == "CausalConv1dNorm":
            child.forward = _make_forward(child)
            patched += 1
    return patched


def build_codec_decoder(model: nn.Module, device: torch.device) -> tuple[nn.Module, dict]:
    """Build the exact NeMo codec path that will be captured by ``torch.export``."""
    from hydra.utils import instantiate

    from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter

    codec = model._codec_model
    if hasattr(codec, "discriminator"):
        del codec.discriminator
    codec = codec.to(device=device).eval().float()
    codec.freeze()
    folded = _remove_weight_norm_recursive(codec.audio_decoder)

    stacking = int(model.frame_stacking_factor)
    num_audio_codebooks = int(model.num_audio_codebooks)
    codebook_size = int(model.codebook_size)
    num_stacked_codebooks = num_audio_codebooks * stacking

    converter = None
    model_vq_cfg = model.cfg.get("vector_quantizer")
    if model_vq_cfg is not None:
        vq_new = instantiate(model_vq_cfg).to(device=device).eval()
        if int(vq_new.num_codebooks) != int(codec.vector_quantizer.num_codebooks):
            converter = (
                VectorQuantizerIndexConverter(
                    vector_quantizer_original=codec.vector_quantizer,
                    vector_quantizer_new=vq_new,
                )
                .to(device=device)
                .eval()
            )

    wrapper = (
        CodecDecoderExportWrapper(
            codec_model=codec,
            converter=converter,
            stacking=stacking,
            clamp_max=codebook_size - 1,
        )
        .to(device=device)
        .eval()
    )
    patched = _patch_codec_for_export(wrapper)
    sample_rate = int(getattr(codec, "output_sample_rate", getattr(codec, "sample_rate", 22050)))
    info = {
        "num_stacked_codebooks": num_stacked_codebooks,
        "codebook_size": codebook_size,
        "frame_stacking_factor": stacking,
        "output_sample_rate": sample_rate,
        "weight_norm_layers_folded": folded,
        "causal_convs_patched": patched,
    }
    return wrapper, info


@torch.no_grad()
def export_codec_decoder(
    model: nn.Module,
    output_path: str | Path,
    metadata_path: str | Path,
    *,
    frames: int,
    max_batch_size: int,
    device: torch.device,
    atol: float = 2e-3,
) -> dict:
    """Export, save, reload, and verify a fixed-frame, dynamic-batch codec."""
    if frames <= 0:
        raise ValueError(f"frames must be positive, got {frames}")
    if max_batch_size <= 0:
        raise ValueError(f"max_batch_size must be positive, got {max_batch_size}")

    wrapper, info = build_codec_decoder(model, device)
    q = int(info["num_stacked_codebooks"])
    codebook_size = int(info["codebook_size"])
    example_batch = min(2, max_batch_size)
    example = torch.randint(0, codebook_size, (example_batch, frames, q), dtype=torch.long, device=device)

    dynamic_shapes = None
    if max_batch_size > 1:
        batch = torch.export.Dim("batch", min=1, max=max_batch_size)
        dynamic_shapes = {"audio_codes": {0: batch}}

    exported = torch.export.export(wrapper, (example,), dynamic_shapes=dynamic_shapes, strict=False)
    output_path = Path(output_path)
    metadata_path = Path(metadata_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.export.save(exported, output_path)

    loaded_module = torch.export.load(output_path).module().to(device=device)
    reference = wrapper(example)
    actual = loaded_module(example)
    max_abs_diff = float((reference - actual).abs().max().item())
    if max_abs_diff > atol:
        raise RuntimeError(
            f"Reloaded codec ExportedProgram parity failed: max_abs_diff={max_abs_diff:.6g}, atol={atol}"
        )

    samples_per_frame = int(actual.shape[-1]) // frames
    if samples_per_frame <= 0 or int(actual.shape[-1]) != frames * samples_per_frame:
        raise RuntimeError(
            f"Codec output length {int(actual.shape[-1])} is not a positive integer multiple of {frames} frames"
        )

    metadata = {
        "format": "torch.export",
        "format_version": 1,
        "torch_version": torch.__version__,
        "frames": frames,
        "max_batch_size": max_batch_size,
        "samples_per_frame": samples_per_frame,
        "parity_max_abs_diff": max_abs_diff,
        **info,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata
