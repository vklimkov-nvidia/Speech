#!/usr/bin/env python3
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
"""Compare converted native weights with a NeMo or torch.export reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from easymagpie_codec_vllm.codec import EasyMagpieCodec
from easymagpie_codec_vllm.config import EasyMagpieCodecConfig
from safetensors.torch import load_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path, help="Original codec .nemo or exported codec_decoder.pt2")
    parser.add_argument("converted", type=Path, help="Output of convert_codec.py")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--atol", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def build_nemo_reference(codec_path: Path, config: EasyMagpieCodecConfig, device: torch.device):
    from nemo.collections.tts.models import AudioCodecModel
    from nemo.collections.tts.modules.audio_codec_modules import (
        GroupFiniteScalarQuantizer,
        VectorQuantizerIndexConverter,
    )

    codec_config = AudioCodecModel.restore_from(str(codec_path), return_config=True)
    if "use_scl_loss" in codec_config:
        codec_config.use_scl_loss = False
    codec = AudioCodecModel.restore_from(
        str(codec_path),
        strict=False,
        override_config_path=codec_config,
        map_location="cpu",
    )
    if hasattr(codec, "discriminator"):
        del codec.discriminator
    codec = codec.to(device=device).eval().float()
    codec.freeze()
    codec.audio_decoder.remove_weight_norm()

    input_vq = GroupFiniteScalarQuantizer(
        num_groups=config.num_codebooks,
        num_levels_per_group=config.num_levels_per_group,
    ).to(device=device)
    converter = VectorQuantizerIndexConverter(
        vector_quantizer_original=codec.vector_quantizer,
        vector_quantizer_new=input_vq,
    ).to(device=device)

    @torch.no_grad()
    def decode(codes: torch.Tensor) -> torch.Tensor:
        tokens = codes.transpose(1, 2).contiguous()
        batch, stacked_codebooks, frames = tokens.shape
        tokens = (
            tokens.view(batch, stacked_codebooks // config.frame_stacking_factor, config.frame_stacking_factor, frames)
            .permute(0, 1, 3, 2)
            .reshape(batch, config.num_codebooks, frames * config.frame_stacking_factor)
        )
        lengths = torch.full((batch,), tokens.shape[-1], dtype=torch.long, device=device)
        original_tokens = converter.convert_new_to_original(tokens, lengths)
        audio, _ = codec.decode(original_tokens, lengths)
        return audio

    return decode


def build_exported_reference(codec_path: Path, device: torch.device):
    module = torch.export.load(codec_path).module().to(device=device)

    @torch.no_grad()
    def decode(codes: torch.Tensor) -> torch.Tensor:
        return module(codes)

    return decode


@torch.no_grad()
def main() -> None:
    args = parse_args()
    if sum(args.chunks) != args.frames:
        raise ValueError(f"chunk sizes {args.chunks} must sum to --frames={args.frames}")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    config = EasyMagpieCodecConfig.from_pretrained(args.converted)
    native = EasyMagpieCodec(config).to(device=device).eval()
    native.load_state_dict(load_file(args.converted / "model.safetensors", device=str(device)), strict=True)
    reference = (
        build_exported_reference(args.reference, device)
        if args.reference.suffix == ".pt2"
        else build_nemo_reference(args.reference, config, device)
    )

    codes = torch.randint(
        0,
        config.codebook_size,
        (1, args.frames, config.num_stacked_codebooks),
        dtype=torch.long,
        device=device,
    )
    expected = reference(codes)
    full = native(codes)
    full_diff = float((full - expected).abs().max().item())

    state = None
    offset = 0
    pieces = []
    for chunk in args.chunks:
        output, state = native.stream(codes[:, offset : offset + chunk], state)
        pieces.append(output)
        offset += chunk
    streamed = torch.cat(pieces, dim=-1)
    stream_diff = float((streamed - full).abs().max().item())

    print(f"reference vs native max abs diff: {full_diff:.8f}")
    print(f"full vs streamed max abs diff: {stream_diff:.8f}")
    print(f"output shape: {tuple(full.shape)} ({config.samples_per_frame} samples/model-frame)")
    if full_diff > args.atol or stream_diff > args.atol:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
