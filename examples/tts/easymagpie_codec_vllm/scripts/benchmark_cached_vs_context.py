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
"""Benchmark cached native codec chunks against stateless CUDA-graph context replay."""

from __future__ import annotations

import argparse
import gc
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from easymagpie_codec_vllm.codec import EasyMagpieCodec
from easymagpie_codec_vllm.config import EasyMagpieCodecConfig
from easymagpie_codec_vllm.demo import (
    StatefulCodecRunner,
    _decode_metadata,
    _prefill_metadata,
    load_acoustic_tokens,
)
from safetensors.torch import load_file
from vllm.forward_context import set_forward_context


@dataclass
class Timing:
    wall_p50_ms: float
    wall_p95_ms: float
    cuda_p50_ms: float
    cuda_p95_ms: float
    saturated_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-checkpoint", type=Path, required=True)
    parser.add_argument("--exported-codec", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--left-context", type=int, default=11)
    parser.add_argument("--chunks", type=int, nargs="+", default=[1, 2, 5, 6, 8])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def measure(call: Callable[[], Any], *, warmup: int, iterations: int) -> Timing:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_ms: list[float] = []
    cuda_ms: list[float] = []
    result = None
    for _ in range(iterations):
        started = time.perf_counter()
        start_event.record()
        result = call()
        end_event.record()
        end_event.synchronize()
        wall_ms.append((time.perf_counter() - started) * 1000.0)
        cuda_ms.append(start_event.elapsed_time(end_event))
    torch.cuda.synchronize()
    saturated_started = time.perf_counter()
    for _ in range(iterations):
        result = call()
    torch.cuda.synchronize()
    saturated_ms = (time.perf_counter() - saturated_started) * 1000.0 / iterations

    del result
    return Timing(
        wall_p50_ms=float(np.percentile(wall_ms, 50)),
        wall_p95_ms=float(np.percentile(wall_ms, 95)),
        cuda_p50_ms=float(np.percentile(cuda_ms, 50)),
        cuda_p95_ms=float(np.percentile(cuda_ms, 95)),
        saturated_ms=saturated_ms,
    )


class CapturedNativeCodec:
    """Capture the complete native cached forward."""

    def __init__(self, runner: StatefulCodecRunner, codes: torch.Tensor, *, total_frames: int) -> None:
        self.static_codes = codes.clone()
        self.call = native_call(runner, self.static_codes, total_frames=total_frames)
        for _ in range(3):
            self.call()
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            output = self.call()
            self.static_output = output.multimodal_outputs["model_outputs"][0]

    def replay_only(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.static_codes.copy_(codes)
        self.graph.replay()
        return self.static_output.clone()


class CapturedStatelessCodec:
    """Capture an exact-shape full-context call around the dense codec oracle."""

    def __init__(self, model: EasyMagpieCodec, codes: torch.Tensor) -> None:
        self.static_input = torch.empty_like(codes)
        self.static_input.copy_(codes)
        for _ in range(3):
            model(self.static_input)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_output = model(self.static_input)

    def replay_only(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.static_input.copy_(codes)
        self.graph.replay()
        return self.static_output.clone()


def load_deployed_graph(exported_codec: Path, device: torch.device):
    from easymagpie_vllm_omni.code2wav import _ExportedCodecDecoder
    from easymagpie_vllm_omni.cuda_graph_codec_wrapper import CUDAGraphCodecDecoder

    suffix = exported_codec.stem.removeprefix("codec_decoder")
    metadata_path = exported_codec.with_name(f"codec_export{suffix}.json")
    metadata = json.loads(metadata_path.read_text())
    frames = int(metadata["frames"])
    quantizers = int(metadata["num_stacked_codebooks"])
    samples_per_frame = int(metadata["samples_per_frame"])
    module = _ExportedCodecDecoder(str(exported_codec), device)
    graph = CUDAGraphCodecDecoder(
        module,
        num_stacked_codebooks=quantizers,
        samples_per_frame=samples_per_frame,
        capture_frames=[frames],
        capture_batch_sizes=[1],
    )
    graph.warmup(device)
    if (1, frames) not in graph.graphs:
        raise RuntimeError(f"failed to capture deployed codec graph for shape (1, {frames}, {quantizers})")
    return module, graph, metadata


def native_call(runner: StatefulCodecRunner, codes: torch.Tensor, *, total_frames: int) -> Callable[[], Any]:
    chunk = int(codes.shape[0])
    if chunk == 1:
        metadata = _decode_metadata(total_frames=total_frames, device=runner.device)
    else:
        metadata = _prefill_metadata(
            chunk,
            has_initial=True,
            total_frames=total_frames,
            device=runner.device,
        )
    layer_metadata = {layer.prefix: metadata for layer in runner.state_layers}
    input_ids = torch.zeros((chunk,), dtype=torch.long, device=runner.device)

    def call():
        with set_forward_context(layer_metadata, runner.vllm_config, num_tokens=chunk):
            return runner.model(input_ids=input_ids, codec_codes=codes)

    return call


def print_timing(name: str, timing: Timing) -> None:
    print(
        f"{name:<28} wall p50/p95={timing.wall_p50_ms:7.3f}/{timing.wall_p95_ms:7.3f} ms  "
        f"CUDA p50/p95={timing.cuda_p50_ms:7.3f}/{timing.cuda_p95_ms:7.3f} ms  "
        f"saturated={timing.saturated_ms:7.3f} ms"
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.left_context <= 0 or any(chunk <= 0 for chunk in args.chunks):
        raise ValueError("left context and chunk sizes must be positive")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    device = torch.device("cuda")
    tokens = load_acoustic_tokens(args.tokens)
    required = args.left_context + max(args.chunks)
    if tokens.shape[0] < required:
        repeats = (required + tokens.shape[0] - 1) // tokens.shape[0]
        tokens = tokens.repeat((repeats, 1))
    tokens = tokens[:required].to(device)

    properties = torch.cuda.get_device_properties(device)
    report: dict[str, Any] = {
        "gpu": properties.name,
        "left_context_frames": args.left_context,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "chunks": {},
    }

    print(f"GPU: {properties.name}")
    print(f"Warmup={args.warmup}, measured iterations={args.iterations}, left context={args.left_context} frames")

    runner = StatefulCodecRunner(args.native_checkpoint, device=device)
    runner.warmup()
    dense_config = EasyMagpieCodecConfig.from_pretrained(args.native_checkpoint)
    dense = EasyMagpieCodec(dense_config).to(device=device).eval()
    dense.load_state_dict(load_file(args.native_checkpoint / "model.safetensors", device=str(device)), strict=True)

    exported_module, deployed_graph, deployed_metadata = load_deployed_graph(args.exported_codec, device)
    deployed_frames = int(deployed_metadata["frames"])
    if deployed_frames < required:
        raise ValueError(
            f"deployed graph has {deployed_frames} frames, but {required} are required for "
            f"left_context={args.left_context} plus max chunk={max(args.chunks)}"
        )
    deployed_codes = tokens[-1:].expand(deployed_frames, -1).clone().unsqueeze(0)
    deployed_codes[:, :required] = tokens.unsqueeze(0)
    deployed_api = measure(
        lambda: deployed_graph.decode(deployed_codes),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    deployed_replay = measure(
        lambda: deployed_graph.graphs[(1, deployed_frames)].replay(),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    print()
    print(f"Deployed stateless graph: fixed {deployed_frames} frames")
    print_timing("graph replay only", deployed_replay)
    print_timing("copy + replay + clone", deployed_api)
    report["deployed_fixed_graph"] = {
        "frames": deployed_frames,
        "replay_only": asdict(deployed_replay),
        "api": asdict(deployed_api),
    }

    del deployed_graph, exported_module
    gc.collect()
    torch.cuda.empty_cache()

    print()
    for chunk in args.chunks:
        context = tokens[: args.left_context]
        new_codes = tokens[args.left_context : args.left_context + chunk]

        runner.reset()
        runner.decode(context.cpu(), [args.left_context], reset=False)
        cached_call = native_call(
            runner,
            new_codes,
            total_frames=args.left_context + chunk,
        )
        cached_timing = measure(cached_call, warmup=args.warmup, iterations=args.iterations)

        captured_native = CapturedNativeCodec(
            runner,
            new_codes,
            total_frames=args.left_context + chunk,
        )
        native_graph_replay = measure(
            captured_native.replay_only,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        native_graph_api = measure(
            lambda: captured_native.forward(new_codes),
            warmup=args.warmup,
            iterations=args.iterations,
        )

        full_window = torch.cat((context, new_codes), dim=0).unsqueeze(0)
        exact_graph = CapturedStatelessCodec(dense, full_window)
        exact_replay = measure(
            exact_graph.replay_only,
            warmup=args.warmup,
            iterations=args.iterations,
        )
        exact_api = measure(
            lambda: exact_graph.forward(full_window),
            warmup=args.warmup,
            iterations=args.iterations,
        )

        speedup_exact = exact_api.wall_p50_ms / cached_timing.wall_p50_ms
        speedup_deployed = deployed_api.wall_p50_ms / cached_timing.wall_p50_ms
        audio_ms = chunk * dense_config.samples_per_frame * 1000.0 / dense_config.output_sample_rate
        realtime = audio_ms / cached_timing.wall_p50_ms
        print(
            f"chunk={chunk} predictor frames ({audio_ms:.0f} ms audio), stateless exact window={args.left_context + chunk}"
        )
        print_timing("native cached forward", cached_timing)
        print_timing("native graph replay only", native_graph_replay)
        print_timing("native graph API", native_graph_api)
        print_timing("exact graph replay only", exact_replay)
        print_timing("exact copy+replay+clone", exact_api)
        print(
            f"  speedup vs exact stateless={speedup_exact:.2f}x; "
            f"vs deployed fixed-{deployed_frames}={speedup_deployed:.2f}x; "
            f"native throughput={realtime:.1f}x realtime"
        )
        print()
        report["chunks"][str(chunk)] = {
            "audio_ms": audio_ms,
            "exact_stateless_frames": args.left_context + chunk,
            "native_cached": asdict(cached_timing),
            "native_graph_replay_only": asdict(native_graph_replay),
            "native_graph_api": asdict(native_graph_api),
            "exact_graph_replay_only": asdict(exact_replay),
            "exact_graph_api": asdict(exact_api),
            "speedup_vs_exact": speedup_exact,
            "speedup_vs_deployed": speedup_deployed,
            "native_realtime": realtime,
        }

        del exact_graph, captured_native
        gc.collect()
        torch.cuda.empty_cache()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
