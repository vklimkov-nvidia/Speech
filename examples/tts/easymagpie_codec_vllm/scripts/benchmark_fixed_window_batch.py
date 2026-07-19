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
"""Compare captured native cache reuse with fixed 15/19-frame stateless graphs."""

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
from easymagpie_codec_vllm.demo import StatefulCodecRunner, load_acoustic_tokens
from easymagpie_codec_vllm.packed import CODEC_STATE_ELEMENTS
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.mamba1_attn import Mamba1AttentionMetadata


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
    parser.add_argument("--exported-codec-15", type=Path, required=True)
    parser.add_argument("--exported-codec-19", type=Path, required=True)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--left-context", type=int, default=11)
    parser.add_argument("--native-dtype", choices=["float32", "bfloat16", "float16"], default="float32")
    parser.add_argument("--total-frames", type=int, nargs="+", default=[15, 19], choices=[15, 19])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
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


def batch_metadata(
    batch_size: int,
    frames: int,
    *,
    has_initial: bool,
    total_frames: int,
    device: torch.device,
) -> Mamba1AttentionMetadata:
    query_start_loc = torch.arange(batch_size + 1, dtype=torch.int32, device=device) * frames
    metadata = Mamba1AttentionMetadata(
        num_prefills=batch_size,
        num_prefill_tokens=batch_size * frames,
        num_decodes=0,
        num_decode_tokens=0,
        num_reqs=batch_size,
        has_initial_states_p=torch.full((batch_size,), has_initial, dtype=torch.bool, device=device),
        query_start_loc_p=query_start_loc,
        num_computed_tokens_p=None,
        state_indices_tensor_p=torch.arange(batch_size, dtype=torch.int32, device=device),
        state_indices_tensor_d=torch.empty((0, 1), dtype=torch.int32, device=device),
        query_start_loc_d=None,
        num_accepted_tokens=None,
        block_idx_last_scheduled_token=None,
        block_idx_first_scheduled_token_p=None,
        block_idx_last_computed_token=None,
        block_idx_last_scheduled_token_prev_step=None,
        seq_lens=torch.full((batch_size,), total_frames, dtype=torch.int32, device=device),
    )
    metadata.codec_max_query_len = frames
    metadata.codec_uniform = True
    return metadata


def allocate_state_pages(runner: StatefulCodecRunner, batch_size: int) -> None:
    for layer in runner.state_layers:
        layer.kv_cache = [
            torch.zeros(
                (batch_size, CODEC_STATE_ELEMENTS),
                dtype=layer.dtype,
                device=runner.device,
            )
        ]


def native_forward(
    runner: StatefulCodecRunner,
    codes: torch.Tensor,
    metadata: Mamba1AttentionMetadata,
) -> Any:
    input_ids = torch.zeros((codes.shape[0],), dtype=torch.long, device=runner.device)
    layer_metadata = {layer.prefix: metadata for layer in runner.state_layers}
    with set_forward_context(layer_metadata, runner.vllm_config, num_tokens=codes.shape[0]):
        return runner.model(input_ids=input_ids, codec_codes=codes)


class CapturedNativeBatch:
    def __init__(
        self,
        runner: StatefulCodecRunner,
        codes: torch.Tensor,
        metadata: Mamba1AttentionMetadata,
    ) -> None:
        self.runner = runner
        self.static_codes = codes.clone()
        self.input_ids = torch.zeros((codes.shape[0],), dtype=torch.long, device=runner.device)
        self.layer_metadata = {layer.prefix: metadata for layer in runner.state_layers}

        for _ in range(3):
            self._call()
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            output = self._call()
            self.static_output = output.multimodal_outputs["model_outputs"][0]

    def _call(self):
        with set_forward_context(
            self.layer_metadata,
            self.runner.vllm_config,
            num_tokens=self.static_codes.shape[0],
        ):
            return self.runner.model(input_ids=self.input_ids, codec_codes=self.static_codes)

    def replay_only(self) -> torch.Tensor:
        self.graph.replay()
        return self.static_output

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        self.static_codes.copy_(codes)
        self.graph.replay()
        return self.static_output.clone()


def load_stateless_graph(exported_codec: Path, *, batch_size: int, device: torch.device):
    from easymagpie_vllm_omni.code2wav import _ExportedCodecDecoder
    from easymagpie_vllm_omni.cuda_graph_codec_wrapper import CUDAGraphCodecDecoder

    suffix = exported_codec.stem.removeprefix("codec_decoder")
    metadata = json.loads(exported_codec.with_name(f"codec_export{suffix}.json").read_text())
    frames = int(metadata["frames"])
    max_batch_size = int(metadata["max_batch_size"])
    if batch_size > max_batch_size:
        raise ValueError(f"batch size {batch_size} exceeds exported maximum {max_batch_size}")
    module = _ExportedCodecDecoder(str(exported_codec), device)
    graph = CUDAGraphCodecDecoder(
        module,
        num_stacked_codebooks=int(metadata["num_stacked_codebooks"]),
        samples_per_frame=int(metadata["samples_per_frame"]),
        capture_frames=[frames],
        capture_batch_sizes=[batch_size],
    )
    graph.warmup(device)
    if (batch_size, frames) not in graph.graphs:
        raise RuntimeError(f"failed to capture stateless graph for {(batch_size, frames)}")
    return module, graph, metadata


def expand_tokens(tokens: torch.Tensor, batch_size: int, total_frames: int, device: torch.device) -> torch.Tensor:
    required = batch_size * total_frames
    repeats = (required + tokens.shape[0] - 1) // tokens.shape[0]
    return tokens.repeat((repeats, 1))[:required].view(batch_size, total_frames, -1).to(device)


def print_result(total_frames: int, batch_size: int, native: Timing, stateless: Timing) -> None:
    speedup = stateless.wall_p50_ms / native.wall_p50_ms
    print(
        f"T={total_frames:2d} B={batch_size:2d}  "
        f"native={native.wall_p50_ms:7.3f} ms  stateless={stateless.wall_p50_ms:7.3f} ms  "
        f"speedup={speedup:6.2f}x  native/item={native.wall_p50_ms / batch_size:7.3f} ms"
    )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.left_context <= 0 or any(batch <= 0 for batch in args.batch_sizes):
        raise ValueError("left context and batch sizes must be positive")
    if any(total <= args.left_context for total in args.total_frames):
        raise ValueError("every total frame count must exceed left context")
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be nonnegative and iterations must be positive")

    device = torch.device("cuda")
    source_tokens = load_acoustic_tokens(args.tokens)
    runner = StatefulCodecRunner(args.native_checkpoint, device=device, dtype=args.native_dtype)
    runner.warmup()
    properties = torch.cuda.get_device_properties(device)
    exported_paths = {15: args.exported_codec_15, 19: args.exported_codec_19}
    report: dict[str, Any] = {
        "gpu": properties.name,
        "left_context_frames": args.left_context,
        "native_dtype": args.native_dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "results": {},
    }

    print(f"GPU: {properties.name}")
    print(f"Fixed totals={args.total_frames}; batch sizes={args.batch_sizes}; left context={args.left_context}")
    for total_frames in args.total_frames:
        new_frames = total_frames - args.left_context
        for batch_size in args.batch_sizes:
            codes = expand_tokens(source_tokens, batch_size, total_frames, device)
            context = codes[:, : args.left_context].reshape(-1, codes.shape[-1]).contiguous()
            new_codes = codes[:, args.left_context :].reshape(-1, codes.shape[-1]).contiguous()

            allocate_state_pages(runner, batch_size)
            initial_metadata = batch_metadata(
                batch_size,
                args.left_context,
                has_initial=False,
                total_frames=args.left_context,
                device=device,
            )
            native_forward(runner, context, initial_metadata)
            continuation_metadata = batch_metadata(
                batch_size,
                new_frames,
                has_initial=True,
                total_frames=total_frames,
                device=device,
            )
            captured_native = CapturedNativeBatch(runner, new_codes, continuation_metadata)

            exported_module, stateless_graph, stateless_metadata = load_stateless_graph(
                exported_paths[total_frames],
                batch_size=batch_size,
                device=device,
            )
            if int(stateless_metadata["frames"]) != total_frames:
                raise ValueError(
                    f"{exported_paths[total_frames]} has {stateless_metadata['frames']} frames, "
                    f"expected {total_frames}"
                )

            native_timing = measure(
                lambda: captured_native.forward(new_codes),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            stateless_timing = measure(
                lambda: stateless_graph.decode(codes),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print_result(total_frames, batch_size, native_timing, stateless_timing)
            report["results"][f"t{total_frames}_b{batch_size}"] = {
                "total_frames": total_frames,
                "new_frames": new_frames,
                "batch_size": batch_size,
                "native": asdict(native_timing),
                "stateless": asdict(stateless_timing),
                "speedup": stateless_timing.wall_p50_ms / native_timing.wall_p50_ms,
            }

            del captured_native, stateless_graph, exported_module
            gc.collect()
            torch.cuda.empty_cache()

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Wrote {args.output_json}")


if __name__ == "__main__":
    main()
