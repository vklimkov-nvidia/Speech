# EasyMagpie codec for vLLM

This directory is an isolated vLLM/vLLM-Omni plugin draft for the causal
EasyMagpie spectral codec. It converts the NeMo decoder weights to a local
Hugging Face-style model bundle, decodes EasyMagpie FSQ indices without NeMo,
and stores every causal convolution history in vLLM-managed fixed-size state
pages. The sibling `easymagpie_vllm_omni` pipeline loads this plugin as its
stateful Stage-1 codec.

## Quick start

Run the conversion wrapper from the NeMo repository root:

```bash
examples/tts/easymagpie_codec_vllm/scripts/convert_nemo_to_vllm.sh \
  easymagpietts_NEXT/25fps_spectral_codec_with_bandwidth_extension.nemo \
  /tmp/easymagpie_codec_native
```

The script uses the `easymagpie-vllm` conda environment by default. Override it
with `EASYMAGPIE_VLLM_ENV=<name>`. It folds weight normalization, strictly loads
the result into the native PyTorch model, and writes:

```text
/tmp/easymagpie_codec_native/
├── config.json
└── model.safetensors
```

Open
[`notebooks/chunked_decode.ipynb`](notebooks/chunked_decode.ipynb) with the same
conda environment as the Jupyter kernel. It loads a real `[T, 16]` predictor
sample already present under `easymagpie_vllm_omni/dumped_codes`, warms the
packed kernels, decodes chunks `[1, 1, 2, 6, 6, ...]`, displays a playable audio
widget, and checks the chunked waveform against one-shot decoding.

To generate fresh acoustic tokens with the talker-only pipeline:

```bash
conda run -n easymagpie-vllm python \
  examples/tts/easymagpie_vllm_omni/scripts/benchmark_model.py \
  --model examples/tts/easymagpie_vllm_omni/converted_model_multiturn \
  --audio-codes-dir /tmp/easymagpie_audio_codes \
  --no-warmup -n 1 -c 1 --max-new-tokens 256
```

Point the notebook's `TOKEN_FILE` at
`/tmp/easymagpie_audio_codes/concurrency_1/request_0000.pt`.

## Predictor packing

The EasyMagpie acoustic predictor emits `[T, 16]`, where one row is ordered as:

```text
[c0@2t, c0@2t+1, c1@2t, c1@2t+1, ..., c7@2t, c7@2t+1]
```

This is NeMo `stack_codes` ordering. `unstack_acoustic_codes` changes the view
from `[T, 8*2]` to `[T, 8, 2]`, transposes the last two dimensions, and reshapes
to `[2T, 8]`. Both the eager reference and packed vLLM model call this shared
implementation. The caller should submit predictor output directly; it should
not pre-transpose the 16 codebooks.

One scheduled vLLM placeholder is one predictor frame, not one codec frame or
one codebook. Since each predictor frame contains two 25-fps codec frames, it
produces 1,764 samples, or 80 ms at 22.05 kHz. A one-frame first call therefore
provides 80-ms TTFA granularity. Later calls can use 5–6 predictor frames for
400–480 ms of audio per kernel launch.

## Packed model design

The codec changes packed time resolution using views rather than padding
sequences into a batch:

| Operation | Packed rows per predictor frame | Channels |
| --- | ---: | ---: |
| Input acoustic codes | 1 | 16 indices |
| Unstack (`[BT,C*S] -> [BT*S,C]`) | 2 | 8 indices / 40 latent values |
| Pre-upsample, stride 2 | 4 | 768 |
| Upsample, stride 9 | 36 | 384 |
| Upsample, stride 7 | 252 | 128 |
| Upsample, stride 7 | 1764 | 32, then one waveform sample |

There is no attention in this decoder. Its persistent state consists of 22
causal `Conv1d` histories (`kernel_size - 1` input rows) and four causal grouped
`ConvTranspose1d` histories (one input row). All 26 layers expose a uniform
`MambaSpec((4608,), fp32)` cache page to vLLM; 4608 values is the largest real
history (`6 * 768`). The CUDA implementation operates directly on packed
sequences using base-resolution Mamba metadata plus a static time factor.

Uniform prefill/extend and decode batches use one batched cuDNN operation per
convolution. Fused Triton kernels gather arbitrary vLLM state pages into the
dense convolution input, update the ring state, and evaluate HalfSnake. Ragged
or mixed batches retain the direct packed Triton convolution fallback. The
codec metadata builder derives uniformity and maximum chunk length from vLLM's
CPU metadata, so the fast path has no device-to-host synchronization.

The direct `StatefulCodecRunner` used by the notebook follows the intended
scheduler behavior:

- The first chunk uses prefill metadata with zero initial state.
- A one-frame continuation uses decode metadata and `state_indices_tensor_d`.
- A multi-frame continuation uses prefill/extend metadata with initial state.
- Every call returns only new waveform samples; no left context is resubmitted.

## Fixed-window CUDA graph benchmark

The primary benchmark always measures total time `T=15` or `T=19` with 11
history frames. The native graph therefore processes only 4 or 8 new frames,
while the deployed stateless graph processes the entire 15/19-frame window.
The complete native forward, including cache gather/update, all decoder layers,
and waveform production, is captured in one `torch.cuda.CUDAGraph`. The script
varies batch size and reports synchronized wall/CUDA p50/p95 plus saturated
throughput.

```bash
conda run -n easymagpie-vllm env \
  PYTHONPATH=examples/tts/easymagpie_codec_vllm:examples/tts/easymagpie_vllm_omni \
  python examples/tts/easymagpie_codec_vllm/scripts/benchmark_fixed_window_batch.py \
  --native-checkpoint /tmp/easymagpie_codec_native \
  --native-dtype float32 \
  --exported-codec-15 examples/tts/easymagpie_vllm_omni/converted_model/codec/codec_decoder_15.pt2 \
  --exported-codec-19 examples/tts/easymagpie_vllm_omni/converted_model/codec/codec_decoder.pt2 \
  --tokens examples/tts/easymagpie_vllm_omni/dumped_codes/concurrency_1/request_0000.pt \
  --batch-sizes 1 2 4 8 16 32 --warmup 20 --iterations 100
```

Clean RTX A6000 wall-p50 results are below. Each cell is `native ms (speedup
over stateless FP32 context replay)`:

| Batch | FP32 T=15 | FP32 T=19 | FP16 T=15 | FP16 T=19 |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 2.285 (1.80x) | 2.511 (1.78x) | 2.544 (1.58x) | 2.784 (1.65x) |
| 2 | 2.773 (2.13x) | 2.960 (2.38x) | 2.778 (2.15x) | 3.078 (2.32x) |
| 4 | 3.253 (2.86x) | 4.198 (2.60x) | 3.071 (3.08x) | 3.832 (2.77x) |
| 8 | 4.828 (3.08x) | 7.566 (2.34x) | 3.845 (3.80x) | 5.263 (3.43x) |
| 16 | 7.766 (3.27x) | 12.179 (2.64x) | 5.183 (5.03x) | 8.457 (3.82x) |
| 32 | 12.462 (3.73x) | 21.691 (2.67x) | 8.367 (5.57x) | 14.285 (4.06x) |

FP32 remains the default. On the checked 19-frame sample, FP16 versus native
FP32 measured 53.8 dB SNR, 0.999998 correlation, and 0.00157 maximum absolute
waveform error. Use FP16 only after validating audio quality on the target set.
The machine-readable runs are in
`benchmark_fixed_window_batch_fp32_a6000.json` and
`benchmark_fixed_window_batch_fp16_a6000.json`.

## Validate and install

```bash
export PYTHONPATH=examples/tts/easymagpie_codec_vllm

conda run -n easymagpie-vllm python \
  examples/tts/easymagpie_codec_vllm/scripts/validate_codec.py \
  examples/tts/easymagpie_vllm_omni/converted_model/codec/codec_decoder.pt2 \
  /tmp/easymagpie_codec_native --frames 15 --chunks 1 3 11

conda run -n easymagpie-vllm pytest -q \
  examples/tts/easymagpie_codec_vllm/tests
```

For normal vLLM plugin discovery, install the directory editable in the serving
environment:

```bash
conda run -n easymagpie-vllm pip install -e \
  examples/tts/easymagpie_codec_vllm
```

## vLLM-Omni integration contract

The native codec stage must append only newly predicted `[T, 16]` rows to the
same logical request. It must schedule one placeholder per predictor frame and
send only the corresponding new rows. The packed model scales base
`query_start_loc` by the time factor at each layer.

The model supports packed prefill/extend batches, one-frame decode
continuations, and mixed prefill/decode batches. The one-frame CUDA path uses
`state_indices_tensor_d` directly and computes uniform sequence offsets inside
the kernel, without allocating a synthetic `query_start_loc`. The custom
metadata builder also permits full CUDA-graph capture for uniform multi-frame
chunks; ragged prefill remains eager/Triton by design.

The integrated vLLM-Omni stage currently requires `enforce_eager: true`.
vLLM-Omni graph replay pads scheduled placeholders to a captured size while
leaving the dynamic connector payload unpadded, which breaks their one-to-one
frame contract. This does not affect the standalone fixed-window graph runner.
