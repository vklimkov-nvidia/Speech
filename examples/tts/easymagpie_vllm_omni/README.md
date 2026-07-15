## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **EasyMagpieTTS** (Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

The in-engine **two-stage pipeline** chains:

| Stage | Role |
|-------|------|
| **0 — talker** | Autoregressive Nemotron-H + local transformer → stacked acoustic codes |
| **1 — code2wav** | Bundled NeMo codec decoder (`torch.export`, optional CUDA graph) → 22.05 kHz waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

### Setup

Requires **vLLM / vLLM-Omni 0.24+**, a GPU, and this package installed into that env:

```bash
pip install -e examples/tts/easymagpie_vllm_omni
```

### Quick start — offline synthesis

Open [`scripts/offline_demo.ipynb`](scripts/offline_demo.ipynb).
Set `MODEL_PATH` to a converted checkpoint. The notebook runs both stages in
one `AsyncOmni` engine, writes `out.wav`, and plays it.

### Serve over HTTP

```bash
cd examples/tts/easymagpie_vllm_omni/scripts
./run_server.sh /path/to/converted-model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091.
Query it from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello world.","voice":"eng","response_format":"pcm","stream":true,"stream_format":"audio"}' \
  --output out.pcm
```

### Benchmark

```bash
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 100 -c 8
```

Text file format: `<uttid>\t<text>` per line. Reports req/s, TTFA, inter-chunk
latency, and playback underrun rate.

### Tests

```bash
pytest examples/tts/easymagpie_vllm_omni/tests -v
```

Unit tests cover config math, local-transformer parity, audio-output parsing,
and the HTTP benchmark client. No GPU or model directory required for most tests.

### Layout

```
easymagpie_vllm_omni/          # model + pipeline + codec stage
vllm_plugin_easymagpie_omni/   # vLLM plugin entry point
deploy/easymagpie.yaml         # two-stage deploy config
scripts/
  offline_demo.ipynb           # offline synthesis demo
  run_server.sh                # HTTP server
  benchmark_server.py          # throughput / latency benchmark
tests/                         # unit tests
```
