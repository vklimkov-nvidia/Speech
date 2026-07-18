## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + per-codebook local
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

### Convert a NeMo checkpoint

This step turns the training-time `.nemo` checkpoints into a self-contained
vLLM-Omni model directory: a) converts the talker weights to Safetensors,
b) precomputes the text-embedding lookup, c) saves the tokenizer and optional speaker embedding, d) exports the codec graph and weights as a NeMo-free `torch.export` artifact. Run the converter in the **NeMo environment** from the repository root:

```bash
python examples/tts/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir examples/tts/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

### Setup the serving environment

Create a separate, clean environment for inference. It needs a GPU,
matching **vLLM 0.24 / vLLM-Omni 0.24** versions, and this package. It does not
need NeMo when using the default exported codec bundle:

```bash
cd examples/tts/easymagpie_vllm_omni
conda create -n easymagpie-vllm python=3.12 -y
conda activate easymagpie-vllm
pip install -r requirements.txt
pip install -e .
# optionally for notebook
pip install ipykernel
python -m ipykernel install --user \
  --name easymagpie-vllm \
  --display-name "Python (easymagpie-vllm)"
```

### Quick start — offline synthesis

Run [`scripts/offline_demo.ipynb`](scripts/offline_demo.ipynb) to check how `AsyncOmni` is initialized and used.

### Serve over HTTP and WebSocket

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091. Two serving
APIs are available:

- `POST /v1/audio/speech` with a complete text input.
- `WS /v1/audio/speech/stream` with incremental text/token updates and
  asynchronous PCM audio output.

Query the HTTP endpoint from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

Run [`scripts/server_request.ipynb`](scripts/server_request.ipynb) for examples
of both serving APIs.

### Benchmarks

Benchmark running service:

```bash
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32
```

#### RTX A6000 reference (2026-07-18)

The current deployment uses an 11-frame (0.88 s) codec left context, a
17-frame initial body, an 8-frame steady body, and a fixed 19-frame exported
codec graph. With 128 requests per level and streaming PCM output:

| Concurrency | req/s | RTF | TTFA mean | ITL mean | Playback underruns |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.16 | 7.15x | 246.2 ms | 77.3 ms | 0 / 1,148 chunks |
| 2 | 2.04 | 11.94x | 301.9 ms | 89.9 ms | 0 / 1,083 chunks |

These numbers include client and HTTP transport overhead and are intended as a
regression reference, not a hardware-independent model claim.

Benchmark incremental WebSocket input with five tokenizer IDs per message:

```bash
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```

Benchmark acoustic tokens prediction only (no codec)

```bash
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32
```
