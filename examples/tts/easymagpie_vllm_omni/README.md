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

### Serve over HTTP

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091.
Query it from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

### Benchmarks

Benchmark running service:

```bash
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32
```

Benchmark acoustic tokens prediction only (no codec)

```bash
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32
```
