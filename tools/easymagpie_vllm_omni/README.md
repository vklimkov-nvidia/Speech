## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + configurable
codebook predictor over a 25 fps spectral codec) via
[vLLM-Omni](https://github.com/vllm-project/vllm-omni).

EasyMagpieTTS decomposes into an EasyMagpie LM and a causal spectral codec:

| Stage | Role |
|-------|------|
| **0 — EasyMagpie LM** | Nemotron-H backbone + configured codebook predictor → stacked acoustic codes |
| **1 — SpectralCodec** | Stateful native vLLM codec → checkpoint-native-rate waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

Stage 0 supports three `codebook_prediction_mode` values in `config.json`:

- `autoregressive` (default) runs the intra-frame local transformer and its
  per-codebook projection heads.
- `parallel` omits the local transformer and uses one direct backbone
  projection to produce every stacked-codebook distribution in a single graph.
  A real checkpoint must provide `parallel_codebook_out_projection.{weight,bias}`;
  dummy loading initializes these parameters automatically.
- `backbone` also omits the local transformer. It keeps
  `backbone_codebook_start_layer` ordinary temporal blocks, then assigns one
  trailing logical block to each stacked codebook. After each tail block, the
  sampled code is embedded and added to the live hidden/residual stream before
  the next block. `backbone_codebook_layer_type` selects `attention_ffn`,
  `mamba_ffn`, or `attention`. The `hybrid_override_pattern` tail must contain
  one `M` entry per codebook for `mamba_ffn`, or one `*` entry per codebook for
  either attention variant. A trained checkpoint must
  provide `backbone_codebook_output_norms.*` and
  `backbone_codebook_output_projections.*` plus tail block weights. Pipeline
  parallelism is not supported for this experimental mode.

  `backbone_codebook_layers_per_group` and
  `backbone_codebooks_per_group` generalize the tail. Each group runs the
  configured number of blocks, predicts its codebooks in parallel, then
  averages their sampled embeddings into the live stream before the next
  group. Both values default to one.

The review dummy checkpoints use 32 logical layers and retain the current
model's first 16 entries verbatim (`MEMEM*EMEMEM*EME`). Their 16-entry tails
compare attention+FFN, Mamba+FFN, and attention-only codebook blocks. The two
composite variants therefore execute two mixer sublayers inside each logical
tail block while preserving one codebook prediction per layer entry.
The corresponding directories are
`converted_model_roy_fullsize_32khz_backbone_attn_ffn_dummy`,
`converted_model_roy_fullsize_32khz_backbone_mamba_ffn_dummy`, and
`converted_model_roy_fullsize_32khz_backbone_attn_dummy`.

`converted_model_roy_fullsize_32khz_backbone_grouped_3x4_attn_ffn_dummy`
retains all 31 original backbone entries, then runs four groups of three
attention+dense-FFN blocks. Each group predicts four codebooks in parallel and
feeds their sampled embeddings into the next group.

The converter currently emits `autoregressive` checkpoints because the NeMo
training checkpoints contain local-transformer weights.

### Convert a NeMo checkpoint

This step turns the training-time `.nemo` checkpoints into a self-contained
vLLM-Omni model directory: it converts EasyMagpie LM and the causal codec to native
vLLM models, precomputes the text-embedding lookup, and saves the tokenizer and
optional speaker embedding. Run it in the **NeMo environment** from the repository root:

```bash
python tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir tools/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

The codec converter accepts either a packaged `.nemo` file or a NeMo
Lightning `.ckpt`. It infers the FSQ group count and levels from checkpoint
metadata, including 32 kHz codecs with 4096-entry codebooks and alternate
upsampling layouts.

### Setup the serving environment

Serving needs a GPU, matching **vLLM 0.24 / vLLM-Omni 0.24** versions, and this package.
It does not need NeMo after conversion:

```bash
cd tools/easymagpie_vllm_omni
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

Mamba's selective-state-update kernel requires shape- and GPU-specific tuning, so an untuned cache can give
suboptimal performance. Reuse the same Triton/vLLM cache directories across launches so repeated runs accumulate
better kernels; for an explicit sweep, run `python scripts/tune_mamba_ssu.py --model converted_model` and restart.

### Quick start — offline synthesis

See the [`offline_demo.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/offline_demo.ipynb) tutorial to check how
`AsyncOmni` is initialized and used.

### Serve over HTTP and WebSocket

```bash
bash ./scripts/run_server.sh ./converted_model 8091
```

This starts `vllm serve` with the EasyMagpie plugin on port 8091. Two serving
APIs are available:

- `POST /v1/audio/speech` with a complete text input.
- `WS /v1/audio/speech/stream` with incremental text/token updates and
  asynchronous PCM audio output.

Converted checkpoints with `enable_phoneme_text_input=true` accept inline IPA
spans such as `Turn <bop>lɛft<eop> here`. The markers are syntax only: ordinary
segments use the exported text tokenizer, while span contents use the bundled
IPA tokenizer and the checkpoint's reserved text-token range.

For delayed-stream checkpoints, the adapter folds the known text-led positions
into the causal prefill. The current `phoneme_delay=3`, `speech_delay=5` model
therefore prefills four target positions: text-only positions 0–2 and position
3 with the known phoneme BOS input. Whole-text HTTP requests satisfy this
automatically. Incremental WebSocket input buffers initial updates until at
least `phoneme_delay + 1` tokens are available. Marker strings and IPA spans may
cross `input.text` messages. An unclosed IPA span is rejected at `input.done`;
`input.tokens` remains an exact tokenization bypass and is accepted only when
there is no incomplete text marker or IPA span.

Query the HTTP endpoint from any OpenAI-compatible client:

```bash
curl -X POST http://localhost:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"This is a TTS service test.","voice":"eng","response_format":"wav","stream":true,"stream_format":"audio"}' \
  --output out.wav
```

See the [`server_request.ipynb`](../../tutorials/tts/easymagpie_vllm_omni/server_request.ipynb) tutorial for examples
of both serving APIs.

### Benchmarks

```bash
# Benchmark acoustic token prediction only (no codec).
python scripts/benchmark_model.py --model ./converted_model -n 128 -c 1 32 \
    [--streaming --tokens-per-chunk 5]

# Benchmark the service's HTTP API.
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 1 32

# Benchmark both service stages with dummy weights and a fixed 128-step decode.
EASYMAGPIE_DEPLOY_CONFIG=deploy/easymagpie_dummy.yaml \\
    bash scripts/run_server.sh ./converted_model 8091
python scripts/benchmark_server.py --text-file vctk_subset.txt -n 128 -c 32 \\
    --max-new-tokens 128

# Benchmark the service's incremental synthesis via its WebSocket API.
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```

The dummy deployment loads random weights for both stages. For a dummy-loaded
talker, the HTTP adapter ignores its synthetic audio-EOS signal so
`--max-new-tokens` is the exact decode length for every request.
Use `deploy/easymagpie_dummy_talker.yaml` to keep that fixed-length dummy
talker while loading the converted codec weights from `codec_native`.
