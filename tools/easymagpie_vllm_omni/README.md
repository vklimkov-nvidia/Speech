## EasyMagpieTTS — vLLM-Omni two-stage inference

Streaming TTS for **NemotronTTS** (Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec) via [vLLM-Omni](https://github.com/vllm-project/vllm-omni).

EasyMagpieTTS decomposes into EasyMagpie LM and SpectralCodec-BWE-22kHz:

| Stage | Role |
|-------|------|
| **0 — EasyMagpie LM** | `EasyMagpie_LM_Backbone` (Nemotron-H) + `EasyMagpie_LM_LT` → stacked acoustic codes |
| **1 — SpectralCodec-BWE-22kHz** | Stateful native vLLM codec → 22.05 kHz waveform |

Model definition and pipeline registration live in
[`easymagpie_vllm_omni/`](easymagpie_vllm_omni/) and
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni/).
Deployment knobs are in [`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

### Convert a NeMo checkpoint

This step converts the EasyMagpie LM and Stage-1 codec decoder, precomputes
the text-embedding lookup, and saves the tokenizer and optional precomputed
speaker embedding. The codec decoder is always included so the documented
two-stage pipeline is immediately usable. The codec encoder and reference-speaker
encoder are deliberately omitted by default, so the resulting artifact accepts
only speaker names or precomputed embeddings.

Pass `--bundle-audio-encoders` only when you intend to package both the codec
encoder and the reference-speaker Transformer. This opt-in enables request-time
reference audio and therefore zero-shot voice cloning. It does not make a
single-turn checkpoint multi-turn; raw user-audio history additionally requires
the source checkpoint's independent multi-turn configuration. Run conversion in the **NeMo environment** from
the repository root. Prepending the repository root to `PYTHONPATH` is
important when the environment has an editable NeMo install pointing at a
different checkout:

```bash
PYTHONPATH="$PWD" python tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py \
  --nemo_file /path/to/emptts.nemo \
  --codec_model_path /path/to/25fps_spectral_codec.nemo \
  --phoneme_tokenizer_path /path/to/bpe_ipa_tokenizer.json \
  --outdir tools/easymagpie_vllm_omni/converted_model \
  --context_audio /path/to/reference_voice.wav \
  --speaker_name eng
```

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

### Request-time reference and user audio

A converted artifact accepts raw reference or user audio only when it was
created from a compatible checkpoint with `--bundle-audio-encoders`. That flag
exports `codec_encoder.safetensors` and `codec_encoder.json`; the bundled tower
returns both acoustic tokens and reference-speaker conditioning. Multi-turn is
independent: user-audio history is accepted only when the source configuration
also has `use_multiturn_dataset` and `condition_on_user_speech` enabled.

Pass audio as `(waveform, sample_rate)`. The waveform must already be mono and
match `arch.codec_input_sample_rate`; the serving path deliberately does not
downmix or resample it.

Identify each audio item's purpose using request-level `audio_roles`. This
first history-preserving turn batches reference and user audio:

```python
prompt = {
    "prompt_token_ids": (
        [0] * task_rows
        + [arch.audio_input_token_id]
        + [0] * len(context_token_ids)
        + [arch.audio_input_token_id]
    ),
    "multi_modal_data": {
        "audio": [
            (reference_waveform, reference_sample_rate),
            (user_waveform, user_sample_rate),
        ]
    },
    "mm_processor_kwargs": {
        "audio_roles": ["speaker_reference", "user"],
    },
    "additional_information": {
        "context_text": "[EN]",
        "speaker_reference_audio": True,
        "user_audio_prefill": True,
        "text": response_text,
        "text_prefill_num": arch.text_prefill_num,
        "temperature": 0.7,
        "top_k": 80,
        "reset_codec_on_segment": True,
    },
}
```

For independent synthesis, submit only the reference item, set
`audio_roles=["speaker_reference"]` and `user_audio_prefill=False`, and replace
the final user-audio marker above with `arch.text_prefill_num` zero rows. Also
provide `prefill_text_tokens`, the first `arch.text_prefill_num` encoded target
tokens.

For later turns on the same Stage 0 request, submit only the new user audio
marker/waveform and target text. Set `audio_roles=["user"]`,
`speaker_reference_audio=False`, and `user_audio_prefill=True`; do not repeat
the reference/context rows.

Yielding the turns as `StreamingInput` items from one async input generator
keeps the existing Stage 0 causal/Mamba state. Using a new `omni.generate(...)`
request instead starts synthesis from scratch. vLLM-Omni marks each appended
audio-bearing span as prefill and switches back to decoding after that span; the
client does not select an engine stage manually. Set
`reset_codec_on_segment=True` on each dialogue reply so Stage 1 flushes and
resets its response-local codec state while Stage 0 remains resumable.

The OpenAI-compatible speech and WebSocket endpoints currently accept TTS text
input only. Use the direct `AsyncOmni` input path for raw user speech until an
audio-dialog request adapter is added.

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

# Benchmark the service's incremental synthesis via its WebSocket API.
python scripts/benchmark_incremental_server.py --model ./converted_model \
    --text-file vctk_subset.txt --tokens-per-chunk 5 -n 128 -c 1 32
```
