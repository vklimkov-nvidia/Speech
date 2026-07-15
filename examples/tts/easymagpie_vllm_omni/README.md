## EasyMagpie TTS — vLLM-Omni + Triton service

Streaming TTS server for **EasyMagpieTTS** (NeMo model
`nemo.collections.tts.models.easy_magpietts.EasyMagpieTTSModel` /
`EasyMagpieTTSInferenceModel`, Nemotron-H backbone + per-codebook local
transformer over a 25 fps spectral codec).

The vLLM-Omni model definition (talker that runs the backbone + local
transformer as a single CUDA graph during uniform-batch decoding, piecewise
during prefill/mixed) lives in
[`vllm_plugin_easymagpie_omni/`](vllm_plugin_easymagpie_omni).

There are **two ways** to turn the talker's acoustic codes into a waveform:

* **In-engine two-stage pipeline (recommended, no external service)** — a
  Stage-1 `EasyMagpieCode2Wav` loads a `torch.export` capture of the original
  NeMo codec decoder inside vLLM-Omni (optionally CUDA-graphed). NeMo is needed
  for conversion but not serving. See
  [In-engine two-stage pipeline](#in-engine-two-stage-pipeline) below.
* **Triton + TensorRT codec service** — a Triton ensemble wraps the talker
  together with a TensorRT codec decoder for gRPC streaming. See
  [Triton service pipeline](#triton-service-pipeline) below.

## In-engine two-stage pipeline

The two-stage pipeline (`EASYMAGPIE_PIPELINE` in
[`easymagpie_vllm_omni/pipeline.py`](easymagpie_vllm_omni/pipeline.py)) chains
the talker (Stage 0, autoregressive) into `EasyMagpieCode2Wav` (Stage 1,
generative) with the same structure as the Qwen3-TTS reference pipeline:

- **Stage 0** — the existing talker; emits stacked acoustic codes under
  `codes.audio`.
- **Stage 1** — [`easymagpie_vllm_omni/code2wav.py`](easymagpie_vllm_omni/code2wav.py)
  loads the exported original decode path (clamp specials → unstack → FSQ
  index-convert → `AudioCodecModel.decode`) and optionally captures the loaded
  `ExportedProgram` as a CUDA graph
  ([`cuda_graph_codec_wrapper.py`](easymagpie_vllm_omni/cuda_graph_codec_wrapper.py)).
- **Stage plumbing** — [`stage_processors.py`](easymagpie_vllm_omni/stage_processors.py)
  turns the talker's per-frame codes into the codebook-major flat stream the
  codec consumes (full-payload for end-to-end mode, chunked windows for
  `async_chunk` streaming).

Steps:

1. **Convert with the codec exported** (the `--bundle_codec` default writes
   `<model>/codec/codec_decoder.pt2` and its metadata, so Stage 1 is
   self-contained and has no NeMo dependency):

   ```bash
   python examples/tts/easymagpie_vllm_omni/scripts/easy_magpietts_convert_to_vllm.py \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --outdir examples/tts/easymagpie_vllm_omni/easymp_vllm_model \
       --context_audio english_sample.wav --speaker_name eng \
       --phoneme_tokenizer_path <ckpt>/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh_ko-KR_pt-BR_ar.json
   ```

   The default export fixes the codec input at 15 model frames and allows
   dynamic batches up to 32. Override these with `--codec_export_frames` and
   `--codec_export_max_batch_size`; the frame count must match
   `codec_fixed_chunk_frames` in `deploy/easymagpie.yaml`.

   To validate a checkpoint's export and CUDA-graph capture in the `emp`
   environment before conversion:

   ```bash
   conda run -n emp python \
       examples/tts/easymagpie_vllm_omni/scripts/debug_codec_export.py \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --frames 15 --batches 1 2 4
   ```

2. **Offline synthesis** (single engine, both stages):

   ```bash
   python examples/tts/easymagpie_vllm_omni/scripts/synthesize_two_stage.py \
       --model examples/tts/easymagpie_vllm_omni/easymp_vllm_model \
       --text "Hello, welcome to the voice synthesis demo." --out out.wav
   ```

3. **Or serve it** (OpenAI-compatible vLLM-Omni server):

   ```bash
   examples/tts/easymagpie_vllm_omni/scripts/run_server.sh \
       examples/tts/easymagpie_vllm_omni/easymp_vllm_model 8091
   ```

Deployment knobs (stage placement, `async_chunk` streaming vs end-to-end, codec
chunking, Code2Wav CUDA-graph capture buckets) live in
[`deploy/easymagpie.yaml`](deploy/easymagpie.yaml).

## Triton service pipeline

The following steps build the Triton ensemble (talker python backend + TensorRT
codec plan). Use this only if you specifically need the standalone TRT codec
service; the in-engine pipeline above needs no ONNX/TRT export.

### Pipeline

1. **Convert the NeMo checkpoint to a vLLM-Omni model directory** — bakes the
   text embedding + CAS lookup, dumps `config.json`, `model.safetensors`, the
   text tokenizer, and optional reference speaker embeddings.

   ```bash
   python examples/tts/easymagpie_vllm_omni/scripts/easy_magpietts_convert_to_vllm.py \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --outdir examples/tts/easymagpie_vllm_omni/easymp_vllm_model \
       --context_audio english_sample.wav --speaker_name eng \
       --phoneme_tokenizer_path <ckpt>/bpe_ipa_tokenizer_2048_en_de_es_fr_hi_it_vi_zh_ko-KR_pt-BR_ar.json
   ```

   Checkpoints: <https://huggingface.co/nvidia/easymagpietts_NEXT/tree/main/2605_NemotronTTS_V0.2/v2>.

2. **Export the codec decoder to ONNX** — wraps `AudioCodecModel` so a single
   `(B, T, C*S)` int tensor of stacked model codes decodes to a 22.05 kHz
   waveform (clamp specials → unstack → FSQ index-convert → decode baked in).

   ```bash
   python examples/tts/easymagpie_vllm_omni/scripts/export_codec_decoder_onnx.py \
       --codec_model_path <ckpt>/25fps_spectral_codec_with_bandwidth_extension.nemo \
       --nemo_file <ckpt>/2605_EMTTS_SmallMamba_Step150k_posttrained_epoch12.nemo \
       --onnx-path examples/tts/easymagpie_vllm_omni/codec.onnx \
       --frames 15 --device cuda
   ```

3. **Build the serving container** (Triton 26.02 + vLLM 0.21.0 +
   vllm-omni 0.21.0rc1 + this plugin).

   ```bash
   docker build --network=host -t easymp-vllm-omni examples/tts/easymagpie_vllm_omni/
   ```

4. **Launch the container** with the workspace and a GPU mounted.

   ```bash
   docker run --rm -it --gpus all --network host --shm-size=8g \
       -v "$PWD":/workspace -w /workspace \
       easymp-vllm-omni bash
   ```

5. **Build the TensorRT engine from the ONNX** (inside the container) and drop
   it into the Triton repo as `model.plan`. For now fp32 seems to be mandatory.

   ```bash
   python examples/tts/easymagpie_vllm_omni/scripts/export_codec_decoder_trt.py \
       --onnx-path examples/tts/easymagpie_vllm_omni/codec.onnx \
       --trt-path  examples/tts/easymagpie_vllm_omni/model_repository/codec/1/model.plan \
       --batch-profile 1 8 32 --frames-profile 15 15 15 --fp32
   ```

6. **Start the Triton inference server** against
   [`model_repository/`](model_repository) (two models: `easymp` python
   backend + `codec` TRT plan).

   ```bash
   tritonserver --model-repository=examples/tts/easymagpie_vllm_omni/model_repository
   ```

7. **Send a request.** End-to-end gRPC streaming example in
   [`scripts/run_service_request.ipynb`](scripts/run_service_request.ipynb) —
   sends `text`, receives streamed `audio` chunks at 22.05 kHz.
