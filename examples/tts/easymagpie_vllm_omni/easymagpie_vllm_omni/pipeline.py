# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""EasyMagpieTTS pipeline: Talker (text -> stacked acoustic codes) -> Code2Wav
(codes -> audio), fully in-engine (no external codec service).

Structured exactly like the Qwen3-TTS reference pipeline
(``vllm_omni/model_executor/models/qwen3_tts/pipeline.py``): a Stage-0
autoregressive talker followed by a Stage-1 generative Code2Wav. Chunked vs
end-to-end mode is dispatched from ``deploy.async_chunk``.

The talker's HF config reports ``model_type="nemotron_h"`` (a generic backbone
type), so this pipeline is routed by ``hf_architectures`` — the
``StageConfigFactory`` matches ``config.architectures`` against it. A deploy YAML
may also force routing with ``pipeline: easymagpie``.
"""
from vllm_omni.config.stage_config import (
    PipelineConfig,
    StageExecutionType,
    StagePipelineConfig,
)

_PROC = "easymagpie_vllm_omni.stage_processors"

# The talker repurposes a 2-wide dummy backbone vocab as a continue/stop signal;
# the last index is the audio-EOS stop token (see
# ``EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id``).
_AUDIO_EOS_STOP_TOKEN_ID = 1

EASYMAGPIE_PIPELINE = PipelineConfig(
    model_type="easymagpie",
    model_arch="EasyMagpieTTSForConditionalGeneration",
    hf_architectures=(
        "EasyMagpieTTSForConditionalGeneration",
        "EasyMagpieTTS",
    ),
    stages=(
        StagePipelineConfig(
            stage_id=0,
            model_stage="easymagpie",
            execution_type=StageExecutionType.LLM_AR,
            input_sources=(),
            owns_tokenizer=True,
            engine_output_type="latent",
            async_chunk_process_next_stage_input_func=f"{_PROC}.talker2code2wav_async_chunk",
            custom_process_next_stage_input_func=f"{_PROC}.talker2code2wav_full_payload",
            # NOTE: The installed vllm_omni (0.21.0rc1) StagePipelineConfig has no
            # ``scheduler_cls`` field — the AR scheduler is resolved from
            # ``execution_type`` + ``async_scheduling`` (LLM_AR -> OmniARAsyncScheduler).
            # The custom EasyMagpieARAsyncScheduler is only needed for the
            # streaming-text (multi-chunk StreamingInput) path, which the two-stage
            # synthesis flow does not use, so it is intentionally not wired here.
            sampling_constraints={
                "detokenize": False,
                "stop_token_ids": [_AUDIO_EOS_STOP_TOKEN_ID],
            },
        ),
        StagePipelineConfig(
            stage_id=1,
            model_stage="easymagpie_code2wav",
            execution_type=StageExecutionType.LLM_GENERATION,
            input_sources=(0,),
            final_output=True,
            final_output_type="audio",
            engine_output_type="audio",
            model_arch="EasyMagpieCode2Wav",
            # Sync (non-async-chunk) mode: a length-only placeholder input; the
            # bulk codec payload ships via the worker connector from Stage 0's
            # ``talker2code2wav_full_payload`` producer. Under async_chunk mode
            # chunks are delivered directly to the consumer.
            sync_process_input_func=f"{_PROC}.talker2code2wav_token_only",
            sampling_constraints={"detokenize": True},
        ),
    ),
)
