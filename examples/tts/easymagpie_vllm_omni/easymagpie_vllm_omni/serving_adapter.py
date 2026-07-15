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
"""``/v1/audio/speech`` support for EasyMagpieTTS on vLLM-Omni 0.24+."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable

logger = logging.getLogger(__name__)

MODEL_TYPE = "easymagpie"
_TALKER_STAGE = "easymagpie"
_TALKER_ARCH = "EasyMagpieTTSForConditionalGeneration"
_SERVING_MODULE = "vllm_omni.entrypoints.openai.serving_speech"

_DEFAULT_SPEAKER = "eng"
_DEFAULT_CONTEXT_TEXT = "[EN]"
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_K = 80

if TYPE_CHECKING:
    from vllm_omni.entrypoints.openai.protocol.audio import OpenAICreateSpeechRequest


def _build_adapter_cls() -> type:
    """Define the adapter class lazily (needs vllm_omni imported first)."""
    from vllm_omni.entrypoints.openai.tts_adapters.base import ARTTSAdapter, PreparedRequest

    class EasyMagpieTTSAdapter(ARTTSAdapter):
        """Build speaker-conditioned prompts for ``/v1/audio/speech`` requests."""

        name = MODEL_TYPE
        stage_keys = frozenset({_TALKER_STAGE})

        def __init__(self, ctx: Any) -> None:
            super().__init__(ctx)
            self._tokenizer: Any = None
            self._model_path_cache: str | None = None
            self._prompt_len_cache: dict[str, int] = {}

        def _model_path(self) -> str:
            if self._model_path_cache is not None:
                return self._model_path_cache
            engine_client = getattr(self.ctx, "engine_client", None)
            model_config = getattr(engine_client, "model_config", None)
            path = getattr(model_config, "model", None)
            if not path:
                for stage in getattr(engine_client, "stage_configs", None) or []:
                    stage_path = getattr(getattr(stage, "engine_args", None), "model", None)
                    if stage_path:
                        path = stage_path
                        break
            if not path:
                raise RuntimeError("EasyMagpie serving adapter could not resolve the model path.")
            self._model_path_cache = path
            return path

        def _tokenize(self) -> Callable[[str], Any]:
            if self._tokenizer is None:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self._model_path(), trust_remote_code=True)
            return lambda text: self._tokenizer.encode(text)

        def _prompt_len(self, speaker_id: str) -> int:
            cached = self._prompt_len_cache.get(speaker_id)
            if cached is not None:
                return cached
            from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

            plen = int(
                EasyMagpieTTSForConditionalGeneration.get_prompt_len(
                    speaker_id, self._model_path(), tokenize=self._tokenize()
                )
            )
            self._prompt_len_cache[speaker_id] = plen
            return plen

        def validate(self, request: "OpenAICreateSpeechRequest") -> str | None:
            if not request.input or not request.input.strip():
                return "Input text cannot be empty"
            extra = request.extra_params
            if extra is not None and not isinstance(extra, dict):
                return "extra_params must be a JSON object/dict"
            return None

        async def build(
            self,
            request: "OpenAICreateSpeechRequest",
            sampling_params_list: list,
            has_inline_ref_audio: bool,
        ) -> "PreparedRequest":
            del sampling_params_list, has_inline_ref_audio  # EasyMagpie needs neither.
            speaker_id = (request.voice or _DEFAULT_SPEAKER).strip()
            extra = request.extra_params or {}
            prompt = {
                "prompt_token_ids": [0] * self._prompt_len(speaker_id),
                "additional_information": {
                    "context_text": extra.get("context_text", _DEFAULT_CONTEXT_TEXT),
                    "text": request.input,
                    "temperature": float(extra.get("temperature", _DEFAULT_TEMPERATURE)),
                    "top_k": int(extra.get("top_k", _DEFAULT_TOP_K)),
                    "speaker_id": speaker_id,
                },
            }
            return PreparedRequest(prompt=prompt, tts_params={}, model_type=MODEL_TYPE)

    return EasyMagpieTTSAdapter


def _register_adapter() -> None:
    from vllm_omni.entrypoints.openai import tts_adapters

    if MODEL_TYPE in tts_adapters.TTS_ADAPTER_REGISTRY:
        return
    tts_adapters.register_tts_adapter(_build_adapter_cls())


def _patch_detection() -> None:
    from vllm_omni.entrypoints.openai import serving_speech as ss

    ss._TTS_MODEL_STAGES.add(_TALKER_STAGE)

    detect = ss.OmniOpenAIServingSpeech._detect_tts_model_type
    if getattr(detect, "_easymagpie_patched", False):
        return
    _orig_detect = detect

    def _detect_tts_model_type(self):
        stage = getattr(self, "_tts_stage", None)
        if stage is not None:
            engine_args = getattr(stage, "engine_args", None)
            model_stage = getattr(engine_args, "model_stage", None)
            model_arch = getattr(engine_args, "model_arch", None)
            if model_stage == _TALKER_STAGE or model_arch == _TALKER_ARCH:
                return MODEL_TYPE
        return _orig_detect(self)

    _detect_tts_model_type._easymagpie_patched = True
    ss.OmniOpenAIServingSpeech._detect_tts_model_type = _detect_tts_model_type


def apply_serving_patches(force: bool = False) -> None:
    """Install speech support in the API-server process."""
    import sys

    if not force and _SERVING_MODULE not in sys.modules:
        return
    try:
        _patch_detection()
        _register_adapter()
        logger.info("EasyMagpie: /v1/audio/speech serving adapter registered (model_type=%r).", MODEL_TYPE)
    except Exception:  # never let a serving-layer change break model/pipeline loading
        logger.exception("EasyMagpie: failed to install /v1/audio/speech serving support.")
