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
"""Register EasyMagpieTTS models and pipeline with vLLM-Omni."""

_TARGET = "easymagpie_vllm_omni.easymagpie:EasyMagpieTTSForConditionalGeneration"
_ARCHS = ("EasyMagpieTTS", "EasyMagpieTTSForConditionalGeneration")

_CODE2WAV_TARGET = "easymagpie_vllm_omni.code2wav:EasyMagpieCode2Wav"
_CODE2WAV_ARCH = "EasyMagpieCode2Wav"


def register() -> None:
    """Register model architectures in both vLLM registries."""
    from vllm import ModelRegistry

    registries = [ModelRegistry]
    omni_available = False
    try:
        from vllm_omni.model_executor.models import OmniModelRegistry

        registries.append(OmniModelRegistry)
        omni_available = True
    except Exception:
        # vllm_omni not installed — stock vLLM registration is enough.
        pass

    for registry in registries:
        for arch in _ARCHS:
            if arch not in registry.get_supported_archs():
                registry.register_model(arch, _TARGET)
        # Model validation and stage resolution consult different registries.
        if _CODE2WAV_ARCH not in registry.get_supported_archs():
            registry.register_model(_CODE2WAV_ARCH, _CODE2WAV_TARGET)

    if omni_available:
        _register_pipeline()
        _register_serving_adapter()


def _register_serving_adapter() -> None:
    """Install optional ``/v1/audio/speech`` support."""
    import logging

    try:
        from easymagpie_vllm_omni.serving_adapter import apply_serving_patches

        apply_serving_patches()
    except Exception:  # pragma: no cover - serving support is best-effort
        logging.getLogger(__name__).exception(
            "EasyMagpie: /v1/audio/speech serving support could not be installed "
            "(model + pipeline registration still succeeded)."
        )


def _register_pipeline() -> None:
    """Register the two-stage and talker-only pipelines."""
    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE, EASYMAGPIE_TALKER_ONLY_PIPELINE
    from vllm_omni.config.pipeline_registry import register_pipeline

    register_pipeline(EASYMAGPIE_PIPELINE)
    register_pipeline(EASYMAGPIE_TALKER_ONLY_PIPELINE)
