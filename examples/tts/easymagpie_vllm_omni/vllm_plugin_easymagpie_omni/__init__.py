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
"""vLLM plugin: register ``EasyMagpieTTS`` as a model architecture for vLLM-Omni.

Loaded by vLLM in the parent process and each EngineCore subprocess via the
``vllm.general_plugins`` entry point. The lazy ``<module>:<class>`` target means
the (NeMo-free) model module is only imported when vLLM resolves the
architecture, keeping heavy imports out of the parent process.
"""

_TARGET = "easymagpie_vllm_omni.easymagpie:EasyMagpieTTSForConditionalGeneration"
_ARCHS = ("EasyMagpieTTS", "EasyMagpieTTSForConditionalGeneration")

# Stage-1 code2wav architecture (in-engine two-stage pipeline). Registered as a
# lazy ``<module>:<class>`` target so the (NeMo-dependent) codec module is only
# imported in the Stage-1 worker when the arch is resolved.
_CODE2WAV_TARGET = "easymagpie_vllm_omni.code2wav:EasyMagpieCode2Wav"
_CODE2WAV_ARCH = "EasyMagpieCode2Wav"


def register() -> None:
    """Register the model class under all supported arch names.

    The architecture must be registered in **both** registries:

    * ``vllm.ModelRegistry`` — the stock vLLM global registry.
    * ``vllm_omni``'s ``OmniModelRegistry`` — a *separate* ``_ModelRegistry``
      instance that the vLLM-Omni engine actually consults when resolving a
      model architecture. Registering only in the stock registry leaves the
      omni engine reporting ``Model architectures [...] are not supported``.

    The two-stage EasyMagpie pipeline (talker -> code2wav) is also registered
    with vLLM-Omni's pipeline registry so the engine can resolve it from the
    talker checkpoint's ``config.json`` (matched via ``hf_architectures``, since
    the backbone reports the generic ``model_type="nemotron_h"``).
    """
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
        # Code2Wav (Stage-1) must be registered in *both* registries: the
        # omni engine resolves stage archs via OmniModelRegistry, but vLLM's
        # ModelConfig validation (create_model_config) checks the stock
        # ModelRegistry — without the stock registration the Stage-1 engine
        # aborts with "Model architectures ['EasyMagpieCode2Wav'] are not
        # supported". Registration is lazy (``module:class`` string) and
        # ``code2wav.py`` keeps every NeMo import inside methods, so this does
        # not import NeMo in the parent/orchestrator process.
        if _CODE2WAV_ARCH not in registry.get_supported_archs():
            registry.register_model(_CODE2WAV_ARCH, _CODE2WAV_TARGET)

    if omni_available:
        _register_pipeline()


def _register_pipeline() -> None:
    """Register the two-stage EasyMagpie pipeline (idempotent).

    ``register_pipeline`` lives in ``vllm_omni.config.stage_config`` (the
    ``pipeline_registry`` module only holds the in-tree ``_OMNI_PIPELINES`` lazy
    map). It writes into ``_PIPELINE_REGISTRY`` unconditionally, so calling it
    every time is safe.

    A failure here is fatal for the two-stage pipeline: if the ``easymagpie``
    key is never registered, ``StageConfigFactory`` cannot resolve it and
    silently falls back to vLLM-Omni's default single-stage **diffusion** stage
    (``create_default_diffusion``) — which then fails trying to load the talker
    as a diffusion model. So surface the error loudly instead of swallowing it.
    """
    import logging

    try:
        from vllm_omni.config.stage_config import register_pipeline
    except Exception:  # pragma: no cover - very old vllm_omni without the registry
        logging.getLogger(__name__).exception(
            "vllm_omni has no register_pipeline; the two-stage EasyMagpie pipeline "
            "cannot be registered and the engine will fall back to a diffusion stage."
        )
        raise

    from easymagpie_vllm_omni.pipeline import EASYMAGPIE_PIPELINE

    register_pipeline(EASYMAGPIE_PIPELINE)
