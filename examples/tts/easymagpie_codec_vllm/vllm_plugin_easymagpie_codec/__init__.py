# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""vLLM plugin registration for the standalone EasyMagpie codec."""

_ARCH = "EasyMagpieCodecForConditionalGeneration"
_TARGET = "easymagpie_codec_vllm.model:EasyMagpieCodecForConditionalGeneration"


def register() -> None:
    from easymagpie_codec_vllm.config import EasyMagpieCodecConfig
    from transformers import AutoConfig
    from vllm import ModelRegistry
    from vllm.model_executor.models.config import MODELS_CONFIG_MAP, MambaModelConfig

    try:
        AutoConfig.register(EasyMagpieCodecConfig.model_type, EasyMagpieCodecConfig)
    except ValueError:
        # Another worker in the same process may already have imported us.
        pass

    MODELS_CONFIG_MAP.setdefault(_ARCH, MambaModelConfig)
    if _ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(_ARCH, _TARGET)

    try:
        from vllm_omni.model_executor.models import OmniModelRegistry

        if _ARCH not in OmniModelRegistry.get_supported_archs():
            OmniModelRegistry.register_model(_ARCH, _TARGET)
    except ImportError:
        pass
