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

"""
Factory for creating native PyTorch LLM and TTS inference backends.

NemotronVoiceChat exposes two native components:

- ``"native_llm"``    -- wraps the PyTorch model directly (LLM component)
- ``"native_eartts"`` -- wraps the PyTorch DuplexEARTTS model (TTS component)

The user-facing ``engine_type`` value selects between the native path and the
``vllm_omni`` path (which replaces both LLM and TTS components with a vLLM-Omni
streaming pipeline; see
:class:`~nemo.collections.speechlm2.inference.model_wrappers.nemotron_voicechat_inference_wrapper.NemotronVoicechatInferenceWrapper`).
The factory is only used on the native path.

Usage::

    from nemo.collections.speechlm2.inference.model_wrappers.factory import create_model

    llm = create_model(engine_type="native_llm", model=voicechat_model)
    tts = create_model(engine_type="native_eartts", model=voicechat_model.tts_model)
"""

from nemo.collections.speechlm2.inference.model_wrappers.backend.interface import ModelInterface


def create_model(
    engine_type: str,
    model=None,
    special_token_ids: set[int] | None = None,
    top_p: float = 1.0,
    repetition_penalty: float = 1.0,
    temperature: float = 1.0,
    **kwargs,
) -> ModelInterface:
    """Factory function to create a single native inference backend for one component.

    Args:
        engine_type: One of ``"native_llm"``, ``"native_eartts"``.
        model: The PyTorch model to wrap (NemotronVoiceChat for LLM,
            DuplexEARTTS for TTS).
        special_token_ids: Set of special token IDs (pad, eos, bos) that should
            bypass sampling and always use greedy decoding.
        top_p: Top-p (nucleus) sampling threshold. 1.0 disables it (greedy). Default: 1.0
        repetition_penalty: Penalty for repeated tokens. 1.0 disables it. Default: 1.0
        temperature: Temperature for sampling. 1.0 = no change, 0.0 = greedy. Default: 1.0
        **kwargs: Additional arguments passed to the backend constructor.

    Returns:
        A ModelInterface instance ready for inference.
    """
    engine_type = engine_type.lower()

    if engine_type == "native_llm":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.model import PyTorchLLM

        if model is None:
            raise ValueError("model must be provided for native_llm engine")
        return PyTorchLLM(
            model=model,
            special_token_ids=special_token_ids,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            temperature=temperature,
        )

    elif engine_type == "native_eartts":
        from nemo.collections.speechlm2.inference.model_wrappers.backend.pytorch.eartts import PyTorchEarTTS

        if model is None:
            raise ValueError("model (DuplexEARTTS instance) must be provided for native_eartts engine")
        return PyTorchEarTTS(tts_model=model)

    else:
        raise ValueError(
            f"Unknown engine_type: {engine_type}. Supported types: 'native_llm', 'native_eartts'."
        )
