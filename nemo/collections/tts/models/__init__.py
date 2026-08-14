# Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from nemo.collections.tts.models.acoustic_model import AcousticModel
from nemo.collections.tts.models.acoustic_model_ar_with_context import AcousticModelAutoregressiveWithContext
from nemo.collections.tts.models.acoustic_model_ar_with_text import AcousticModelAutoregressiveWithText
from nemo.collections.tts.models.acoustic_model_with_context import AcousticModelWithContext
from nemo.collections.tts.models.acoustic_model_with_text import AcousticModelWithText
from nemo.collections.tts.models.aligner import AlignerModel
from nemo.collections.tts.models.audio_codec import AudioCodecModel
from nemo.collections.tts.models.audio_codec_moss import MossAudioCodecModel
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_acoustic import EasyMagpieTTSAcousticModel
from nemo.collections.tts.models.easy_magpietts_acoustic_linear import EasyMagpieTTSAcousticLinearModel
from nemo.collections.tts.models.easy_magpietts_acoustic_transformer import EasyMagpieTTSAcousticTransformerModel
from nemo.collections.tts.models.easy_magpietts_semantic import EasyMagpieTTSSemanticModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.models.easy_magpietts_preference_optimization import EasyMagpieTTSModelOnlinePO
from nemo.collections.tts.models.discrete_speech import DiscreteSpeechModel
from nemo.collections.tts.models.fastpitch import FastPitchModel
from nemo.collections.tts.models.fastpitch_ssl import FastPitchModel_SSL
from nemo.collections.tts.models.hifigan import HifiGanModel
from nemo.collections.tts.models.magpietts import InferBatchOutput, MagpieTTSModel
from nemo.collections.tts.models.magpietts_cfg_distillation import OnlineCFGDistillation
from nemo.collections.tts.models.magpietts_preference_optimization import (
    MagpieTTSModelOfflinePO,
    MagpieTTSModelOfflinePODataGen,
    MagpieTTSModelOnlinePO,
)
from nemo.collections.tts.models.ssl_tts import SSLDisentangler

__all__ = [
    "AlignerModel",
    "AudioCodecModel",
    "FastPitchModel",
    "DiscreteSpeechModel",
    "FastPitchModel_SSL",
    "SSLDisentangler",
    "HifiGanModel",
    "InferBatchOutput",
    "MagpieTTSModel",
    "OnlineCFGDistillation",
    "EasyMagpieTTSModel",
    "EasyMagpieTTSAcousticModel",
    "EasyMagpieTTSInferenceModel",
    "EasyMagpieTTSModelOnlinePO",
    "MagpieTTSModelOfflinePODataGen",
    "MagpieTTSModelOfflinePO",
    "MagpieTTSModelOnlinePO",
]
