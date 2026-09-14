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

import convert_codec as converter
import pytest
import torch
from omegaconf import OmegaConf


def _valid_decoder_config() -> dict:
    return {
        "_target_": "nemo.collections.tts.modules.audio_codec_modules.ResNetDecoder",
        "is_causal": True,
        "activation": "half_snake",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("_target_", "some.OtherDecoder", "ResNetDecoder"),
        ("is_causal", False, "causal"),
        ("activation", "snake", "half_snake"),
    ],
)
def test_validate_decoder_config_rejects_unsupported_codec(field, value, message):
    config = _valid_decoder_config()
    config[field] = value

    with pytest.raises(ValueError, match=message):
        converter.validate_decoder_config(config)


def test_read_lightning_checkpoint(tmp_path):
    path = tmp_path / "codec.ckpt"
    config = {"sample_rate": 32000, "audio_decoder": _valid_decoder_config()}
    state = {"audio_decoder.weight": torch.arange(3)}
    torch.save(
        {
            "hyper_parameters": OmegaConf.create({"cfg": config}),
            "state_dict": state,
        },
        path,
    )

    actual_config, actual_state = converter._read_lightning_checkpoint(path)

    assert actual_config == config
    assert torch.equal(actual_state["audio_decoder.weight"], state["audio_decoder.weight"])


def test_resolve_quantizer_config_infers_checkpoint_geometry():
    config = {
        "vector_quantizer": {
            "num_groups": 8,
            "num_levels_per_group": [4, 4, 4, 4, 4, 4],
        }
    }

    assert converter.resolve_quantizer_config(config, None, None) == (8, [4, 4, 4, 4, 4, 4])
    assert converter.resolve_quantizer_config(config, 4, [3, 3]) == (4, [3, 3])
