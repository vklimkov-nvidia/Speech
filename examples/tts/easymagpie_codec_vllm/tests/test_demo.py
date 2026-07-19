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

import pytest
import torch
from easymagpie_codec_vllm.demo import load_acoustic_tokens, make_chunk_plan, normalize_acoustic_tokens


def test_make_chunk_plan() -> None:
    assert make_chunk_plan(0) == []
    assert make_chunk_plan(1) == [1]
    assert make_chunk_plan(16) == [1, 1, 2, 6, 6]
    assert make_chunk_plan(13, startup=(2,), steady=5) == [2, 5, 5, 1]


@pytest.mark.parametrize("invalid", [-1])
def test_make_chunk_plan_rejects_invalid_lengths(invalid: int) -> None:
    with pytest.raises(ValueError):
        make_chunk_plan(invalid)


def test_normalize_acoustic_tokens_accepts_predictor_and_codebook_major_layouts() -> None:
    expected = torch.arange(48).view(3, 16)
    torch.testing.assert_close(normalize_acoustic_tokens(expected), expected)
    torch.testing.assert_close(normalize_acoustic_tokens(expected.transpose(0, 1)), expected)
    torch.testing.assert_close(normalize_acoustic_tokens(expected.unsqueeze(0)), expected)


def test_load_benchmark_audio_codes(tmp_path) -> None:
    expected = torch.arange(48).view(3, 16)
    path = tmp_path / "request_0000.pt"
    torch.save({"text": "hello", "audio_codes": expected}, path)
    torch.testing.assert_close(load_acoustic_tokens(path), expected)
