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
"""Shape regression tests for the EasyMagpie local transformer."""
from __future__ import annotations

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

from easymagpie_vllm_omni.local_transformer import EasyMagpieLTSelfAttention  # noqa: E402


@pytest.mark.unit
def test_local_self_attention_keeps_batch_dimension_inferred():
    """The vLLM compiled graph may be replayed with a different decode batch size."""
    source = inspect.getsource(EasyMagpieLTSelfAttention.forward)
    assert ".reshape(-1," in source
    assert ".reshape(b," not in source
    assert ".view(b," not in source

    attn = EasyMagpieLTSelfAttention(d_model=64, n_heads=4).eval()
    first = torch.randn(1, 16, 64)
    later = torch.randn(7, 16, 64)
    with torch.no_grad():
        assert attn(first).shape == first.shape
        assert attn(later).shape == later.shape
