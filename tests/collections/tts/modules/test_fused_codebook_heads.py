# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
from torch import nn

from nemo.collections.tts.modules.magpietts_modules import FusedCodebookHeads

pytestmark = pytest.mark.unit

NUM_CODEBOOKS = 4
NUM_TOKENS_PER_CODEBOOK = 6
IN_FEATURES = 8


def _separate_heads():
    torch.manual_seed(0)
    return nn.ModuleList(
        [nn.Linear(IN_FEATURES, NUM_TOKENS_PER_CODEBOOK) for _ in range(NUM_CODEBOOKS)],
    )


def _fused_heads():
    torch.manual_seed(1)
    return FusedCodebookHeads(
        in_features=IN_FEATURES,
        num_codebooks=NUM_CODEBOOKS,
        num_tokens_per_codebook=NUM_TOKENS_PER_CODEBOOK,
    )


def test_fused_heads_index_like_a_module_list_of_heads():
    """Indexing has to keep working, the sampling loop taking one codebook's head at a time."""
    heads = _fused_heads()
    x = torch.randn(3, IN_FEATURES)

    assert len(heads) == NUM_CODEBOOKS
    for codebook in range(NUM_CODEBOOKS):
        assert torch.equal(heads[codebook](x), heads(x)[:, codebook, :])
    # a ModuleList of heads indexes from the end too
    assert torch.equal(heads[-1](x), heads(x)[:, -1, :])


def test_fused_heads_reject_an_out_of_range_codebook():
    heads = _fused_heads()
    with pytest.raises(IndexError, match="out of range"):
        heads[NUM_CODEBOOKS]


def test_fused_heads_keep_the_batch_dimensions_they_are_given():
    heads = _fused_heads()
    logits = heads(torch.randn(2, 5, IN_FEATURES))
    assert logits.shape == (2, 5, NUM_CODEBOOKS, NUM_TOKENS_PER_CODEBOOK)


def test_fused_heads_match_separate_heads_holding_the_same_weights():
    """The fused weight is the separate heads concatenated, so it has to read back as them."""
    separate = _separate_heads()
    heads = _fused_heads()
    with torch.no_grad():
        heads.proj.weight.copy_(torch.cat([head.weight for head in separate], dim=0))
        heads.proj.bias.copy_(torch.cat([head.bias for head in separate], dim=0))
    x = torch.randn(3, IN_FEATURES)

    for codebook in range(NUM_CODEBOOKS):
        assert torch.allclose(heads[codebook](x), separate[codebook](x))
        assert torch.allclose(heads(x)[:, codebook, :], separate[codebook](x))


def test_fused_heads_load_their_own_checkpoint():
    heads = _fused_heads()
    reloaded = FusedCodebookHeads(
        in_features=IN_FEATURES,
        num_codebooks=NUM_CODEBOOKS,
        num_tokens_per_codebook=NUM_TOKENS_PER_CODEBOOK,
    )

    reloaded.load_state_dict(heads.state_dict())

    x = torch.randn(3, IN_FEATURES)
    assert torch.equal(reloaded(x), heads(x))
