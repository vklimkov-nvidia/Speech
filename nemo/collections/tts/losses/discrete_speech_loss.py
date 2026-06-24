# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import torch

from nemo.core.classes import Loss, typecheck
from nemo.core.neural_types.elements import (
    LogitsType,
    LossType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType


class SpeakingRateLoss(Loss):
    def __init__(self):
        super(SpeakingRateLoss, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')

    @property
    def input_types(self):
        return {
            "logits": NeuralType(('B', 'C'), LogitsType()),
            "target_index": NeuralType(tuple('B'), TokenIndex()),
        }

    @property
    def output_types(self):
        return {
            "loss": NeuralType(elements_type=LossType()),
        }

    @typecheck()
    def forward(self, logits, target_index):
        loss = self.loss_fn(input=logits, target=target_index.long())
        return loss