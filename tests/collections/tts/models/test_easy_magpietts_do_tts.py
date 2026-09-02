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
Unit tests for EasyMagpieTTSInferenceModel.do_tts local-transformer selection.

With use_local_transformer=None, do_tts derives it from local_transformer_type (AR -> use it;
MASKGIT/NO_LT -> parallel sampling, since EasyMagpie only supports an AR local transformer and raises
otherwise); an explicit value overrides that. The test drives the real do_tts with a mock self and
asserts the use_local_transformer_for_inference passed to infer_batch.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo.collections.tts.models.easy_magpietts_acoustic_transformer import EasyMagpieTTSAcousticTransformerModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyMagpieTTSInferenceModel
from nemo.collections.tts.modules.magpietts_modules import LocalTransformerType, add_special_tokens


def _make_easy_mock_model(local_transformer_type):
    """An EasyMagpieTTSInferenceModel mock wired enough for do_tts to reach the infer_batch call."""
    model = MagicMock(spec=EasyMagpieTTSInferenceModel)
    model.local_transformer_type = local_transformer_type
    model.parameters.side_effect = lambda: iter([torch.zeros(1)])  # do_tts reads device off a param
    model.cfg = MagicMock()
    model.cfg.text_tokenizers = {"english_phoneme": object()}
    model.tokenizer = MagicMock()
    model.tokenizer.tokenizers = {"english_phoneme": object()}
    model.tokenizer.encode.return_value = [1, 2, 3]
    model.eos_id = 0
    model.text_conditioning_tokenizer_name = "text_ce_tokenizer"
    model.data_num_audio_codebooks = 4
    model.infer_batch.return_value = SimpleNamespace(
        predicted_audio=torch.zeros(1, 1), predicted_audio_lens=torch.zeros(1, dtype=torch.long)
    )
    return model


def _local_transformer_flag_passed_to_infer_batch(model):
    model.infer_batch.assert_called_once()
    _args, kwargs = model.infer_batch.call_args
    return kwargs["use_local_transformer_for_inference"]


class TestEasyDoTtsLocalTransformerSelection:
    @pytest.mark.run_only_on('CPU')
    @pytest.mark.unit
    @pytest.mark.parametrize(
        "local_transformer_type, use_local_transformer, expected_use_lt",
        [
            # use_local_transformer=None -> derive from local_transformer_type (EasyMagpie uses the
            # local transformer only for AR; MASKGIT/NO_LT decode via parallel sampling).
            (LocalTransformerType.NO_LT, None, False),
            (LocalTransformerType.AR, None, True),
            (LocalTransformerType.MASKGIT, None, False),
            # An explicit value overrides the derived default -- e.g. False flips AR's derived True.
            (LocalTransformerType.AR, True, True),
            (LocalTransformerType.AR, False, False),
            (LocalTransformerType.MASKGIT, False, False),
        ],
    )
    def test_easy_do_tts_local_transformer_selection(
        self, local_transformer_type, use_local_transformer, expected_use_lt
    ):
        """do_tts derives use_local_transformer from local_transformer_type when None, else honors the explicit value."""
        model = _make_easy_mock_model(local_transformer_type)

        EasyMagpieTTSInferenceModel.do_tts(model, "hello world", use_local_transformer=use_local_transformer)

        assert _local_transformer_flag_passed_to_infer_batch(model) is expected_use_lt


class _NoSemanticRefiner(torch.nn.Module):
    num_codebooks = 16
    num_tokens_per_codebook = 16

    def __init__(self):
        super().__init__()
        self.semantic_tokens = object()

    def forward(self, inputs, audio_lens, semantic_tokens, vector_quantizer, **kwargs):
        self.semantic_tokens = semantic_tokens
        batch_size, num_steps, _ = inputs.shape
        predictions = torch.zeros(batch_size, self.num_codebooks, num_steps, dtype=torch.long)
        logits = torch.zeros(batch_size, num_steps, self.num_codebooks * self.num_tokens_per_codebook)
        return predictions, logits, None


def test_predict_audio_codes_supports_zero_semantic_stacked_refinement():
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    torch.nn.Module.__init__(model)
    model.num_audio_codebooks_train = 0
    model.frame_stacking_factor = 2
    model.audio_out_projection = torch.nn.Identity()
    model.final_proj = lambda hidden: hidden.new_empty(hidden.size(0), 0)
    model.acoustic_decoder_linear = None
    model.acoustic_decoder_transformer = _NoSemanticRefiner()
    model._codec_model = SimpleNamespace(vector_quantizer=object())
    state = SimpleNamespace(
        last_hidden=torch.randn(1, 1, 4),
        config=SimpleNamespace(
            batch_size=1,
            use_cfg=False,
            cfg_scale=1.0,
            temperature=0.0,
            topk=8,
            use_local_transformer=False,
        ),
    )

    sampled, argmax = EasyMagpieTTSInferenceModel._predict_audio_codes(model, state)

    assert sampled.shape == (1, 16)
    assert argmax.shape == (1, 16)
    assert model.acoustic_decoder_transformer.semantic_tokens is None


def test_process_predictions_keeps_appended_acoustic_codebooks():
    model = MagicMock(spec=EasyMagpieTTSInferenceModel)
    model.frame_stacking_factor = 1
    model.num_audio_codebooks = 13
    model.num_audio_codebooks_pred = 1
    model.acoustic_decoder_linear = None
    model.acoustic_decoder_transformer = object()
    model.phoneme_tokenizer = None
    model.audio_eos_id = 4097
    model._predict_audio_codes.return_value = (torch.zeros(1, 13, dtype=torch.long),) * 2

    state = SimpleNamespace(
        config=SimpleNamespace(batch_size=1, device=torch.device('cpu')),
        context_position=torch.zeros(1, dtype=torch.long),
        text_tokens_seen=torch.zeros(1, dtype=torch.long),
        phoneme_steps=torch.zeros(1, dtype=torch.long),
        audio_steps=torch.zeros(1, dtype=torch.long),
        audio_prediction_start_idx=torch.full((1,), -1, dtype=torch.long),
        audio_prediction_end_idx=torch.full((1,), -1, dtype=torch.long),
        all_predictions=[],
        last_audio_codes=None,
        gt_audio_embeddings=None,
        gt_audio_lens=None,
        finished=torch.zeros(1, dtype=torch.bool),
    )

    audio_codes, _ = EasyMagpieTTSInferenceModel._process_predictions(
        model,
        state,
        needs_context=torch.zeros(1, dtype=torch.bool),
        needs_phoneme=torch.zeros(1, dtype=torch.bool),
        needs_audio=torch.ones(1, dtype=torch.bool),
    )

    assert audio_codes.shape == (1, 13, 1)
    assert state.all_predictions[0].shape == (1, 13, 1)


def test_frame_stacking_keeps_bos_and_eos_in_dedicated_steps():
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    torch.nn.Module.__init__(model)
    raw_codes = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 1, 2, 3]],
            [[4, 5, 6, 0, 0], [7, 1, 2, 0, 0]],
        ]
    )
    raw_lens = torch.tensor([5, 3])
    with_special, special_lens = add_special_tokens(raw_codes, raw_lens, bos_id=8, eos_id=9)

    stacked, stacked_lens = model.stack_codes(
        with_special,
        special_lens,
        bos_id=8,
        eos_id=9,
        stacking_factor=2,
        num_codebooks=2,
    )

    assert stacked.shape == (2, 4, 5)
    assert torch.equal(stacked_lens, torch.tensor([5, 4]))
    assert torch.equal(stacked[:, :, 0], torch.full((2, 4), 8))
    for batch_index, length in enumerate(stacked_lens):
        assert torch.equal(stacked[batch_index, :, length - 1], torch.full((4,), 9))

    unstacked, _ = model.unstack_codes(stacked[:, :, 1:-1], stacked_lens - 2, stacking_factor=2)
    for batch_index, length in enumerate(raw_lens):
        assert torch.equal(unstacked[batch_index, :, :length], raw_codes[batch_index, :, :length])


def test_frame_stacking_keeps_bos_and_eos_in_dedicated_steps():
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    torch.nn.Module.__init__(model)
    raw_codes = torch.tensor(
        [
            [[1, 2, 3, 4, 5], [6, 7, 1, 2, 3]],
            [[4, 5, 6, 0, 0], [7, 1, 2, 0, 0]],
        ]
    )
    raw_lens = torch.tensor([5, 3])
    with_special, special_lens = add_special_tokens(raw_codes, raw_lens, bos_id=8, eos_id=9)

    stacked, stacked_lens = model.stack_codes(
        with_special,
        special_lens,
        bos_id=8,
        eos_id=9,
        stacking_factor=2,
        num_codebooks=2,
    )

    assert stacked.shape == (2, 4, 5)
    assert torch.equal(stacked_lens, torch.tensor([5, 4]))
    assert torch.equal(stacked[:, :, 0], torch.full((2, 4), 8))
    for batch_index, length in enumerate(stacked_lens):
        assert torch.equal(stacked[batch_index, :, length - 1], torch.full((4,), 9))

    unstacked, _ = model.unstack_codes(stacked[:, :, 1:-1], stacked_lens - 2, stacking_factor=2)
    for batch_index, length in enumerate(raw_lens):
        assert torch.equal(unstacked[batch_index, :, :length], raw_codes[batch_index, :, :length])


def test_non_strict_state_dict_allows_new_child_modules():
    model = EasyMagpieTTSInferenceModel.__new__(EasyMagpieTTSInferenceModel)
    torch.nn.Module.__init__(model)
    model.backbone = torch.nn.Linear(2, 2)
    model.acoustic_decoder_transformer = torch.nn.Linear(2, 2)
    state_dict = {
        'backbone.weight': torch.ones_like(model.backbone.weight),
        'backbone.bias': torch.ones_like(model.backbone.bias),
    }

    model.load_state_dict(state_dict, strict=False)

    assert torch.equal(model.backbone.weight, state_dict['backbone.weight'])
    assert torch.equal(model.backbone.bias, state_dict['backbone.bias'])


def test_freeze_backbone_for_refinement_only_trains_prediction_heads():
    model = EasyMagpieTTSAcousticTransformerModel.__new__(EasyMagpieTTSAcousticTransformerModel)
    torch.nn.Module.__init__(model)
    model.decoder = torch.nn.Linear(2, 2)
    model.context_code_proj = torch.nn.Linear(2, 2)
    model.acoustic_decoder_transformer = torch.nn.Linear(2, 2)
    model.audio_out_projection = torch.nn.Linear(2, 2)
    model.final_proj = torch.nn.Linear(2, 2)
    model.phoneme_final_proj = torch.nn.Linear(2, 2)

    model._freeze_backbone_for_refinement()

    assert not any(parameter.requires_grad for parameter in model.decoder.parameters())
    assert not any(parameter.requires_grad for parameter in model.context_code_proj.parameters())
    for module_name in (
        'acoustic_decoder_transformer',
        'audio_out_projection',
        'final_proj',
        'phoneme_final_proj',
    ):
        assert all(parameter.requires_grad for parameter in getattr(model, module_name).parameters())
