from types import MethodType

import pytest
import torch
from torch import nn

from easymagpie_vllm_omni.backbone_codebook import EasyMagpieBackboneCodebookModel


class _RecordingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.effective_inputs = []

    def forward(self, positions, hidden_states, residual):
        effective = hidden_states if residual is None else hidden_states + residual
        self.effective_inputs.append(effective.detach().clone())
        return torch.zeros_like(hidden_states), effective


class _FinalNorm(nn.Module):
    def forward(self, hidden_states, residual):
        return hidden_states + residual, None


def _fixed_code(self, codebook_idx, hidden_states, gumbel_noise, temperature):
    return torch.ones(hidden_states.shape[0], dtype=torch.long)


def _make_tiny_backbone():
    model = EasyMagpieBackboneCodebookModel.__new__(EasyMagpieBackboneCodebookModel)
    nn.Module.__init__(model)
    model.codebook_start_layer = 0
    model.num_codebooks = 2
    first = _RecordingBlock()
    second = _RecordingBlock()
    model.layers = nn.ModuleList([first, second])
    model.codebook_output_norms = nn.ModuleList([nn.Identity(), nn.Identity()])
    model.audio_embeddings = nn.ModuleList([nn.Embedding(3, 1), nn.Embedding(3, 1)])
    with torch.no_grad():
        model.audio_embeddings[0].weight.zero_()
        model.audio_embeddings[0].weight[1].fill_(5.0)
    model.audio_in_projection = nn.Identity()
    model.norm_f = _FinalNorm()
    model._sample_codebook = MethodType(_fixed_code, model)
    return model, second


@pytest.mark.parametrize(("predict", "expected_second_input"), [(True, 7.0), (False, 2.0)])
def test_sampled_code_is_added_to_next_backbone_block_only_for_prediction_rows(predict, expected_second_input):
    model, second = _make_tiny_backbone()

    _, codes = EasyMagpieBackboneCodebookModel.forward(
        model,
        input_ids=torch.zeros(1, dtype=torch.long),
        positions=torch.zeros(1, dtype=torch.long),
        inputs_embeds=torch.tensor([[2.0]]),
        code_prediction_mask=torch.tensor([predict]),
        gumbel_noise=torch.zeros(1, 2, 1),
        temperature=torch.ones(1),
    )

    torch.testing.assert_close(second.effective_inputs[0], torch.tensor([[expected_second_input]]))
    torch.testing.assert_close(codes, torch.ones(1, 2, dtype=torch.long))
