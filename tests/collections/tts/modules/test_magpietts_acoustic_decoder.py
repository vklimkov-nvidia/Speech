# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_modules import AcousticDecoderTransformer
from nemo.collections.tts.modules.transformer_2501 import Transformer


class _SemanticLayer(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.proj = nn.Linear(1, d_model)

    def forward(self, audio_codes, audio_lens):
        return self.proj(audio_codes)


class _VectorQuantizer:
    @staticmethod
    def decode(indices, input_len):
        return indices.float().mean(dim=0, keepdim=False).unsqueeze(1)


def _make_decoder(device="cpu"):
    d_model = 16
    transformer = Transformer(
        n_layers=8,
        d_model=d_model,
        d_ffn=32,
        sa_n_heads=4,
        kernel_size=1,
        p_dropout=0.0,
        has_xattn=False,
        is_causal=True,
        ffn_type="swiglu",
    )
    return AcousticDecoderTransformer(
        input_dim=d_model,
        d_model=d_model,
        semantic_layer=_SemanticLayer(d_model),
        num_codebooks=12,
        codebook_size=8,
        transformer=transformer,
        num_prediction_steps=4,
    ).to(device)


def test_selects_three_new_codebooks_per_stage():
    confidence = torch.arange(12, dtype=torch.float).view(1, 1, 12)
    unresolved = torch.ones_like(confidence, dtype=torch.bool)

    selected = AcousticDecoderTransformer.select_codebooks(confidence, unresolved, num_to_select=3)
    assert selected.sum().item() == 3
    assert selected[..., 9:].all()

    unresolved &= ~selected
    selected = AcousticDecoderTransformer.select_codebooks(confidence, unresolved, num_to_select=3)
    assert selected.sum().item() == 3
    assert selected[..., 6:9].all()


def test_four_stage_loss_is_finite_and_backpropagates():
    torch.manual_seed(0)
    decoder = _make_decoder()
    inputs = torch.randn(2, 4, 16)
    audio_lens = torch.tensor([4, 3])
    semantic_tokens = torch.randint(0, 8, (2, 1, 4))
    acoustic_tokens = torch.randint(0, 8, (2, 12, 4))

    predicted, logits, loss = decoder(
        inputs, audio_lens, semantic_tokens, _VectorQuantizer(), acoustic_tokens=acoustic_tokens
    )
    loss.backward()

    assert predicted.shape == acoustic_tokens.shape
    assert logits.shape == (2, 4, 12 * 8)
    assert loss.dtype == torch.float32
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_four_stage_cached_inference_is_finite():
    torch.manual_seed(0)
    decoder = _make_decoder().eval()
    inputs = torch.randn(1, 2, 16)
    semantic_tokens = torch.randint(0, 8, (1, 1, 2))
    decoder.transformer.reset_cache(use_cache=True)

    for length in (1, 2):
        predicted, logits, loss = decoder(
            inputs[:, :length],
            torch.tensor([length]),
            semantic_tokens[:, :, :length],
            _VectorQuantizer(),
        )
        assert predicted.shape == (1, 12, length)
        assert logits.shape == (1, length, 12 * 8)
        assert torch.isfinite(logits).all()
        assert loss is None

    decoder.transformer.reset_cache(use_cache=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for the BF16 mixed-precision check")
def test_four_stage_bf16_loss_is_finite_and_decreases():
    torch.manual_seed(0)
    decoder = _make_decoder("cuda")
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-3)
    inputs = torch.randn(1, 3, 16, device="cuda")
    audio_lens = torch.tensor([3], device="cuda")
    semantic_tokens = torch.randint(0, 8, (1, 1, 3), device="cuda")
    acoustic_tokens = torch.randint(0, 8, (1, 12, 3), device="cuda")
    losses = []

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            _, _, loss = decoder(
                inputs, audio_lens, semantic_tokens, _VectorQuantizer(), acoustic_tokens=acoustic_tokens
            )
        assert loss.dtype == torch.float32
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in decoder.parameters()
        )
        optimizer.step()
        losses.append(loss.detach())

    assert losses[-1] < 0.5 * losses[0]
