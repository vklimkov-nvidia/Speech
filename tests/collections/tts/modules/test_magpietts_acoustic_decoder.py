# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy

import pytest
import torch
from torch import nn

from nemo.collections.tts.modules.magpietts_modules import AcousticDecoderTransformer, CodecHelper
from nemo.collections.tts.modules.transformer_2501 import Transformer


class _Codec:
    num_codebooks = 5

    def __init__(self):
        self.received_codes = None

    def eval(self):
        return self

    def decode(self, tokens, tokens_len):
        self.received_codes = tokens
        return torch.zeros(tokens.size(0), 1), tokens_len


class _Converter:
    def convert_new_to_original(self, audio_tokens, audio_lens):
        raise AssertionError("native codec tokens must not be converted a second time")


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


def _make_decoder(device='cpu', n_layers=3):
    d_model = 16
    transformer = Transformer(
        n_layers=n_layers,
        d_model=d_model,
        d_ffn=32,
        sa_n_heads=4,
        kernel_size=1,
        p_dropout=0.0,
        has_xattn=False,
        is_causal=True,
        ffn_type='swiglu',
        use_learnable_pos_emb=True,
    )
    return AcousticDecoderTransformer(
        input_dim=d_model,
        d_model=d_model,
        semantic_layer=_SemanticLayer(d_model),
        num_codebooks=12,
        codebook_size=8,
        transformer=transformer,
    ).to(device)


def _make_no_semantic_decoder(device='cpu'):
    d_model = 16
    transformer = Transformer(
        n_layers=3,
        d_model=d_model,
        d_ffn=32,
        sa_n_heads=4,
        kernel_size=1,
        p_dropout=0.0,
        has_xattn=False,
        is_causal=True,
        ffn_type='swiglu',
        use_learnable_pos_emb=True,
    )
    return AcousticDecoderTransformer(
        input_dim=d_model,
        d_model=d_model,
        semantic_layer=None,
        num_codebooks=16,
        codebook_size=8,
        prediction_schedule=(1, 3, 4, 8),
        predict_eos=True,
        transformer=transformer,
    ).to(device)


def test_codec_helper_can_decode_native_reference_tokens_without_conversion():
    codec = _Codec()
    helper = CodecHelper(codec_model=codec, codec_converter=_Converter())
    native_codes = torch.randint(0, 8, (1, 5, 3))

    helper.codes_to_audio(native_codes, torch.tensor([3]), codes_are_native=True)

    assert codec.received_codes is native_codes


def test_requires_three_causal_layers_per_stage():
    with pytest.raises(ValueError, match='exactly 3 layers'):
        _make_decoder(n_layers=6)


def test_uses_requested_prediction_schedule():
    decoder = _make_decoder()
    assert decoder.prediction_schedule == (1, 3, 4, 4)

    confidence = torch.arange(12, dtype=torch.float).view(1, 1, 12)
    unresolved = torch.ones_like(confidence, dtype=torch.bool)
    for count, expected_start in zip(decoder.prediction_schedule, (11, 8, 4, 0)):
        selected = decoder.select_codebooks(confidence, unresolved, num_to_select=count)
        assert selected.sum().item() == count
        assert selected[..., expected_start:].logical_or(~unresolved[..., expected_start:]).all()
        unresolved &= ~selected
    assert not unresolved.any()


def test_uses_four_distinct_three_layer_stage_transformers():
    torch.manual_seed(0)
    decoder = _make_decoder().eval()
    layer_calls = [[0] * 3 for _ in decoder.transformers]
    handles = []

    for stage, transformer in enumerate(decoder.transformers):
        for index, layer in enumerate(transformer.layers):

            def count_call(module, args, output, stage=stage, index=index):
                layer_calls[stage][index] += 1

            handles.append(layer.register_forward_hook(count_call))

    decoder(
        inputs=torch.randn(1, 2, 16),
        audio_lens=torch.tensor([2]),
        semantic_tokens=torch.randint(0, 8, (1, 1, 2)),
        vector_quantizer=_VectorQuantizer(),
    )
    for handle in handles:
        handle.remove()

    assert layer_calls == [[1, 1, 1]] * 4
    assert len({id(transformer.layers[0]) for transformer in decoder.transformers}) == 4
    assert not hasattr(decoder, 'maskgit_step_embedding')
    assert not hasattr(decoder, 'maskgit_step_projection')


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
    assert all(embedding.weight.grad is not None for embedding in decoder.sampled_token_embeddings)
    assert all(
        all(any(parameter.grad is not None for parameter in layer.parameters()) for layer in transformer.layers)
        for transformer in decoder.transformers
    )


def test_no_semantic_decoder_predicts_all_stacked_acoustic_channels_and_eos():
    torch.manual_seed(0)
    decoder = _make_no_semantic_decoder()
    inputs = torch.randn(2, 4, 16)
    audio_lens = torch.tensor([4, 3])
    acoustic_tokens = torch.randint(0, 8, (2, 16, 4))
    acoustic_tokens[:, :, -1] = 9  # AUDIO_EOS for a base codebook size of 8.

    predicted, logits, loss = decoder(
        inputs=inputs,
        audio_lens=audio_lens,
        semantic_tokens=None,
        vector_quantizer=None,
        acoustic_tokens=acoustic_tokens,
    )
    loss.backward()

    assert decoder.prediction_schedule == (1, 3, 4, 8)
    assert decoder.num_tokens_per_codebook == 16
    assert predicted.shape == acoustic_tokens.shape
    assert logits.shape == (2, 4, 16 * 16)
    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_loss_mask_excludes_multiturn_user_special_tokens():
    torch.manual_seed(0)
    decoder = _make_decoder()
    inputs = torch.randn(2, 4, 16)
    audio_lens = torch.tensor([4, 3])
    semantic_tokens = torch.randint(0, 8, (2, 1, 4))
    acoustic_tokens = torch.randint(0, 8, (2, 12, 4))
    acoustic_tokens[:, :, 0] = 15  # Special user token outside this decoder vocab.
    loss_mask = torch.ones(2, 4, dtype=torch.bool)
    loss_mask[:, 0] = False

    _, _, loss = decoder(
        inputs=inputs,
        audio_lens=audio_lens,
        semantic_tokens=semantic_tokens,
        vector_quantizer=_VectorQuantizer(),
        acoustic_tokens=acoustic_tokens,
        loss_mask=loss_mask,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert all(parameter.grad is None or torch.isfinite(parameter.grad).all() for parameter in decoder.parameters())


def test_stage_specific_caches_match_full_sequence_inference():
    torch.manual_seed(0)
    full_decoder = _make_decoder().eval()
    cached_decoder = copy.deepcopy(full_decoder)
    inputs = torch.randn(1, 3, 16)
    semantic_tokens = torch.randint(0, 8, (1, 1, 3))

    full_predictions, full_logits, _ = full_decoder(
        inputs,
        torch.tensor([3]),
        semantic_tokens,
        _VectorQuantizer(),
    )

    cached_decoder.reset_cache(use_cache=True)
    cached_predictions = []
    cached_logits = []
    for index in range(inputs.size(1)):
        predictions, logits, loss = cached_decoder(
            inputs[:, index : index + 1],
            torch.ones(1, dtype=torch.long),
            semantic_tokens[:, :, index : index + 1],
            _VectorQuantizer(),
        )
        cached_predictions.append(predictions)
        cached_logits.append(logits)
        assert cached_decoder.cache_sequence_lengths() == (index + 1,) * 4
        assert loss is None

    cached_decoder.reset_cache(use_cache=False)
    cached_predictions = torch.cat(cached_predictions, dim=-1)
    cached_logits = torch.cat(cached_logits, dim=1)
    assert torch.equal(cached_predictions, full_predictions)
    assert torch.allclose(cached_logits, full_logits, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is required for the BF16 mixed-precision check')
def test_four_stage_bf16_loss_is_finite_and_decreases():
    torch.manual_seed(0)
    decoder = _make_decoder('cuda')
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=3e-3)
    inputs = torch.randn(1, 3, 16, device='cuda')
    audio_lens = torch.tensor([3], device='cuda')
    semantic_tokens = torch.randint(0, 8, (1, 1, 3), device='cuda')
    acoustic_tokens = torch.randint(0, 8, (1, 12, 3), device='cuda')
    losses = []

    for _ in range(40):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
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
