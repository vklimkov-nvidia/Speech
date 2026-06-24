# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import torch
from torch import nn

from easymagpie_vllm_omni.local_transformer import (
    EasyMagpieCodePredictor,
    sample_codebook,
    sample_codebook_with_logprobs,
)


def test_sampler_sanitizes_non_finite_logits_argmax():
    logits = torch.tensor([[float("nan"), float("inf"), float("-inf"), 5.0]])

    sampled = sample_codebook(logits, temperature=0.0, top_k=0, forbidden_mask=None)
    sampled_with_lp, model_lp, sampling_lp = sample_codebook_with_logprobs(
        logits,
        temperature=0.0,
        top_k=0,
        forbidden_mask=None,
    )

    assert sampled.tolist() == [1]
    assert sampled_with_lp.tolist() == [1]
    assert torch.isfinite(model_lp).all()
    assert torch.isfinite(sampling_lp).all()


def test_sampler_sanitizes_non_finite_logits_with_temperature_topk():
    logits = torch.tensor(
        [
            [float("nan"), float("inf"), float("-inf"), 5.0],
            [float("nan"), float("-inf"), -4.0, 2.0],
        ]
    )

    torch.manual_seed(0)
    sampled, model_lp, sampling_lp = sample_codebook_with_logprobs(
        logits,
        temperature=0.7,
        top_k=3,
        forbidden_mask=None,
    )

    assert sampled.shape == (2,)
    assert torch.isfinite(model_lp).all()
    assert torch.isfinite(sampling_lp).all()


class _RecordingLocalTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def forward(self, inputs_embeds):
        self.shapes.append(tuple(inputs_embeds.shape))
        return inputs_embeds


def test_code_predictor_pads_local_transformer_to_stable_graph_tokens(monkeypatch):
    monkeypatch.setenv("EASYMAGPIE_LOCAL_TRANSFORMER_GRAPH_TOKENS", "8")
    cp = EasyMagpieCodePredictor.__new__(EasyMagpieCodePredictor)
    nn.Module.__init__(cp)
    cp.num_codebooks = 2
    cp.num_tokens_per_codebook = 8
    cp._buf_inputs = torch.zeros(8, cp.num_codebooks, 4)
    cp._out_codes = torch.zeros(8, cp.num_codebooks, dtype=torch.long)
    cp._out_code_logprobs = torch.zeros(8, cp.num_codebooks, dtype=torch.float32)
    cp._out_code_sampling_logprobs = torch.zeros(8, cp.num_codebooks, dtype=torch.float32)
    cp.debug_collect_logits = False
    cp.forbidden_mask = torch.zeros(cp.num_tokens_per_codebook, dtype=torch.bool)
    cp._local_transformer_graph_tokens = EasyMagpieCodePredictor._resolve_local_transformer_graph_tokens(8)
    cp.local_transformer_in_projection = nn.Identity()
    cp.local_transformer_audio_out_projection = nn.Identity()
    cp.local_transformer_out_projections = nn.ModuleList(
        [nn.Linear(4, cp.num_tokens_per_codebook), nn.Linear(4, cp.num_tokens_per_codebook)]
    )
    cp.audio_embeddings = nn.ModuleList(
        [nn.Embedding(cp.num_tokens_per_codebook, 4), nn.Embedding(cp.num_tokens_per_codebook, 4)]
    )
    cp.audio_in_projection = nn.Identity()
    cp.temperature = 0.0
    cp.top_k = 0
    recorder = _RecordingLocalTransformer()
    cp.local_transformer = recorder

    codes_6, model_lp_6, sampling_lp_6 = cp.generate_codes_with_logprobs(
        torch.randn(6, 4)
    )
    assert codes_6.shape == (6, cp.num_codebooks)
    assert model_lp_6.shape == (6, cp.num_codebooks)
    assert sampling_lp_6.shape == (6, cp.num_codebooks)
    assert recorder.shapes
    assert set(recorder.shapes) == {(8, cp.num_codebooks, cp._buf_inputs.shape[-1])}

    recorder.shapes.clear()
    codes_7, _, _ = cp.generate_codes_with_logprobs(
        torch.randn(7, 4)
    )
    assert codes_7.shape == (7, cp.num_codebooks)
    assert set(recorder.shapes) == {(8, cp.num_codebooks, cp._buf_inputs.shape[-1])}


def test_max_concurrent_requests_does_not_enable_local_transformer_padding(monkeypatch):
    monkeypatch.delenv("EASYMAGPIE_LOCAL_TRANSFORMER_GRAPH_TOKENS", raising=False)
    monkeypatch.setenv("EASYMAGPIE_VLLM_MAX_CONCURRENT_REQUESTS", "8")

    assert EasyMagpieCodePredictor._resolve_local_transformer_graph_tokens(8) == 0
