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
"""Tests for EasyMagpie's vLLM Nemotron-H backbone call contract."""
from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")


def _install_omni_output_stub() -> None:
    if "vllm_omni.model_executor.models.output_templates" in sys.modules:
        return

    omni_pkg = types.ModuleType("vllm_omni")
    omni_pkg.__path__ = []
    model_executor_pkg = types.ModuleType("vllm_omni.model_executor")
    model_executor_pkg.__path__ = []
    models_pkg = types.ModuleType("vllm_omni.model_executor.models")
    models_pkg.__path__ = []
    output_templates = types.ModuleType("vllm_omni.model_executor.models.output_templates")

    class OmniOutput:
        def __init__(self, text_hidden_states=None, multimodal_outputs=None):
            self.text_hidden_states = text_hidden_states
            self.multimodal_outputs = multimodal_outputs or {}

    output_templates.OmniOutput = OmniOutput
    sys.modules.setdefault("vllm_omni", omni_pkg)
    sys.modules.setdefault("vllm_omni.model_executor", model_executor_pkg)
    sys.modules.setdefault("vllm_omni.model_executor.models", models_pkg)
    sys.modules["vllm_omni.model_executor.models.output_templates"] = output_templates


_install_omni_output_stub()

import easymagpie_vllm_omni.easymagpie as easymagpie_module  # noqa: E402
from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration  # noqa: E402


class _BackboneRequiringMambaCache(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_mamba_cache_params = None

    def forward(
        self,
        *,
        input_ids,
        positions,
        mamba_cache_params,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):
        self.seen_mamba_cache_params = mamba_cache_params
        return inputs_embeds


class _FakeMambaCache:
    def __init__(self) -> None:
        self.capture_batch_size = None
        self.conv_state = torch.ones(1, 2, 3)
        self.ssm_state = torch.ones(1, 2, 4)
        self.state_indices_tensor = torch.tensor([0, 1], dtype=torch.int32)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        self.capture_batch_size = batch_size
        return (self.conv_state, self.ssm_state), self.state_indices_tensor[:batch_size]


class _FakeCurrentRunMambaCache(_FakeMambaCache):
    def __init__(self) -> None:
        super().__init__()
        self.current_run_kwargs = None

    def current_run_tensors(self, **kwargs):
        self.current_run_kwargs = dict(kwargs)
        return SimpleNamespace(
            conv_state=self.conv_state,
            ssm_state=self.ssm_state,
            state_indices_tensor=self.state_indices_tensor,
        )


def _minimal_model(backbone: torch.nn.Module) -> EasyMagpieTTSForConditionalGeneration:
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.backbone = backbone
    model._backbone_accepts_mamba_cache_params = True
    model.has_phoneme = False
    model.mamba_cache = None

    max_tokens = 4
    hidden_dim = 3
    num_codebooks = 2
    model._combined_embeddings = torch.zeros(max_tokens, hidden_dim)
    model._token_stop = torch.zeros(max_tokens, dtype=torch.bool)
    model._sample_stop = torch.zeros(max_tokens, dtype=torch.bool)
    model._out_codes = torch.zeros(max_tokens, num_codebooks, dtype=torch.long)
    model._out_code_logprobs = torch.zeros(max_tokens, num_codebooks)
    model._out_code_sampling_logprobs = torch.zeros(max_tokens, num_codebooks)
    model._out_frame_logprobs = torch.zeros(max_tokens)
    model._debug_combined_pre_norm = torch.zeros(max_tokens)
    model._debug_hidden_norm = torch.zeros(max_tokens)
    model._debug_outputs_enabled = False
    model._debug_backbone_layers_enabled = False
    model._debug_backbone_active_tokens = 0
    model.code_predictor = SimpleNamespace(debug_collect_logits=False)
    model._get_decode_idxs = lambda: (torch.empty(0, dtype=torch.long), 0)
    return model


def test_forward_passes_mamba_cache_params_to_vllm_backbone():
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)
    mamba_cache_params = object()

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        inputs_embeds=torch.ones(2, 3),
        mamba_cache_params=mamba_cache_params,
    )

    assert backbone.seen_mamba_cache_params is mamba_cache_params
    assert torch.equal(output, torch.ones(2, 3))


def test_forward_without_v0_cache_metadata_uses_forward_context_cache(monkeypatch):
    monkeypatch.setattr(easymagpie_module.envs, "VLLM_USE_V1", False)
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        inputs_embeds=torch.ones(2, 3),
    )

    assert backbone.seen_mamba_cache_params is None
    assert torch.equal(output, torch.ones(2, 3))


def test_missing_vllm_mamba_cache_manager_disables_v0_cache(monkeypatch):
    monkeypatch.setattr(easymagpie_module, "_HAS_VLLM_MAMBA_CACHE_MANAGER", False)
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)
    model.vllm_config = object()
    model.model_config = SimpleNamespace(get_num_layers_by_block_type=lambda *_args, **_kwargs: 1)

    assert model._ensure_v0_mamba_cache() is None


def test_forward_without_v0_cache_metadata_uses_profile_forward_context_cache(monkeypatch):
    monkeypatch.setattr(easymagpie_module.envs, "VLLM_USE_V1", False)
    monkeypatch.setattr(
        easymagpie_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(
                num_prefills=1,
                num_decode_tokens=1,
            )
        ),
    )
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)
    cache = _FakeMambaCache()
    model.mamba_cache = cache

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        inputs_embeds=torch.ones(2, 3),
    )

    assert cache.capture_batch_size == 2
    assert backbone.seen_mamba_cache_params is not None
    assert backbone.seen_mamba_cache_params.conv_state is cache.conv_state
    assert backbone.seen_mamba_cache_params.ssm_state is cache.ssm_state
    assert torch.equal(backbone.seen_mamba_cache_params.state_indices_tensor, cache.state_indices_tensor)
    assert torch.equal(output, torch.ones(2, 3))


def test_forward_without_v0_cache_metadata_uses_explicit_profile_cache_batch_size(monkeypatch):
    monkeypatch.setattr(easymagpie_module.envs, "VLLM_USE_V1", False)
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)
    cache = _FakeMambaCache()
    model.mamba_cache = cache

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        inputs_embeds=torch.ones(2, 3),
        easymagpie_mamba_cache_batch_size=2,
    )

    assert cache.capture_batch_size == 2
    assert backbone.seen_mamba_cache_params is not None
    assert backbone.seen_mamba_cache_params.conv_state is cache.conv_state
    assert backbone.seen_mamba_cache_params.ssm_state is cache.ssm_state
    assert torch.equal(backbone.seen_mamba_cache_params.state_indices_tensor, cache.state_indices_tensor)
    assert torch.equal(output, torch.ones(2, 3))


def test_forward_prefers_request_cache_kwargs_over_profile_batch_size(monkeypatch):
    monkeypatch.setattr(easymagpie_module.envs, "VLLM_USE_V1", True)
    backbone = _BackboneRequiringMambaCache()
    model = _minimal_model(backbone)
    cache = _FakeCurrentRunMambaCache()
    model.mamba_cache = cache

    output = model(
        input_ids=torch.tensor([1, 2]),
        positions=torch.tensor([0, 1]),
        inputs_embeds=torch.ones(2, 3),
        easymagpie_mamba_cache_batch_size=2,
        request_ids_to_seq_ids={"req-a": [0], "req-b": [1]},
        finished_requests_ids=["old-req"],
    )

    assert cache.capture_batch_size is None
    assert cache.current_run_kwargs["request_ids_to_seq_ids"] == {"req-a": [0], "req-b": [1]}
    assert cache.current_run_kwargs["finished_requests_ids"] == ["old-req"]
    assert backbone.seen_mamba_cache_params is not None
    assert backbone.seen_mamba_cache_params.conv_state is cache.conv_state
    assert torch.equal(output, torch.ones(2, 3))


def test_mamba_prefill_only_metadata_returns_empty_decode_indices(monkeypatch):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._combined_embeddings = torch.zeros(4, 3)
    monkeypatch.setattr(
        easymagpie_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(
                num_prefills=1,
                num_decode_tokens=0,
            )
        ),
    )

    decode_idx, num_req = model._get_decode_idxs()

    assert num_req == 0
    assert decode_idx is not None
    assert decode_idx.numel() == 0


def test_mamba_decode_only_metadata_keeps_decode_all_path(monkeypatch):
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._combined_embeddings = torch.zeros(4, 3)
    monkeypatch.setattr(
        easymagpie_module,
        "get_forward_context",
        lambda: SimpleNamespace(
            attn_metadata=SimpleNamespace(
                num_prefills=0,
                num_decode_tokens=2,
            )
        ),
    )

    decode_idx, num_req = model._get_decode_idxs()

    assert decode_idx is None
    assert num_req == 0
