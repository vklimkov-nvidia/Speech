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


class _BackboneForRefit(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 3
        num_experts = 2
        self.embed_tokens = torch.nn.Embedding(8, hidden_dim)
        self.layers = torch.nn.ModuleList([_RefitLayer(hidden_dim, num_experts)])

    def load_weights(self, weights):
        params = dict(self.named_parameters())
        loaded = set()
        for name, tensor in weights:
            target = params.get(name)
            if target is None:
                continue
            with torch.no_grad():
                target.copy_(tensor.to(dtype=target.dtype, device=target.device))
            loaded.add(name)
        return loaded


class _RefitLayer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.norm = torch.nn.LayerNorm(hidden_dim)
        self.mixer = _RefitMixer(hidden_dim, num_experts)


class _RefitMixer(torch.nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int) -> None:
        super().__init__()
        self.gate = torch.nn.Linear(hidden_dim, num_experts, bias=False)
        self.gate.e_score_correction_bias = torch.nn.Parameter(
            torch.zeros(num_experts),
            requires_grad=False,
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
    model.code_predictor = SimpleNamespace()
    model._get_decode_idxs = lambda: (torch.empty(0, dtype=torch.long), 0)
    return model


def _minimal_refit_model() -> EasyMagpieTTSForConditionalGeneration:
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.backbone = _BackboneForRefit()
    model.text_embedding = torch.nn.Embedding(8, 3)
    model.context_text_embedding = torch.nn.Embedding(8, 3)
    model.task_embedding = None
    model.code_predictor = SimpleNamespace(init_forbidden_mask=lambda: None)
    return model


def test_cfg_unconditional_prefill_replaces_the_full_context_with_cfg_unk():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model._combined_embeddings = torch.zeros(8, 3)
    model.task_embedding = None
    model.context_text_embedding = torch.nn.Embedding(10, 3)
    model.cfg_unk_token_id = 9
    model._encode_context_text = lambda _text, device: torch.tensor([2, 4], device=device)
    speaker_embedding = torch.randn(4, 3)

    result = model._build_prefill_embeds(
        torch.device("cpu"),
        {
            "speaker_embedding": speaker_embedding,
            "context_text": "[EN]",
            "cfg_role": "unconditional",
        },
    )

    expected_row = model.context_text_embedding(torch.tensor([9]))
    assert result.shape == (6, 3)
    assert torch.equal(result, expected_row.expand(6, -1))


def test_cfg_unconditional_decode_keeps_audio_but_masks_text():
    model = EasyMagpieTTSForConditionalGeneration.__new__(EasyMagpieTTSForConditionalGeneration)
    torch.nn.Module.__init__(model)
    model.has_phoneme = False
    model._cfg_roles = torch.tensor([1, 2], dtype=torch.long)
    model._dec_audio_codes = torch.tensor([[1, 2], [3, 4]], dtype=torch.long)
    model._dec_audio_valid = torch.ones(2, dtype=torch.long)
    model._dec_text_tokens = torch.tensor([1, 1], dtype=torch.long)
    model._dec_text_mask = torch.ones(2, dtype=torch.long)
    model.text_embedding = torch.nn.Embedding(3, 2)
    with torch.no_grad():
        model.text_embedding.weight.zero_()
        model.text_embedding.weight[1] = torch.tensor([10.0, 20.0])
    model.code_predictor = SimpleNamespace(
        embed_audio_frame=lambda codes: codes.float(),
    )
    combined = torch.zeros(2, 2)

    model._assemble_decode_embeddings(combined, torch.tensor([0, 1]))

    assert torch.equal(combined[0], torch.tensor([11.0, 22.0]))
    assert torch.equal(combined[1], torch.tensor([3.0, 4.0]))


def test_non_text_refit_allows_static_backbone_and_text_targets():
    model = _minimal_refit_model()
    model._easymagpie_allow_missing_text_tables_refit = True
    weights = [
        ("decoder.layers.0.norm.weight", torch.ones(3)),
        ("decoder.layers.0.norm.bias", torch.zeros(3)),
        ("decoder.layers.0.mixer.gate.weight", torch.ones(2, 3)),
    ]

    loaded = model.load_weights(weights)

    summary = model._last_easy_magpie_load_weights_summary
    assert "backbone.layers.0.norm.weight" in loaded
    assert summary["ok"] is True
    assert summary["num_blocking_missing_model_targets"] == 0
    assert "backbone.embed_tokens.weight" in summary["allowed_missing_model_targets"]
    assert "backbone.layers.0.mixer.gate.e_score_correction_bias" in summary[
        "allowed_missing_model_targets"
    ]
    assert "text_embedding.weight" in summary["allowed_missing_model_targets"]
    assert "context_text_embedding.weight" in summary["allowed_missing_model_targets"]


def test_non_text_refit_still_blocks_unexpected_missing_targets():
    model = _minimal_refit_model()
    model.backbone.unexpected_projection = torch.nn.Linear(3, 3, bias=False)
    model._easymagpie_allow_missing_text_tables_refit = True
    weights = [
        ("decoder.layers.0.norm.weight", torch.ones(3)),
        ("decoder.layers.0.norm.bias", torch.zeros(3)),
        ("decoder.layers.0.mixer.gate.weight", torch.ones(2, 3)),
    ]

    model.load_weights(weights)

    summary = model._last_easy_magpie_load_weights_summary
    assert summary["ok"] is False
    assert summary["blocking_missing_model_targets"] == ["backbone.unexpected_projection.weight"]


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
