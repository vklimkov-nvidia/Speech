from types import SimpleNamespace

import pytest

from easymagpie_vllm_omni.config import EasyMagpieOmniArch


def test_default_codebook_prediction_uses_local_transformer():
    arch = EasyMagpieOmniArch.from_hf_config(SimpleNamespace())

    assert arch.codebook_prediction_mode == "autoregressive"
    assert arch.uses_local_transformer


def test_parallel_codebook_prediction_disables_local_transformer_requirements():
    arch = EasyMagpieOmniArch.from_hf_config(
        SimpleNamespace(codebook_prediction_mode="parallel", local_transformer_n_layers=0)
    )

    assert arch.codebook_prediction_mode == "parallel"
    assert not arch.uses_local_transformer


def test_autoregressive_codebook_prediction_requires_local_transformer_layers():
    with pytest.raises(ValueError, match="local_transformer_n_layers must be positive"):
        EasyMagpieOmniArch.from_hf_config(
            SimpleNamespace(codebook_prediction_mode="autoregressive", local_transformer_n_layers=0)
        )


def test_rejects_unknown_codebook_prediction_mode():
    with pytest.raises(ValueError, match="codebook_prediction_mode"):
        EasyMagpieOmniArch.from_hf_config(SimpleNamespace(codebook_prediction_mode="unknown"))
