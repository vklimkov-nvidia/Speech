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


@pytest.mark.parametrize(
    ("layer_type", "tail_symbol"),
    [
        ("attention_ffn", "*"),
        ("mamba_ffn", "M"),
        ("attention", "*"),
    ],
)
def test_backbone_codebook_prediction_accepts_one_tail_block_per_codebook(layer_type, tail_symbol):
    prefix = "MEMEM*EMEMEM*EME"
    arch = EasyMagpieOmniArch.from_hf_config(
        SimpleNamespace(
            codebook_prediction_mode="backbone",
            backbone_codebook_start_layer=16,
            backbone_codebook_layer_type=layer_type,
            num_hidden_layers=32,
            hybrid_override_pattern=prefix + tail_symbol * 16,
            local_transformer_n_layers=0,
        )
    )

    assert arch.uses_backbone_codebook_layers
    assert not arch.uses_local_transformer
    assert arch.num_stacked_codebooks == 16


def test_backbone_codebook_prediction_requires_one_tail_block_per_codebook():
    with pytest.raises(ValueError, match="one logical tail block per stacked codebook"):
        EasyMagpieOmniArch.from_hf_config(
            SimpleNamespace(
                codebook_prediction_mode="backbone",
                backbone_codebook_start_layer=16,
                num_hidden_layers=31,
                hybrid_override_pattern="MEMEM*EMEMEM*EME" + "*" * 15,
                local_transformer_n_layers=0,
            )
        )


def test_backbone_codebook_prediction_requires_matching_tail_cache_type():
    with pytest.raises(ValueError, match="requires a tail"):
        EasyMagpieOmniArch.from_hf_config(
            SimpleNamespace(
                codebook_prediction_mode="backbone",
                backbone_codebook_start_layer=16,
                backbone_codebook_layer_type="mamba_ffn",
                num_hidden_layers=32,
                hybrid_override_pattern="MEMEM*EMEMEM*EME" + "*" * 16,
                local_transformer_n_layers=0,
            )
        )


def test_rejects_unknown_backbone_codebook_layer_type():
    with pytest.raises(ValueError, match="backbone_codebook_layer_type"):
        EasyMagpieOmniArch.from_hf_config(
            SimpleNamespace(
                codebook_prediction_mode="backbone",
                backbone_codebook_layer_type="unknown",
                local_transformer_n_layers=0,
            )
        )
