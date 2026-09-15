# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import random
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from huggingface_hub import snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError
from omegaconf import OmegaConf
from torch import nn

from nemo.collections.tts.models import AudioCodecModel
from nemo.collections.tts.models.easy_magpietts import EasyMagpieTTSModel
from nemo.collections.tts.models.easy_magpietts_inference import EasyModelInferenceParameters, TrainingMode
from nemo.collections.tts.modules.magpietts_modules import AcousticRefiner, SpecialAudioToken, create_feature_mask
from tests.collections.tts.models.test_audio_codec import create_codec_config


pytestmark = pytest.mark.unit

BPE_TOKENIZER_NAME = "nemotron_bpe"
BPE_TOKENIZER_MODEL = "nvidia/NVIDIA-Nemotron-Nano-9B-v2"
try:
    # Attempt to resolve local cache for CI tests
    BPE_TOKENIZER_MODEL = snapshot_download(BPE_TOKENIZER_MODEL, local_files_only=True)
except LocalEntryNotFoundError:
    # For local tests, can call into HF servers to download as needed
    pass


@pytest.fixture(autouse=True, scope="module")
def _default_device_cuda():
    """Run this module with a CUDA default device without leaking it to later modules."""
    if not torch.cuda.is_available():
        yield
        return

    prev = torch.get_default_device()
    torch.set_default_device("cuda")
    try:
        yield
    finally:
        torch.set_default_device(prev)


def _restore_codec_as_random_initialized_model(*args, **kwargs):
    del args
    if kwargs.get("return_config", False):
        return create_codec_config()

    codec_cfg = kwargs.get("override_config_path", None)
    if codec_cfg is None:
        codec_cfg = create_codec_config()
    codec_model = AudioCodecModel(cfg=codec_cfg)
    codec_model.freeze()
    return codec_model


@contextmanager
def _codec_restore_uses_random_initialized_audio_codec():
    from nemo.collections.tts.models import easy_magpietts_inference

    with patch.object(
        easy_magpietts_inference.AudioCodecModel,
        "restore_from",
        staticmethod(_restore_codec_as_random_initialized_model),
    ):
        yield


def _seed_everything():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


def tiny_easy_magpie_cfg(overrides=None):
    cfg = OmegaConf.create(
        {
            "codecmodel_path": "dummy_codec.nemo",
            "decoder_type": "nemotron_h",
            "embedding_dim": 32,
            "hidden_dim": 32,
            "audio_embedding_dim": 16,
            "frame_stacking_factor": 1,
            "local_transformer_type": "none",
            "disable_lm_text_head": True,
            "disable_subword_embedding": False,
            "use_bpe_char_tokenizer": True,
            "text_conditioning_tokenizer_name": BPE_TOKENIZER_NAME,
            "use_multiturn_dataset": False,
            "run_val_inference": False,
            "use_utmos": False,
            "cfg_unconditional_prob": 0.0,
            "dropout_text_input_prob": 0.0,
            "phoneme_corruption_batch_prob": 0.0,
            "phoneme_corruption_timestep_ratio": 0.0,
            "phoneme_as_text_prob": 0.0,
            "mask_user_on_loss": True,
            "text_tokenizers": {
                BPE_TOKENIZER_NAME: {
                    "_target_": "AutoTokenizer",
                    "pretrained_model": BPE_TOKENIZER_MODEL,
                }
            },
            "training_modes": [
                {
                    "text_input_mode": "streaming",
                    "streaming_phonemes_delay": 1,
                    "streaming_speech_delay": 2,
                }
            ],
            "nemotron_h_config": {
                "hidden_size": 32,
                "num_hidden_layers": 1,
                "vocab_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "mamba_num_heads": 4,
                "mamba_head_dim": 8,
                "ssm_state_size": 8,
                "n_groups": 2,
                "intermediate_size": 64,
                "hybrid_override_pattern": "*",
                "use_cache": True,
                "_attn_implementation": "sdpa",
            },
            "optimizer": {
                "_target_": "torch.optim.AdamW",
                "lr": 0.001,
            },
        }
    )
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)
    return cfg


def _make_easy_magpie_model(cfg=None):
    _seed_everything()
    with _codec_restore_uses_random_initialized_audio_codec():
        model = EasyMagpieTTSModel(cfg or tiny_easy_magpie_cfg())
    model.eval()
    return model


@pytest.fixture()
def model():
    return _make_easy_magpie_model()


def _padded_token_tensor(model, texts):
    tokenized = [model.tokenizer.encode(text, tokenizer_name=BPE_TOKENIZER_NAME) + [model.eos_id] for text in texts]
    lens = torch.tensor([len(tokens) for tokens in tokenized], dtype=torch.long)
    max_len = int(lens.max().item())
    padded = torch.full((len(tokenized), max_len), model.pad_id, dtype=torch.long)
    for idx, tokens in enumerate(tokenized):
        padded[idx, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return padded, lens


def _toy_codes(model, batch_size, num_frames):
    codes = torch.zeros(batch_size, model.num_audio_codebooks, num_frames, dtype=torch.long)
    frame_ids = torch.arange(num_frames, dtype=torch.long)
    for batch_idx in range(batch_size):
        for codebook_idx in range(model.num_audio_codebooks):
            codes[batch_idx, codebook_idx] = (frame_ids + batch_idx + codebook_idx * 7) % model.codebook_size
    return codes


def _toy_batch(model):
    text, text_lens = _padded_token_tensor(model, ["abc", "de"])
    context_text_tokens, context_text_tokens_lens = _padded_token_tensor(model, ["hi", "ok"])

    audio_codes = _toy_codes(model, batch_size=2, num_frames=4)
    audio_codes_lens = torch.tensor([4, 3], dtype=torch.long)
    context_audio_codes = _toy_codes(model, batch_size=2, num_frames=2)
    context_audio_codes[1, :, 1] = 0
    context_audio_codes_lens = torch.tensor([2, 1], dtype=torch.long)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, False],
            [True, False, True, False, False],
        ],
        dtype=torch.bool,
    )

    return {
        "text": text,
        "text_lens": text_lens,
        "context_text_tokens": context_text_tokens,
        "context_text_tokens_lens": context_text_tokens_lens,
        "audio_codes": audio_codes,
        "audio_codes_lens": audio_codes_lens,
        "context_audio_codes": context_audio_codes,
        "context_audio_codes_lens": context_audio_codes_lens,
        "agent_mask": agent_mask,
        "task": ["tts", "tts"],
    }


@pytest.fixture()
def toy_batch(model):
    return _toy_batch(model)


def test_training_mode_and_inference_parameters():
    mode = TrainingMode(
        text_input_mode="streaming",
        streaming_phonemes_delay=4,
        streaming_speech_delay=8,
        mode_idx=2,
    )
    assert mode.name == "streaming_4_8"

    params = EasyModelInferenceParameters.from_dict(
        {
            "max_decoder_steps": 11,
            "temperature": 0.25,
            "topk": 7,
            "cfg_scale": 1.5,
            "unknown_key": "ignored",
        }
    )
    assert params == EasyModelInferenceParameters(
        max_decoder_steps=11,
        temperature=0.25,
        topk=7,
        cfg_scale=1.5,
    )


def test_easy_magpietts_model_construction(model):
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert next(model.parameters()).device.type == expected_device
    assert model.tokenizer is not None
    assert model.decoder is not None
    assert len(model.audio_embeddings) == model.num_audio_codebooks * model.frame_stacking_factor
    assert isinstance(model.audio_in_projection, nn.Linear)
    assert isinstance(model.audio_out_projection, nn.Linear)
    assert model.final_proj.out_features == model.num_audio_codebooks * model.num_all_tokens_per_codebook
    assert model.audio_bos_id == model.codebook_size
    assert model.audio_eos_id == model.codebook_size + 1
    assert model.training_modes[0].name == "streaming_1_2"
    assert model.default_inference_mode == "streaming_1_2"
    assert model.lm_text_head is None
    assert model.use_bpe_char_tokenizer
    assert model.text_conditioning_tokenizer_name == BPE_TOKENIZER_NAME
    assert hasattr(model, "cas_encoder")


def test_state_dict_excludes_codec(model):
    state = model.state_dict()
    assert state
    assert not any("_codec_model" in key for key in state)
    assert any(key.startswith("audio_embeddings.") for key in state)
    assert any(key.startswith("final_proj.") for key in state)


def test_audio_and_text_embedding_shapes(model):
    audio_tokens = _toy_codes(model, batch_size=2, num_frames=3)
    audio_tokens[0, :, -1] = model.audio_eos_id
    audio_embedded = model.embed_audio_tokens(audio_tokens)
    assert audio_embedded.shape == (2, 3, model.cfg.embedding_dim)
    assert audio_embedded.dtype == torch.float32
    assert torch.isfinite(audio_embedded).all()

    text_tokens, text_lens = _padded_token_tensor(model, ["abc", "de"])
    text_embedded = model.embed_text_tokens(text_tokens, text_lens=text_lens)
    assert text_embedded.shape == (2, text_tokens.size(1), model.cfg.embedding_dim)
    assert text_embedded.dtype == torch.float32
    assert torch.isfinite(text_embedded).all()


def test_create_feature_mask_hides_the_requested_share_of_valid_timesteps():
    _seed_everything()
    lengths = torch.tensor([10, 8, 4], dtype=torch.long)
    # pinning the share takes the randomness out of how many timesteps are hidden, but not of which
    mask = create_feature_mask(lengths, mask_min=0.5, mask_max=0.5, x=torch.zeros(3, 10))

    assert mask.shape == (3, 10)
    assert mask.dtype == torch.bool
    assert mask.sum(dim=1).tolist() == [5, 4, 2]
    # padding is never hidden, there being nothing there to hide
    assert not mask[1, 8:].any()
    assert not mask[2, 4:].any()


@pytest.mark.parametrize("use_codec_latent", [False, True])
def test_audio_history_masking_only_applies_while_training(use_codec_latent):
    model = _make_easy_magpie_model(
        tiny_easy_magpie_cfg(
            {
                "mask_audio_history": True,
                "audio_history_mask_min": 1.0,
                "audio_history_mask_max": 1.0,
                "use_codec_latent_audio_embedding": use_codec_latent,
            }
        )
    )
    codes_kwargs = {
        "audio_codes": _toy_codes(model, batch_size=2, num_frames=3),
        "audio_codes_lens": torch.tensor([3, 3], dtype=torch.long),
        "delay": torch.zeros(2, dtype=torch.long),
    }

    model.eval()
    plain, _, _, target_lens, _ = model.prepare_audio_channel_embeddings(**codes_kwargs)

    model.train()
    masked, _, _, _, _ = model.prepare_audio_channel_embeddings(**codes_kwargs)

    # a share of 1.0 hides every frame, leaving a history of nothing but mask tokens
    num_frames = int(target_lens.max())
    all_masked = torch.full(
        (2, model.num_audio_codebooks * model.frame_stacking_factor, num_frames),
        model.mask_token_id,
        dtype=torch.long,
    )
    torch.testing.assert_close(masked[:, :num_frames], model.embed_audio_tokens(all_masked))
    assert not torch.allclose(plain[:, :num_frames], masked[:, :num_frames])


def _codec_latent_cfg(overrides=None):
    cfg = {"use_codec_latent_audio_embedding": True}
    cfg.update(overrides or {})
    return tiny_easy_magpie_cfg(cfg)


def test_codec_latent_embedding_replaces_the_embedding_tables():
    model = _make_easy_magpie_model(_codec_latent_cfg())

    num_channels = model.num_audio_codebooks * model.frame_stacking_factor
    assert model.audio_embeddings is None
    assert model.audio_in_projection is None
    # the tiny codec quantizes 5 dimensions per codebook, which is all the projection has to read
    assert model.codec_latent_dim == 5
    assert model.audio_code_projection.in_features == num_channels * 5
    assert model.audio_code_projection.out_features == model.cfg.embedding_dim
    # special tokens are no codec tokens, so they keep a table, of one row per channel and token
    assert model.audio_special_embeddings.num_embeddings == num_channels * len(SpecialAudioToken)
    assert model.audio_special_embeddings.embedding_dim == model.cfg.embedding_dim

    state = model.state_dict()
    assert not any(key.startswith("audio_embeddings.") for key in state)
    assert any(key.startswith("audio_code_projection.") for key in state)
    assert any(key.startswith("audio_special_embeddings.") for key in state)

    # the whole point of the flag: the input side no longer scales with the codebook size
    table_model = _make_easy_magpie_model()
    table_params = sum(p.numel() for p in table_model.audio_embeddings.parameters())
    latent_params = model.audio_code_projection.weight.numel() + model.audio_special_embeddings.weight.numel()
    assert latent_params < table_params / 10


def test_codec_latent_embedding_separates_codec_and_special_tokens():
    model = _make_easy_magpie_model(_codec_latent_cfg())
    codes = _toy_codes(model, batch_size=2, num_frames=3)
    codes[0, :, -1] = model.audio_eos_id

    embedded = model.embed_audio_tokens(codes)

    assert embedded.shape == (2, 3, model.cfg.embedding_dim)
    assert embedded.dtype == torch.float32
    assert torch.isfinite(embedded).all()

    # a frame of special tokens has to land somewhere no frame of codec tokens can reach
    eos_frame = torch.full((1, model.num_audio_codebooks, 1), model.audio_eos_id, dtype=torch.long)
    bos_frame = torch.full_like(eos_frame, model.audio_bos_id)
    assert not torch.allclose(model.embed_audio_tokens(eos_frame), model.embed_audio_tokens(bos_frame))
    for codec_code in (0, model.codebook_size - 1):
        codec_frame = torch.full_like(eos_frame, codec_code)
        assert not torch.allclose(model.embed_audio_tokens(codec_frame), model.embed_audio_tokens(eos_frame))


@pytest.mark.parametrize("frame_stacking_factor", [1, 2], ids=["no_stacking", "stacking"])
def test_codec_latent_embedding_per_channel_matches_the_whole_frame(frame_stacking_factor):
    """Channel-by-channel embedding is what the local transformer uses, and has to agree."""
    model = _make_easy_magpie_model(_codec_latent_cfg({"frame_stacking_factor": frame_stacking_factor}))
    num_channels = model.num_audio_codebooks * frame_stacking_factor
    codes = _toy_codes(model, batch_size=2, num_frames=num_channels)[:, :, :frame_stacking_factor]
    codes = codes.reshape(2, num_channels, 1)
    codes[1, -1, 0] = model.mask_token_id

    whole_frame = model.embed_audio_tokens(codes)
    per_channel = sum(model.embed_audio_codebook(channel, codes[:, channel, :]) for channel in range(num_channels))
    # every channel carries the projection bias, which the whole frame only carries once
    per_channel = per_channel - (num_channels - 1) * model.audio_code_projection.bias

    torch.testing.assert_close(per_channel, whole_frame)


def test_process_batch_with_codec_latent_embedding_and_local_transformer():
    _seed_everything()
    model = _make_easy_magpie_model(
        _codec_latent_cfg(
            {
                "local_transformer_type": "autoregressive",
                "local_transformer_hidden_dim": 32,
                "local_transformer_n_layers": 1,
                "local_transformer_n_heads": 4,
            }
        )
    )
    batch = _toy_batch(model)

    output = model.process_batch(
        text=batch["text"],
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=batch["agent_mask"],
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.local_transformer_loss)
    output.loss.backward()
    assert model.audio_code_projection.weight.grad is not None
    assert torch.isfinite(model.audio_code_projection.weight.grad).all()


def test_stack_codes_round_trip_expected_shape(model):
    codes = _toy_codes(model, batch_size=2, num_frames=4)
    codes_lens = torch.tensor([4, 4], dtype=torch.long)

    stacked, stacked_lens = model.stack_codes(
        codes,
        codes_lens,
        bos_id=model.audio_bos_id,
        eos_id=model.audio_eos_id,
        stacking_factor=2,
        num_codebooks=model.num_audio_codebooks,
    )
    unstacked, unstacked_lens = model.unstack_codes(stacked, stacked_lens, stacking_factor=2)

    assert stacked.shape == (2, model.num_audio_codebooks * 2, 2)
    assert stacked_lens.tolist() == [2, 2]
    assert unstacked.shape == codes.shape
    assert unstacked_lens.tolist() == codes_lens.tolist()
    torch.testing.assert_close(unstacked, codes)


def test_compute_loss_with_and_without_agent_mask(model):
    _seed_everything()
    batch_size, num_frames = 2, 5
    audio_codes = torch.randint(
        low=0,
        high=model.num_all_tokens_per_codebook,
        size=(batch_size, model.num_audio_codebooks, num_frames),
    )
    audio_codes_lens = torch.tensor([5, 3], dtype=torch.long)
    logits = torch.randn(
        batch_size,
        num_frames,
        model.num_audio_codebooks * model.num_all_tokens_per_codebook,
    )
    agent_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [False, True, True, False, False],
        ],
        dtype=torch.bool,
    )

    loss, loss_mask = model.compute_loss(logits, audio_codes, audio_codes_lens)
    masked_loss, masked_loss_mask = model.compute_loss(
        logits,
        audio_codes,
        audio_codes_lens,
        agent_mask_target=agent_mask,
    )

    assert loss.ndim == 0
    assert masked_loss.ndim == 0
    assert torch.isfinite(loss)
    assert torch.isfinite(masked_loss)
    assert loss_mask.shape == (batch_size, model.num_audio_codebooks, num_frames)
    assert masked_loss_mask.shape == loss_mask.shape
    assert loss_mask.dtype == torch.bool


def test_prepare_audio_channel_embeddings_shapes(model):
    audio_codes = _toy_codes(model, batch_size=2, num_frames=3)
    audio_codes[1, :, 2] = 0
    audio_codes_lens = torch.tensor([3, 2], dtype=torch.long)
    delay = torch.tensor([2, 1], dtype=torch.long)
    agent_mask = torch.tensor(
        [
            [True, False, False, False],
            [False, True, False, False],
        ],
        dtype=torch.bool,
    )

    embeddings, lens, targets, target_lens, loss_agent_mask = model.prepare_audio_channel_embeddings(
        audio_codes=audio_codes,
        audio_codes_lens=audio_codes_lens,
        delay=delay,
        agent_mask=agent_mask,
    )

    assert embeddings.shape == (2, int((delay + target_lens).max().item()), model.cfg.embedding_dim)
    assert embeddings.dtype == torch.float32
    assert torch.isfinite(embeddings).all()
    assert lens.tolist() == (delay + target_lens).tolist()
    assert targets.shape == (2, model.num_audio_codebooks, int(target_lens.max().item()))
    assert target_lens.tolist() == [4, 3]
    assert loss_agent_mask.shape == (2, targets.size(2))
    assert loss_agent_mask.dtype == torch.bool


def test_forward_with_inputs_embeds(model):
    _seed_everything()
    inputs_embeds = torch.randn(2, 6, model.cfg.embedding_dim)
    attention_mask = torch.ones(2, 6, dtype=torch.bool)

    output = model.forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=True)

    assert output.last_hidden_state.shape == inputs_embeds.shape
    assert output.last_hidden_state.dtype == torch.float32
    assert torch.isfinite(output.last_hidden_state).all()
    assert output.past_key_values is not None


def test_logits_to_audio_codes_schema(model):
    logits = torch.zeros(2, 4, model.num_audio_codebooks * model.num_all_tokens_per_codebook)
    expected_tokens = []
    for codebook_idx in range(model.num_audio_codebooks):
        token_id = codebook_idx + 3
        expected_tokens.append(token_id)
        offset = codebook_idx * model.num_all_tokens_per_codebook
        logits[:, :, offset + token_id] = 5.0
    audio_codes_lens = torch.tensor([4, 2], dtype=torch.long)

    audio_codes = model.logits_to_audio_codes(logits, audio_codes_lens)

    assert audio_codes.shape == (2, model.num_audio_codebooks, 4)
    assert audio_codes.dtype == torch.long
    for codebook_idx, token_id in enumerate(expected_tokens):
        assert audio_codes[0, codebook_idx].tolist() == [token_id] * 4
    assert audio_codes[1, :, 2:].eq(0).all()


def test_process_batch_smoke(model, toy_batch):
    _seed_everything()
    output = model.process_batch(
        text=toy_batch["text"],
        text_lens=toy_batch["text_lens"],
        context_text_tokens=toy_batch["context_text_tokens"],
        context_text_tokens_lens=toy_batch["context_text_tokens_lens"],
        audio_codes=toy_batch["audio_codes"],
        audio_codes_lens=toy_batch["audio_codes_lens"],
        context_audio_codes=toy_batch["context_audio_codes"],
        context_audio_codes_lens=toy_batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=toy_batch["agent_mask"],
    )

    assert output.selected_training_mode == model.default_inference_mode
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert output.phoneme_loss is None
    assert output.local_transformer_loss is None
    assert output.logits.shape[:2] == output.audio_codes_target.shape[0::2]
    assert output.logits.shape[-1] == model.num_audio_codebooks * model.num_all_tokens_per_codebook
    assert output.audio_codes_target.dtype == torch.long
    assert output.context_audio_codes.shape[1] == model.num_audio_codebooks


def test_process_batch_with_multiturn_dataset_enabled():
    _seed_everything()
    model = _make_easy_magpie_model(tiny_easy_magpie_cfg({"use_multiturn_dataset": True}))
    batch = _toy_batch(model)
    text = batch["text"].clone()
    text[0, 0] = model.interruption_token_id

    output = model.process_batch(
        text=text,
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        task=batch["task"],
        agent_mask=batch["agent_mask"],
    )

    assert text[0, 0].item() == model.pad_id
    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert output.local_transformer_loss is None


def _autoregressive_local_transformer_cfg(overrides=None):
    cfg = {
        "local_transformer_type": "autoregressive",
        "local_transformer_hidden_dim": 32,
        "local_transformer_n_layers": 1,
        "local_transformer_n_heads": 4,
        "local_transformer_loss_scale": 0.5,
    }
    cfg.update(overrides or {})
    return tiny_easy_magpie_cfg(cfg)


def test_process_batch_with_autoregressive_local_transformer():
    _seed_everything()
    model = _make_easy_magpie_model(_autoregressive_local_transformer_cfg())
    batch = _toy_batch(model)

    output = model.process_batch(
        text=batch["text"],
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=batch["agent_mask"],
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert torch.isfinite(output.local_transformer_loss)
    assert output.local_transformer_draft_loss is None
    assert output.local_transformer_logits is not None
    assert output.local_transformer_logits.shape == output.logits.shape


def _all_codebook_local_transformer_cfg(overrides=None):
    cfg = {"local_transformer_predict_all_codebooks": True, "local_transformer_draft_loss_scale": 0.25}
    cfg.update(overrides or {})
    return _autoregressive_local_transformer_cfg(cfg)


def test_process_batch_predicting_all_codebooks_per_local_transformer_step():
    _seed_everything()
    model = _make_easy_magpie_model(_all_codebook_local_transformer_cfg())
    batch = _toy_batch(model)

    output = model.process_batch(
        text=batch["text"],
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=batch["agent_mask"],
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.local_transformer_loss)
    assert torch.isfinite(output.local_transformer_draft_loss)
    # the autoregressive predictions stay the ones a head per codebook makes, so they keep their shape
    assert output.local_transformer_logits.shape == output.logits.shape
    # the drafts are weighted on their own, so the autoregressive prediction keeps its weight
    assert torch.allclose(
        output.loss,
        model.parallel_codebook_loss_scale * output.codebook_loss
        + model.local_transformer_loss_scale * output.local_transformer_loss
        + model.local_transformer_draft_loss_scale * output.local_transformer_draft_loss,
    )


def test_predicting_all_codebooks_leaves_the_autoregressive_predictions_alone():
    """The drafts ride along with the autoregressive readout, which has to stay what it was."""
    _seed_everything()
    model = _make_easy_magpie_model(_all_codebook_local_transformer_cfg())
    dec_out = torch.randn(2, 4, model.cfg.hidden_dim)
    audio_codes = _toy_codes(model, batch_size=2, num_frames=4)

    all_code_logits = model._lt_helper.compute_all_codebook_logits(dec_out, audio_codes)
    autoregressive_logits = model._lt_helper.compute_logits(dec_out, audio_codes, targets_offset_by_one=False)

    num_channels = model.num_audio_codebooks * model.frame_stacking_factor
    assert all_code_logits.shape == (2, 4, num_channels, num_channels, model.num_all_tokens_per_codebook)
    assert torch.allclose(
        model._lt_helper.select_autoregressive_logits(all_code_logits), autoregressive_logits, atol=1e-5
    )


def test_predicting_all_codebooks_drafts_only_the_codebooks_a_step_has_not_been_fed():
    _seed_everything()
    model = _make_easy_magpie_model(_all_codebook_local_transformer_cfg({"num_backbone_codebooks": 2}))

    draft_mask = model._lt_helper.draft_pair_mask()

    num_channels = model.num_audio_codebooks * model.frame_stacking_factor
    assert draft_mask.shape == (num_channels, num_channels)
    for step in range(num_channels):
        for codebook in range(num_channels):
            # a step reads the codes before it and predicts its own, so only later ones are drafts,
            # and the backbone's codebooks are given to the local transformer rather than drafted
            expected = codebook > step and codebook >= 2
            assert bool(draft_mask[step, codebook]) == expected, (step, codebook)


def test_draft_loss_covers_every_drafted_prediction_and_nothing_else():
    _seed_everything()
    model = _make_easy_magpie_model(_all_codebook_local_transformer_cfg())
    batch_size, num_frames = 2, 4
    num_channels = model.num_audio_codebooks * model.frame_stacking_factor
    audio_codes = _toy_codes(model, batch_size=batch_size, num_frames=num_frames)
    audio_codes_lens = torch.tensor([num_frames, num_frames - 1], dtype=torch.long)
    # a peak on the target of every pair drives the cross entropy of a drafted pair to ~0, so any
    # pair that enters the loss shows up by flattening its peak back out
    matched_logits = torch.zeros(batch_size, num_frames, num_channels, num_channels, model.num_all_tokens_per_codebook)
    targets = audio_codes.permute(0, 2, 1)[:, :, None, :, None]
    matched_logits.scatter_(-1, targets.expand(-1, -1, num_channels, -1, -1), 30.0)

    def draft_loss(flatten=None, agent_mask_target=None):
        logits = matched_logits.clone()
        if flatten is not None:
            logits[flatten] = 0.0
        return model.compute_draft_loss(logits, audio_codes, audio_codes_lens, agent_mask_target)

    assert draft_loss() < 1e-6

    # a step's own codebook is the autoregressive prediction, which has its own loss
    assert draft_loss(flatten=(slice(None), slice(None), 1, 1)) < 1e-6
    # and the codebooks before it were fed to it, so there was nothing to predict
    assert draft_loss(flatten=(slice(None), slice(None), 3, 1)) < 1e-6
    # the codebooks after it are the drafts, which do count
    assert draft_loss(flatten=(slice(None), slice(None), 1, 3)) > 0.1
    # frames past an item's length are padding
    assert draft_loss(flatten=(1, num_frames - 1)) < 1e-6
    assert draft_loss(flatten=(0, num_frames - 1)) > 0.1
    # and a frame mask keeps whole frames out, down to nothing left to average over
    assert draft_loss(agent_mask_target=torch.zeros(batch_size, num_frames, dtype=torch.bool)) == 0.0


def test_predicting_all_codebooks_needs_the_autoregressive_local_transformer():
    _seed_everything()
    with pytest.raises(ValueError, match="needs the autoregressive local transformer"):
        _make_easy_magpie_model(_all_codebook_local_transformer_cfg({"local_transformer_type": "maskgit"}))


def test_predicting_all_codebooks_needs_the_fused_output_projection():
    _seed_everything()
    model = _make_easy_magpie_model(_autoregressive_local_transformer_cfg())

    with pytest.raises(ValueError, match="fused output projection"):
        model._lt_helper.compute_all_codebook_logits(
            torch.randn(2, 4, model.cfg.hidden_dim), _toy_codes(model, batch_size=2, num_frames=4)
        )


@pytest.mark.parametrize("frame_stacking_factor", [1, 2])
def test_autoregressive_local_transformer_decodes_on_the_backbone_codes(frame_stacking_factor):
    """The backbone owns the leading codebook, so the sampled frame has to carry its code."""
    _seed_everything()
    model = _make_easy_magpie_model(
        _autoregressive_local_transformer_cfg(
            {"num_backbone_codebooks": 1, "frame_stacking_factor": frame_stacking_factor}
        )
    )
    batch_size = 2
    last_hidden = torch.randn(batch_size, 3, model.cfg.hidden_dim)
    all_code_logits_t = model.final_proj(model.audio_out_projection(last_hidden[:, -1, :]))

    audio_codes_next, argmax_codes = model._sample_audio_codes(
        last_hidden=last_hidden,
        all_code_logits_t=all_code_logits_t,
        temperature=0.0,  # argmax, so the backbone's own codes are reproducible below
        topk=8,
        use_local_transformer_for_inference=True,
        use_cfg=False,
        cfg_scale=1.0,
    )
    backbone_codes = model.sample_codes_from_logits(
        all_code_logits_t, temperature=0.0, topk=8, forbid_special_tokens=True
    )

    assert audio_codes_next.shape == (batch_size, model.num_audio_codebooks * frame_stacking_factor)
    # stack_codes writes channel `codebook * S + frame`, so codebook 0 owns the leading S channels
    backbone_channels = slice(None, frame_stacking_factor)
    assert torch.equal(audio_codes_next[:, backbone_channels], backbone_codes[:, backbone_channels])
    assert torch.equal(argmax_codes, audio_codes_next), "argmax sampling already returns the codes it sampled"


def test_autoregressive_local_transformer_decodes_on_the_backbone_codes_under_cfg():
    _seed_everything()
    model = _make_easy_magpie_model(_autoregressive_local_transformer_cfg({"num_backbone_codebooks": 1}))
    batch_size = 2
    # under CFG the hidden states stay doubled while the backbone logits are already guided
    last_hidden = torch.randn(2 * batch_size, 3, model.cfg.hidden_dim)
    all_code_logits_t = model.final_proj(model.audio_out_projection(last_hidden[:batch_size, -1, :]))

    audio_codes_next, _ = model._sample_audio_codes(
        last_hidden=last_hidden,
        all_code_logits_t=all_code_logits_t,
        temperature=0.0,
        topk=8,
        use_local_transformer_for_inference=True,
        use_cfg=True,
        cfg_scale=2.5,
    )
    backbone_codes = model.sample_codes_from_logits(
        all_code_logits_t, temperature=0.0, topk=8, forbid_special_tokens=True
    )

    assert audio_codes_next.shape == (batch_size, model.num_audio_codebooks)
    assert torch.equal(audio_codes_next[:, :1], backbone_codes[:, :1])


def test_autoregressive_local_transformer_streams_on_the_backbone_codes():
    """The backbone's codes have to reach the local transformer through the streaming loop too."""
    _seed_everything()
    model = _make_easy_magpie_model(_autoregressive_local_transformer_cfg({"num_backbone_codebooks": 1}))
    batch = _toy_batch(model)

    state = model.streaming_init(
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        use_local_transformer=True,
        temperature=0.7,
        topk=8,
    )
    for step in range(8):
        state, _, _ = model.streaming_step(state, text_tokens=batch["text"][:, min(step, batch["text"].size(1) - 1)])

    output = model.streaming_finalize(state)
    assert output.audio_codes.size(1) == model.num_audio_codebooks


def test_autoregressive_local_transformer_requires_the_backbone_codes():
    _seed_everything()
    model = _make_easy_magpie_model(_autoregressive_local_transformer_cfg({"num_backbone_codebooks": 1}))

    with pytest.raises(ValueError, match="codes have to be passed in"):
        model._lt_helper.sample_autoregressive(dec_output=torch.randn(2, model.cfg.hidden_dim))


def test_autoregressive_local_transformer_needs_a_codebook_of_its_own():
    _seed_everything()
    with pytest.raises(ValueError, match="at least one of the 8 codebook channels"):
        _make_easy_magpie_model(_autoregressive_local_transformer_cfg({"num_backbone_codebooks": 8}))


def _parallel_local_transformer_cfg(overrides=None):
    # audio_embedding_dim is 16 while hidden_dim is 32, so this also pins down that the refiner
    # runs in the decoder's hidden space rather than the narrower audio embedding space
    cfg = {
        "local_transformer_type": "parallel",
        "local_transformer_hidden_dim": 32,
        "local_transformer_n_layers": 1,
        "local_transformer_n_heads": 4,
        "local_transformer_loss_scale": 0.5,
        # the tiny codec has 8 codebooks, so leave one for the backbone to predict
        "acoustic_refiner_prediction_schedule": [3, 4],
    }
    cfg.update(overrides or {})
    return tiny_easy_magpie_cfg(cfg)


def test_process_batch_with_parallel_local_transformer():
    _seed_everything()
    model = _make_easy_magpie_model(_parallel_local_transformer_cfg())
    batch = _toy_batch(model)

    output = model.process_batch(
        text=batch["text"],
        text_lens=batch["text_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        audio_codes=batch["audio_codes"],
        audio_codes_lens=batch["audio_codes_lens"],
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        mode="val",
        training_mode=model.training_modes[0],
        agent_mask=batch["agent_mask"],
    )

    assert torch.isfinite(output.loss)
    assert torch.isfinite(output.codebook_loss)
    assert torch.isfinite(output.local_transformer_loss)
    # the refiner has no per-codebook heads over special tokens, so it reports no comparable logits
    assert output.local_transformer_logits is None


def test_acoustic_refiner_requires_backbone_codebooks_to_match_its_schedule():
    _seed_everything()
    # the tiny codec has 8 codebooks and the schedule commits 7, leaving 1 to the backbone
    with pytest.raises(ValueError, match="leaves 1 of 8 codebooks to the backbone"):
        _make_easy_magpie_model(_parallel_local_transformer_cfg({"num_backbone_codebooks": 2}))

    model = _make_easy_magpie_model(_parallel_local_transformer_cfg({"num_backbone_codebooks": 1}))
    assert model.num_backbone_codebooks == 1


def test_acoustic_refiner_reads_backbone_codebooks_off_its_schedule():
    _seed_everything()
    model = _make_easy_magpie_model(_parallel_local_transformer_cfg())
    assert model.num_backbone_codebooks == 1


def test_acoustic_refiner_requires_backbone_hidden_dim():
    _seed_everything()
    with pytest.raises(ValueError, match="has to match the backbone dimension"):
        _make_easy_magpie_model(_parallel_local_transformer_cfg({"local_transformer_hidden_dim": 16}))


def test_acoustic_refiner_requires_matching_embedding_and_hidden_dim():
    _seed_everything()
    with pytest.raises(ValueError, match="embedding_dim has to equal hidden_dim"):
        _make_easy_magpie_model(
            _parallel_local_transformer_cfg({"embedding_dim": 16, "nemotron_h_config": {"hidden_size": 16}})
        )


def test_acoustic_refiner_samples_codes_for_the_frame():
    _seed_everything()
    model = _make_easy_magpie_model(_parallel_local_transformer_cfg())
    batch_size, num_codebooks = 2, model.num_audio_codebooks
    last_hidden = torch.randn(batch_size, 3, model.cfg.hidden_dim)
    all_code_logits_t = model.final_proj(model.audio_out_projection(last_hidden[:, -1, :]))

    cache = model._lt_helper.make_cache(batch_size, device=last_hidden.device, dtype=last_hidden.dtype)
    audio_codes_next, argmax_codes = model._sample_audio_codes(
        last_hidden=last_hidden,
        all_code_logits_t=all_code_logits_t,
        temperature=0.7,
        topk=8,
        use_local_transformer_for_inference=True,
        use_cfg=False,
        cfg_scale=1.0,
        refiner_cache=cache,
    )

    # flat (B, C*S) layout, the same shape the autoregressive local transformer returns
    assert audio_codes_next.shape == (batch_size, num_codebooks * model.frame_stacking_factor)
    assert audio_codes_next.dtype == torch.long
    assert (audio_codes_next != model.mask_token_id).all()
    assert argmax_codes.shape == audio_codes_next.shape
    assert AcousticRefiner.cached_frames(cache) == 1
    # the caller reshapes to (B, C, S) and masks with a per-item flag
    assert audio_codes_next.view(batch_size, num_codebooks, model.frame_stacking_factor).shape[1] == num_codebooks


def test_acoustic_refiner_samples_codes_under_cfg():
    _seed_everything()
    model = _make_easy_magpie_model(_parallel_local_transformer_cfg())
    batch_size = 2
    # under CFG the hidden states stay doubled while the backbone logits are already guided
    last_hidden = torch.randn(2 * batch_size, 3, model.cfg.hidden_dim)
    all_code_logits_t = model.final_proj(model.audio_out_projection(last_hidden[:batch_size, -1, :]))

    cache = model._lt_helper.make_cache(batch_size, device=last_hidden.device, dtype=last_hidden.dtype)
    for _ in range(3):
        audio_codes_next, _ = model._sample_audio_codes(
            last_hidden=last_hidden,
            all_code_logits_t=all_code_logits_t,
            temperature=0.7,
            topk=8,
            use_local_transformer_for_inference=True,
            use_cfg=True,
            cfg_scale=2.5,
            refiner_cache=cache,
        )

    assert audio_codes_next.shape == (batch_size, model.num_audio_codebooks * model.frame_stacking_factor)
    assert AcousticRefiner.cached_frames(cache) == 3


@pytest.mark.parametrize("use_cfg", [False, True], ids=["no_cfg", "cfg"])
def test_acoustic_refiner_cache_keeps_up_with_the_backbone(use_cfg):
    """Training feeds the refiner every decoder position, so streaming has to cache every one."""
    _seed_everything()
    model = _make_easy_magpie_model(_parallel_local_transformer_cfg())
    model.eval()
    batch = _toy_batch(model)

    state = model.streaming_init(
        context_audio_codes=batch["context_audio_codes"],
        context_audio_codes_lens=batch["context_audio_codes_lens"],
        context_text_tokens=batch["context_text_tokens"],
        context_text_tokens_lens=batch["context_text_tokens_lens"],
        use_cfg=use_cfg,
        cfg_scale=2.5 if use_cfg else 1.0,
        use_local_transformer=True,
    )

    # the context prefill goes in as one pass, so the refiner is level with the backbone right away
    assert state.cache_seq_len > 1
    assert AcousticRefiner.cached_frames(state.refiner_cache) == state.cache_seq_len

    # the steps before speech starts predict no audio, and still have to reach the refiner
    for step in range(8):
        state, _, _ = model.streaming_step(state, text_tokens=batch["text"][:, min(step, batch["text"].size(1) - 1)])
        cached = AcousticRefiner.cached_frames(state.refiner_cache)
        assert cached == state.cache_seq_len, f"drifted apart on step {step}"

    model.streaming_finalize(state)
    assert state.refiner_cache is None, "the per-utterance cache is released"


def test_training_step_smoke(model, toy_batch):
    _seed_everything()
    model.train()

    with (
        patch.object(model, "log", lambda *args, **kwargs: None),
        patch.object(model, "log_dict", lambda *args, **kwargs: None),
    ):
        loss = model.training_step(toy_batch, batch_idx=0)

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_validation_step_smoke(model, toy_batch, tmp_path):
    _seed_everything()
    model.eval()
    object.__setattr__(
        model,
        "_trainer",
        SimpleNamespace(world_size=1, global_rank=0, local_rank=0, log_dir=str(tmp_path), current_epoch=0),
    )

    with patch.object(model, "log_val_audio_example", lambda *args, **kwargs: {}):
        output = model.validation_step(toy_batch, batch_idx=1)

    assert set(output.keys()) == {"val_loss", "val_codebook_loss", "val_local_transformer_loss"}
    assert torch.isfinite(output["val_loss"])
    assert torch.isfinite(output["val_codebook_loss"])
    assert output["val_local_transformer_loss"] is None
    assert model.validation_step_outputs[-1] == output
