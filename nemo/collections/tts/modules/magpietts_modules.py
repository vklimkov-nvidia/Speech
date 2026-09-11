# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

from __future__ import annotations

import math
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import get_worker_info

from nemo.collections.tts.modules import transformer_2501
from nemo.collections.tts.modules.nemotron_h_decoder import (
    HybridMambaAttentionDynamicCache,
    NemotronHBlock,
    NemotronHConfig,
    NemotronHRMSNorm,
)
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes.common import safe_instantiate
from nemo.core.classes.module import NeuralModule
from nemo.utils import logging
from nemo.utils.enum import PrettyStrEnum


class LocalTransformerType(PrettyStrEnum):
    """
    Enum for the type of local transformer to use in the MagpieTTS model.
    These strings are the values allowed in the YAML config file.
    """

    NO_LT = "none"
    AR = "autoregressive"
    MASKGIT = "maskgit"
    PARALLEL = "parallel"


class AcousticRefinerOrder(PrettyStrEnum):
    """
    Enum for the order in which the AcousticRefiner commits codebooks.
    These strings are the values allowed in the YAML config file.
    """

    # commit the codebooks the current refinement step is most confident about
    CONFIDENCE = "confidence"
    # commit codebooks in codebook order, following the coarse-to-fine hierarchy of the codec
    CODEBOOK = "codebook"


class EOSDetectionMethod(PrettyStrEnum):
    """
    Enum for the EOS detection method to use in the MagpieTTS model.
    These strings are the values allowed in the YAML config file.
    """

    ARGMAX_ANY = "argmax_any"
    ARGMAX_OR_MULTINOMIAL_ANY = "argmax_or_multinomial_any"
    ARGMAX_ALL = "argmax_all"
    ARGMAX_OR_MULTINOMIAL_ALL = "argmax_or_multinomial_all"
    ARGMAX_ZERO_CB = "argmax_zero_cb"
    ARGMAX_OR_MULTINOMIAL_ZERO_CB = "argmax_or_multinomial_zero_cb"

    @staticmethod
    def detection_type(detection_method: EOSDetectionMethod):
        if detection_method in [EOSDetectionMethod.ARGMAX_ANY, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY]:
            return "any"
        elif detection_method in [EOSDetectionMethod.ARGMAX_ALL, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ALL]:
            return "all"
        elif detection_method in [EOSDetectionMethod.ARGMAX_ZERO_CB, EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ZERO_CB]:
            return "zero_cb"
        else:
            raise ValueError(f"Invalid EOS detection method: {detection_method}")

    @staticmethod
    def sampling_type(detection_method: EOSDetectionMethod):
        if detection_method in [
            EOSDetectionMethod.ARGMAX_ANY,
            EOSDetectionMethod.ARGMAX_ALL,
            EOSDetectionMethod.ARGMAX_ZERO_CB,
        ]:
            return "argmax"
        elif detection_method in [
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ANY,
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ALL,
            EOSDetectionMethod.ARGMAX_OR_MULTINOMIAL_ZERO_CB,
        ]:
            return "argmax_or_multinomial"
        else:
            raise ValueError(f"Invalid EOS detection method: {detection_method}")


class SpecialAudioToken(Enum):
    """
    Enum for the special tokens to use in the MagpieTTS model.
    The special tokens are appended at the end of the codebook after the actual audio codec tokens.
    The actual embedding table index is the value below plus the number of codec tokens - do not use the Enum directly.
    """

    AUDIO_BOS = 0
    AUDIO_EOS = 1
    AUDIO_CONTEXT_BOS = 2
    AUDIO_CONTEXT_EOS = 3
    MASK_TOKEN = 4
    # Reserve these values so that if we need to add more special tokens in the future the codebook size will remain the same
    USER_SPEAKING = 5
    USER_SPEAKING_END = 6
    RESERVED_3 = 7

    @staticmethod
    def get_index(token: SpecialAudioToken, base_codebook_size: int):
        """
        Returns the index of the special token in the embedding table.
        """
        return base_codebook_size + token.value

    @staticmethod
    def get_forbidden_tokens(base_codebook_size: int, forbid_audio_eos: bool = False) -> list[int]:
        """
        Returns a list of token indices that should not be sampled or returned to user.
        Args:
            base_codebook_size (int): The size of the codec codebook (which is the first part of the embedding table).
            forbid_audio_eos (bool): Whether AUDIO_EOS should be forbidden. Default: False (i.e. allowed).
        """
        all_special_tokens = list(SpecialAudioToken)
        if not forbid_audio_eos:
            all_special_tokens.remove(SpecialAudioToken.AUDIO_EOS)
        return [SpecialAudioToken.get_index(token, base_codebook_size) for token in all_special_tokens]


def cosine_schedule(x: torch.Tensor):
    """
    Maps input values from [0, 1] to [1, 0] using the first quadrant of the cosine function.
    Used for MaskGit mask scheduling.
    """
    return torch.cos(x * (torch.pi / 2))


def build_vocabs(subword_vocab: dict, subword_padding_idx: int, special_vocab: dict = None) -> tuple[dict, dict]:
    """
    Builds the character vocabulary and the mapping from subword ids to character ids.
    Args:
        subword_vocab (dict): A dictionary of subword vocab items. Eg.
            tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
            subword_vocab = tokenizer.vocab
        subword_padding_idx (int): The padding index for the subword vocabulary.
        special_vocab (dict): items of special token dictionary (usually BOS, EOS)
            eg. special_vocab = {'<BOS>': 0, '<EOS>': 1}
    Returns:
        subword_id_to_char_ids: A dictionary mapping subword ids to character ids.
        char_vocab: A dictionary mapping character ids to their corresponding characters.
    """
    org_char_vocab = {subword: subword_id for subword, subword_id in subword_vocab.items() if len(subword) == 1}

    # Add special tokens directly to char vocab
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in org_char_vocab:
                raise ValueError(f"Special token {special_token} already exists in the character vocabulary.")
            org_char_vocab[special_token] = special_token_id

    sorted_char_vocab = dict(sorted(org_char_vocab.items(), key=lambda x: x[1]))
    char_vocab = {k: i for i, (k, _) in enumerate(sorted_char_vocab.items())}
    assert sorted(char_vocab.values()) == list(range(len(char_vocab)))
    subword_id_to_char_ids = {
        subword_id: tuple(char_vocab[char] for char in subword) for subword, subword_id in subword_vocab.items()
    }

    # Creating mapping from subword ids of special tokens to their char ids
    if special_vocab is not None:
        for special_token, special_token_id in special_vocab.items():
            if special_token in subword_id_to_char_ids:
                raise ValueError(f"Special token {special_token} already exists in the subword id Vocabulary.")
            subword_id_to_char_ids[special_token_id] = (char_vocab[special_token],)

    assert max(subword_id_to_char_ids) == len(subword_id_to_char_ids) - 1

    # Always add padding token to the end of the vocab (this is the convention used in the original code)
    subword_id_to_char_ids[subword_padding_idx] = (len(char_vocab),)

    return subword_id_to_char_ids, char_vocab


class CharAwareSubwordEncoder(NeuralModule):
    """
    Char-aware subword encoder for the MagpieTTS model.
    This module takes subword ids as input, maps them to character ids, and then applies a transformer encoder to the character embeddings.
    The output is a tensor of shape (batch_size, max_subword_length, d_embed).
    """

    def __init__(self, d_embed: int, llm_tokenizer_vocab: dict, subword_padding_idx: int, special_vocab: dict = None):
        """
        Args:
            d_embed (int): The dimension of the embedding.
            llm_tokenizer_vocab (dict): A dictionary of subword vocab items. Eg.
                tokenizer = AutoTokenizer.from_pretrained(pretrained_tokenizer_name)
                llm_tokenizer_vocab = tokenizer.vocab
            subword_padding_idx (int): The padding index for the subword vocabulary.
            special_vocab (dict): items of special token dictionary (usually BOS, EOS)
                eg. special_vocab = {'<BOS>': 30001, '<EOS>': 30002}
        """
        super().__init__()
        self.subword_id_to_char_ids, self.char_vocab = build_vocabs(
            llm_tokenizer_vocab, subword_padding_idx, special_vocab
        )
        self.embed_tokens = torch.nn.Embedding(self.vocab_size + 1, d_embed, padding_idx=self.vocab_size)
        self.encoder = transformer_2501.Transformer(
            n_layers=1,
            d_model=d_embed,
            d_ffn=d_embed * 4,
            sa_n_heads=8,
            kernel_size=1,
            max_length_causal_mask=256,
            use_learnable_pos_emb=True,
        )

    @property
    def vocab_size(self):
        return len(self.char_vocab)

    def prepare_inputs(self, subword_ids: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        device = subword_ids.device

        subword_id_list = torch.masked_select(subword_ids, padding_mask).cpu().tolist()
        char_id_list = [list(self.subword_id_to_char_ids[x]) for x in subword_id_list]

        char_lengths = torch.tensor([len(x) for x in char_id_list], dtype=torch.long, device=device)
        batch_size = char_lengths.size(0)

        char_ids = torch.full((batch_size, int(char_lengths.max().item())), self.vocab_size, dtype=torch.long)
        for i in range(batch_size):
            char_ids[i, : char_lengths[i]] = torch.tensor(char_id_list[i])
        char_ids = char_ids.to(device=device)
        return char_ids, char_lengths

    def forward(self, subword_ids: Tensor, subword_mask: Tensor | None = None) -> Tensor:
        """
        Args:
            subword_ids (Tensor): A tensor of shape (batch_size, max_subword_length) containing the subword ids.
            subword_mask (Tensor | None): A tensor of shape (batch_size, max_subword_length) containing the mask for the subword ids.
                If None, a mask of ones will be used.
        Returns:
            Tensor: A tensor of shape (batch_size, max_subword_length, d_embed) containing the subword embeddings.
        """
        device = subword_ids.device
        if subword_mask is None:
            subword_mask = torch.ones_like(subword_ids).bool()
        else:
            subword_mask = subword_mask.bool()

        if subword_mask.ndim == 3:
            subword_mask = subword_mask.squeeze(-1)

        if not subword_mask.any():
            B, T = subword_ids.shape
            D = self.embed_tokens.embedding_dim
            return torch.zeros((B, T, D), dtype=self.embed_tokens.weight.dtype, device=device)

        char_ids, char_lengths = self.prepare_inputs(subword_ids, subword_mask)
        char_mask = get_mask_from_lengths(char_lengths)
        char_emb = self.embed_tokens(char_ids)
        # char emb has the shape  [B*T, N, channels], where N is the max number of chars tokens decoded from bpe tokens
        x = self.encoder(x=char_emb, x_mask=char_mask)['output']

        # Get average embedding over the chars
        mean_emb = ((x / char_mask.unsqueeze(-1).sum(1, keepdim=True)) * char_mask.unsqueeze(-1)).sum(1)
        subword_emb = torch.zeros((subword_mask.size(0), subword_mask.size(1), mean_emb.size(-1)), device=device)
        subword_emb[subword_mask.unsqueeze(-1).expand(-1, -1, mean_emb.size(-1))] = mean_emb.view(-1)

        return subword_emb


def worker_init_fn(worker_id):
    """Per-worker init for DataLoader workers.

    Sets up tokenizers for the dataset (text and optionally phoneme)
    when using multiprocessing.
    """
    from nemo.collections.tts.data.text_to_speech_dataset_lhotse import setup_tokenizers

    logging.info(f"Worker {worker_id} initializing...")
    worker_info = get_worker_info()
    dataset = worker_info.dataset
    tokenizer = setup_tokenizers(dataset.tokenizer_config, mode=dataset.dataset_type)
    dataset.text_tokenizer = tokenizer
    if hasattr(dataset, 'phoneme_tokenizer_config'):
        dataset.phoneme_tokenizer = safe_instantiate(dataset.phoneme_tokenizer_config)


def add_eos_token(codes, codes_len, eos_id, num_eos_tokens=1):
    """Appends EOS tokens at the end of each sequence in the batch.

    Args:
        codes: (B, C, T')
        codes_len: (B,)
        eos_id: Token id to use as EOS.
        num_eos_tokens: Number of EOS tokens to append.
    """
    codes = torch.nn.functional.pad(input=codes, pad=(0, num_eos_tokens), value=0)
    codes_len = codes_len + num_eos_tokens
    for idx in range(codes.size(0)):
        codes[idx, :, codes_len[idx] - 1] = eos_id
    return codes, codes_len


def add_special_tokens(codes, codes_len, bos_id, eos_id, num_bos_tokens=1, num_eos_tokens=1):
    """Prepends BOS and appends EOS tokens to each sequence.

    Args:
        codes: (B, C, T')
    """
    codes = torch.nn.functional.pad(input=codes, pad=(num_bos_tokens, 0), value=bos_id)
    codes_len = codes_len + num_bos_tokens
    codes, codes_len = add_eos_token(codes=codes, codes_len=codes_len, eos_id=eos_id, num_eos_tokens=num_eos_tokens)
    return codes, codes_len


def remove_bos_token(codes, codes_len, num_tokens=1):
    codes = codes[:, :, num_tokens:]
    codes_len = codes_len - num_tokens
    return codes, codes_len


def remove_embedded_bos_token(embedded, embedded_len):
    embedded = embedded[:, 1:, :]
    embedded_len = embedded_len - 1
    return embedded, embedded_len


def remove_eos_token(codes, codes_len):
    codes_len = codes_len - 1
    codes = codes[:, :, :-1]
    mask = get_mask_from_lengths(lengths=codes_len)
    codes = codes * mask.unsqueeze(1)
    return codes, codes_len


def remove_embedded_eos_token(embedded, embedded_len):
    """Remove the last token from embedded sequences.

    Args:
        embedded: (B, T', D)
    """
    embedded_len = embedded_len - 1
    embedded = embedded[:, :-1, :]
    mask = get_mask_from_lengths(lengths=embedded_len)
    embedded = embedded * mask.unsqueeze(2)
    return embedded, embedded_len


def remove_special_tokens(codes, codes_len, num_bos_tokens=1):
    codes, codes_len = remove_bos_token(codes=codes, codes_len=codes_len, num_tokens=num_bos_tokens)
    codes, codes_len = remove_eos_token(codes=codes, codes_len=codes_len)
    return codes, codes_len


def pad_audio_codes(audio_codes: torch.Tensor, frame_stacking_factor: int) -> torch.Tensor:
    """Pads the time dimension of audio codes to a multiple of *frame_stacking_factor*.

    Args:
        audio_codes: (B, C, T)
        frame_stacking_factor: Factor to pad to.
    Returns:
        (B, C, T_padded)
    """
    T = audio_codes.size(2)
    T_padded = int(np.ceil(T / frame_stacking_factor) * frame_stacking_factor)
    num_pad = T_padded - T
    audio_codes = torch.nn.functional.pad(input=audio_codes, pad=(0, num_pad))
    return audio_codes


def clear_forbidden_logits(logits: torch.Tensor, codebook_size: int, forbid_audio_eos: bool = False) -> torch.Tensor:
    """Sets logits of forbidden tokens to ``-inf`` so they will never be sampled.

    Specifically, we forbid sampling of all special tokens except AUDIO_EOS
    which is allowed by default.

    Args:
        logits: (B, C, num_audio_tokens_per_codebook) or compatible shape.
        codebook_size: Base codebook size (excluding special tokens).
        forbid_audio_eos: If True, also forbid AUDIO_EOS tokens from being sampled.
    """
    logits[
        :,
        :,
        SpecialAudioToken.get_forbidden_tokens(codebook_size, forbid_audio_eos=forbid_audio_eos),
    ] = float('-inf')
    return logits


class CodecHelper:
    """Thin wrapper around a codec model and optional token converter.

    Instantiate once per model and use ``audio_to_codes`` / ``codes_to_audio``
    without having to pass the codec objects every time.
    """

    def __init__(self, codec_model, codec_converter=None):
        self.codec_model = codec_model
        self.codec_converter = codec_converter

    def audio_to_codes(self, audio, audio_len, sample_rate=None):
        """Encode audio waveforms into codec codes."""
        self.codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=audio.device.type, dtype=torch.float32):
            codes, codes_len = self.codec_model.encode(audio=audio, audio_len=audio_len, sample_rate=sample_rate)
            return codes, codes_len

    def codes_to_audio(self, codes, codes_len):
        """Decode codec codes back into audio waveforms.

        ``codes`` must already be unstacked to the shape the codec expects.
        """
        self.codec_model.eval()
        with torch.no_grad(), torch.autocast(device_type=codes.device.type, dtype=torch.float32):
            if self.codec_converter is not None:
                codes = self.codec_converter.convert_new_to_original(audio_tokens=codes, audio_lens=codes_len)
            audio, audio_len = self.codec_model.decode(tokens=codes, tokens_len=codes_len)
            return audio, audio_len, codes


class LocalTransformerHelper:
    """Orchestrates local-transformer forward passes and sampling.

    This is a plain Python class (not ``nn.Module``) that holds *references*
    to nn.Module sub-modules owned by the parent model.  Keeping it non-Module
    preserves checkpoint key compatibility.

    Args:
        local_transformer: The local transformer module.
        audio_embeddings: List/ModuleList of per-codebook embedding layers.
        audio_in_projection: Linear projection applied after per-codebook embedding.
        local_transformer_in_projection: Projection into the local transformer input space.
        local_transformer_audio_out_projection: Projection applied to local transformer output
            before the per-codebook output heads.
        local_transformer_out_projections: List/ModuleList of per-codebook output heads.
        num_audio_codebooks: Number of audio codebooks (C).
        frame_stacking_factor: Frame stacking factor (S).
        audio_eos_id: Token id for audio EOS.
        mask_token_id: Token id used for MaskGit masking.
        codebook_size: Base codebook size (excluding special tokens).
    """

    def __init__(
        self,
        local_transformer,
        audio_embeddings,
        audio_in_projection,
        local_transformer_in_projection,
        local_transformer_audio_out_projection,
        local_transformer_out_projections,
        num_audio_codebooks: int,
        frame_stacking_factor: int,
        audio_eos_id: int,
        mask_token_id: int,
        codebook_size: int,
    ):
        self.local_transformer = local_transformer
        self.audio_embeddings = audio_embeddings
        self.audio_in_projection = audio_in_projection
        self.local_transformer_in_projection = local_transformer_in_projection
        self.local_transformer_audio_out_projection = local_transformer_audio_out_projection
        self.local_transformer_out_projections = local_transformer_out_projections
        self.num_audio_codebooks = num_audio_codebooks
        self.frame_stacking_factor = frame_stacking_factor
        self.audio_eos_id = audio_eos_id
        self.mask_token_id = mask_token_id
        self.codebook_size = codebook_size

    def create_random_mask(self, codes):
        """Creates a mask where True indicates positions that should be replaced with MASK_TOKEN."""
        B, C, T = codes.shape
        rand_values = torch.rand(B, T, device=codes.device)
        frac_masked = cosine_schedule(rand_values)
        n_masked = torch.ceil(frac_masked * C).long()
        random_permutations = torch.argsort(torch.rand(B, C, T, device=codes.device), dim=1)
        mask_indices = torch.arange(C, device=codes.device).view(1, C, 1)
        mask = mask_indices < n_masked.view(B, 1, T)
        mask = torch.gather(mask, 1, random_permutations)
        return mask

    def apply_random_mask(self, codes):
        """Randomly replaces some codes with MASK_TOKEN following the cosine schedule."""
        mask = self.create_random_mask(codes)
        codes_with_mask = torch.where(mask, self.mask_token_id, codes)
        return codes_with_mask, mask

    def compute_logits(self, dec_out, audio_codes_target, targets_offset_by_one=False):
        """Predicts the logits for all codebooks using the local transformer.

        Used in both autoregressive (AR) and MaskGit (MG) modes during
        training and validation (not inference/sampling).

        The sequence layout is slightly different between AR and MG modes, as shown below
        (using an 8-codebook setup as an example)::

            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | AR target  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |   none  |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | MG target  |  none   |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            |   Input    | Magpie  |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |
            |            | Latent  | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK | or MASK |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+
            | Seq. Index |    0    |    1    |    2    |    3    |    4    |    5    |    6    |    7    |    8    |
            +------------+---------+---------+---------+---------+---------+---------+---------+---------+---------+

        Args:
            dec_out: (B, T', E)
            audio_codes_target: (B, C, T')
            targets_offset_by_one: if False, target for index 0 is codebook 0 (AR);
                if True, target for index 1 is codebook 0 (MaskGit).
        """
        C = self.num_audio_codebooks
        dec_out_all = dec_out.reshape(-1, dec_out.size(-1))  # (B*T', E)
        local_transformer_input = [dec_out_all]
        audio_codes_target = pad_audio_codes(audio_codes_target, self.frame_stacking_factor).long()
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(C):
                codes = audio_codes_target[:, codebook_num, fs_index :: self.frame_stacking_factor]
                codes = codes.reshape(-1)
                codebook_embedding = self.audio_embeddings[codebook_num + fs_index * C](codes)
                codebook_embedding = self.audio_in_projection(codebook_embedding)
                local_transformer_input.append(codebook_embedding)

        local_transformer_input = torch.stack(local_transformer_input, dim=1)
        local_transformer_input = self.local_transformer_in_projection(local_transformer_input)
        _mask = torch.ones(
            local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
        )
        local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']
        if not targets_offset_by_one:
            local_transformer_output = local_transformer_output[:, :-1, :]
        else:
            local_transformer_output = local_transformer_output[:, 1:, :]

        local_transformer_output = self.local_transformer_audio_out_projection(local_transformer_output)

        all_code_logits = []
        for fs_index in range(self.frame_stacking_factor):
            for codebook_num in range(audio_codes_target.size(1)):
                codebook_logits = self.local_transformer_out_projections[codebook_num + fs_index * C](
                    local_transformer_output[:, codebook_num + fs_index * C, :]
                )
                all_code_logits.append(codebook_logits)
        all_code_logits = torch.cat(all_code_logits, dim=1)

        all_code_logits = all_code_logits.view(
            audio_codes_target.size(0), audio_codes_target.size(2) // self.frame_stacking_factor, -1
        )

        return all_code_logits

    def sample_autoregressive(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        use_kv_cache: bool = True,
        forbid_audio_eos: bool = False,
        sanitize_logits: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes autoregressively across codebooks using the local transformer.

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature. When <= 0, uses argmax.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation (EOS forbidden).
            finished_items: Batch indices that are completed (EOS forced).
            use_cfg: Whether to use classifier-free guidance (doubled batch).
            cfg_scale: Scale factor for CFG.
            use_kv_cache: Whether to use key-value caching in the local transformer.
            forbid_audio_eos: Whether to globally forbid audio EOS.
            sanitize_logits: Whether to clamp/clean logits before sampling.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        self.local_transformer.reset_cache(use_cache=use_kv_cache)
        dec_output = dec_output.unsqueeze(1)  # (B, 1, E)
        local_transformer_input = self.local_transformer_in_projection(dec_output)
        all_preds = []
        for codebook_num in range(self.num_audio_codebooks * self.frame_stacking_factor):
            _mask = torch.ones(
                local_transformer_input.size(0), local_transformer_input.size(1), device=local_transformer_input.device
            )
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            lt_out_for_proj = self.local_transformer_audio_out_projection(local_transformer_output[:, -1, :])
            codebook_logits = self.local_transformer_out_projections[codebook_num](lt_out_for_proj)

            if use_cfg:
                actual_batch_size = codebook_logits.size(0) // 2
                conditional_logits = codebook_logits[:actual_batch_size]
                unconditional_logits = codebook_logits[actual_batch_size:]
                cfg_logits = cfg_scale * conditional_logits + (1.0 - cfg_scale) * unconditional_logits
                codebook_logits[:actual_batch_size] = cfg_logits

            if sanitize_logits:
                codebook_logits = torch.nan_to_num(codebook_logits, nan=0.0, posinf=100.0, neginf=-100.0)
                codebook_logits = codebook_logits.clamp(min=-100.0, max=100.0)

            for item_idx in unfinished_items:
                codebook_logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                codebook_logits[item_idx, :] = float('-inf')
                codebook_logits[item_idx, self.audio_eos_id] = 0.0

            codebook_logits = clear_forbidden_logits(
                codebook_logits.unsqueeze(1), self.codebook_size, forbid_audio_eos=forbid_audio_eos
            ).squeeze(1)

            codebook_logits_topk = torch.topk(codebook_logits, topk, dim=-1)[0]
            indices_to_remove = codebook_logits < codebook_logits_topk[:, -1].unsqueeze(-1)
            codebook_logits_rescored = codebook_logits.clone()
            codebook_logits_rescored[indices_to_remove] = float('-inf')

            if temperature <= 0.0:
                codebook_preds = codebook_logits_rescored.argmax(dim=-1, keepdim=True)
            else:
                codebook_probs = torch.softmax(codebook_logits_rescored / temperature, dim=-1)
                codebook_preds = torch.multinomial(codebook_probs, 1)

            if use_cfg:
                codebook_preds[actual_batch_size:] = codebook_preds[:actual_batch_size]
            all_preds.append(codebook_preds)

            next_local_transformer_input = self.audio_embeddings[codebook_num](codebook_preds.squeeze(-1)).unsqueeze(1)
            next_local_transformer_input = self.audio_in_projection(next_local_transformer_input)
            next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
            local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

        all_preds = torch.cat(all_preds, dim=1)  # (B, num_codebooks * frame_stacking_factor)
        all_preds = all_preds.reshape(-1, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)
        if use_cfg:
            all_preds = all_preds[:actual_batch_size]

        return all_preds

    def sample_maskgit(
        self,
        dec_output: torch.Tensor,
        temperature: float = 0.7,
        topk: int = 80,
        unfinished_items: Dict[int, bool] = {},
        finished_items: Dict[int, bool] = {},
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        n_steps: int = 3,
        noise_scale: float = 0.0,
        fixed_schedule: Optional[List[int]] = None,
        dynamic_cfg_scale: bool = False,
        sampling_type: Optional[str] = None,
        forbid_audio_eos: bool = False,
    ) -> torch.Tensor:
        """Sample audio codes using MaskGit-like iterative prediction with the local transformer.

        Args:
            dec_output: Decoder output tensor (B, E).
            temperature: Sampling temperature.
            topk: Number of top-probability tokens to consider.
            unfinished_items: Batch indices that have not completed generation.
            finished_items: Batch indices that are completed.
            use_cfg: Whether to use classifier-free guidance.
            cfg_scale: Scale factor for CFG.
            n_steps: Number of iterative refinement steps.
            noise_scale: Scale factor for noise added to confidence scores.
            fixed_schedule: Fixed schedule for number of tokens to unmask per step.
            dynamic_cfg_scale: Whether to dynamically adjust CFG scale.
            sampling_type: Sampling strategy.
            forbid_audio_eos: Whether to globally forbid audio EOS.

        Returns:
            Sampled audio codes (B, num_codebooks, frame_stacking_factor).
        """
        device = dec_output.device
        self.local_transformer.reset_cache(use_cache=False)
        dec_output = dec_output.unsqueeze(1)
        local_transformer_input_init = self.local_transformer_in_projection(dec_output)
        codebook_seq_len = self.num_audio_codebooks * self.frame_stacking_factor
        B = dec_output.size(0)

        min_confidence = 0
        max_confidence = 5
        confidences = min_confidence * torch.ones(B, codebook_seq_len, device=device)
        codes = self.mask_token_id * torch.ones((B, codebook_seq_len), device=device, dtype=torch.long)
        sampled_codes = codes.clone()
        if fixed_schedule is not None:
            n_steps = len(fixed_schedule)
        for step in range(n_steps):
            progress = step / n_steps
            frac_masked = cosine_schedule(torch.tensor(progress))
            if sampling_type == "causal" or sampling_type == "purity_causal":
                frac_masked = torch.ones_like(frac_masked) * (1.0 - progress)
            if fixed_schedule is None:
                n_masked = torch.ceil(codebook_seq_len * frac_masked).long()
            else:
                n_masked = codebook_seq_len - fixed_schedule[step]
            n_unmasked = codebook_seq_len - n_masked

            if sampling_type == "causal" or sampling_type == "purity_causal":
                n_frames_to_allow = int(np.floor(progress * self.frame_stacking_factor + 1))
                confidences[:, n_frames_to_allow * self.num_audio_codebooks :] = min_confidence - 1

            _, topk_indices = torch.topk(confidences, k=n_unmasked, dim=1)
            if use_cfg:
                actual_batch_size = topk_indices.size(0) // 2
                assert (
                    topk_indices[actual_batch_size:] == topk_indices[:actual_batch_size]
                ).all(), "Topk indices are not the same for conditional and unconditional codes"

            unmasked_codes = torch.gather(sampled_codes, dim=1, index=topk_indices)
            codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)

            local_transformer_input = local_transformer_input_init
            for codebook_num in range(codebook_seq_len):
                next_local_transformer_input = self.audio_embeddings[codebook_num](codes[:, codebook_num]).unsqueeze(1)
                next_local_transformer_input = self.local_transformer_in_projection(next_local_transformer_input)
                local_transformer_input = torch.cat([local_transformer_input, next_local_transformer_input], dim=1)

            _mask = torch.ones(B, codebook_seq_len + 1, device=device)
            local_transformer_output = self.local_transformer(local_transformer_input, _mask)['output']

            logits = []
            for codebook_num in range(codebook_seq_len):
                codebook_logits = self.local_transformer_out_projections[codebook_num](
                    local_transformer_output[:, codebook_num + 1, :]
                )
                logits.append(codebook_logits)
            logits = torch.stack(logits, dim=1)

            if use_cfg:
                actual_batch_size = logits.size(0) // 2
                conditional_logits = logits[:actual_batch_size]
                unconditional_logits = logits[actual_batch_size:]
                if not dynamic_cfg_scale:
                    current_cfg_scale = cfg_scale
                else:
                    progress = step / (n_steps - 1)
                    interp = progress
                    current_cfg_scale = (cfg_scale - 1) * interp + 1.0
                cfg_logits = current_cfg_scale * conditional_logits + (1.0 - current_cfg_scale) * unconditional_logits
                logits[:actual_batch_size] = cfg_logits

            logits = clear_forbidden_logits(logits, self.codebook_size, forbid_audio_eos=forbid_audio_eos)

            for item_idx in unfinished_items:
                logits[item_idx, self.audio_eos_id] = float('-inf')
            for item_idx in finished_items:
                logits[item_idx, :, :] = float('-inf')
                logits[item_idx, :, self.audio_eos_id] = 0.0

            logits_topk = torch.topk(logits, topk, dim=-1)[0]
            indices_to_remove = logits < logits_topk[:, :, -1].unsqueeze(-1)
            logits_rescored = logits.clone()
            logits_rescored[indices_to_remove] = float('-inf')
            probs = torch.softmax(logits_rescored / temperature, dim=-1)
            sampled_codes = torch.multinomial(probs.view(B * codebook_seq_len, -1), 1).view(B, codebook_seq_len)
            if use_cfg:
                sampled_codes[actual_batch_size:] = sampled_codes[:actual_batch_size]
                probs[actual_batch_size:] = probs[:actual_batch_size]
            if sampling_type != "purity_causal" and sampling_type != "purity_default":
                confidences = torch.gather(probs, dim=2, index=sampled_codes.unsqueeze(-1)).squeeze(-1)
            else:
                confidences = probs.max(dim=2)[0]
            sampled_codes.scatter_(dim=1, index=topk_indices, src=unmasked_codes)
            if noise_scale > 0.0:
                noise = (torch.rand_like(confidences) - 0.5) * noise_scale * (1 - (step + 2) / n_steps)
                confidences += noise
                confidences[actual_batch_size:] = confidences[:actual_batch_size]
            confidence_eps = 0.1
            assert (
                confidences.max() + confidence_eps < max_confidence
            ), f"Predicted confidence is approaching max_confidence: {confidences.max()}"
            confidences.scatter_(
                index=topk_indices, dim=1, src=max_confidence * torch.ones_like(topk_indices, dtype=torch.float)
            )
        codes = sampled_codes
        assert not (
            codes == self.mask_token_id
        ).any(), "Codes contain mask tokens after completion of MaskGit sampling"

        codes = codes.reshape(B, self.frame_stacking_factor, self.num_audio_codebooks).permute(0, 2, 1)

        if use_cfg:
            codes = codes[:actual_batch_size]
        return codes


# target value ignored by cross entropy, matching its default
_IGNORE_INDEX = -100


class CausalTransformerStack(torch.nn.Module):
    """
    `n_layers` attention and feed forward blocks, run causally over time.

    Built from the same NemotronH blocks the decoder backbone is made of, so a stack taken to
    another runtime reuses the kernels and the weight layout that runtime already has for the
    backbone. The blocks carry no positional encoding of their own, which is what a stack that
    reads backbone hidden states wants, and they attend through `scaled_dot_product_attention`,
    so the attention matrix is never materialized.

    Positions come in through `forward`, which takes any number of them at a time. A cache from
    `make_cache` holds the ones already seen, and only key and value tensors go into it, so the
    positions themselves never have to be passed again.

    No padding mask is taken. Attention is causal and batches are right padded, so a real position
    never reaches a padded key; positions past the end hold values that nothing reads.
    """

    def __init__(self, n_layers: int, d_model: int, d_ffn: int, n_heads: int, n_kv_heads: Optional[int] = None):
        super().__init__()
        # one attention block followed by one feed forward block is how the backbone's pattern
        # string spells what is elsewhere called a single transformer layer
        self.config = NemotronHConfig(
            hidden_size=d_model,
            num_hidden_layers=2 * n_layers,
            hybrid_override_pattern="*-" * n_layers,
            num_attention_heads=n_heads,
            num_key_value_heads=n_kv_heads or n_heads,
            intermediate_size=d_ffn,
            # a handful of layers deep, so the residual stays in the activation dtype
            residual_in_fp32=False,
        )
        self.blocks = torch.nn.ModuleList(
            [NemotronHBlock(self.config, layer_idx=idx) for idx in range(self.config.num_hidden_layers)]
        )
        self.norm_out = NemotronHRMSNorm(d_model, eps=self.config.layer_norm_epsilon)
        self._init_weights()

    def make_cache(self, batch_size: int, device, dtype) -> HybridMambaAttentionDynamicCache:
        """
        A key value cache covering one sequence of this stack.

        Hand it back on every later call; dropping it is all that ends the sequence.
        """
        return HybridMambaAttentionDynamicCache(self.config, batch_size=batch_size, dtype=dtype, device=device)

    def forward(
        self, hidden_states: torch.Tensor, cache: Optional[HybridMambaAttentionDynamicCache] = None
    ) -> torch.Tensor:
        """
        hidden_states  (batch x frames x d_model), the positions following whatever `cache` holds
        cache  filled in place when given, absent during training

        returns  (batch x frames x d_model)
        """
        attention_mask = self._offset_causal_mask(hidden_states, cache)
        for block in self.blocks:
            hidden_states = block(hidden_states, cache_params=cache, attention_mask=attention_mask)
        return self.norm_out(hidden_states)

    @staticmethod
    def _offset_causal_mask(
        hidden_states: torch.Tensor, cache: Optional[HybridMambaAttentionDynamicCache]
    ) -> Optional[torch.Tensor]:
        """
        The additive mask for queries that follow a warm cache, or None when the blocks work it out.

        `scaled_dot_product_attention` lines its own causal mask up with the top left corner, which
        is only where the triangle belongs when the queries are the whole sequence. Behind a warm
        cache it sits further right and has to be spelled out. A single query needs no mask either
        way, every key it can reach being in its past.
        """
        frames = hidden_states.size(1)
        cached = cache.get_seq_length() if cache is not None else 0
        if cached == 0 or frames == 1:
            return None
        device = hidden_states.device
        query_positions = torch.arange(cached, cached + frames, device=device).unsqueeze(1)
        blocked = torch.arange(cached + frames, device=device) > query_positions
        mask = torch.zeros(blocked.shape, dtype=hidden_states.dtype, device=device)
        return mask.masked_fill(blocked, torch.finfo(hidden_states.dtype).min)[None, None]

    def _init_weights(self):
        """Applies the backbone's initialization, which the blocks expect but do not apply."""
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

        if self.config.rescale_prenorm_residual:
            for name, parameter in self.named_parameters():
                if name.endswith(("o_proj.weight", "down_proj.weight")):
                    with torch.no_grad():
                        parameter /= math.sqrt(self.config.num_hidden_layers)


class AcousticRefiner(torch.nn.Module):
    """
    Refines acoustic codes predicted in parallel.
    Defines N transformers, that take as input hidden state, currently predicted
    acoustic codes and predict missing codes.
    Input embeddings and output projections are shared between all transformers.

    Codes are embedded by `embed_codes`, provided by the backbone: it maps
    (batch x time x num_audio_codebooks) codes to (batch x time x d_model) embeddings and has to
    accept the mask token, which is not a codec token.

    The stack works directly in the backbone's audio space, so `d_model` has to match
    `backbone_dim` and no input or output projection is needed on either side.

    It spans the backbone's timeline rather than just the frames it refines, so the positions the
    backbone takes without predicting audio go through `advance` first. Those positions carry no
    codes, only the mask token, which is also what `compute_loss` reads for them during training.
    """

    # this represents how many transformers there are,
    # and how many codebooks we commit after each of them.
    PREDICTION_SCHEDULE = (1, 3, 4, 4)

    def __init__(
        self,
        # params applied to each transformer
        n_layers,
        d_model,
        d_ffn,
        sa_n_heads,
        # defines overall refinement process
        embed_codes,  # backbone callable, embeds (batch x num_audio_codebooks x time) codes into d_model
        backbone_dim,  # dimension of the backbone hidden states and code embeddings fed to this stack
        num_audio_codebooks,
        audio_eos_id,
        mask_token_id,
        codebook_size,
        prediction_schedule=PREDICTION_SCHEDULE,
        commit_order=AcousticRefinerOrder.CODEBOOK,
    ):
        super().__init__()

        # hidden states and code embeddings are added together and read back by `out_proj`, so the
        # stack has to live in the backbone's audio space; requiring a match keeps it projection free
        if d_model != backbone_dim:
            raise ValueError(
                f"AcousticRefiner runs in the backbone's audio space and adds no projection, so its "
                f"d_model ({d_model}) has to match the backbone dimension ({backbone_dim})"
            )

        # the backbone predicts the leading codebook(s) on its own, the schedule covers the rest
        num_backbone_codebooks = num_audio_codebooks - sum(prediction_schedule)
        if num_backbone_codebooks < 1:
            raise ValueError(
                f"prediction_schedule {prediction_schedule} commits {sum(prediction_schedule)} of "
                f"{num_audio_codebooks} codebooks, leaving none for the backbone to predict"
            )

        # separate transformer for every refinement step
        self.transformers = torch.nn.ModuleList(
            [
                CausalTransformerStack(n_layers=n_layers, d_model=d_model, d_ffn=d_ffn, n_heads=sa_n_heads)
                for _ in range(len(prediction_schedule))
            ]
        )

        # codes are embedded the way the backbone does it, so the embedding tables stay owned by the
        # backbone: holding them here would register the very same parameters a second time
        self.embed_codes = embed_codes

        # one projection layer for all codebooks, covering codec tokens only (no special tokens)
        self.out_proj = torch.nn.Linear(d_model, num_audio_codebooks * codebook_size)

        # codebook range every step commits when committing in codebook order
        self.codebook_order_ranges = []
        next_codebook = num_backbone_codebooks
        for num_to_commit in prediction_schedule:
            self.codebook_order_ranges.append((next_codebook, next_codebook + num_to_commit))
            next_codebook += num_to_commit

        # access
        self.prediction_schedule = prediction_schedule
        self.commit_order = AcousticRefinerOrder(commit_order)
        self.num_backbone_codebooks = num_backbone_codebooks
        self.num_audio_codebooks = num_audio_codebooks
        self.audio_eos_id = audio_eos_id
        self.mask_token_id = mask_token_id
        self.codebook_size = codebook_size

    def make_cache(self, batch_size: int, device, dtype) -> List[HybridMambaAttentionDynamicCache]:
        """
        A key value cache per refinement step, covering one utterance and one batch layout.

        Keep it in the caller's state and hand it to every `advance` and `refine_codes` call.
        """
        return [stack.make_cache(batch_size, device=device, dtype=dtype) for stack in self.transformers]

    @staticmethod
    def cached_frames(cache: List[HybridMambaAttentionDynamicCache]) -> int:
        """How many positions the cache holds. Every refinement step holds the same count."""
        return cache[0].get_seq_length()

    def compute_loss(self, hidden_states, target_codes, lengths) -> torch.Tensor:
        """
        computes loss for the refiner.
        for that iteratively predicts codes with each `self.transformers`

        Codebooks a step has not committed yet are hidden behind the mask token, while committed
        ones are teacher forced with the target codes. Which codebooks a step commits follows the
        same rule as at inference time, so the stack is trained on the inputs it will see there.

        Runs over the backbone's whole timeline, so `hidden_states` and `target_codes` cover the
        same positions. The target codes decide what a position is: the ones the backbone takes
        without predicting audio hold the mask token, which is what `advance` feeds them at
        inference and which is not a codec token, so they condition the frames that follow without
        entering the loss.

        hidden_states  (batch x time x d_model)
        target_codes  (batch x time x num_audio_codebooks)
        lengths  (batch)

        loss: cross entropy averaged over every codebook prediction made by any step
        """
        target_codes = target_codes.long()
        # right padding needs no attention mask, the stack being causal, so lengths only keep the
        # padded positions out of the loss
        time_mask = get_mask_from_lengths(lengths, x=target_codes[:, :, 0])

        # `out_proj` only covers codec tokens, so frames holding a special token (the mask token of
        # a position without audio, the trailing EOS frame) have no valid target and stay out of the loss.
        has_target = time_mask.unsqueeze(-1) & (target_codes < self.codebook_size)
        ignored = torch.full_like(target_codes, _IGNORE_INDEX)

        committed = torch.zeros_like(target_codes, dtype=torch.bool)
        committed[:, :, : self.num_backbone_codebooks] = True

        loss_sum = hidden_states.new_zeros(())
        num_predictions = torch.zeros((), dtype=torch.long, device=target_codes.device)
        last_step = len(self.prediction_schedule) - 1
        for step, num_to_commit in enumerate(self.prediction_schedule):
            input_codes = torch.where(committed, target_codes, self.mask_token_id)
            hidden_states = self.transformers[step](hidden_states + self._embed(input_codes))
            logits = self.out_proj(hidden_states).reshape(target_codes.shape + (self.codebook_size,))

            if self.commit_order == AcousticRefinerOrder.CODEBOOK:
                selected = self._select_next_codebooks(step, committed)
            elif step < last_step:
                selected = self._select_most_confident(logits, committed, num_to_commit)
            else:
                # the last step commits whatever confidence based selection has left over
                selected = ~committed

            # a step is supervised only on the codebooks it commits, so every codebook contributes
            # exactly one prediction to the loss and none of them is weighted more than the others
            step_targets = torch.where(has_target & selected, target_codes, ignored)
            step_count = (step_targets != _IGNORE_INDEX).sum()
            step_sum = torch.nn.functional.cross_entropy(
                logits.reshape(-1, self.codebook_size),
                step_targets.reshape(-1),
                ignore_index=_IGNORE_INDEX,
                reduction='sum',
            )
            loss_sum = loss_sum + step_sum
            num_predictions = num_predictions + step_count
            committed = committed | selected

        return loss_sum / num_predictions.clamp(min=1)

    @torch.no_grad()
    def advance(self, hidden_states: torch.Tensor, cache: List[HybridMambaAttentionDynamicCache]) -> None:
        """
        Takes positions that carry no audio, so that the cache covers them and the frames refined
        later read them as left context.

        The backbone's prefill and every later step that predicts no audio come through here. They
        go in as the mask token, which is what `compute_loss` teacher forces at a position without
        audio. Nothing is sampled and nothing is returned; the pass is only there for the cache.
        Any number of positions can go in at once, warm cache or not.

        hidden_states  (batch [doubled under CFG] x frames x d_model)
        """
        codes = torch.full(
            (hidden_states.size(0), hidden_states.size(1), self.num_audio_codebooks),
            self.mask_token_id,
            dtype=torch.long,
            device=hidden_states.device,
        )
        # no code is ever committed here, so one embedding serves every step
        embedded = self._embed(codes)
        for step, stack in enumerate(self.transformers):
            hidden_states = stack(hidden_states + embedded, cache=cache[step])

    @torch.no_grad()
    def refine_codes(
        self,
        hidden_states: torch.Tensor,  # (batch [doubled under CFG] x frames x d_model)
        pred_codes: torch.Tensor,  # (batch x frames x num_audio_codebooks)
        cache: Optional[List[HybridMambaAttentionDynamicCache]] = None,
        temperature: float = 1.0,
        topk: int = 80,
        use_cfg: bool = False,
        cfg_scale: float = 1.0,
        sanitize_logits: bool = False,
        refine: Optional[torch.Tensor] = None,  # (batch)
    ) -> torch.Tensor:
        """
        Samples the acoustic codes of the given frames, one refinement step at a time.

        Mirrors `compute_loss`, with the sampled codes taking the place of the teacher forced ones:
        the leading codebooks come from the backbone through `pred_codes` and the rest start masked,
        then every step samples the codebooks it commits and hands them to the next step.

        Pass only the frames the backbone has just produced; the positions before them are the ones
        `cache` holds, having gone through `advance` or an earlier call here. Without a cache the
        whole sequence is refined at once, which is what training does.

        The stack takes the batch as one, so every row gets a cache entry whether or not it has a
        frame here. `refine` marks the rows that do; the rest keep the mask token through every
        step and commit nothing, which is what a position without audio holds during training. Its
        default of every row is what a caller handing over nothing but real frames wants.

        Under CFG `hidden_states` holds the conditional stream followed by the unconditional one,
        while `pred_codes` holds the conditional stream alone, the backbone having already sampled
        it from guided logits. Guidance is applied once per step, to this stack's own logits, and
        the codes committed from them are shared by both streams.

        returns the refined codes of the given frames, the mask token on the rows left alone
            (batch x frames x num_audio_codebooks)
        """
        num_streams = hidden_states.size(0) // 2 if use_cfg else hidden_states.size(0)
        if pred_codes.size(0) != num_streams:
            raise ValueError(
                f"expected {num_streams} rows of predicted codes to go with {hidden_states.size(0)} "
                f"rows of hidden states, but got {pred_codes.size(0)}"
            )

        if refine is None:
            refine = torch.ones(num_streams, dtype=torch.bool, device=hidden_states.device)
        elif refine.shape != (num_streams,):
            raise ValueError(f"expected which rows to refine as ({num_streams},), but got {tuple(refine.shape)}")
        refined_row = refine.view(-1, 1, 1)

        committed = torch.zeros_like(pred_codes, dtype=torch.bool)
        committed[:, :, : self.num_backbone_codebooks] = refined_row
        codes = torch.where(committed, pred_codes, self.mask_token_id)

        last_step = len(self.prediction_schedule) - 1
        for step, num_to_commit in enumerate(self.prediction_schedule):
            # both streams are refined from the same codes, only their hidden states differ
            embedded = self._embed(codes)
            if use_cfg:
                embedded = embedded.repeat(2, 1, 1)
            hidden_states = self.transformers[step](
                hidden_states + embedded, cache=None if cache is None else cache[step]
            )
            logits = self.out_proj(hidden_states)
            logits = logits.reshape(logits.shape[:2] + (self.num_audio_codebooks, self.codebook_size))

            if use_cfg:
                logits = cfg_scale * logits[:num_streams] + (1.0 - cfg_scale) * logits[num_streams:]
            if sanitize_logits:
                logits = torch.nan_to_num(logits, nan=0.0, posinf=100.0, neginf=-100.0)
                logits = logits.clamp(min=-100.0, max=100.0)

            if self.commit_order == AcousticRefinerOrder.CODEBOOK:
                selected = self._select_next_codebooks(step, committed)
            elif step < last_step:
                selected = self._select_most_confident(logits, committed, num_to_commit)
            else:
                # the last step commits whatever confidence based selection has left over
                selected = ~committed
            # the rows without a frame here hold still, so they cache nothing but the mask token
            selected = selected & refined_row

            sampled = self._sample_codes(logits, temperature, topk)
            committed = committed | selected
            codes = torch.where(selected, sampled, codes)

        return codes

    def _embed(self, codes):
        """
        Embeds codes through the backbone, which indexes them one codebook at a time.

        This stack holds its codes frame major, the layout `out_proj` writes its logits in, so the
        one tensor that has to be realigned is the smallest one: the transpose below is a view over
        the codes, where realigning the logits instead would copy them.

        codes  (batch x time x num_audio_codebooks)

        returns code embeddings  (batch x time x d_model)
        """
        return self.embed_codes(codes.transpose(1, 2))

    def _sample_codes(self, logits, temperature, topk):
        """
        Samples one token per codebook, keeping only the `topk` most likely tokens of each codebook.

        logits  (batch x time x num_audio_codebooks x codebook_size)

        returns sampled codes  (batch x time x num_audio_codebooks)
        """
        cutoff = logits.topk(min(topk, self.codebook_size), dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < cutoff, float('-inf'))
        if temperature <= 0.0:
            return logits.argmax(dim=-1)
        probs = torch.softmax(logits / temperature, dim=-1)
        return torch.multinomial(probs.reshape(-1, self.codebook_size), num_samples=1).reshape(probs.shape[:-1])

    def _select_next_codebooks(self, step, committed):
        """
        Selects the codebooks `step` commits when going in codebook order, which follows the
        coarse-to-fine hierarchy of the codec rather than what the model happens to be sure about.

        committed  (batch x time x num_audio_codebooks)

        returns the mask of newly selected codebooks  (batch x time x num_audio_codebooks)
        """
        first_codebook, last_codebook = self.codebook_order_ranges[step]
        selected = torch.zeros_like(committed)
        selected[:, :, first_codebook:last_codebook] = True
        return selected

    @torch.no_grad()
    def _select_most_confident(self, logits, committed, num_to_commit):
        """
        Picks `num_to_commit` of the not yet committed codebooks per timestamp, taking the ones the
        current step is most confident about. Selection is discrete, so it stays out of the graph.

        logits  (batch x time x num_audio_codebooks x codebook_size)
        committed  (batch x time x num_audio_codebooks)

        returns the mask of newly selected codebooks  (batch x time x num_audio_codebooks)
        """
        # Confidence of a codebook is the probability of its most likely token. Raw logits are not
        # comparable across codebooks, since every codebook has its own output head with its own
        # scale, so normalizing per codebook is what makes the top-k across codebooks meaningful.
        confidence = logits.softmax(dim=-1).amax(dim=-1)
        # confidence is in [0, 1], so -1 keeps already committed codebooks out of the selection
        confidence = confidence.masked_fill(committed, -1.0)
        selected = confidence.topk(num_to_commit, dim=-1).indices
        return torch.zeros_like(committed).scatter(dim=-1, index=selected, value=True)
