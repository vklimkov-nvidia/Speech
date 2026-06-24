# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import math
from pathlib import Path
from typing import List

import torch
from einops import rearrange
from hydra.utils import instantiate
from lightning.pytorch import Trainer
from omegaconf import DictConfig

from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset, stack_tensors
from nemo.collections.tts.losses.acoustic_model_loss import AudioTokenLoss
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import model_utils
from nemo.utils.decorators import experimental


@experimental
class AcousticModelAutoregressiveWithContext(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = instantiate(cfg.text_tokenizer)
        self.inference_phoneme_probability = cfg.get("inference_phoneme_probability", 1.0)

        super().__init__(cfg=cfg, trainer=trainer)

        # Text tokenizer information
        num_text_embed = len(self.text_tokenizer.tokens)
        self.pad_with_space = self.text_tokenizer.pad_with_space
        self.text_pad_token = self.text_tokenizer.pad
        self.space_token = self.text_tokenizer.space

        # Context length in terms of number of audio tokens
        self.target_min_len = cfg.get("target_min_len", 10)
        self.context_min_len = cfg.get("context_min_len", 25)
        self.context_max_len = cfg.get("context_max_len", 125)
        self.context_len_noise = cfg.get("context_len_noise", 5)

        # Quantizer definitions
        self.semantic_codebook_num = cfg.get("semantic_codebook_num")
        self.semantic_codebook_dim = cfg.get("semantic_codebook_dim")
        self.acoustic_codebook_num = cfg.get("acoustic_codebook_num")
        self.acoustic_codebook_dim = cfg.get("acoustic_codebook_dim")

        self.text_encoder = instantiate(cfg.text_encoder, n_embed=num_text_embed, padding_idx=self.text_pad_token)
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)
        self.context_encoder = instantiate(cfg.context_encoder)
        self.semantic_layer = instantiate(cfg.semantic_layer)

        self.vector_quantizer = instantiate(cfg.vector_quantizer)

        if "vector_quantizer_codec" in cfg:
            vector_quantizer_codec = instantiate(cfg.vector_quantizer_codec)
            self.vector_quantizer_converter_codec = VectorQuantizerIndexConverter(
                vector_quantizer_original=vector_quantizer_codec,
                vector_quantizer_new=self.vector_quantizer,
            )
        else:
            self.vector_quantizer_converter_codec = None

        if "vector_quantizer_acoustic" in cfg:
            self.vector_quantizer_acoustic = instantiate(cfg.vector_quantizer_acoustic)
            self.vector_quantizer_converter_acoustic = VectorQuantizerIndexConverter(
                vector_quantizer_original=self.vector_quantizer,
                vector_quantizer_new=self.vector_quantizer_acoustic,
            )
        else:
            self.vector_quantizer_acoustic = self.vector_quantizer
            self.vector_quantizer_converter_acoustic = None

        self.aligner = instantiate(cfg.aligner, num_text_emb=num_text_embed)

        # Audio infill hyperparemters
        self.audio_infill_min = cfg.get("audio_infill_min", 0.1)
        self.audio_infill_max = cfg.get("audio_infill_max", 1.0)
        audio_infill_beta = cfg.get("audio_infill_beta", 2.0)
        self.audio_infill_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=audio_infill_beta)

        # Semantic masking hyperparemters
        self.semantic_mask_min = cfg.get("semantic_mask_min", 0.0)
        self.semantic_mask_max = cfg.get("semantic_mask_max", 0.5)
        semantic_mask_beta = cfg.get("semantic_mask_beta", 2.0)
        self.semantic_mask_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=semantic_mask_beta)

        # Reconstruction losses
        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        self.audio_token_loss_fn = AudioTokenLoss(num_codebooks=self.acoustic_codebook_num)

        self.aligner_bin_loss_scale = cfg.get("aligner_bin_loss_scale", 0.01)
        self.aligner_ctc_loss_scale = cfg.get("aligner_ctc_loss_scale", 0.01)
        self.bin_loss_start_epoch = cfg.get("bin_loss_start_epoch", 0)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 10)

        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.log_config = cfg.get("log_config", None)

    def parse(self, str_input: str) -> torch.tensor:
        if not hasattr(self.text_tokenizer, "set_phone_prob"):
            text_tokens = self.text_tokenizer.encode(str_input)
        else:
            with self.text_tokenizer.set_phone_prob(prob=self.inference_phoneme_probability):
                text_tokens = self.text_tokenizer.encode(str_input)

        token_tensor = torch.tensor(text_tokens).unsqueeze_(0).long().to(self.device)
        return token_tensor

    def create_infill_mask(self, input_lens, dist, infill_min, infill_max):
        batch_size = input_lens.shape[0]
        len_mask = get_mask_from_lengths(input_lens)
        max_len = len_mask.shape[1]

        infill_percent = dist.sample(sample_shape=torch.Size([batch_size])).to(input_lens.device)
        infill_percent = infill_min + (infill_max - infill_min) * infill_percent
        infill_len = infill_percent * input_lens.float()
        infill_rank = torch.clamp_min(infill_len - 1, 0).long()
        infill_rank = rearrange(infill_rank, 'B -> B 1')

        # [batch_size, time]
        infill_vals = torch.rand(size=len_mask.shape, device=input_lens.device)
        infill_vals = infill_vals * len_mask
        infill_topk = torch.topk(infill_vals, k=max_len, dim=1, sorted=True).values
        infill_min_val = torch.gather(infill_topk, index=infill_rank, dim=1)
        infill_mask = infill_vals >= infill_min_val

        infill_mask = infill_mask * len_mask

        return infill_mask

    def _sample_lens(self, batch_size, audio_lens, random_sample, max_len=None):
        if max_len is None:
            max_len = self.context_max_len

        # [B]
        max_lens = torch.clamp_max(audio_lens, max=max_len)
        min_lens = torch.clamp_max(max_lens, max=self.context_min_len)

        sample_len_list = []
        for i in range(batch_size):
            if random_sample:
                # Sample between minimum and maximum sample size
                sample_len = torch.randint(low=min_lens[i], high=max_lens[i] + 1, size=[]).item()
            else:
                # Use maximum sample size
                sample_len = max_lens[i].item()

            sample_len_list.append(sample_len)

        sample_lens = torch.tensor(sample_len_list, device=audio_lens.device)
        return sample_lens

    def _find_space_endings(self, text, durs, max_audio_len):
        # [B, T_text]
        cum_ends = torch.cumsum(durs, dim=1).long()
        space_ends = torch.where(text == self.space_token, cum_ends, torch.zeros_like(cum_ends))
        space_ends_invert = torch.where(text == self.space_token, cum_ends, max_audio_len * torch.ones_like(cum_ends))
        # [B, 1]
        if self.pad_with_space:
            min_space = space_ends_invert.topk(k=2, dim=1, largest=False).values[:, 1:2]
            max_space = space_ends.topk(k=2, dim=1).values[:, 1:2]
        else:
            min_space = space_ends_invert.topk(k=1, dim=1, largest=False).values[:, :1]
            max_space = space_ends.topk(k=1, dim=1).values[:, :1]

        return space_ends, min_space, max_space

    def _slice_context_information(
        self,
        text,
        audio_tokens,
        audio_codes,
        batch_size,
        context_starts,
        context_ends,
        context_lens,
        text_starts,
        text_ends,
        target_text_lens,
        audio_starts,
        audio_ends,
        target_audio_lens,
    ):
        context_list = []
        target_text_list = []
        target_audio_token_list = []
        target_audio_code_list = []
        for i in range(batch_size):
            context_start_i = context_starts[i].item()
            context_end_i = context_ends[i].item()
            context_len_i = context_lens[i].item()
            text_start_i = text_starts[i].item()
            text_end_i = text_ends[i].item()
            audio_start_i = audio_starts[i].item()
            audio_end_i = audio_ends[i].item()

            context_i = audio_codes[i, :, context_start_i:context_end_i]
            if context_i.shape[1] < self.context_max_len:
                num_repeats = int(math.ceil(self.context_max_len / context_len_i))
                context_i = torch.tile(context_i, dims=(1, num_repeats))
                context_i = context_i[:, : self.context_max_len]

            text_i = text[i, text_start_i:text_end_i]

            audio_tokens_i = audio_tokens[i, :, audio_start_i:audio_end_i]
            audio_codes_i = audio_codes[i, :, audio_start_i:audio_end_i]

            context_list.append(context_i)
            target_text_list.append(text_i)
            target_audio_token_list.append(audio_tokens_i)
            target_audio_code_list.append(audio_codes_i)

        context_codes = stack_tensors(tensors=context_list, max_lens=[self.context_max_len]).to(audio_codes.device)
        context_output_lens = self.context_max_len * torch.ones_like(context_lens)

        max_text_len = target_text_lens.max().item()
        target_text = stack_tensors(tensors=target_text_list, max_lens=[max_text_len]).to(audio_codes.device)

        max_audio_len = target_audio_lens.max().item()
        target_audio_tokens = stack_tensors(tensors=target_audio_token_list, max_lens=[max_audio_len]).to(
            audio_codes.device
        )
        target_audio_codes = stack_tensors(tensors=target_audio_code_list, max_lens=[max_audio_len]).to(
            audio_codes.device
        )

        return context_codes, context_output_lens, target_text, target_audio_tokens, target_audio_codes

    def sample_context_audio_start(
        self, audio_tokens, audio_codes, audio_lens, text, durs, text_lens, random_sample, max_len=None
    ):
        batch_size = audio_codes.shape[0]
        max_audio_len = audio_tokens.shape[2]
        # [B, 1]
        context_ends = self._sample_lens(
            batch_size=batch_size, audio_lens=audio_lens, random_sample=random_sample, max_len=max_len
        )
        context_ends = rearrange(context_ends, 'B -> B 1')
        space_ends, min_space, max_space = self._find_space_endings(text=text, durs=durs, max_audio_len=max_audio_len)

        # [B, T]
        valid_space = torch.logical_and(space_ends >= min_space, space_ends <= max_space)
        valid_space = torch.logical_and(valid_space, space_ends <= context_ends)
        valid_space = torch.logical_or(valid_space, space_ends == min_space)
        space_ends = torch.where(valid_space, space_ends, torch.zeros_like(space_ends))

        # [B]
        context_end_topk = space_ends.topk(k=1, dim=1)
        context_lens = context_end_topk.values[:, 0]
        target_audio_lens = audio_lens - context_lens

        target_text_starts = context_end_topk.indices[:, 0] + torch.tensor(1)
        target_text_lens = text_lens - target_text_starts

        audio_starts = context_lens
        target_audio_lens = torch.maximum(target_audio_lens, target_text_lens)

        if self.context_len_noise and random_sample:
            min_context_len = torch.clamp_max(input=context_lens, max=self.context_min_len)
            context_len_noise = torch.randint_like(input=context_lens, low=0, high=self.context_len_noise + 1)
            context_lens = context_lens - context_len_noise
            context_lens = torch.maximum(input=context_lens, other=min_context_len)

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_audio_tokens, target_audio_codes = (
            self._slice_context_information(
                text=text,
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                batch_size=batch_size,
                context_starts=zero_starts,
                context_ends=context_lens,
                context_lens=context_lens,
                text_starts=target_text_starts,
                text_ends=text_lens,
                target_text_lens=target_text_lens,
                audio_starts=audio_starts,
                audio_ends=audio_lens,
                target_audio_lens=target_audio_lens,
            )
        )

        return (
            target_text,
            target_text_lens,
            target_audio_tokens,
            target_audio_codes,
            target_audio_lens,
            context_codes,
            context_lens,
        )

    def sample_context_audio_end(self, audio_tokens, audio_codes, audio_lens, text, durs):
        batch_size = audio_codes.shape[0]
        max_audio_len = audio_tokens.shape[2]
        # [B, 1]
        context_rand_lens = self._sample_lens(batch_size=batch_size, audio_lens=audio_lens, random_sample=True)
        context_starts = audio_lens - context_rand_lens
        context_starts = rearrange(context_starts, 'B -> B 1')
        space_ends, min_space, max_space = self._find_space_endings(text=text, durs=durs, max_audio_len=max_audio_len)

        # [B, T]
        valid_space = torch.logical_and(space_ends >= min_space, space_ends <= max_space)
        valid_space = torch.logical_and(valid_space, space_ends >= context_starts)
        valid_space = torch.logical_or(valid_space, space_ends == max_space)
        space_ends = torch.where(valid_space, space_ends, max_audio_len * torch.ones_like(space_ends))

        # [B]
        context_start_topk = space_ends.topk(k=1, dim=1, largest=False)
        target_audio_lens = context_start_topk.values[:, 0]
        context_lens = audio_lens - target_audio_lens
        target_text_lens = context_start_topk.indices[:, 0] + torch.tensor(1)

        target_audio_lens = torch.maximum(target_audio_lens, target_text_lens)

        if self.context_len_noise:
            min_context_len = torch.clamp_max(input=context_lens, max=self.context_min_len)
            context_len_noise = torch.randint_like(input=context_lens, low=0, high=self.context_len_noise + 1)
            context_lens = context_lens - context_len_noise
            context_lens = torch.maximum(input=context_lens, other=min_context_len)
            context_starts = audio_lens - context_lens

        zero_starts = torch.zeros(batch_size, dtype=torch.int32)
        context_codes, context_lens, target_text, target_audio_tokens, target_audio_codes = (
            self._slice_context_information(
                text=text,
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                batch_size=batch_size,
                context_starts=context_starts,
                context_ends=audio_lens,
                context_lens=context_lens,
                text_starts=zero_starts,
                text_ends=target_text_lens,
                target_text_lens=target_text_lens,
                audio_starts=zero_starts,
                audio_ends=target_audio_lens,
                target_audio_lens=target_audio_lens,
            )
        )

        return (
            target_text,
            target_text_lens,
            target_audio_tokens,
            target_audio_codes,
            target_audio_lens,
            context_codes,
            context_lens,
        )

    def _concat_tensors(self, inputs1, inputs2, pad_value=0.0):
        len1 = inputs1.shape[-1]
        len2 = inputs2.shape[-1]
        padding1 = max(len2 - len1, 0)
        padding2 = max(len1 - len2, 0)
        out1 = torch.nn.functional.pad(inputs1, pad=[0, padding1], value=pad_value)
        out2 = torch.nn.functional.pad(inputs2, pad=[0, padding2], value=pad_value)
        out = torch.cat([out1, out2], dim=0)
        return out

    def sample_context_audio_batch(self, audio_tokens, audio_codes, audio_lens, text, text_lens, durs):
        half_batch_size = audio_tokens.shape[0] // 2
        (
            text_sample1,
            text_sample_lens1,
            audio_token_sample1,
            audio_codes_sample1,
            audio_token_sample_lens1,
            context_codes1,
            context_lens1,
        ) = self.sample_context_audio_start(
            audio_tokens=audio_tokens[:half_batch_size],
            audio_codes=audio_codes[:half_batch_size],
            audio_lens=audio_lens[:half_batch_size],
            text=text[:half_batch_size],
            text_lens=text_lens[:half_batch_size],
            durs=durs[:half_batch_size],
            random_sample=True,
        )
        (
            text_sample2,
            text_sample_lens2,
            audio_token_sample2,
            audio_codes_sample2,
            audio_token_sample_lens2,
            context_codes2,
            context_lens2,
        ) = self.sample_context_audio_end(
            audio_tokens=audio_tokens[half_batch_size:],
            audio_codes=audio_codes[half_batch_size:],
            audio_lens=audio_lens[half_batch_size:],
            text=text[half_batch_size:],
            durs=durs[half_batch_size:],
        )
        text_sample = self._concat_tensors(text_sample1, text_sample2, pad_value=self.text_pad_token)
        text_sample_lens = torch.cat([text_sample_lens1, text_sample_lens2], dim=0)
        audio_token_sample = self._concat_tensors(audio_token_sample1, audio_token_sample2)
        audio_codes_sample = self._concat_tensors(audio_codes_sample1, audio_codes_sample2)
        audio_token_sample_lens = torch.cat([audio_token_sample_lens1, audio_token_sample_lens2], dim=0)
        context_codes = self._concat_tensors(context_codes1, context_codes2)
        context_lens = torch.cat([context_lens1, context_lens2], dim=0)

        return (
            text_sample,
            text_sample_lens,
            audio_token_sample,
            audio_codes_sample,
            audio_token_sample_lens,
            context_codes,
            context_lens,
        )

    def get_context(self, audio_tokens, audio_lens, max_len=None):
        if max_len is None:
            max_len = self.context_max_len

        batch_size = audio_tokens.shape[0]
        context_lens = torch.clamp_max(audio_lens, max=max_len)
        context_list = []
        context_len_list = []
        for i in range(batch_size):
            context_len_i = context_lens[i]
            context_i = audio_tokens[i, :, :context_len_i]

            if context_i.shape[1] < self.context_max_len:
                num_repeats = int(math.ceil(self.context_max_len / context_len_i))
                context_i = torch.tile(context_i, dims=(1, num_repeats))
                context_i = context_i[:, : self.context_max_len]
                context_len_i = self.context_max_len

            context_list.append(context_i)
            context_len_list.append(context_len_i)

        context_lens = self.context_max_len * torch.ones_like(context_lens)
        max_context_len = max(context_len_list)
        context_tokens = stack_tensors(tensors=context_list, max_lens=[max_context_len]).to(audio_tokens.device)

        if self.vector_quantizer_converter_codec is not None:
            context_tokens = self.vector_quantizer_converter_codec.convert_original_to_new(
                audio_tokens=context_tokens, audio_lens=context_lens
            )

        context_tokens_rearrange = rearrange(context_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        context_codes = self.vector_quantizer.decode(indices=context_tokens_rearrange, input_len=context_lens)

        context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context, context_lens

    @typecheck(
        input_types={
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "audio_token_sample": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        },
    )
    def forward(self, audio_tokens, audio_token_lens, text, text_lens):
        if self.vector_quantizer_converter_codec is not None:
            audio_tokens = self.vector_quantizer_converter_codec.convert_original_to_new(
                audio_tokens=audio_tokens, audio_lens=audio_token_lens
            )

        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens).detach()

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, align_hard, align_soft, align_logits = self.aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
        )

        if self.training:
            (
                text_sample,
                text_sample_lens,
                audio_token_sample,
                audio_codes_sample,
                audio_token_sample_lens,
                context_codes,
                context_lens,
            ) = self.sample_context_audio_batch(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
            )
        else:
            (
                text_sample,
                text_sample_lens,
                audio_token_sample,
                audio_codes_sample,
                audio_token_sample_lens,
                context_codes,
                context_lens,
            ) = self.sample_context_audio_start(
                audio_tokens=audio_tokens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                durs=durs,
                random_sample=False,
            )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)
        text_mask = get_mask_from_lengths(text_sample_lens)
        text_enc = self.text_encoder(text=text_sample, text_mask=text_mask)
        context = self.context_encoder(audio_codes=context_codes, audio_lens=context_lens)

        if self.training:
            audio_maskin = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                dist=self.audio_infill_dist,
                infill_min=self.audio_infill_min,
                infill_max=self.audio_infill_max,
            )
            semantic_mask = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                dist=self.semantic_mask_dist,
                infill_min=self.semantic_mask_min,
                infill_max=self.semantic_mask_max,
            )
        else:
            audio_maskin = None
            semantic_mask = None

        audio_token_sample = audio_token_sample[:, self.semantic_codebook_num :, :]
        semantic_codes = audio_codes_sample[:, : self.semantic_codebook_dim, :]
        audio_codes = audio_codes_sample[:, self.semantic_codebook_dim :, :]

        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)
        context = rearrange(context, 'B D T -> B T D')

        semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
        encoder_input = self.semantic_layer(
            semantic_codes=semantic_codes, audio_mask=audio_mask, semantic_mask=semantic_mask
        )
        encoded = self.encoder(
            inputs=encoder_input,
            audio_mask=audio_mask,
            context=context,
            context_mask=context_mask,
            text_enc=text_enc,
            text_mask=text_mask,
        )

        audio_codes = rearrange(audio_codes, 'B C T -> B T C')
        audio_tokens_pred, audio_logits = self.decoder(
            inputs=encoded,
            audio_mask=audio_mask,
            audio_codes=audio_codes,
            audio_maskin=audio_maskin,
        )

        if self.vector_quantizer_converter_acoustic is not None:
            audio_token_sample = self.vector_quantizer_converter_acoustic.convert_original_to_new(
                audio_tokens=audio_token_sample, audio_lens=audio_token_sample_lens
            )

        return (
            audio_token_sample,
            audio_token_sample_lens,
            audio_tokens_pred,
            audio_logits,
            align_hard,
            align_soft,
            align_logits,
        )

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            audio_token_sample,
            audio_token_sample_lens,
            _,
            audio_token_logits,
            align_hard,
            align_soft,
            align_logits,
        ) = self(audio_tokens=audio_tokens, audio_token_lens=audio_token_lens, text=text, text_lens=text_lens)

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_token_sample, mask=audio_mask
        )
        train_audio_token_loss = self.audio_token_loss_scale * audio_token_loss

        ctc_loss = self.forward_sum_loss_fn(attn_logprob=align_logits, in_lens=text_lens, out_lens=audio_token_lens)
        train_ctc_loss = self.aligner_ctc_loss_scale * ctc_loss

        if self.current_epoch < self.bin_loss_start_epoch:
            bin_loss_weight = 0.0
        elif self.current_epoch >= self.bin_loss_warmup_epochs:
            bin_loss_weight = 1.0
        else:
            bin_loss_weight = (self.current_epoch - self.bin_loss_start_epoch) / (
                self.bin_loss_warmup_epochs - self.bin_loss_start_epoch
            )

        bin_loss = self.bin_loss_fn(hard_attention=align_hard, soft_attention=align_soft)
        train_bin_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_loss

        loss = train_audio_token_loss + train_ctc_loss + train_bin_loss
        metrics = {
            "t_audio_token_loss": audio_token_loss,
            "t_ctc_loss": ctc_loss,
            "t_bin_loss": bin_loss,
        }
        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", audio_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            audio_token_sample,
            audio_token_sample_lens,
            audio_tokens_pred,
            audio_token_logits,
            _,
            _,
            _,
        ) = self(audio_tokens=audio_tokens, audio_token_lens=audio_token_lens, text=text, text_lens=text_lens)

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)
        num_audio_tokens = max(1, audio_token_sample_lens.sum() * self.acoustic_codebook_num)

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=audio_token_sample, mask=audio_mask
        )
        audio_token_correct = (audio_token_sample == audio_tokens_pred) * rearrange(audio_mask, 'B T -> B 1 T')
        audio_token_accuracy = audio_token_correct.sum() / num_audio_tokens

        metrics = {
            "val_loss": audio_token_loss,
            "val_audio_token_loss": audio_token_loss,
            "val_audio_token_accuracy": audio_token_accuracy,
        }
        self.log_dict(metrics, on_epoch=True, sync_dist=True)

    def _setup_train_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=True,
        )

        sampler = dataset.get_sampler(dataloader_params.batch_size, world_size=self.trainer.world_size)
        return torch.utils.data.DataLoader(
            dataset, collate_fn=dataset.collate_fn, sampler=sampler, **dataloader_params
        )

    def _setup_test_dataloader(self, dataset_config, dataloader_params):
        dataset = create_text_to_speech_dataset(
            dataset_type=dataset_config.dataset_type,
            text_tokenizer=self.text_tokenizer,
            phoneme_probability=self.inference_phoneme_probability,
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=False,
        )
        return torch.utils.data.DataLoader(dataset, collate_fn=dataset.collate_fn, **dataloader_params)

    def setup_training_data(self, cfg):
        self._train_dl = self._setup_train_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_validation_data(self, cfg):
        self._validation_dl = self._setup_test_dataloader(
            dataset_config=cfg.dataset, dataloader_params=cfg.dataloader_params
        )

    def setup_test_data(self, cfg):
        """Omitted."""
        pass

    def configure_callbacks(self):
        if not self.log_config:
            return []

        data_loader = self._setup_test_dataloader(
            dataset_config=self.log_config.dataset, dataloader_params=self.log_config.dataloader_params
        )
        generators = instantiate(self.log_config.generators)
        log_dir = Path(self.log_config.log_dir) if self.log_config.log_dir else None
        log_callback = LoggingCallback(
            generators=generators,
            data_loader=data_loader,
            log_epochs=self.log_config.log_epochs,
            epoch_frequency=self.log_config.epoch_frequency,
            output_dir=log_dir,
            loggers=self.trainer.loggers,
            log_tensorboard=self.log_config.log_tensorboard,
            log_wandb=self.log_config.log_wandb,
            max_filename_len=self.log_config.max_filename_len,
        )

        return [log_callback]

    @classmethod
    def list_available_models(cls) -> 'List[PretrainedModelInfo]':
        return []

    def _audio_token_infer(
        self,
        inputs,
        audio_lens,
        semantic_tokens,
        frames_per_iter,
        num_audio_iters,
        temperature=None,
        topk=None,
    ):
        if num_audio_iters > 0:
            audio_tokens = self.decoder.infer_diffusion(
                inputs=inputs,
                audio_lens=audio_lens,
                num_iters=num_audio_iters,
                vector_quantizer=self.vector_quantizer_acoustic,
                temperature=temperature,
                topk=topk,
            )
        else:
            audio_tokens = self.decoder.infer(
                inputs=inputs,
                audio_lens=audio_lens,
                frames_per_iter=frames_per_iter,
                vector_quantizer=self.vector_quantizer_acoustic,
                temperature=temperature,
                topk=topk,
            )

        if self.vector_quantizer_converter_acoustic is not None:
            audio_tokens = self.vector_quantizer_converter_acoustic.convert_new_to_original(
                audio_tokens=audio_tokens, audio_lens=audio_lens
            )

        audio_tokens = torch.concat([semantic_tokens, audio_tokens], dim=1)

        if self.vector_quantizer_converter_codec is not None:
            audio_tokens = self.vector_quantizer_converter_codec.convert_new_to_original(
                audio_tokens=audio_tokens, audio_lens=audio_lens
            )

        return audio_tokens

    @typecheck(
        input_types={
            "semantic_tokens": NeuralType(('B', 'C', 'T'), TokenIndex()),
            "semantic_lens": NeuralType(tuple('B'), LengthsType()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "frames_per_iter": NeuralType((), IntType(), optional=True),
            "num_audio_iters": NeuralType((), IntType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T'), TokenIndex()),
        },
    )
    def infer(
        self,
        semantic_tokens,
        semantic_lens,
        context,
        context_lens,
        text,
        text_lens,
        frames_per_iter=1,
        num_audio_iters=0,
        audio_topk=None,
        audio_temperature=None,
    ):
        with torch.no_grad():
            # [batch_size, context_len]
            audio_mask = get_mask_from_lengths(semantic_lens)

            context_mask = get_mask_from_lengths(context_lens)
            text_mask = get_mask_from_lengths(text_lens)

            context = rearrange(context, 'B D T -> B T D')

            text_enc = self.text_encoder(text=text, text_mask=text_mask)

            semantic_tokens_rearrange = rearrange(semantic_tokens, 'B C T -> C B T')
            # [batch_size, code_dim, audio_token_len]
            semantic_codes = self.vector_quantizer.decode(indices=semantic_tokens_rearrange, input_len=semantic_lens)

            semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
            encoder_input = self.semantic_layer(semantic_codes=semantic_codes, audio_mask=audio_mask)
            encoded = self.encoder(
                inputs=encoder_input,
                audio_mask=audio_mask,
                context=context,
                context_mask=context_mask,
                text_enc=text_enc,
                text_mask=text_mask,
            )
            # [B, C_acoustic, T]
            audio_tokens = self._audio_token_infer(
                inputs=encoded,
                audio_lens=semantic_lens,
                semantic_tokens=semantic_tokens,
                frames_per_iter=frames_per_iter,
                num_audio_iters=num_audio_iters,
                temperature=audio_temperature,
                topk=audio_topk,
            )

        return audio_tokens
