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

from nemo.collections.common.parts.utils import mask_sequence_tensor
from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.acoustic_model_loss import AudioTokenLoss, MaskedSoftmax
from nemo.collections.tts.losses.aligner_loss import BinLoss, ForwardSumLoss
from nemo.collections.tts.losses.discrete_speech_loss import SpeakingRateLoss
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths, regulate_len
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import (
    BoolType,
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    LogprobsType,
    MaskType,
    ProbsType,
    TokenDurationType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import model_utils
from nemo.utils.decorators import experimental


@experimental
class DiscreteSpeechModel(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = self._create_tokenizer(cfg.text_tokenizer)
        self.pad_with_space = self.text_tokenizer.pad_with_space
        self.inference_phoneme_probability = cfg.get("inference_phoneme_probability", 1.0)

        super().__init__(cfg=cfg, trainer=trainer)

        # Text tokenizer information
        num_text_embed = len(self.text_tokenizer.tokens)
        self.text_pad_token = self.text_tokenizer.pad
        self.space_token = self.text_tokenizer.space
        self.bos_token = self.text_tokenizer.bos
        self.eos_token = self.text_tokenizer.eos

        # Maximum duration of a single multiphone
        self.max_token_duration = cfg.get("max_token_duration")

        # Context length in terms of number of audio tokens
        self.context_min_len = cfg.get("context_min_len", 50)
        self.context_max_len = cfg.get("context_max_len", 125)
        self.context_len_noise = cfg.get("context_len_noise", 5)

        # Quantizer definitions
        self.semantic_codebook_num = cfg.get("semantic_codebook_num")
        self.semantic_codebook_dim = cfg.get("semantic_codebook_dim")
        self.vector_quantizer = instantiate(cfg.vector_quantizer)

        if "vector_quantizer_codec" in cfg:
            vector_quantizer_codec = instantiate(cfg.vector_quantizer_codec)
            self.vector_quantizer_converter_codec = VectorQuantizerIndexConverter(
                vector_quantizer_original=vector_quantizer_codec,
                vector_quantizer_new=self.vector_quantizer,
            )
        else:
            self.vector_quantizer_converter_codec = None

        self.speaking_rate_quantizer = instantiate(cfg.speaking_rate_quantizer)

        self.text_down_sample_rate = cfg.get("text_down_sample_rate", 1)
        self.space_dur = cfg.get("space_dur", 1)

        # Encoder, decoder definitions
        self.text_encoder = instantiate(
            cfg.text_encoder,
            n_embed=num_text_embed,
            padding_idx=self.text_pad_token,
            down_sample_rate=self.text_down_sample_rate,
            bos_id=self.bos_token,
            eos_id=self.eos_token,
            space_id=self.space_token,
            space_dur=self.space_dur,
        )
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)
        self.duration_encoder = instantiate(cfg.duration_encoder)
        self.duration_decoder = instantiate(cfg.duration_decoder)
        self.speaking_rate_predictor = instantiate(cfg.speaking_rate_predictor)

        # Context encoder definition
        self.context_encoder = instantiate(cfg.context_encoder)

        if "context_aligner_encoder" in cfg:
            self.context_aligner_encoder = instantiate(cfg.context_aligner_encoder)
        else:
            self.context_aligner_encoder = None

        # Aligner definition
        self.phoneme_aligner = instantiate(cfg.aligner, num_text_emb=num_text_embed)

        if self.text_down_sample_rate == 1:
            self.multiphone_aligner = None
        elif self.text_down_sample_rate > 1:
            self.multiphone_aligner = instantiate(
                cfg.aligner,
                num_text_emb=num_text_embed,
                down_sample_rate=self.text_down_sample_rate,
                bos_id=self.bos_token,
                eos_id=self.eos_token,
                space_id=self.space_token,
                space_dur=self.space_dur,
            )
        else:
            raise ValueError(f"text_down_sample_rate must be >= 1")

        # Infilling hyperparameters
        self.semantic_infill_min = cfg.get("semantic_infill_min", 0.05)
        self.semantic_infill_max = cfg.get("semantic_infill_max", 0.5)
        semantic_infill_beta = cfg.get("semantic_infill_beta", 2.0)
        self.semantic_infill_dist = torch.distributions.beta.Beta(
            concentration1=1.0, concentration0=semantic_infill_beta
        )

        self.duration_infill_min = cfg.get("duration_infill_min", 0.05)
        self.duration_infill_max = cfg.get("duration_infill_max", 0.5)
        duration_infill_beta = cfg.get("duration_infill_beta", 2.0)
        self.duration_infill_dist = torch.distributions.beta.Beta(
            concentration1=1.0, concentration0=duration_infill_beta
        )

        self.encoder_mask_min = cfg.get("encoder_mask_min", 0.0)
        self.encoder_mask_max = cfg.get("encoder_mask_max", 0.5)
        encoder_mask_beta = cfg.get("encoder_mask_beta", 2.0)
        self.encoder_mask_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=encoder_mask_beta)

        # Rate at which noise is added to ground truth alignment seen by the decoder
        self.decoder_duration_noise_percent = cfg.get("decoder_duration_noise_percent", 0.0)

        # Reconstruction losses
        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        self.semantic_token_loss_fn = AudioTokenLoss(num_codebooks=self.semantic_codebook_num)

        self.duration_loss_scale = cfg.get("duration_loss_scale", 0.01)
        self.duration_loss_fn = MaskedSoftmax()

        self.speaking_rate_loss_scale = cfg.get("speaking_rate_loss_scale", 1e-3)
        self.speaking_rate_loss_fn = SpeakingRateLoss()

        # Aligner losses
        self.aligner_bin_loss_scale = cfg.get("aligner_bin_loss_scale", 0.01)
        self.aligner_ctc_loss_scale = cfg.get("aligner_ctc_loss_scale", 0.01)
        self.bin_loss_start_epoch = cfg.get("bin_loss_start_epoch", 0)
        self.bin_loss_warmup_epochs = cfg.get("bin_loss_warmup_epochs", 10)

        self.forward_sum_loss_fn = ForwardSumLoss()
        self.bin_loss_fn = BinLoss()

        self.log_config = cfg.get("log_config", None)

    def _create_tokenizer(self, tokenizer_config):
        if "g2p" in tokenizer_config:
            if "phoneme_dict" in tokenizer_config.g2p:
                tokenizer_config.g2p.phoneme_dict = self.register_artifact(
                    'text_tokenizer.g2p.phoneme_dict',
                    tokenizer_config.g2p.phoneme_dict,
                )

            if "heteronyms" in tokenizer_config.g2p:
                tokenizer_config.g2p.heteronyms = self.register_artifact(
                    'text_tokenizer.g2p.heteronyms',
                    tokenizer_config.g2p.heteronyms,
                )

        text_tokenizer = instantiate(tokenizer_config)
        return text_tokenizer

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
        infill_loss_mask = ~infill_mask * len_mask

        return infill_mask, infill_loss_mask

    def _add_decoder_duration_noise(self, durs, text_lens):
        max_text_len = durs.shape[1]
        mask = get_mask_from_lengths(text_lens)

        indices = torch.arange(max_text_len, device=durs.device) + 1
        noise_mask = torch.where(
            rearrange(indices, 'T -> 1 T') == rearrange(text_lens, 'B -> B 1'), torch.zeros_like(mask), mask
        )

        end_indices = torch.cumsum(durs, dim=1)
        end_indices = end_indices * mask

        add_noise = torch.rand(size=durs.shape, device=durs.device) < self.decoder_duration_noise_percent
        noise_mask = noise_mask * add_noise
        shift_backward = torch.rand(size=durs.shape, device=durs.device) <= 0.5
        shift_backward = shift_backward * noise_mask
        shift_forward = ~shift_backward * noise_mask

        min_end_indices = torch.nn.functional.pad(end_indices[:, :-1], pad=[1, 0]) + 1
        end_indices_noise = end_indices + shift_backward.int()
        end_indices_noise = torch.maximum(end_indices_noise, min_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        max_end_indices = torch.nn.functional.pad(end_indices_noise[:, 1:] - 1, pad=[0, 1])
        end_indices_noise = end_indices_noise + shift_forward.int()
        end_indices_noise = torch.minimum(end_indices_noise, max_end_indices)
        end_indices_noise = torch.where(noise_mask, end_indices_noise, end_indices)

        end_indices_noise = end_indices_noise.int()

        durs_noise = end_indices_noise - torch.nn.functional.pad(end_indices_noise[:, :-1], pad=[1, 0])
        durs_noise = durs_noise * mask

        return durs_noise

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
        durs,
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
        target_dur_list = []
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
            dur_i = durs[i, text_start_i:text_end_i]

            audio_tokens_i = audio_tokens[i, :, audio_start_i:audio_end_i]
            audio_codes_i = audio_codes[i, :, audio_start_i:audio_end_i]

            context_list.append(context_i)
            target_text_list.append(text_i)
            target_dur_list.append(dur_i)
            target_audio_token_list.append(audio_tokens_i)
            target_audio_code_list.append(audio_codes_i)

        context_codes = stack_tensors(tensors=context_list, max_lens=[self.context_max_len]).to(audio_codes.device)
        context_output_lens = self.context_max_len * torch.ones_like(context_lens)

        max_text_len = target_text_lens.max().item()
        target_text = stack_tensors(tensors=target_text_list, max_lens=[max_text_len]).to(audio_codes.device)
        target_durs = stack_tensors(tensors=target_dur_list, max_lens=[max_text_len]).to(audio_codes.device)

        max_audio_len = target_audio_lens.max().item()
        target_audio_tokens = stack_tensors(tensors=target_audio_token_list, max_lens=[max_audio_len]).to(
            audio_codes.device
        )
        target_audio_codes = stack_tensors(tensors=target_audio_code_list, max_lens=[max_audio_len]).to(
            audio_codes.device
        )

        return context_codes, context_output_lens, target_text, target_durs, target_audio_tokens, target_audio_codes

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
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = (
            self._slice_context_information(
                text=text,
                durs=durs,
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
            target_durs,
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
        context_codes, context_lens, target_text, target_durs, target_audio_tokens, target_audio_codes = (
            self._slice_context_information(
                text=text,
                durs=durs,
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
            target_durs,
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
            dur_sample1,
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
            dur_sample2,
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
        dur_sample = self._concat_tensors(dur_sample1, dur_sample2)
        text_sample_lens = torch.cat([text_sample_lens1, text_sample_lens2], dim=0)
        audio_token_sample = self._concat_tensors(audio_token_sample1, audio_token_sample2)
        audio_codes_sample = self._concat_tensors(audio_codes_sample1, audio_codes_sample2)
        audio_token_sample_lens = torch.cat([audio_token_sample_lens1, audio_token_sample_lens2], dim=0)
        context_codes = self._concat_tensors(context_codes1, context_codes2)
        context_lens = torch.cat([context_lens1, context_lens2], dim=0)

        return (
            text_sample,
            dur_sample,
            text_sample_lens,
            audio_token_sample,
            audio_codes_sample,
            audio_token_sample_lens,
            context_codes,
            context_lens,
        )

    def get_context(self, audio_tokens, audio_lens, text, text_lens, max_len=None):
        if self.vector_quantizer_converter_codec is not None:
            audio_tokens = self.vector_quantizer_converter_codec.convert_original_to_new(
                audio_tokens=audio_tokens, audio_lens=audio_lens
            )

        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_lens)

        if self.context_aligner_encoder is not None:
            context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_lens)
        else:
            context_aligner_emb = None

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, _, _, _ = self.phoneme_aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
            context_emb=context_aligner_emb,
        )

        _, _, _, _, _, _, context_codes, context_lens = self.sample_context_audio_start(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            random_sample=False,
            max_len=max_len,
        )
        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    def get_context_audio(self, audio_tokens, audio_lens):
        batch_size = audio_tokens.shape[0]
        context_lens = torch.clamp_max(audio_lens, max=self.context_max_len)
        context_token_list = []
        for i in range(batch_size):
            context_len_i = context_lens[i]
            context_tokens_i = audio_tokens[i, :, :context_len_i]
            context_token_list.append(context_tokens_i)

        max_context_len = max(context_lens)
        context_tokens = stack_tensors(tensors=context_token_list, max_lens=[max_context_len]).to(audio_tokens.device)

        context_tokens_rearrange = rearrange(context_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        context_codes = self.vector_quantizer.decode(indices=context_tokens_rearrange, input_len=context_lens)

        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )
        return context_emb, context, context_lens

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "sample_context": NeuralType((), BoolType(), optional=True),
        },
        output_types={
            "semantic_token_sample": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_sample_lens": NeuralType(tuple('B'), LengthsType()),
            "semantic_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_tokens_pred_post": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "semantic_logits_post": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
            "semantic_loss_mask": NeuralType(('B', 'T_text'), MaskType()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_lens": NeuralType(tuple('B'), LengthsType()),
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "dur_indices_pred_post": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits_post": NeuralType(('B', 'D', 'T_text'), LogitsType()),
            "dur_loss_mask": NeuralType(('B', 'T_text'), MaskType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_indices_pred": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_logits": NeuralType(('B', 'C'), LogitsType()),
            "align_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "align_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
            "balign_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "balign_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "balign_logits": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        },
    )
    def forward(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
        sample_context=False,
    ):
        if self.vector_quantizer_converter_codec is not None:
            audio_tokens = self.vector_quantizer_converter_codec.convert_original_to_new(
                audio_tokens=audio_tokens, audio_lens=audio_token_lens
            )

        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens).detach()

        if self.context_aligner_encoder is not None:
            context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_token_lens)
        else:
            context_aligner_emb = None

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, align_hard, align_soft, align_logits = self.phoneme_aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            context_emb=context_aligner_emb,
        )

        if sample_context:
            (
                text_sample,
                dur_sample,
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
                dur_sample,
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

        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )

        speaking_rate, speaking_rate_indices = self.get_speaking_rate(text_lens=text_lens, durs=durs)
        speaking_rate = speaking_rate.detach()

        if self.multiphone_aligner is not None:
            dur_sample, dur_lens, balign_hard, balign_soft, balign_logits = self.multiphone_aligner(
                text=text_sample,
                text_lens=text_sample_lens,
                audio_codes=audio_codes_sample,
                audio_lens=audio_token_sample_lens,
                context_emb=context_aligner_emb,
            )
        else:
            dur_lens = text_sample_lens
            balign_hard = None
            balign_soft = None
            balign_logits = None

        dur_indices = self.duration_to_index(durs=dur_sample, lengths=dur_lens)

        if sample_context:
            semantic_maskin, semantic_loss_mask = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                dist=self.semantic_infill_dist,
                infill_min=self.semantic_infill_min,
                infill_max=self.semantic_infill_max,
            )
            duration_maskin, duration_loss_mask = self.create_infill_mask(
                input_lens=dur_lens,
                dist=self.duration_infill_dist,
                infill_min=self.duration_infill_min,
                infill_max=self.duration_infill_max,
            )
            dur_noise = self._add_decoder_duration_noise(durs=dur_sample, text_lens=dur_lens)
            encoder_mask, _ = self.create_infill_mask(
                input_lens=audio_token_sample_lens,
                dist=self.encoder_mask_dist,
                infill_min=self.encoder_mask_min,
                infill_max=self.encoder_mask_max,
            )
        else:
            audio_mask = get_mask_from_lengths(audio_token_sample_lens)
            semantic_maskin = torch.zeros(
                [audio_codes_sample.shape[0], audio_codes_sample.shape[2]],
                dtype=torch.bool,
                device=audio_tokens.device,
            )
            # Unmask every 10th element
            semantic_maskin[:, ::10] = True
            semantic_maskin = semantic_maskin * audio_mask
            semantic_loss_mask = ~semantic_maskin * audio_mask

            dur_mask = get_mask_from_lengths(dur_lens)
            duration_maskin = torch.zeros_like(dur_sample, dtype=torch.bool)
            # Unmask every 5th element
            duration_maskin[:, ::5] = True
            duration_maskin = duration_maskin * dur_mask
            duration_loss_mask = ~duration_maskin * dur_mask

            dur_noise = dur_sample

            encoder_mask = None

        semantic_token_sample = audio_token_sample[:, : self.semantic_codebook_num, :]
        semantic_codes = audio_codes_sample[:, : self.semantic_codebook_dim, :]

        (
            semantic_tokens_pred,
            semantic_token_logits,
            semantic_tokens_pred_post,
            semantic_token_logits_post,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_post,
            dur_logits_post,
            speaking_rate_indices_pred,
            speaking_rate_logits,
        ) = self.forward_internal(
            text=text_sample,
            text_lens=text_sample_lens,
            context_emb=context_emb,
            context=context,
            context_lens=context_lens,
            speaking_rate=speaking_rate,
            semantic_codes=semantic_codes,
            audio_lens=audio_token_sample_lens,
            semantic_maskin=semantic_maskin,
            durs=dur_noise,
            dur_indices=dur_indices,
            duration_maskin=duration_maskin,
            encoder_mask=encoder_mask,
        )

        return (
            semantic_token_sample,
            audio_token_sample_lens,
            semantic_tokens_pred,
            semantic_token_logits,
            semantic_tokens_pred_post,
            semantic_token_logits_post,
            semantic_loss_mask,
            dur_indices,
            dur_lens,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_post,
            dur_logits_post,
            duration_loss_mask,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            align_hard,
            align_soft,
            align_logits,
            balign_hard,
            balign_soft,
            balign_logits,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "word_stride": NeuralType((), IntType()),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
        },
        output_types={
            "semantic_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "dur_lens": NeuralType(tuple('B'), LengthsType()),
            "align_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "balign_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
        },
    )
    def infer_gta(
        self,
        text,
        text_lens,
        audio_tokens,
        audio_token_lens,
        word_stride=2,
        audio_temperature=None,
        audio_topk=None,
    ):
        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T').detach()
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens)

        if self.context_aligner_encoder is not None:
            context_aligner_emb = self.context_aligner_encoder(audio_codes=audio_codes, audio_lens=audio_token_lens)
        else:
            context_aligner_emb = None

        # [batch_size, text_len], [batch_size, audio_token_len, text_len], ...
        durs, _, align_hard, align_soft, align_logits = self.phoneme_aligner(
            text=text,
            text_lens=text_lens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            context_emb=context_aligner_emb,
        )

        _, _, _, _, _, _, context_codes, context_lens = self.sample_context_audio_start(
            audio_tokens=audio_tokens,
            audio_codes=audio_codes,
            audio_lens=audio_token_lens,
            text=text,
            text_lens=text_lens,
            durs=durs,
            random_sample=False,
        )
        context_emb, context = self.context_encoder(
            audio_codes=context_codes,
            audio_lens=context_lens,
        )

        if self.multiphone_aligner is not None:
            durs, _, _, balign_soft, _ = self.multiphone_aligner(
                text=text,
                text_lens=text_lens,
                audio_codes=audio_codes,
                audio_lens=audio_token_lens,
                context_emb=context_aligner_emb,
            )
        else:
            balign_soft = None

        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)
        context = rearrange(context, 'B D T -> B T D')
        # [batch_size, text_len, hidden_dim]
        text_enc, dur_lens, text_durs = self.text_encoder(text=text, text_lens=text_lens, context_emb=context_emb)

        text_enc_repeated, semantic_lens = regulate_len(durs, text_enc, pace=1.0)
        semantic_mask = get_mask_from_lengths(semantic_lens)

        semantic_enc = self.encoder(
            inputs=text_enc_repeated, audio_mask=semantic_mask, context=context, context_mask=context_mask
        )
        # [B, C_semantic, T]
        semantic_tokens = self._semantic_token_infer(
            inputs=semantic_enc,
            audio_lens=semantic_lens,
            text=text,
            text_durs=text_durs,
            durs=durs,
            word_stride=word_stride,
            temperature=audio_temperature,
            topk=audio_topk,
        )

        return semantic_tokens, context, context_lens, dur_lens, align_soft, balign_soft

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            semantic_token_sample,
            audio_token_sample_lens,
            _,
            semantic_token_logits,
            _,
            semantic_token_logits_post,
            semantic_loss_mask,
            dur_indices,
            dur_lens,
            _,
            dur_logits,
            _,
            dur_logits_post,
            dur_loss_mask,
            speaking_rate_indices,
            _,
            speaking_rate_logits,
            align_hard,
            align_soft,
            align_logits,
            balign_hard,
            balign_soft,
            balign_logits,
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
            sample_context=True,
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits, target_tokens=semantic_token_sample, mask=audio_mask
        )
        train_semantic_token_loss = self.audio_token_loss_scale * semantic_token_loss

        semantic_token_post_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits_post, target_tokens=semantic_token_sample, mask=semantic_loss_mask
        )
        train_semantic_token_post_loss = self.audio_token_loss_scale * semantic_token_post_loss

        dur_mask = get_mask_from_lengths(dur_lens)

        duration_loss = self.duration_loss_fn(logits=dur_logits, target_index=dur_indices.detach(), mask=dur_mask)
        train_dur_loss = self.duration_loss_scale * duration_loss

        duration_post_loss = self.duration_loss_fn(
            logits=dur_logits_post, target_index=dur_indices.detach(), mask=dur_loss_mask
        )
        train_dur_post_loss = self.duration_loss_scale * duration_post_loss

        speaking_rate_loss = self.speaking_rate_loss_fn(
            logits=speaking_rate_logits, target_index=speaking_rate_indices.detach()
        )
        train_speaking_rate_loss = self.speaking_rate_loss_scale * speaking_rate_loss

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

        loss = (
            train_semantic_token_loss
            + train_semantic_token_post_loss
            + train_dur_loss
            + train_dur_post_loss
            + train_speaking_rate_loss
            + train_ctc_loss
            + train_bin_loss
        )

        metrics = {
            "t_semantic_token_loss": semantic_token_loss,
            "t_semantic_token_post_loss": semantic_token_post_loss,
            "t_duration_loss": duration_loss,
            "t_duration_post_loss": duration_post_loss,
            "t_speaking_rate_loss": speaking_rate_loss,
            "t_ctc_loss": ctc_loss,
            "t_bin_loss": bin_loss,
        }

        if balign_logits is not None:
            ctc_multiphone_loss = self.forward_sum_loss_fn(
                attn_logprob=balign_logits, in_lens=dur_lens, out_lens=audio_token_sample_lens
            )
            train_ctc_multiphone_loss = self.aligner_ctc_loss_scale * ctc_multiphone_loss

            bin_multiphone_loss = self.bin_loss_fn(hard_attention=balign_hard, soft_attention=balign_soft)
            train_bin_multiphone_loss = bin_loss_weight * self.aligner_bin_loss_scale * bin_multiphone_loss

            loss += train_ctc_multiphone_loss + train_bin_multiphone_loss

            metrics['t_ctc_multiphone_loss'] = ctc_multiphone_loss
            metrics['t_bin_multiphone_loss'] = bin_multiphone_loss

        self.log_dict(metrics, on_step=True, sync_dist=True)
        self.log("t_loss", semantic_token_loss, prog_bar=True, logger=False, sync_dist=True)

        return loss

    def validation_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            semantic_token_sample,
            audio_token_sample_lens,
            semantic_tokens_pred,
            semantic_token_logits,
            semantic_tokens_pred_post,
            semantic_token_logits_post,
            semantic_loss_mask,
            dur_indices,
            dur_lens,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_post,
            dur_logits_post,
            dur_loss_mask,
            speaking_rate_indices,
            speaking_rate_indices_pred,
            speaking_rate_logits,
            _,
            _,
            _,
            _,
            _,
            _,
        ) = self(
            text=text,
            text_lens=text_lens,
            audio_tokens=audio_tokens,
            audio_token_lens=audio_token_lens,
        )

        audio_mask = get_mask_from_lengths(audio_token_sample_lens)

        semantic_token_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits, target_tokens=semantic_token_sample, mask=audio_mask
        )
        semantic_token_correct = (semantic_token_sample == semantic_tokens_pred) * rearrange(
            audio_mask, 'B T -> B 1 T'
        )
        semantic_token_accuracy = (
            semantic_token_correct.sum() / audio_token_sample_lens.sum() / self.semantic_codebook_num
        )

        semantic_token_post_loss = self.semantic_token_loss_fn(
            logits=semantic_token_logits_post, target_tokens=semantic_token_sample, mask=semantic_loss_mask
        )
        semantic_token_correct_post = (semantic_token_sample == semantic_tokens_pred_post) * rearrange(
            semantic_loss_mask, 'B T -> B 1 T'
        )
        semantic_token_post_accuracy = (
            semantic_token_correct_post.sum() / semantic_loss_mask.sum() / self.semantic_codebook_num
        )

        dur_mask = get_mask_from_lengths(dur_lens)

        duration_loss = self.duration_loss_fn(logits=dur_logits, target_index=dur_indices, mask=dur_mask)
        dur_token_correct = (dur_indices == dur_indices_pred) * dur_mask
        dur_token_accuracy = dur_token_correct.sum() / dur_lens.sum()

        duration_post_loss = self.duration_loss_fn(
            logits=dur_logits_post,
            target_index=dur_indices,
            mask=dur_loss_mask,
        )
        dur_token_correct_post = (dur_indices == dur_indices_pred_post) * dur_loss_mask
        dur_token_post_accuracy = dur_token_correct_post.sum() / dur_loss_mask.sum()

        speaking_rate_loss = self.speaking_rate_loss_fn(
            logits=speaking_rate_logits, target_index=speaking_rate_indices
        )
        speaking_rate_correct = speaking_rate_indices == speaking_rate_indices_pred
        speaking_rate_accuracy = speaking_rate_correct.float().mean()

        metrics = {
            "val_loss": semantic_token_loss,
            "val_semantic_token_loss": semantic_token_loss,
            "val_semantic_token_accuracy": semantic_token_accuracy,
            "val_semantic_token_post_loss": semantic_token_post_loss,
            "val_semantic_token_post_accuracy": semantic_token_post_accuracy,
            "val_duration_loss": duration_loss,
            "val_dur_token_accuracy": dur_token_accuracy,
            "val_duration_post_loss": duration_post_loss,
            "val_dur_token_post_accuracy": dur_token_post_accuracy,
            "val_speaking_rate_loss": speaking_rate_loss,
            "val_speaking_rate_accuracy": speaking_rate_accuracy,
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
            global_rank=self.trainer.global_rank,
            world_size=self.trainer.world_size,
            dataset_args=dataset_config.dataset_args,
            is_train=False,
            phoneme_probability=self.inference_phoneme_probability,
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

    def _find_word_starts(self, text, text_durs, durs, audio_lens, word_stride):
        batch_size = text.shape[0]
        max_dur_len = durs.shape[1]
        # [B, T_dur]
        cum_ends = torch.cumsum(durs, dim=1).long()
        # [B, T_text, 1]
        text_repeated, _ = regulate_len(durations=text_durs, enc_out=text.unsqueeze(2).float())
        # [B, T_text]
        text_repeated = text_repeated.squeeze(2).long()
        pad_len = text_repeated.shape[1] % self.text_down_sample_rate
        text_repeated = torch.nn.functional.pad(text_repeated, (0, pad_len))
        # [B, T_dur, down_sample_rate]
        text_repeated = text_repeated.reshape([batch_size, max_dur_len, self.text_down_sample_rate])

        # [B, T_dur]
        is_space = torch.logical_or(text_repeated == self.space_token, text_repeated == self.bos_token).all(dim=2)
        word_start_indices = torch.where(is_space, cum_ends + 1, torch.zeros_like(cum_ends))

        word_num = torch.cumsum(is_space, dim=1).long() - 1
        word_mask = word_num % word_stride == 0
        word_start_indices = torch.where(word_mask, word_start_indices, torch.zeros_like(word_mask))

        # [B, 1, T_text]
        word_start_indices_3d = word_start_indices.unsqueeze(1)

        # [T_audio]
        max_audio_len = audio_lens.max()
        audio_indices = torch.arange(1, max_audio_len + 1, device=text.device)
        # [B, T_audio, 1]
        audio_indices_3d = audio_indices.unsqueeze(0).tile([batch_size, 1]).unsqueeze(2)

        # [B, T_audio]
        word_starts = (audio_indices_3d == word_start_indices_3d).any(dim=2)

        num_iters = 0
        for i in range(batch_size):
            indices = torch.nonzero(word_start_indices[i])
            for j in range(1, indices.shape[0]):
                word_dur = word_start_indices[i, indices[j]] - word_start_indices[i, indices[j - 1]]
                if word_dur > num_iters:
                    num_iters = word_dur

        return word_starts, num_iters

    def _find_word_starts_text(self, text, word_stride):
        batch_size = text.shape[0]
        max_text_len = text.shape[1]
        # [B]
        text_indices = torch.arange(1, max_text_len + 1, device=text.device)
        # [B, T]
        text_indices = text_indices.unsqueeze(0).tile([batch_size, 1])

        # [B, T]
        is_space = torch.logical_or(text == self.space_token, text == self.bos_token)
        word_start_indices = torch.where(is_space, text_indices, torch.zeros_like(text_indices))

        word_num = torch.cumsum(is_space, dim=1).long() - 1
        word_mask = word_num % word_stride == 0
        word_start_indices = torch.where(word_mask, word_start_indices, torch.zeros_like(word_mask))

        word_starts = text_indices == word_start_indices

        num_iters = 0
        for i in range(batch_size):
            indices = torch.nonzero(word_start_indices[i])
            for j in range(1, indices.shape[0]):
                word_dur = word_start_indices[i, indices[j]] - word_start_indices[i, indices[j - 1]]
                if word_dur > num_iters:
                    num_iters = word_dur

        return word_starts, num_iters

    def _semantic_token_infer(
        self,
        inputs,
        audio_lens,
        text,
        text_durs,
        durs,
        word_stride,
        temperature=None,
        topk=None,
    ):
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)

        # [B, T]
        maskin, num_iters = self._find_word_starts(
            text=text, text_durs=text_durs, durs=durs, audio_lens=audio_lens, word_stride=word_stride
        )

        # [B, T]
        audio_maskin = torch.zeros_like(audio_mask, dtype=torch.bool)
        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.decoder.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1], self.decoder.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self.decoder.forward_parallel(
                    inputs=inputs,
                    audio_mask=audio_mask,
                    temperature=temperature,
                    topk=topk,
                )
            else:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self.decoder(
                    inputs=inputs,
                    audio_mask=audio_mask,
                    audio_codes=audio_codes,
                    audio_maskin=audio_maskin,
                    temperature=temperature,
                    topk=topk,
                )
            audio_tokens_i = rearrange(audio_tokens_i, 'B C T -> B T C')
            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            audio_codes_i = self.vector_quantizer.decode(indices=audio_tokens_rearrange_i, input_len=audio_lens)
            audio_codes_i = rearrange(audio_codes_i, 'B D T -> B T D')

            # [B, T]
            maskin_i = torch.where(audio_mask, maskin, False)
            maskin_i = torch.where(audio_maskin, False, maskin_i)
            maskin_3d_i = rearrange(maskin_i, 'B T -> B T 1')

            audio_tokens = torch.where(maskin_3d_i, audio_tokens_i, audio_tokens)
            audio_codes = torch.where(maskin_3d_i, audio_codes_i, audio_codes)

            audio_maskin = torch.logical_or(audio_maskin, maskin_i)
            maskin = torch.logical_or(maskin, torch.nn.functional.pad(maskin[:, :-1], pad=(1, 0), value=True))

        audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
        audio_tokens = torch.where(audio_maskin_3d, audio_tokens, audio_tokens_i)

        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens

    def _duration_infer(
        self,
        inputs,
        text,
        text_lens,
        word_stride,
        temperature=None,
        topk=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        # [B, T]
        text_mask = get_mask_from_lengths(text_lens)

        # [B, T]
        maskin, num_iters = self._find_word_starts_text(text=text, word_stride=word_stride)

        duration_maskin = torch.zeros_like(text_mask, dtype=torch.bool)
        dur_indices = torch.zeros_like(text_mask, dtype=torch.int)

        if silence_pad_start:
            for i in range(dur_indices.shape[0]):
                dur_indices[i, 0] = silence_pad_start - 1
                duration_maskin[i, 0] = True

        if silence_pad_end:
            for i in range(dur_indices.shape[0]):
                last_i = text_lens[i] - 1
                dur_indices[i, last_i] = silence_pad_end - 1
                duration_maskin[i, last_i] = True

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                dur_indices_i, dur_logits = self.duration_decoder.forward_parallel(
                    inputs=inputs, text_mask=text_mask, temperature=temperature, topk=topk
                )
            else:
                dur_indices_i, dur_logits = self.duration_decoder(
                    inputs=inputs,
                    dur_indices=dur_indices,
                    text_mask=text_mask,
                    duration_maskin=duration_maskin,
                    temperature=temperature,
                    topk=topk,
                )

            maskin_i = torch.where(text_mask, maskin, False)
            maskin_i = torch.where(duration_maskin, False, maskin_i)

            dur_indices = torch.where(maskin_i, dur_indices_i, dur_indices)
            duration_maskin = torch.logical_or(duration_maskin, maskin_i)
            maskin = torch.logical_or(maskin, torch.nn.functional.pad(maskin[:, :-1], pad=(1, 0), value=True))

        dur_indices = torch.where(duration_maskin, dur_indices, dur_indices_i)
        # [B, T]
        durs = self.index_to_duration(dur_indices=dur_indices, mask=text_mask)

        return durs, dur_indices

    def _duration_infer_iters(
        self,
        inputs,
        text_lens,
        num_iters,
        temperature=None,
        topk=None,
        silence_pad_start=None,
        silence_pad_end=None,
    ):
        # [B, T]
        text_mask = get_mask_from_lengths(text_lens)

        num_tokens = inputs.shape[1]
        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        duration_maskin = torch.zeros_like(text_mask, dtype=torch.bool)
        dur_indices = torch.zeros_like(text_mask, dtype=torch.int)

        if silence_pad_start:
            for i in range(dur_indices.shape[0]):
                dur_indices[i, 0] = silence_pad_start - 1
                duration_maskin[i, 0] = True

        if silence_pad_end:
            for i in range(dur_indices.shape[0]):
                last_i = text_lens[i] - 1
                dur_indices[i, last_i] = silence_pad_end - 1
                duration_maskin[i, last_i] = True

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                dur_indices_i, dur_logits = self.duration_decoder.forward_parallel(
                    inputs=inputs, text_mask=text_mask, temperature=temperature, topk=topk
                )
            else:
                dur_indices_i, dur_logits = self.duration_decoder(
                    inputs=inputs,
                    dur_indices=dur_indices,
                    text_mask=text_mask,
                    duration_maskin=duration_maskin,
                    temperature=temperature,
                    topk=topk,
                )

            top_i = torch.clamp_max(index_shift + 1, max=num_tokens - 1)

            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)
            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(text_mask, maskin_i, False)
            maskin_i = torch.where(duration_maskin, False, maskin_i)

            dur_indices = torch.where(maskin_i, dur_indices_i, dur_indices)
            duration_maskin = torch.logical_or(duration_maskin, maskin_i)

        dur_indices = torch.where(duration_maskin, dur_indices, dur_indices_i)
        # [B, T]
        durs = self.index_to_duration(dur_indices=dur_indices, mask=text_mask)

        return durs, dur_indices

    @typecheck(
        input_types={
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
            "lengths": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={"dur_indices": NeuralType(('B', 'T_text'), TokenIndex())},
    )
    def duration_to_index(self, durs, lengths):
        durs = torch.clamp(durs.float(), min=1, max=self.max_token_duration)
        dur_indices = durs - 1
        dur_indices = mask_sequence_tensor(tensor=dur_indices, lengths=lengths)
        dur_indices = dur_indices.int()
        return dur_indices

    def index_to_duration(self, dur_indices, mask):
        # [B, T]
        durs = dur_indices + 1
        durs = durs * mask
        return durs

    @typecheck(
        input_types={
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "durs": NeuralType(('B', 'T_text'), TokenDurationType()),
        },
        output_types={
            "speaking_rate": NeuralType(tuple('B'), FloatType()),
            "speaking_rate_indices": NeuralType(tuple('B'), TokenIndex()),
        },
    )
    def get_speaking_rate(self, text_lens, durs):
        sr_text_len = torch.clamp_min(text_lens - 2, min=1)

        text_mask = get_mask_from_lengths(text_lens)
        max_text_len = text_mask.shape[1]
        indices = torch.arange(max_text_len, device=durs.device) + 1
        dur_mask = torch.where(
            rearrange(indices, 'T -> 1 T') == rearrange(text_lens, 'B -> B 1'), torch.zeros_like(text_mask), text_mask
        )
        sr_durs = torch.clamp(durs.float(), min=1, max=self.max_token_duration)
        sr_durs = sr_durs * dur_mask
        sr_durs = sr_durs[:, 1:]
        sr_audio_lens = sr_durs.sum(dim=1)

        fps = (-1.0) * sr_audio_lens / sr_text_len.float()
        speaking_rate, speaking_rate_indices = self.speaking_rate_quantizer(inputs=fps)

        return speaking_rate, speaking_rate_indices

    def forward_internal(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        speaking_rate,
        semantic_codes,
        audio_lens,
        semantic_maskin,
        durs,
        dur_indices,
        duration_maskin,
        encoder_mask,
    ):
        audio_mask = get_mask_from_lengths(audio_lens)
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')
        speaking_rate_indices_pred, speaking_rate_logits = self.speaking_rate_predictor(context_emb=context_emb)
        # [batch_size, text_len, hidden_dim]
        text_enc, dur_lens, _ = self.text_encoder(text=text, text_lens=text_lens, context_emb=context_emb)
        dur_mask = get_mask_from_lengths(dur_lens)
        dur_enc = self.duration_encoder(
            text_enc=text_enc,
            text_mask=dur_mask,
            speaking_rate=speaking_rate,
            context=context,
            context_mask=context_mask,
        )
        dur_indices_pred, dur_logits = self.duration_decoder.forward_parallel(inputs=dur_enc, text_mask=dur_mask)

        dur_indices_pred_post, dur_logits_post = self.duration_decoder(
            inputs=dur_enc,
            dur_indices=dur_indices,
            text_mask=dur_mask,
            duration_maskin=duration_maskin,
        )

        text_enc_repeated, _ = regulate_len(durs, text_enc, pace=1.0)

        semantic_enc = self.encoder(
            inputs=text_enc_repeated,
            audio_mask=audio_mask,
            encoder_mask=encoder_mask,
            context=context,
            context_mask=context_mask,
        )
        semantic_tokens_pred, semantic_logits = self.decoder.forward_parallel(
            inputs=semantic_enc, audio_mask=audio_mask
        )

        semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
        semantic_tokens_pred_post, semantic_logits_post = self.decoder(
            inputs=semantic_enc,
            audio_mask=audio_mask,
            audio_codes=semantic_codes,
            audio_maskin=semantic_maskin,
        )

        return (
            semantic_tokens_pred,
            semantic_logits,
            semantic_tokens_pred_post,
            semantic_logits_post,
            dur_indices_pred,
            dur_logits,
            dur_indices_pred_post,
            dur_logits_post,
            speaking_rate_indices_pred,
            speaking_rate_logits,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T_context'), EncodedRepresentation()),
            "context_lens": NeuralType(tuple('B'), LengthsType()),
            "word_stride": NeuralType((), IntType(), optional=True),
            "audio_topk": NeuralType((), IntType(), optional=True),
            "audio_temperature": NeuralType((), FloatType(), optional=True),
            "num_duration_iters": NeuralType((), IntType(), optional=True),
            "duration_topk": NeuralType((), IntType(), optional=True),
            "duration_temperature": NeuralType((), FloatType(), optional=True),
            "speaking_rate": NeuralType(tuple('B'), FloatType(), optional=True),
            "silence_pad_start": NeuralType((), IntType(), optional=True),
            "silence_pad_end": NeuralType((), IntType(), optional=True),
            "min_speaking_rate": NeuralType((), IntType(), optional=True),
        },
        output_types={
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_token'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
        },
    )
    def infer(
        self,
        text,
        text_lens,
        context_emb,
        context,
        context_lens,
        word_stride=2,
        audio_topk=None,
        audio_temperature=None,
        num_duration_iters=None,
        duration_topk=None,
        duration_temperature=None,
        speaking_rate=None,
        silence_pad_start=None,
        silence_pad_end=None,
        min_speaking_rate=-0.5,
        max_speaking_rate=0.5,
    ):
        # [batch_size, context_len]
        context_mask = get_mask_from_lengths(context_lens)

        context = rearrange(context, 'B D T -> B T D')

        if speaking_rate is None:
            speaking_rate_indices, _ = self.speaking_rate_predictor(context_emb=context_emb)
            speaking_rate = self.speaking_rate_quantizer.get_codes(indices=speaking_rate_indices)
            speaking_rate = torch.clamp(speaking_rate, min=min_speaking_rate, max=max_speaking_rate)

        # [batch_size, text_len, hidden_dim]
        text_enc, dur_lens, text_durs = self.text_encoder(text=text, text_lens=text_lens, context_emb=context_emb)
        # [batch_size, text_len]
        dur_mask = get_mask_from_lengths(dur_lens)
        dur_enc = self.duration_encoder(
            text_enc=text_enc,
            text_mask=dur_mask,
            speaking_rate=speaking_rate,
            context=context,
            context_mask=context_mask,
        )

        if num_duration_iters:
            durs, _ = self._duration_infer_iters(
                inputs=dur_enc,
                text_lens=dur_lens,
                num_iters=num_duration_iters,
                temperature=duration_temperature,
                topk=duration_topk,
                silence_pad_start=silence_pad_start,
                silence_pad_end=silence_pad_end,
            )
        else:
            durs, _ = self._duration_infer(
                inputs=dur_enc,
                text=text,
                text_lens=dur_lens,
                word_stride=word_stride,
                temperature=duration_temperature,
                topk=duration_topk,
                silence_pad_start=silence_pad_start,
                silence_pad_end=silence_pad_end,
            )

        text_enc_repeated, semantic_lens = regulate_len(durs, text_enc, pace=1.0)
        semantic_mask = get_mask_from_lengths(semantic_lens)

        semantic_enc = self.encoder(
            inputs=text_enc_repeated, audio_mask=semantic_mask, context=context, context_mask=context_mask
        )
        # [B, C_semantic, T]
        semantic_tokens = self._semantic_token_infer(
            inputs=semantic_enc,
            audio_lens=semantic_lens,
            text=text,
            text_durs=text_durs,
            durs=durs,
            word_stride=word_stride,
            temperature=audio_temperature,
            topk=audio_topk,
        )

        return semantic_tokens, semantic_lens
