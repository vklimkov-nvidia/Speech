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

from nemo.collections.tts.data.text_to_speech_dataset import create_text_to_speech_dataset
from nemo.collections.tts.losses.acoustic_model_loss import AudioTokenLoss
from nemo.collections.tts.modules.audio_codec_modules import VectorQuantizerIndexConverter
from nemo.collections.tts.parts.utils.callbacks import LoggingCallback
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core import ModelPT
from nemo.core.classes.common import PretrainedModelInfo, typecheck
from nemo.core.neural_types.elements import FloatType, IntType, LengthsType, LogitsType, MaskType, TokenIndex
from nemo.core.neural_types.neural_type import NeuralType
from nemo.utils import model_utils
from nemo.utils.decorators import experimental


@experimental
class AcousticModelAutoregressiveWithText(ModelPT):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        # Convert to Hydra 1.0 compatible DictConfig
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)

        self.text_tokenizer = instantiate(cfg.text_tokenizer)
        self.inference_phoneme_probability = cfg.get("inference_phoneme_probability", 1.0)

        super().__init__(cfg=cfg, trainer=trainer)

        # Text tokenizer information
        num_text_embed = len(self.text_tokenizer.tokens)
        self.text_pad_token = self.text_tokenizer.pad

        # Quantizer definitions
        self.semantic_codebook_num = cfg.get("semantic_codebook_num")
        self.semantic_codebook_dim = cfg.get("semantic_codebook_dim")
        self.acoustic_codebook_num = cfg.get("acoustic_codebook_num")
        self.acoustic_codebook_dim = cfg.get("acoustic_codebook_dim")

        self.text_encoder = instantiate(cfg.text_encoder, n_embed=num_text_embed, padding_idx=self.text_pad_token)
        self.encoder = instantiate(cfg.encoder)
        self.decoder = instantiate(cfg.decoder)
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

        # Audio infill hyperparemters
        self.audio_infill_min = cfg.get("audio_infill_min", 0.1)
        self.audio_infill_max = cfg.get("audio_infill_max", 1.0)
        audio_infill_beta = cfg.get("audio_infill_beta", 2.0)
        self.audio_infill_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=audio_infill_beta)

        # Semantic masking hyperparemters
        self.semantic_mask_min = cfg.get("semantic_mask_min", 0.0)
        self.semantic_mask_max = cfg.get("semantic_mask_max", 1.0)
        semantic_mask_beta = cfg.get("semantic_mask_beta", 2.0)
        self.semantic_mask_dist = torch.distributions.beta.Beta(concentration1=1.0, concentration0=semantic_mask_beta)

        # Reconstruction losses
        self.audio_token_loss_scale = cfg.get("audio_token_loss_scale", 1.0)
        self.audio_token_loss_fn = AudioTokenLoss(num_codebooks=self.acoustic_codebook_num)

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

    @typecheck(
        input_types={
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_token_lens": NeuralType(tuple('B'), LengthsType()),
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "acoustic_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_tokens_pred": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
        },
    )
    def forward(self, audio_tokens, audio_token_lens, text, text_lens):
        if self.vector_quantizer_converter_codec is not None:
            audio_tokens = self.vector_quantizer_converter_codec.convert_original_to_new(
                audio_tokens=audio_tokens, audio_lens=audio_token_lens
            )

        acoustic_tokens = audio_tokens[:, self.semantic_codebook_num :, :]
        if self.vector_quantizer_converter_acoustic is not None:
            acoustic_tokens = self.vector_quantizer_converter_acoustic.convert_original_to_new(
                audio_tokens=acoustic_tokens, audio_lens=audio_token_lens
            )

        audio_tokens_rearrange = rearrange(audio_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        audio_codes = self.vector_quantizer.decode(indices=audio_tokens_rearrange, input_len=audio_token_lens).detach()

        audio_mask = get_mask_from_lengths(audio_token_lens)
        text_mask = get_mask_from_lengths(text_lens)
        text_enc = self.text_encoder(text=text, text_mask=text_mask)

        if self.training:
            audio_maskin = self.create_infill_mask(
                input_lens=audio_token_lens,
                dist=self.audio_infill_dist,
                infill_min=self.audio_infill_min,
                infill_max=self.audio_infill_max,
            )
            semantic_mask = self.create_infill_mask(
                input_lens=audio_token_lens,
                dist=self.semantic_mask_dist,
                infill_min=self.semantic_mask_min,
                infill_max=self.semantic_mask_max,
            )
        else:
            audio_maskin = None
            semantic_mask = None

        semantic_codes = audio_codes[:, : self.semantic_codebook_dim, :]
        audio_codes = audio_codes[:, self.semantic_codebook_dim :, :]

        semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
        encoder_input = self.semantic_layer(
            semantic_codes=semantic_codes, audio_mask=audio_mask, semantic_mask=semantic_mask
        )
        encoded = self.encoder(
            inputs=encoder_input,
            audio_mask=audio_mask,
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

        return (
            acoustic_tokens,
            audio_tokens_pred,
            audio_logits,
        )

    def training_step(self, batch_dict, batch_idx):
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        (
            acoustic_tokens,
            _,
            audio_token_logits,
        ) = self(audio_tokens=audio_tokens, audio_token_lens=audio_token_lens, text=text, text_lens=text_lens)

        audio_mask = get_mask_from_lengths(audio_token_lens)
        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=acoustic_tokens, mask=audio_mask
        )
        loss = self.audio_token_loss_scale * audio_token_loss

        metrics = {
            "t_audio_token_loss": loss,
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
            acoustic_tokens,
            audio_tokens_pred,
            audio_token_logits,
        ) = self(audio_tokens=audio_tokens, audio_token_lens=audio_token_lens, text=text, text_lens=text_lens)

        audio_mask = get_mask_from_lengths(audio_token_lens)
        num_audio_tokens = max(1, audio_token_lens.sum() * self.acoustic_codebook_num)

        audio_token_loss = self.audio_token_loss_fn(
            logits=audio_token_logits, target_tokens=acoustic_tokens, mask=audio_mask
        )
        audio_token_correct = (acoustic_tokens == audio_tokens_pred) * rearrange(audio_mask, 'B T -> B 1 T')
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
        text,
        text_lens,
        frames_per_iter=1,
        num_audio_iters=0,
        audio_topk=None,
        audio_temperature=None,
    ):
        audio_mask = get_mask_from_lengths(semantic_lens)
        text_mask = get_mask_from_lengths(text_lens)

        text_enc = self.text_encoder(text=text, text_mask=text_mask)

        semantic_tokens_rearrange = rearrange(semantic_tokens, 'B C T -> C B T')
        # [batch_size, code_dim, audio_token_len]
        semantic_codes = self.vector_quantizer.decode(indices=semantic_tokens_rearrange, input_len=semantic_lens)

        semantic_codes = rearrange(semantic_codes, 'B C T -> B T C')
        encoder_input = self.semantic_layer(semantic_codes=semantic_codes, audio_mask=audio_mask)
        encoded = self.encoder(
            inputs=encoder_input,
            audio_mask=audio_mask,
            text_enc=text_enc,
            text_mask=text_mask,
        )
        # [B, C_acoustic, T]
        audio_tokens = self._audio_token_infer(
            inputs=encoded,
            audio_lens=semantic_lens,
            frames_per_iter=frames_per_iter,
            num_audio_iters=num_audio_iters,
            semantic_tokens=semantic_tokens,
            temperature=audio_temperature,
            topk=audio_topk,
        )

        return audio_tokens
