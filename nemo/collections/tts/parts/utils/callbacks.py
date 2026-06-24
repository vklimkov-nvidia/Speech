# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type

import librosa
import numpy as np
import soundfile as sf
import torch
from einops import rearrange
from lightning.pytorch import Callback, LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from torch import Tensor

from nemo.collections.tts.parts.utils.helpers import create_plot
from nemo.collections.tts.parts.utils.tts_dataset_utils import stack_tensors
from nemo.utils import logging

HAVE_WANDB = True
try:
    import wandb
except ModuleNotFoundError:
    HAVE_WANDB = False


def _get_logger(loggers: List[Logger], logger_type: Type[Logger]):
    for logger in loggers:
        if isinstance(logger, logger_type):
            if hasattr(logger, "experiment"):
                return logger.experiment
            else:
                return logger
    raise ValueError(f"Could not find {logger_type} logger in {loggers}.")


def _load_vocoder(
    model_name: Optional[str], checkpoint_path: Optional[str], type: str, device: Optional[str] = None, strict=True
):
    assert (model_name is None) != (
        checkpoint_path is None
    ), f"Must provide exactly one of vocoder model_name or checkpoint: ({model_name}, {checkpoint_path})"

    checkpoint_path = str(checkpoint_path)
    if type == "hifigan":
        from nemo.collections.tts.models import HifiGanModel

        model_type = HifiGanModel
    elif type == "univnet":
        from nemo.collections.tts.models import UnivNetModel

        model_type = UnivNetModel
    elif type == "audio_codec":
        from nemo.collections.tts.models import AudioCodecModel

        model_type = AudioCodecModel
    else:
        raise ValueError(f"Unknown vocoder type '{type}'")

    if device is not None:
        device = torch.device(device)

    if model_name is not None:
        vocoder = model_type.from_pretrained(model_name, map_location=device, strict=strict)
    elif checkpoint_path.endswith(".nemo"):
        vocoder = model_type.restore_from(checkpoint_path, map_location=device, strict=strict)
    else:
        vocoder = model_type.load_from_checkpoint(checkpoint_path, map_location=device, strict=strict)

    return vocoder.eval()


@dataclass
class AudioArtifact:
    id: str
    data: np.ndarray
    sample_rate: int
    filepath: Path


@dataclass
class ImageArtifact:
    id: str
    data: np.ndarray
    filepath: Path
    x_axis: str
    y_axis: str
    color_limit: Optional[Tuple[float, float]] = None


@dataclass
class LogAudioParams:
    vocoder_type: str
    vocoder_name: str
    vocoder_checkpoint_path: str
    vocoder_device: Optional[str] = None
    log_audio_gta: bool = False


def create_id(filepath: Path) -> str:
    path_prefix = str(filepath.with_suffix(""))
    file_id = path_prefix.replace(os.sep, "_")
    return file_id


class ArtifactGenerator(ABC):
    @abstractmethod
    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:
        """
        Create artifacts for the input model and test batch.

        Args:
            model: Model instance being trained to use for inference.
            batch_dict: Test batch to generate artifacts for.
            initial_log: Flag to denote if this is the initial log, can
                         be used to save ground-truth data only once.

        Returns:
            List of audio and image artifacts to log.
        """


class LoggingCallback(Callback):
    """
    Callback which can log artifacts (eg. model predictions, graphs) to local disk, Tensorboard, and/or WandB.

    Args:
        generators: List of generators to create and log artifacts from.
        data_loader: Data to log artifacts for.
        log_epochs: Optional list of specific training epoch numbers to log artifacts for.
        epoch_frequency: Frequency with which to log
        output_dir: Optional local directory. If provided, artifacts will be saved in output_dir.
        loggers: Optional list of loggers to use if logging to tensorboard or wandb.
        log_tensorboard: Whether to log artifacts to tensorboard.
        log_wandb: Whether to log artifacts to WandB.
    """

    def __init__(
        self,
        generators: List[ArtifactGenerator],
        data_loader: torch.utils.data.DataLoader,
        log_epochs: Optional[List[int]] = None,
        epoch_frequency: int = 1,
        output_dir: Optional[Path] = None,
        loggers: Optional[List[Logger]] = None,
        log_tensorboard: bool = False,
        log_wandb: bool = False,
        max_filename_len: Optional[int] = None,
    ):
        self.generators = generators
        self.data_loader = data_loader
        self.log_epochs = log_epochs if log_epochs else []
        self.epoch_frequency = epoch_frequency
        self.output_dir = Path(output_dir) if output_dir else None
        self.loggers = loggers if loggers else []
        self.log_tensorboard = log_tensorboard
        self.log_wandb = log_wandb
        self.max_filename_len = max_filename_len

        if log_tensorboard:
            logging.info('Creating tensorboard logger')
            self.tensorboard_logger = _get_logger(self.loggers, TensorBoardLogger)
        else:
            logging.debug('Not using tensorbord logger')
            self.tensorboard_logger = None

        if log_wandb:
            if not HAVE_WANDB:
                raise ValueError("Wandb not installed.")
            logging.info('Creating wandb logger')
            self.wandb_logger = _get_logger(self.loggers, WandbLogger)
        else:
            logging.debug('Not using wandb logger')
            self.wandb_logger = None

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tlog_epochs:      %s', self.log_epochs)
        logging.debug('\tepoch_frequency: %s', self.epoch_frequency)
        logging.debug('\toutput_dir:      %s', self.output_dir)
        logging.debug('\tlog_tensorboard: %s', self.log_tensorboard)
        logging.debug('\tlog_wandb:       %s', self.log_wandb)

    def _log_audio(self, audio: AudioArtifact, log_dir: Path, step: int):
        if log_dir:
            filepath = log_dir / audio.filepath
            if self.max_filename_len and len(filepath.parts[-1]) > self.max_filename_len:
                suffix = filepath.suffix
                filepath = (filepath.parent / filepath.stem[: self.max_filename_len - len(suffix)]).with_suffix(suffix)

            filepath.parent.mkdir(parents=True, exist_ok=True)
            sf.write(file=filepath, data=audio.data, samplerate=audio.sample_rate)

        if self.tensorboard_logger:
            self.tensorboard_logger.add_audio(
                tag=audio.id,
                snd_tensor=audio.data,
                global_step=step,
                sample_rate=audio.sample_rate,
            )

        if self.wandb_logger:
            wandb_audio = (wandb.Audio(audio.data, sample_rate=audio.sample_rate, caption=audio.id),)
            self.wandb_logger.log({audio.id: wandb_audio})

    def _log_image(self, image: ImageArtifact, log_dir: Path, step: int):
        if log_dir:
            filepath = log_dir / image.filepath
            if self.max_filename_len:
                filepath = (filepath.parent / filepath.stem[: self.max_filename_len]).with_suffix(filepath.suffix)
            filepath.parent.mkdir(parents=True, exist_ok=True)
        else:
            filepath = None

        image_plot = create_plot(
            output_filepath=filepath,
            data=image.data,
            x_axis=image.x_axis,
            y_axis=image.y_axis,
            color_limit=image.color_limit,
        )

        if self.tensorboard_logger:
            self.tensorboard_logger.add_image(
                tag=image.id,
                img_tensor=image_plot,
                global_step=step,
                dataformats="HWC",
            )

        if self.wandb_logger:
            wandb_image = (wandb.Image(image_plot, caption=image.id),)
            self.wandb_logger.log({image.id: wandb_image})

    def _log_artifacts(self, audio_list: list, image_list: list, log_dir: Optional[Path] = None, global_step: int = 0):
        """Log audio and image artifacts."""
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)

        for audio in audio_list:
            self._log_audio(audio=audio, log_dir=log_dir, step=global_step)

        for image in image_list:
            self._log_image(image=image, log_dir=log_dir, step=global_step)

    def on_fit_start(self, trainer: Trainer, model: LightningModule):
        """Log initial data artifacts."""
        audio_list = []
        image_list = []
        for batch_dict in self.data_loader:
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(model.device)

            for generator in self.generators:
                audio, images = generator.generate_artifacts(model=model, batch_dict=batch_dict, initial_log=True)
                audio_list += audio
                image_list += images

        if len(audio_list) == len(image_list) == 0:
            logging.debug('List are empty, no initial artifacts to log.')
            return

        log_dir = self.output_dir / "initial" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir)

    def on_train_epoch_end(self, trainer: Trainer, model: LightningModule):
        """Log artifacts at the end of an epoch."""
        epoch = 1 + model.current_epoch

        if (epoch not in self.log_epochs) and (epoch % self.epoch_frequency != 0):
            return

        audio_list = []
        image_list = []
        for batch_dict in self.data_loader:
            for key, value in batch_dict.items():
                if isinstance(value, torch.Tensor):
                    batch_dict[key] = value.to(model.device)

            for generator in self.generators:
                audio, images = generator.generate_artifacts(model=model, batch_dict=batch_dict)
                audio_list += audio
                image_list += images

        if len(audio_list) == len(image_list) == 0:
            logging.debug('List are empty, no artifacts to log at epoch %d.', epoch)
            return

        log_dir = self.output_dir / f"epoch_{epoch}" if self.output_dir else None

        self._log_artifacts(audio_list=audio_list, image_list=image_list, log_dir=log_dir)


class VocoderArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging Vocoder model outputs.
    """

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        audio = batch_dict.get("audio")
        audio_len = batch_dict.get("audio_lens")

        audio_artifacts = []

        if initial_log:
            # Log ground truth audio
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
                audio_gt_i = audio[i, : audio_len[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=model.sample_rate,
                )
                audio_artifacts.append(audio_artifact)
            return audio_artifacts, []

        spec, spec_len = model.audio_to_melspec_precessor(audio, audio_len)

        with torch.no_grad():
            audio_pred = model.forward(spec=spec)
            audio_pred = rearrange(audio_pred, "B 1 T -> B T")

        for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
            audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
            audio_pred_i = audio_pred[i, : audio_len[i]].cpu().numpy()
            audio_artifact = AudioArtifact(
                id=f"audio_{audio_id}",
                data=audio_pred_i,
                filepath=audio_pred_path,
                sample_rate=model.sample_rate,
            )
            audio_artifacts.append(audio_artifact)

        return audio_artifacts, []


class AudioCodecArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging Audio Codec model outputs.
    """

    def __init__(self, log_audio: bool = True, log_encoding: bool = False, log_dequantized: bool = False):
        # Log reconstructed audio (decoder output)
        self.log_audio = log_audio
        # Log encoded representation of the input audio (encoder output)
        self.log_encoding = log_encoding
        # Log dequantized encoded representation of the input audio (decoder input)
        self.log_dequantized = log_dequantized

        logging.debug('Initialized %s with', self.__class__.__name__)
        logging.debug('\tlog_audio:       %s', self.log_audio)
        logging.debug('\tlog_encoding:    %s', self.log_encoding)
        logging.debug('\tlog_dequantized: %s', self.log_dequantized)

    def _generate_audio(
        self,
        model: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        audio: Tensor,
        audio_len: Tensor,
        save_input: bool = False,
        audio_sample_rate: Optional[int] = 22050,
    ):
        """Generate audio artifacts.

        Args:
            model: callable model, outputs (audio_pred, audio_pred_len)
            dataset_names: list of dataset names for the examples in audio batch
            audio_ids: list of IDs for the examples in audio batch
            audio: tensor of input audio signals, shape (B, T)
            audio_len: tensor of lengths for each example in the batch, shape (B,)
            save_input: if True, save input audio signals
            audio_sample_rate: sample rate of the provided audio signals
        """
        if not self.log_audio:
            return []

        with torch.no_grad():
            # [B, T]
            audio_pred, audio_pred_len = model(audio=audio, audio_len=audio_len, sample_rate=audio_sample_rate)

        audio_artifacts = []
        # Log output audio
        for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
            audio_pred_path = Path(f"{dataset_name}/{audio_id}_audio_out.wav")
            audio_pred_i = audio_pred[i, : audio_pred_len[i]].cpu().numpy()
            audio_artifact = AudioArtifact(
                id=f"audio_out_{audio_id}",
                data=audio_pred_i,
                filepath=audio_pred_path,
                sample_rate=model.output_sample_rate,
            )
            audio_artifacts.append(audio_artifact)

        if save_input:
            # save input audio
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_in_path = Path(f"{dataset_name}/{audio_id}_audio_in.wav")
                audio_in_i = audio[i, : audio_len[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_in_{audio_id}",
                    data=audio_in_i,
                    filepath=audio_in_path,
                    sample_rate=model.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def _generate_images(
        self, model: LightningModule, dataset_names: List[str], audio_ids: List[str], audio: Tensor, audio_len: Tensor
    ):
        """Generate image artifacts.

        Args:
            model: model, needs to support `model.encode_audio`, `model.quantize` and `model.dequantize`
            dataset_names: list of dataset names for the examples in audio batch
            audio_ids: list of IDs for the examples in audio batch
            audio: tensor of input audio signals, shape (B, T)
            audio_len: tensor of lengths for each example in the batch, shape (B,)
        """
        image_artifacts = []

        if not self.log_encoding and not self.log_dequantized:
            return image_artifacts

        with torch.no_grad():
            # [B, D, T]
            encoded, encoded_len = model.encode_audio(
                audio=audio, audio_len=audio_len, sample_rate=model.output_sample_rate
            )

        if self.log_encoding:
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                encoded_path = Path(f"{dataset_name}/{audio_id}_encoded.png")
                encoded_i = encoded[i, :, : encoded_len[i]].cpu().numpy()
                encoded_artifact = ImageArtifact(
                    id=f"encoded_{audio_id}",
                    data=encoded_i,
                    filepath=encoded_path,
                    x_axis="Audio Frames",
                    y_axis="Channels",
                )
                image_artifacts.append(encoded_artifact)

        if not self.log_dequantized or not model.vector_quantizer:
            return image_artifacts

        with torch.no_grad():
            # [B, D, T]
            tokens = model.quantize(encoded=encoded, encoded_len=encoded_len)
            dequantized = model.dequantize(tokens=tokens, tokens_len=encoded_len)

        for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
            dequantized_path = Path(f"{dataset_name}/{audio_id}_dequantized.png")
            dequantized_i = dequantized[i, :, : encoded_len[i]].cpu().numpy()
            dequantized_artifact = ImageArtifact(
                id=f"dequantized_{audio_id}",
                data=dequantized_i,
                filepath=dequantized_path,
                x_axis="Audio Frames",
                y_axis="Channels",
            )
            image_artifacts.append(dequantized_artifact)

        return image_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:
        """
        Args:
            model: model used to process input to generate artifacts
            batch_dict: dictionary obtained form the dataloader
            initial_log: save input audio for the initial log
        """

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        audio = batch_dict.get("audio")
        audio_len = batch_dict.get("audio_lens")

        audio_artifacts = self._generate_audio(
            model=model,
            dataset_names=dataset_names,
            audio_ids=audio_ids,
            audio=audio,
            audio_len=audio_len,
            save_input=initial_log,
            audio_sample_rate=model.output_sample_rate,
        )
        image_artifacts = self._generate_images(
            model=model, dataset_names=dataset_names, audio_ids=audio_ids, audio=audio, audio_len=audio_len
        )

        return audio_artifacts, image_artifacts


class FastPitchArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging FastPitch model outputs.

    Args:
        log_spectrogram: Whether to log predicted spectrograms.
        log_alignment: Whether to log alignment graphs.
        audio_params: Optional parameters for saving predicted audio.
            Requires a vocoder model checkpoint for generating audio from predicted spectrograms.
    """

    def __init__(
        self,
        log_spectrogram: bool = False,
        log_alignment: bool = False,
        audio_params: Optional[LogAudioParams] = None,
    ):
        self.log_spectrogram = log_spectrogram
        self.log_alignment = log_alignment

        if not audio_params:
            self.log_audio = False
            self.log_audio_gta = False
            self.vocoder = None
        else:
            self.log_audio = True
            self.log_audio_gta = audio_params.log_audio_gta
            self.vocoder_type = audio_params.vocoder_type
            self.vocoder = _load_vocoder(
                model_name=audio_params.vocoder_name,
                checkpoint_path=audio_params.vocoder_checkpoint_path,
                type=self.vocoder_type,
                device=audio_params.vocoder_device,
            )

    def _create_ground_truth_artifacts(
        self, model: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio = batch_dict.get("audio")
        audio_lens = batch_dict.get("audio_lens")
        spec, spec_len = model.preprocessor(input_signal=audio, length=audio_lens)

        audio_artifacts = []
        image_artifacts = []
        for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
            audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
            audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
            audio_artifact = AudioArtifact(
                id=f"audio_gt_{audio_id}",
                data=audio_gt_i,
                filepath=audio_gt_path,
                sample_rate=model.preprocessor._sample_rate,
            )
            audio_artifacts.append(audio_artifact)

            spec_gt_path = Path(f"{dataset_name}/{audio_id}_spec_gt.png")
            spec_gt_i = spec[i, :, : spec_len[i]].cpu().numpy()
            spec_artifact = ImageArtifact(
                id=f"spec_{audio_id}",
                data=spec_gt_i,
                filepath=spec_gt_path,
                x_axis="Audio Frames",
                y_axis="Channels",
            )
            image_artifacts.append(spec_artifact)

        return audio_artifacts, image_artifacts

    def _generate_audio(self, mels: Tensor, mels_len: Tensor, hop_length: int):
        voc_input = mels.to(self.vocoder.device)
        with torch.no_grad():
            if self.vocoder_type == "audio_codec":
                audio_pred, _ = self.vocoder.decode_audio(inputs=voc_input, input_len=mels_len)
            else:
                audio_pred = self.vocoder.convert_spectrogram_to_audio(spec=voc_input)

        mels_len_array = mels_len.cpu().numpy()
        audio_pred_lens = librosa.core.frames_to_samples(mels_len_array, hop_length=hop_length)
        return audio_pred, audio_pred_lens

    def _generate_predictions(
        self, model: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        image_artifacts = []

        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        speaker = batch_dict.get("speaker_id", None)

        with torch.no_grad():
            # [B, C, T_spec]
            mels_pred, mels_pred_len, *_ = model.forward(
                text=text,
                input_lens=text_lens,
                speaker=speaker,
            )

        if self.log_spectrogram:
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                spec_path = Path(f"{dataset_name}/{audio_id}_spec.png")
                spec_i = mels_pred[i, :, : mels_pred_len[i]].cpu().numpy()
                spec_artifact = ImageArtifact(
                    id=f"spec_{audio_id}",
                    data=spec_i,
                    filepath=spec_path,
                    x_axis="Audio Frames",
                    y_axis="Channels",
                )
                image_artifacts.append(spec_artifact)

        if self.log_audio:
            # [B, T_audio]
            audio_pred, audio_pred_lens = self._generate_audio(
                mels=mels_pred, mels_len=mels_pred_len, hop_length=model.preprocessor.hop_length
            )
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
                audio_pred_i = audio_pred[i, : audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=self.vocoder.sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts, image_artifacts

    def _generate_gta_predictions(
        self, model: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        image_artifacts = []

        audio = batch_dict.get("audio")
        audio_lens = batch_dict.get("audio_lens")
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        pitch = batch_dict.get("pitch", None)
        energy = batch_dict.get("energy", None)
        speaker = batch_dict.get("speaker_id", None)

        mels, spec_len = model.preprocessor(input_signal=audio, length=audio_lens)
        with torch.no_grad():
            mels_pred, mels_pred_len, _, _, _, attn, _, _, _, _, _, _, _, _, _ = model.forward(
                text=text,
                input_lens=text_lens,
                pitch=pitch,
                energy=energy,
                speaker=speaker,
                spec=mels,
                mel_lens=spec_len,
            )

        if self.log_alignment:
            attn = rearrange(attn, "B 1 T_spec T_text -> B T_text T_spec")
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                attn_path = Path(f"{dataset_name}/{audio_id}_align.png")
                attn_i = attn[i, : text_lens[i], : mels_pred_len[i]].cpu().numpy()
                alignment_artifact = ImageArtifact(
                    id=f"align_{audio_id}",
                    data=attn_i,
                    filepath=attn_path,
                    x_axis="Audio Frames",
                    y_axis="Text Tokens",
                )
                image_artifacts.append(alignment_artifact)

        if self.log_audio_gta:
            # [B, T_audio]
            audio_pred, audio_pred_lens = self._generate_audio(
                mels=mels_pred, mels_len=mels_pred_len, hop_length=model.preprocessor.hop_length
            )
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}_gta.wav")
                audio_pred_i = audio_pred[i, : audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gta_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=self.vocoder.sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts, image_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        if initial_log:
            # Log ground truth audio and spectrograms
            audio_gt, spec_gt = self._create_ground_truth_artifacts(
                model=model, dataset_names=dataset_names, audio_ids=audio_ids, batch_dict=batch_dict
            )
            return audio_gt, spec_gt

        audio_artifacts = []
        image_artifacts = []

        if self.log_audio or self.log_spectrogram:
            audio_pred, spec_pred = self._generate_predictions(
                model=model, dataset_names=dataset_names, audio_ids=audio_ids, batch_dict=batch_dict
            )
            audio_artifacts += audio_pred
            image_artifacts += spec_pred

        if self.log_audio_gta or self.log_alignment:
            audio_gta_pred, alignments = self._generate_gta_predictions(
                model=model, dataset_names=dataset_names, audio_ids=audio_ids, batch_dict=batch_dict
            )
            audio_artifacts += audio_gta_pred
            image_artifacts += alignments

        return audio_artifacts, image_artifacts


class AcousticModelArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging AcousticModel outputs.

    Args:
    """

    def __init__(
        self,
        audio_codec_name,
        audio_codec_path,
        audio_codec_type: str = "audio_codec",
        log_audio: bool = False,
        log_semantic: bool = False,
        num_audio_iters: int = 1,
        num_audio_denoise_iters: int = 0,
        audio_topk: int = 1,
        audio_temperature: float = 1.0,
    ) -> None:
        self.log_audio = log_audio
        self.log_semantic = log_semantic
        self.num_audio_iters = num_audio_iters
        self.num_audio_denoise_iters = num_audio_denoise_iters
        self.audio_topk = audio_topk
        self.audio_temperature = audio_temperature
        self.audio_codec = _load_vocoder(
            model_name=audio_codec_name,
            checkpoint_path=audio_codec_path,
            type=audio_codec_type,
            strict=False,
        )

    def _create_ground_truth_artifacts(
        self, audio_codec: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        if self.log_audio:
            with torch.no_grad():
                audio, audio_lens = audio_codec.decode(tokens=audio_tokens, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        if self.log_semantic:
            with torch.no_grad():
                semantic_tokens = audio_tokens[:, :1, :]
                audio, audio_lens = audio_codec.semantic_codec.decode(
                    tokens=semantic_tokens, tokens_len=audio_token_lens
                )

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt_semantic.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_semantic_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.semantic_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def _generate_predictions(
        self,
        model: LightningModule,
        audio_codec: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        batch_dict: Dict,
    ) -> List[AudioArtifact]:

        audio_artifacts = []

        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        with torch.no_grad():
            semantic_tokens = audio_tokens[:, :1, :]
            audio_tokens_pred = model.infer(
                semantic_tokens=semantic_tokens,
                semantic_lens=audio_token_lens,
                num_audio_iters=self.num_audio_iters,
                num_audio_denoise_iters=self.num_audio_denoise_iters,
                audio_topk=self.audio_topk,
                audio_temperature=self.audio_temperature,
            )

        if self.log_audio:
            with torch.no_grad():
                # [B, T_audio]
                audio_pred, audio_pred_lens = audio_codec.decode(tokens=audio_tokens_pred, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
                audio_pred_i = audio_pred[i][: audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        is_train = model.training
        model = model.eval()

        audio_codec = self.audio_codec.to(model.device).eval()

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        image_artifacts = []

        if initial_log:
            audio_artifacts = self._create_ground_truth_artifacts(
                audio_codec=audio_codec, batch_dict=batch_dict, dataset_names=dataset_names, audio_ids=audio_ids
            )
        else:
            if self.log_audio:
                audio_artifacts = self._generate_predictions(
                    audio_codec=audio_codec,
                    model=model,
                    batch_dict=batch_dict,
                    dataset_names=dataset_names,
                    audio_ids=audio_ids,
                )
            else:
                audio_artifacts = []

        if is_train:
            model.train()

        return audio_artifacts, image_artifacts


class AcousticModelWithTextArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging AcousticModel outputs.

    Args:
    """

    def __init__(
        self,
        audio_codec_name,
        audio_codec_path,
        audio_codec_type: str = "audio_codec",
        log_audio: bool = False,
        log_semantic: bool = False,
        num_audio_iters: int = 1,
        audio_topk: int = 1,
        audio_temperature: float = 1.0,
    ) -> None:
        self.log_audio = log_audio
        self.log_semantic = log_semantic
        self.num_audio_iters = num_audio_iters
        self.audio_topk = audio_topk
        self.audio_temperature = audio_temperature
        self.audio_codec = _load_vocoder(
            model_name=audio_codec_name,
            checkpoint_path=audio_codec_path,
            type=audio_codec_type,
            strict=False,
        )

    def _create_ground_truth_artifacts(
        self, audio_codec: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        if self.log_audio:
            with torch.no_grad():
                audio, audio_lens = audio_codec.decode(tokens=audio_tokens, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        if self.log_semantic:
            with torch.no_grad():
                semantic_tokens = audio_tokens[:, :1, :]
                audio, audio_lens = audio_codec.semantic_codec.decode(
                    tokens=semantic_tokens, tokens_len=audio_token_lens
                )

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt_semantic.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_semantic_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.semantic_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def _generate_predictions(
        self,
        model: LightningModule,
        audio_codec: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        batch_dict: Dict,
    ) -> List[AudioArtifact]:

        audio_artifacts = []

        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")

        with torch.no_grad():
            semantic_tokens = audio_tokens[:, :1, :]
            audio_tokens_pred = model.infer(
                semantic_tokens=semantic_tokens,
                semantic_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
                num_audio_iters=self.num_audio_iters,
                audio_topk=self.audio_topk,
                audio_temperature=self.audio_temperature,
            )

        if self.log_audio:
            with torch.no_grad():
                # [B, T_audio]
                audio_pred, audio_pred_lens = audio_codec.decode(tokens=audio_tokens_pred, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
                audio_pred_i = audio_pred[i][: audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        is_train = model.training
        model = model.eval()

        audio_codec = self.audio_codec.to(model.device).eval()

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        image_artifacts = []

        if initial_log:
            audio_artifacts = self._create_ground_truth_artifacts(
                audio_codec=audio_codec, batch_dict=batch_dict, dataset_names=dataset_names, audio_ids=audio_ids
            )
        else:
            if self.log_audio:
                audio_artifacts = self._generate_predictions(
                    audio_codec=audio_codec,
                    model=model,
                    batch_dict=batch_dict,
                    dataset_names=dataset_names,
                    audio_ids=audio_ids,
                )
            else:
                audio_artifacts = []

        if is_train:
            model.train()

        return audio_artifacts, image_artifacts


class AcousticModelWithContextArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging AcousticModel outputs.

    Args:
    """

    def __init__(
        self,
        audio_codec_name,
        audio_codec_path,
        audio_codec_type: str = "audio_codec",
        log_audio: bool = False,
        log_semantic: bool = False,
        num_audio_iters: int = 1,
        audio_topk: int = 1,
        audio_temperature: float = 1.0,
        max_context_len: int = 50,
    ) -> None:
        self.log_audio = log_audio
        self.log_semantic = log_semantic
        self.num_audio_iters = num_audio_iters
        self.audio_topk = audio_topk
        self.audio_temperature = audio_temperature
        self.max_context_len = max_context_len
        self.audio_codec = _load_vocoder(
            model_name=audio_codec_name,
            checkpoint_path=audio_codec_path,
            type=audio_codec_type,
            strict=False,
        )

    def _create_ground_truth_artifacts(
        self, audio_codec: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        if self.log_audio:
            with torch.no_grad():
                audio, audio_lens = audio_codec.decode(tokens=audio_tokens, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        if self.log_semantic:
            with torch.no_grad():
                semantic_tokens = audio_tokens[:, :1, :]
                audio, audio_lens = audio_codec.semantic_codec.decode(
                    tokens=semantic_tokens, tokens_len=audio_token_lens
                )

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt_semantic.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_semantic_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.semantic_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def _generate_predictions(
        self,
        model: LightningModule,
        audio_codec: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        batch_dict: Dict,
    ) -> List[AudioArtifact]:

        audio_artifacts = []

        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")

        with torch.no_grad():
            context, context_lens = model.get_context(
                audio_tokens=audio_tokens,
                audio_lens=audio_token_lens,
                max_len=self.max_context_len,
            )
            semantic_tokens = audio_tokens[:, :1, :]
            audio_tokens_pred = model.infer(
                semantic_tokens=semantic_tokens,
                semantic_lens=audio_token_lens,
                context=context,
                context_lens=context_lens,
                text=text,
                text_lens=text_lens,
                num_audio_iters=self.num_audio_iters,
                audio_topk=self.audio_topk,
                audio_temperature=self.audio_temperature,
            )

        if self.log_audio:
            with torch.no_grad():
                # [B, T_audio]
                audio_pred, audio_pred_lens = audio_codec.decode(tokens=audio_tokens_pred, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
                audio_pred_i = audio_pred[i][: audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        is_train = model.training
        model = model.eval()

        audio_codec = self.audio_codec.to(model.device).eval()

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        image_artifacts = []

        if initial_log:
            audio_artifacts = self._create_ground_truth_artifacts(
                audio_codec=audio_codec, batch_dict=batch_dict, dataset_names=dataset_names, audio_ids=audio_ids
            )
        else:
            if self.log_audio:
                audio_artifacts = self._generate_predictions(
                    audio_codec=audio_codec,
                    model=model,
                    batch_dict=batch_dict,
                    dataset_names=dataset_names,
                    audio_ids=audio_ids,
                )
            else:
                audio_artifacts = []

        if is_train:
            model.train()

        return audio_artifacts, image_artifacts


class DiscreteSpeechArtifactGenerator(ArtifactGenerator):
    """
    Generator for logging DiscreteSpeech model outputs.

    Args:
        log_audio: Whether to log predicted audio.
        log_audio_gta: Whether to log predicted ground-truth aligned audio.
        log_dequantized: Whether to log dequantized token predictions.
        log_alignment: Whether to log alignment graphs.
    """

    def __init__(
        self,
        audio_codec_path,
        acoustic_model_path,
        audio_codec_name: Optional[str] = None,
        audio_codec_type: str = "audio_codec",
        log_audio: bool = False,
        log_audio_gta: bool = False,
        log_alignment: bool = False,
        word_stride: int = 2,
        num_audio_iters: int = 1,
        audio_topk: int = 1,
        audio_temperature: float = 1.0,
        num_duration_iters: Optional[int] = None,
        duration_topk: int = 1,
        duration_temperature: float = 1.0,
        silence_pad_start: int = None,
        silence_pad_end: int = None,
        acoustic_model_type="text",
    ) -> None:
        self.log_audio = log_audio
        self.log_audio_gta = log_audio_gta
        self.log_alignment = log_alignment
        self.word_stride = word_stride
        self.num_audio_iters = num_audio_iters
        self.audio_topk = audio_topk
        self.audio_temperature = audio_temperature
        self.num_duration_iters = num_duration_iters
        self.duration_topk = duration_topk
        self.duration_temperature = duration_temperature
        self.silence_pad_start = silence_pad_start
        self.silence_pad_end = silence_pad_end
        self.audio_codec = _load_vocoder(
            model_name=audio_codec_name,
            checkpoint_path=audio_codec_path,
            type=audio_codec_type,
            strict=False,
        )
        self.acoustic_model_type = acoustic_model_type
        if self.acoustic_model_type == "default":
            from nemo.collections.tts.models import AcousticModel

            self.acoustic_model = AcousticModel.restore_from(acoustic_model_path)
        elif self.acoustic_model_type == "text":
            from nemo.collections.tts.models import AcousticModelWithText

            self.acoustic_model = AcousticModelWithText.restore_from(acoustic_model_path)
        elif self.acoustic_model_type == "context":
            from nemo.collections.tts.models import AcousticModelWithContext

            self.acoustic_model = AcousticModelWithContext.restore_from(acoustic_model_path)
        else:
            raise ValueError(f"Unknown acoustic model type {self.acoustic_model_type}")

    def _create_ground_truth_artifacts(
        self, audio_codec: LightningModule, dataset_names: List[str], audio_ids: List[str], batch_dict: Dict
    ):
        audio_artifacts = []
        image_artifacts = []
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")

        if self.log_audio:
            with torch.no_grad():
                audio, audio_lens = audio_codec.decode(tokens=audio_tokens, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_gt_path = Path(f"{dataset_name}/{audio_id}_gt.wav")
                audio_gt_i = audio[i, : audio_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gt_{audio_id}",
                    data=audio_gt_i,
                    filepath=audio_gt_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts, image_artifacts

    def _generate_predictions(
        self,
        model: LightningModule,
        audio_codec: LightningModule,
        acoustic_model: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        batch_dict: Dict,
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        audio_artifacts = []
        image_artifacts = []
        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        text_strings = batch_dict.get("text_string")

        with torch.no_grad():
            context_emb, context, context_lens = model.get_context(
                audio_tokens=audio_tokens,
                audio_lens=audio_token_lens,
                text=text,
                text_lens=text_lens,
            )
            semantic_tokens_pred, audio_token_lens = model.infer(
                text=text,
                text_lens=text_lens,
                context_emb=context_emb,
                context=context,
                context_lens=context_lens,
                word_stride=self.word_stride,
                audio_topk=self.audio_topk,
                audio_temperature=self.audio_temperature,
                num_duration_iters=self.num_duration_iters,
                duration_topk=self.duration_topk,
                duration_temperature=self.duration_temperature,
                silence_pad_start=self.silence_pad_start,
                silence_pad_end=self.silence_pad_end,
            )

            if self.acoustic_model_type == "default":
                audio_tokens_pred = acoustic_model.infer(
                    semantic_tokens=semantic_tokens_pred,
                    semantic_lens=audio_token_lens,
                    num_audio_iters=self.num_audio_iters,
                    audio_topk=self.audio_topk,
                    audio_temperature=self.audio_temperature,
                )
            else:
                token_list = []
                token_len_list = []
                for text_str in text_strings:
                    tokens = acoustic_model.parse(text_str)[0]
                    token_len = tokens.shape[0]
                    token_list.append(tokens)
                    token_len_list.append(token_len)
                token_lens = torch.tensor(token_len_list, dtype=torch.int32, device=acoustic_model.device)
                token_max_len = int(token_lens.max().item())
                tokens = stack_tensors(
                    token_list, max_lens=[token_max_len], pad_value=acoustic_model.text_tokenizer.pad
                ).to(acoustic_model.device)
                if self.acoustic_model_type == "text":
                    audio_tokens_pred = acoustic_model.infer(
                        semantic_tokens=semantic_tokens_pred,
                        semantic_lens=audio_token_lens,
                        text=tokens,
                        text_lens=token_lens,
                        num_audio_iters=self.num_audio_iters,
                        audio_topk=self.audio_topk,
                        audio_temperature=self.audio_temperature,
                    )
                elif self.acoustic_model_type == "context":
                    acoustic_context, acoustic_context_lens = acoustic_model.get_context(
                        audio_tokens=audio_tokens, audio_lens=audio_token_lens
                    )
                    audio_tokens_pred = acoustic_model.infer(
                        semantic_tokens=semantic_tokens_pred,
                        semantic_lens=audio_token_lens,
                        context=acoustic_context,
                        context_lens=acoustic_context_lens,
                        text=tokens,
                        text_lens=token_lens,
                        num_audio_iters=self.num_audio_iters,
                        audio_topk=self.audio_topk,
                        audio_temperature=self.audio_temperature,
                    )
                else:
                    raise ValueError(f"Unknown acoustic model type {self.acoustic_model_type}")

        if self.log_audio:
            with torch.no_grad():
                # [B, T_audio]
                audio_pred, audio_pred_lens = audio_codec.decode(tokens=audio_tokens_pred, tokens_len=audio_token_lens)

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}.wav")
                audio_pred_i = audio_pred[i][: audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=audio_codec.output_sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts, image_artifacts

    def _generate_gta_predictions(
        self,
        model: LightningModule,
        audio_codec: LightningModule,
        acoustic_model: LightningModule,
        dataset_names: List[str],
        audio_ids: List[str],
        batch_dict: Dict,
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:
        audio_artifacts = []
        image_artifacts = []

        text = batch_dict.get("text")
        text_lens = batch_dict.get("text_lens")
        audio_tokens = batch_dict.get("audio_tokens")
        audio_token_lens = batch_dict.get("audio_token_lens")
        text_strings = batch_dict.get("text_string")

        with torch.no_grad():
            semantic_tokens_pred, context, context_lens, dur_lens, align, balign = model.infer_gta(
                text=text,
                text_lens=text_lens,
                audio_tokens=audio_tokens,
                audio_token_lens=audio_token_lens,
                word_stride=self.word_stride,
                audio_temperature=self.audio_temperature,
                audio_topk=self.audio_topk,
            )

            token_list = []
            token_len_list = []
            for text_str in text_strings:
                tokens = acoustic_model.parse(text_str)[0]
                token_len = tokens.shape[0]
                token_list.append(tokens)
                token_len_list.append(token_len)
            token_lens = torch.tensor(token_len_list, dtype=torch.int32, device=acoustic_model.device)
            token_max_len = int(token_lens.max().item())
            tokens = stack_tensors(
                token_list, max_lens=[token_max_len], pad_value=acoustic_model.text_tokenizer.pad
            ).to(acoustic_model.device)

            audio_tokens_pred = acoustic_model.infer(
                semantic_tokens=semantic_tokens_pred,
                semantic_lens=audio_token_lens,
                text=tokens,
                text_lens=token_lens,
                num_audio_iters=self.num_audio_iters,
                audio_topk=self.audio_topk,
                audio_temperature=self.audio_temperature,
            )

        if self.log_alignment:
            align = rearrange(align, "B 1 T_audio T_text -> B T_text T_audio")

            if balign is not None:
                balign = rearrange(balign, "B 1 T_audio T_text -> B T_text T_audio")

            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                align_path = Path(f"{dataset_name}/{audio_id}_align.png")
                align_i = align[i, : text_lens[i], : audio_token_lens[i]].cpu().numpy()
                alignment_artifact = ImageArtifact(
                    id=f"align_{audio_id}",
                    data=align_i,
                    filepath=align_path,
                    x_axis="Audio Tokens",
                    y_axis="Text Tokens",
                )
                image_artifacts.append(alignment_artifact)

                if balign is not None:
                    balign_path = Path(f"{dataset_name}/{audio_id}_align_biphone.png")
                    balign_i = balign[i, : dur_lens[i], : audio_token_lens[i]].cpu().numpy()
                    alignment_artifact = ImageArtifact(
                        id=f"align_biphone_{audio_id}",
                        data=balign_i,
                        filepath=balign_path,
                        x_axis="Audio Tokens",
                        y_axis="Biphone Tokens",
                    )
                    image_artifacts.append(alignment_artifact)

        if self.log_audio_gta:
            with torch.no_grad():
                # [B, T_audio]
                audio_pred, audio_pred_lens = audio_codec.decode(tokens=audio_tokens_pred, tokens_len=audio_token_lens)
            for i, (dataset_name, audio_id) in enumerate(zip(dataset_names, audio_ids)):
                audio_pred_path = Path(f"{dataset_name}/{audio_id}_gta.wav")
                audio_pred_i = audio_pred[i][: audio_pred_lens[i]].cpu().numpy()
                audio_artifact = AudioArtifact(
                    id=f"audio_gta_{audio_id}",
                    data=audio_pred_i,
                    filepath=audio_pred_path,
                    sample_rate=audio_codec.sample_rate,
                )
                audio_artifacts.append(audio_artifact)

        return audio_artifacts, image_artifacts

    def generate_artifacts(
        self, model: LightningModule, batch_dict: Dict, initial_log: bool = False
    ) -> Tuple[List[AudioArtifact], List[ImageArtifact]]:

        is_train = model.training
        model.eval()

        audio_codec = self.audio_codec.to(model.device).eval()
        acoustic_model = self.acoustic_model.to(model.device).eval()

        dataset_names = batch_dict.get("dataset_names")
        audio_filepaths = batch_dict.get("audio_filepaths")
        audio_ids = [create_id(p) for p in audio_filepaths]

        if initial_log:
            audio_artifacts, image_artifacts = self._create_ground_truth_artifacts(
                audio_codec=audio_codec, batch_dict=batch_dict, dataset_names=dataset_names, audio_ids=audio_ids
            )
        else:
            audio_artifacts = []
            image_artifacts = []

            if self.log_audio_gta or self.log_alignment:
                audio_gta_pred, alignments = self._generate_gta_predictions(
                    audio_codec=audio_codec,
                    acoustic_model=acoustic_model,
                    model=model,
                    batch_dict=batch_dict,
                    dataset_names=dataset_names,
                    audio_ids=audio_ids,
                )
                if self.log_audio_gta:
                    audio_artifacts += audio_gta_pred

                if self.log_alignment:
                    image_artifacts += alignments

            if self.log_audio:
                audio_pred, dequantized_pred = self._generate_predictions(
                    audio_codec=audio_codec,
                    acoustic_model=acoustic_model,
                    model=model,
                    batch_dict=batch_dict,
                    dataset_names=dataset_names,
                    audio_ids=audio_ids,
                )
                audio_artifacts += audio_pred
                image_artifacts += dequantized_pred

        if is_train:
            model.train()

        return audio_artifacts, image_artifacts
