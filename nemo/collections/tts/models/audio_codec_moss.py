from typing import Optional, Tuple

import torch
from einops import rearrange
from transformers import AutoModel

from nemo.collections.tts.parts.utils.tts_dataset_utils import resample_batch


class MossAudioCodecModel(torch.nn.Module):
    def __init__(
        self,
        model_name: str = "OpenMOSS-Team/MOSS-Audio-Tokenizer-Nano",
        num_codebooks: Optional[int] = None,
    ):
        super().__init__()

        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).eval()

        self.sample_rate = self.model.sampling_rate
        self.output_sample_rate = self.model.sampling_rate
        self.samples_per_frame = self.model.downsample_rate
        self.codebook_size = 1024
        if num_codebooks:
            self.num_codebooks = num_codebooks
        else:
            self.num_codebooks = 16

    def _pad_audio(self, audio, audio_len, samples_per_frame):
        """Zero pad the end of the audio so that we do not have a partial end frame.
        The output will be zero-padded to have an integer number of frames of
        length `self.samples_per_frame`.

        Args:
            audio: input time-domain signal
            audio_len: valid length for each example in the batch

        Returns:
            Padded time-domain signal `padded_audio` and its length `padded_len`.
        """
        num_frames = audio_len / samples_per_frame
        # To avoid rounding issues at lower precisions, do not call torch.ceil when the length is divisible by the frame rate
        num_frames = torch.where(audio_len % samples_per_frame == 0, num_frames, torch.ceil(num_frames))
        padded_len = samples_per_frame * num_frames.int()
        max_len = padded_len.max().item()
        num_padding = max_len - audio.shape[1]
        padded_audio = torch.nn.functional.pad(audio, (0, num_padding))
        return padded_audio, padded_len

    def _preprocess_audio(self, audio, audio_len, sample_rate):
        if sample_rate and sample_rate != self.sample_rate:
            audio, audio_len = resample_batch(
                audio=audio, audio_len=audio_len, input_sample_rate=sample_rate, output_sample_rate=self.sample_rate
            )

        # [B, T]
        audio, audio_len = self._pad_audio(audio=audio, audio_len=audio_len, samples_per_frame=self.samples_per_frame)
        # [B, 2, T]
        audio = audio.unsqueeze(1).repeat(1, self.model.config.number_channels, 1)
        return audio, audio_len

    def freeze(self):
        return self

    def eval(self):
        return self

    def encode(
        self, audio: torch.Tensor, audio_len: torch.Tensor, sample_rate: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert input time-domain audio signal into a discrete representation (tokens).

        Args:
            audio: input time-domain signal, shape `(batch, number of samples)`
            audio_len: valid length for each example in the batch, shape `(batch size,)`
            sample_rate: sample rate of input audio (int)

        Returns:
            Tokens for each codebook for each frame, shape `(batch, number of codebooks, number of frames)`,
            and the corresponding valid lengths, shape `(batch,)`
        """
        audio, audio_len = self._preprocess_audio(audio=audio, audio_len=audio_len, sample_rate=sample_rate)
        tokens = self.model.encode(audio, return_dict=True).audio_codes
        tokens = tokens[: self.num_codebooks]
        tokens = rearrange(tokens, 'C B T -> B C T')

        encoded_len = torch.round(audio_len // self.samples_per_frame).int()

        return tokens, encoded_len

    def decode(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Convert discrete tokens into a continuous time-domain signal.

        Args:
            tokens: discrete tokens for each codebook for each time frame, shape `(batch, number of codebooks, number of frames)`
            tokens_len: valid lengths, shape `(batch,)`

        Returns:
            Decoded output `audio` in the time domain and its length in number of samples `audio_len`.
            Note that `audio_len` will be a multiple of `self.samples_per_frame`.
        """
        tokens = rearrange(tokens, 'B C T -> C B T')

        # [B, 2, T]
        audio = self.model.decode(tokens, return_dict=True).audio
        audio = audio[:, 0, :]
        audio_len = self.samples_per_frame * tokens_len

        return audio, audio_len
