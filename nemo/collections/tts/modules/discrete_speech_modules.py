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

import torch
from einops import rearrange

from nemo.collections.tts.modules.acoustic_model_modules import Conv1d, sample_tokens
from nemo.collections.tts.parts.utils.helpers import get_mask_from_lengths
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
    EncodedRepresentation,
    FloatType,
    IntType,
    LengthsType,
    LogitsType,
    MaskType,
    TokenIndex,
)
from nemo.core.neural_types.neural_type import NeuralType


class ContextEmbeddingEncoder(NeuralModule):

    def __init__(
        self,
        input_dim,
        encoder,
        rnn_layers,
        rnn_dim,
    ):
        super(ContextEmbeddingEncoder, self).__init__()
        d_model = encoder.d_model
        self.pre_conv = Conv1d(in_channels=input_dim, out_channels=d_model)
        self.encoder = encoder
        self.rnn = torch.nn.LSTM(input_size=d_model, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True)
        self.emb_layer = torch.nn.Linear(in_features=rnn_dim, out_features=d_model)

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation()),
            "context": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)
        context = rearrange(context, 'B D T -> B T D')
        context = self.encoder(inputs=context, mask=mask)

        out = torch.nn.utils.rnn.pack_padded_sequence(
            context, audio_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.rnn(out)
        out, padded_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # [B, D]
        out = out[torch.arange(len(padded_lens)), (padded_lens - 1), :]
        context_emb = self.emb_layer(out)

        context = rearrange(context, 'B T D -> B D T')

        return context_emb, context


class SpeakingRatePredictor(NeuralModule):

    def __init__(self, num_speaking_rate, context_dim):
        super(SpeakingRatePredictor, self).__init__()
        self.hidden_layer = torch.nn.Linear(context_dim, context_dim)
        self.speaking_rate_layer = torch.nn.Linear(context_dim, num_speaking_rate)

    @property
    def input_types(self):
        return {"context_emb": NeuralType(('B', 'D'), EncodedRepresentation())}

    @property
    def output_types(self):
        return {
            "speaking_rate_indices_pred": NeuralType(tuple('B'), TokenIndex()),
            "speaking_rate_logits": NeuralType(('B', 'C'), LogitsType()),
        }

    @typecheck()
    def forward(self, context_emb):
        out = self.hidden_layer(context_emb)
        # [B, num_sr]
        speaking_rate_logits = self.speaking_rate_layer(out)
        # [B]
        speaking_rate_indices_pred = speaking_rate_logits.max(dim=1).indices
        return speaking_rate_indices_pred, speaking_rate_logits


class DurationDecoder(NeuralModule):

    def __init__(self, fft, num_duration, dropout_rate=0.1):
        super(DurationDecoder, self).__init__()
        self.d_model = fft.d_model
        self.fft = fft
        self.num_duration = num_duration

        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))

        self.dur_cond_layer = torch.nn.Linear(1, self.d_model)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_norm = torch.nn.LayerNorm(self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.duration_layer = torch.nn.Linear(self.d_model, self.num_duration)
        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.duration_layer_prallel = torch.nn.Linear(self.d_model, self.num_duration)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "dur_indices": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
            "duration_maskin": NeuralType(('B', 'T_text'), MaskType()),
            "temperature": NeuralType((), FloatType(), optional=True),
            "topk": NeuralType((), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "dur_indices_pred": NeuralType(('B', 'T_text'), TokenIndex()),
            "dur_logits": NeuralType(('B', 'C', 'T_text'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, dur_indices, text_mask, duration_maskin, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        log_dur = torch.log(dur_indices + 1.0).detach()
        log_dur = rearrange(log_dur, 'B T -> B T 1')
        dur_res = self.dur_cond_layer(log_dur)
        dur_res = self.dropout(dur_res)
        dur_res = dur_res * rearrange(duration_maskin, 'B T -> B T 1')

        masked_mask = ~duration_maskin * text_mask
        mask_res = self.mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + dur_res + mask_res

        # [B, T, D]
        dec_input = dec_input * rearrange(text_mask, 'B T -> B T 1')
        dec_out = self.fft(inputs=dec_input, text_mask=text_mask)

        # [B, T, num_codes]
        dec_out = self.layer_norm(dec_out)
        dur_logits = self.duration_layer(dec_out)
        dur_logits = dur_logits * text_mask_3d

        # [B, T]
        if temperature is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, temperature=temperature, topk=topk)

        dur_indices_pred = dur_indices_pred * text_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits

    def forward_parallel(self, inputs, text_mask, temperature=None, topk=None):
        text_mask_3d = rearrange(text_mask, 'B T -> B T 1')

        # [B, T, num_codes]
        dec_out = self.layer_norm_parallel(inputs)
        dur_logits = self.duration_layer_prallel(dec_out)
        dur_logits = dur_logits * text_mask_3d

        # [B, T]
        if temperature is None:
            dur_indices_pred = dur_logits.max(dim=2).indices
        else:
            dur_indices_pred = sample_tokens(logits=dur_logits, temperature=temperature, topk=topk)

        dur_indices_pred = dur_indices_pred * text_mask
        dur_logits = rearrange(dur_logits, 'B T N -> B N T')

        return dur_indices_pred, dur_logits


class SpeakingRateQuantizer(NeuralModule):

    def __init__(self, num_bins, min_value, max_value):
        super().__init__()
        self.num_bins = num_bins
        self.max_bin = num_bins - 1
        self.shift = (min_value + max_value) / 2
        self.scale = max_value - self.shift

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(tuple('B'), EncodedRepresentation()),
        }

    @property
    def output_types(self):
        return {
            "codes": NeuralType(tuple('B'), EncodedRepresentation()),
            "indices": NeuralType(tuple('B'), TokenIndex()),
        }

    @typecheck()
    def forward(self, inputs):
        scaled = (inputs - self.shift) / self.scale
        # [-1, 1]
        scaled = torch.clamp(scaled, min=-1.0, max=1.0)
        # [0, 1]
        shifted = (scaled + 1.0) / 2.0
        # [0, num_bins]
        indices = torch.round(shifted * self.max_bin)
        indices = torch.clamp(indices, min=0, max=self.max_bin).int()
        codes = self.get_codes(indices)

        return codes, indices

    def get_codes(self, indices):
        codes = indices.float() / self.max_bin
        codes = 2.0 * codes - 1.0
        return codes


class AudioDecoder(NeuralModule):

    def __init__(self, fft, num_codebooks, codebook_size, codebook_dim):
        super(AudioDecoder, self).__init__()
        self.d_model = fft.d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))
        self.fft = fft

        self.audio_hidden_layer = torch.nn.Linear(codebook_dim, self.d_model)
        self.audio_cond_layer = torch.nn.Linear(self.d_model, self.d_model)

        self.layer_norm = torch.nn.LayerNorm(self.d_model)
        self.audio_token_layer = torch.nn.Linear(self.d_model, self.num_logits)
        self.layer_norm_parallel = torch.nn.LayerNorm(self.d_model)
        self.audio_token_layer_parallel = torch.nn.Linear(self.d_model, self.num_logits)

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
            "audio_maskin": NeuralType(('B', 'T_audio'), MaskType()),
            "temperature": NeuralType((), FloatType(), optional=True),
            "topk": NeuralType((), IntType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "audio_tokens": NeuralType(('B', 'C', 'T_audio'), TokenIndex()),
            "audio_logits": NeuralType(('B', 'C', 'W', 'T_audio'), LogitsType()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, audio_codes, audio_maskin, temperature=None, topk=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        audio_res = self.audio_hidden_layer(audio_codes)
        audio_res = self.audio_cond_layer(audio_res)
        audio_res = audio_res * rearrange(audio_maskin, 'B T -> B T 1')

        masked_mask = ~audio_maskin * audio_mask
        mask_res = self.audio_mask_emb * rearrange(masked_mask, 'B T -> B T 1')

        dec_input = inputs + audio_res + mask_res

        dec_input = dec_input * audio_mask_3d
        dec_out = self.fft(inputs=dec_input, audio_mask=audio_mask)

        # [batch_size, audio_len, num_codebook * codebook_size]
        dec_out = self.layer_norm(dec_out)
        audio_logits = self.audio_token_layer(dec_out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)

        audio_logits = torch.reshape(audio_logits, logit_shape)
        # [batch_size, audio_len, num_codebook]
        if temperature is None:
            audio_tokens = audio_logits.max(dim=3).indices
        else:
            audio_tokens = sample_tokens(logits=audio_logits, temperature=temperature, topk=topk)

        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits

    def forward_parallel(self, inputs, audio_mask, temperature=None, topk=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        # [batch_size, audio_len, num_codebook * codebook_size]
        out = self.layer_norm_parallel(inputs)
        audio_logits = self.audio_token_layer_parallel(out)
        audio_logits = audio_logits * audio_mask_3d

        # [batch_size, audio_len, num_codebook, codebook_size]
        logit_shape = (audio_logits.shape[0], audio_logits.shape[1], self.num_codebooks, self.codebook_size)
        audio_logits = torch.reshape(audio_logits, logit_shape)

        # [batch_size, audio_len, num_codebook]
        if temperature is None:
            audio_tokens = audio_logits.max(dim=3).indices
        else:
            audio_tokens = sample_tokens(logits=audio_logits, temperature=temperature, topk=topk)

        audio_tokens = audio_tokens * audio_mask_3d

        audio_logits = rearrange(audio_logits, 'B T C W -> B C W T')
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens, audio_logits
