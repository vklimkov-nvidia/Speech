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
import math

from nemo.collections.tts.parts.utils.helpers import binarize_attention_parallel, get_mask_from_lengths, regulate_len
from nemo.collections.tts.parts.utils.tts_dataset_utils import beta_binomial_prior_distribution_torch
from nemo.core.classes import NeuralModule, typecheck
from nemo.core.neural_types.elements import (
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
    VoidType,
)
from nemo.core.neural_types.neural_type import NeuralType


def sample_tokens(logits, temperature, topk):
    batch_shape = logits.shape[:-1]
    # [B, codebook_size]
    logits = logits.reshape(batch_shape.numel(), -1)
    # [B, k]
    logits_topk = torch.topk(logits, topk, dim=1)[0]
    # [B, 1]
    min_logits = logits_topk[:, -1:]
    # [B, codebook_size]
    indices_to_remove = logits < min_logits
    # [B, codebook_size]
    logits_rescored = logits.clone()
    logits_rescored = logits_rescored / temperature
    logits_rescored[indices_to_remove] = float('-inf')

    probs = torch.softmax(logits_rescored, dim=1)
    # [(B * T * num_codebook), 1]
    tokens = torch.multinomial(input=probs, num_samples=1)
    tokens = tokens.reshape(batch_shape)
    return tokens


class Conv1d(NeuralModule):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation=None):
        super().__init__()
        padding = kernel_size // 2
        self.conv = torch.nn.Conv1d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        if activation is None:
            self.activation = None
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "elu":
            self.activation = torch.nn.ELU()
        else:
            raise ValueError(f"Unknown activation {activation}")

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'C', 'T'), VoidType()),
            "mask": NeuralType(('B', 'T'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'C', 'T'), VoidType()),
        }

    @typecheck()
    def forward(self, inputs, mask):
        out = self.conv(inputs)
        if self.activation:
            out = self.activation(out)
        out = out * rearrange(mask, 'B T -> B 1 T')
        return out


class SemanticInputLayer(NeuralModule):

    def __init__(self, input_dim, output_dim):
        super(SemanticInputLayer, self).__init__()
        self.hidden_layer = torch.nn.Linear(input_dim, output_dim)
        self.output_layer = torch.nn.Linear(output_dim, output_dim)
        self.mask_emb = torch.nn.Parameter(torch.zeros([1, 1, output_dim]))

    @property
    def input_types(self):
        return {
            "semantic_codes": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T'), MaskType()),
            "semantic_mask": NeuralType(('B', 'T_audio'), MaskType(), optional=True),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'C'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, semantic_codes, audio_mask, semantic_mask=None):
        out = self.hidden_layer(semantic_codes)
        out = self.output_layer(out)

        if semantic_mask is not None:
            semantic_mask_3d = rearrange(semantic_mask, 'B T -> B T 1')
            out = torch.where(semantic_mask_3d, self.mask_emb, out)

        out = out * rearrange(audio_mask, 'B T -> B T 1')
        return out


class TextDownSampling(NeuralModule):

    def __init__(self, input_dim, down_sample_rate, bos_id, eos_id, space_id, space_dur):
        super().__init__()
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.space_id = space_id
        self.space_dur = space_dur
        self.down_sample_rate = down_sample_rate
        kernel_size = 2 * self.down_sample_rate - 1
        self.downsample_layer = Conv1d(
            in_channels=input_dim,
            out_channels=input_dim,
            kernel_size=kernel_size,
            stride=self.down_sample_rate,
        )

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T'), EncodedRepresentation()),
            "text_emb": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
        },
        output_types={
            "outputs": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
            "output_lens": NeuralType(tuple('B'), LengthsType()),
            "text_durs": NeuralType(('B', 'T'), LengthsType()),
        },
    )
    def forward(self, text, text_emb, text_lens):
        text_mask = get_mask_from_lengths(text_lens)
        is_bos_eos = torch.logical_or(text == self.bos_id, text == self.eos_id)
        is_space = text == self.space_id
        text_durs = torch.where(is_bos_eos, self.down_sample_rate * torch.ones_like(text), torch.ones_like(text))
        text_durs = torch.where(is_space, self.space_dur * torch.ones_like(text), text_durs)
        text_durs = text_durs * text_mask
        text_emb = rearrange(text_emb, 'B D T -> B T D')
        text_emb_repeated, output_lens = regulate_len(durations=text_durs, enc_out=text_emb)
        text_emb_repeated = rearrange(text_emb_repeated, 'B T D -> B D T')
        output_lens = torch.ceil(output_lens / self.down_sample_rate).int()
        out_mask = get_mask_from_lengths(output_lens)
        outputs = self.downsample_layer(inputs=text_emb_repeated, mask=out_mask)
        return outputs, output_lens, text_durs


class Aligner(NeuralModule):

    def __init__(
        self,
        alignment_encoder,
        num_text_emb,
        text_emb_dim,
        prior_scaling_factor=0.2,
        down_sample_rate=None,
        space_id=None,
        bos_id=None,
        eos_id=None,
        space_dur=None,
    ):
        super().__init__()
        self.alignment_encoder = alignment_encoder
        self.text_emb = torch.nn.Embedding(num_text_emb, text_emb_dim)
        self.prior_scaling_factor = prior_scaling_factor

        if down_sample_rate and down_sample_rate > 1:
            self.downsample_layer = TextDownSampling(
                input_dim=text_emb_dim,
                down_sample_rate=down_sample_rate,
                bos_id=bos_id,
                eos_id=eos_id,
                space_id=space_id,
                space_dur=space_dur,
            )
        else:
            self.downsample_layer = None

    def _create_alignment_prior(self, text_lens, text_max_len, audio_lens, audio_max_len):
        batch_size = text_lens.shape[0]
        prior_batch = torch.zeros([batch_size, audio_max_len, text_max_len], device=text_lens.device)
        for i in range(batch_size):
            text_len = text_lens[i].item()
            audio_len = audio_lens[i].item()
            prior = beta_binomial_prior_distribution_torch(
                phoneme_count=text_len,
                mel_count=audio_len,
                scaling_factor=self.prior_scaling_factor,
                device=text_lens.device,
            ).to(text_lens.device)
            prior_batch[i, :audio_len, :text_len] = prior

        return prior_batch

    @typecheck(
        input_types={
            "text": NeuralType(('B', 'T_text'), TokenIndex()),
            "text_lens": NeuralType(tuple('B'), LengthsType()),
            "audio_codes": NeuralType(('B', 'D', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
            "context_emb": NeuralType(('B', 'D'), EncodedRepresentation(), optional=True),
        },
        output_types={
            "durations": NeuralType(('B', 'T_text'), TokenDurationType()),
            "duration_lens": NeuralType(tuple('B'), LengthsType()),
            "attn_hard": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_soft": NeuralType(('B', 'S', 'T_audio', 'T_text'), ProbsType()),
            "attn_logprob": NeuralType(('B', 'S', 'T_audio', 'T_text'), LogprobsType()),
        },
    )
    def forward(self, text, text_lens, audio_codes, audio_lens, context_emb=None):
        audio_mask = get_mask_from_lengths(audio_lens)
        text_mask = get_mask_from_lengths(text_lens)
        # [batch_size, text_len, hidden_dim]
        text_emb = self.text_emb(text)
        text_emb = text_emb * rearrange(text_mask, "B T -> B T 1")
        text_emb = rearrange(text_emb, "B T D -> B D T")

        if self.downsample_layer is not None:
            text_emb, text_lens, _ = self.downsample_layer(text=text, text_emb=text_emb, text_lens=text_lens)
            text_mask = get_mask_from_lengths(text_lens)

        attn_mask = rearrange(audio_mask, "B T -> B 1 T 1") * rearrange(text_mask, "B T -> B 1 1 T")
        # Aligner requires an inverted mask
        aligner_text_mask = ~rearrange(text_mask, "B T -> B T 1")

        if context_emb is not None:
            context_emb = rearrange(context_emb, 'B D -> B 1 D')

        text_max_len = text_emb.shape[2]
        audio_max_len = audio_codes.shape[2]
        attn_prior = self._create_alignment_prior(
            text_lens=text_lens,
            text_max_len=text_max_len,
            audio_lens=audio_lens,
            audio_max_len=audio_max_len,
        )
        # [batch_size, 1, audio_len, text_len]
        attn_soft, attn_logprob = self.alignment_encoder(
            queries=audio_codes, keys=text_emb, mask=aligner_text_mask, attn_prior=attn_prior, conditioning=context_emb
        )
        attn_soft = attn_soft * attn_mask
        attn_logprob = attn_logprob * attn_mask
        attn_hard = binarize_attention_parallel(attn=attn_soft, in_lens=text_lens, out_lens=audio_lens)

        durations = attn_hard.sum(2)
        durations = rearrange(durations, 'B 1 T -> B T')

        return durations, text_lens, attn_hard, attn_soft, attn_logprob


class ContextUtteranceEncoder(NeuralModule):

    def __init__(
        self,
        input_dim,
        context_emb_dim,
        filters,
        rnn_layers,
        rnn_dim,
    ):
        super(ContextUtteranceEncoder, self).__init__()
        self.conv1 = Conv1d(in_channels=input_dim, out_channels=filters, activation="elu")
        self.conv2 = Conv1d(in_channels=filters, out_channels=filters, activation="elu")
        self.rnn = torch.nn.LSTM(input_size=filters, hidden_size=rnn_dim, num_layers=rnn_layers, batch_first=True)
        self.emb_layer = torch.nn.Linear(in_features=rnn_dim, out_features=context_emb_dim)

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {"context_emb": NeuralType(('B', 'D'), EncodedRepresentation())}

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)

        out = self.conv1(inputs=audio_codes, mask=mask)
        out = self.conv2(inputs=out, mask=mask)

        out = rearrange(out, 'B D T -> B T D')
        out = torch.nn.utils.rnn.pack_padded_sequence(out, audio_lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.rnn(out)
        out, padded_lens = torch.nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        # [B, D]
        out = out[torch.arange(len(padded_lens)), (padded_lens - 1), :]
        context_emb = self.emb_layer(out)
        return context_emb


class TextEncoder(NeuralModule):
    def __init__(self, transformer, d_model, n_embed, padding_idx):
        super(TextEncoder, self).__init__()
        self.word_emb = torch.nn.Embedding(n_embed, d_model, padding_idx=padding_idx)
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "text": NeuralType(('B', 'T_text'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_audio'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "out": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, text, text_mask):
        text_emb = self.word_emb(text)
        text_enc = self.transformer(x=text_emb, x_mask=text_mask)['output']
        return text_enc


class ContextEncoder(NeuralModule):

    def __init__(self, input_dim, d_model, transformer):
        super(ContextEncoder, self).__init__()
        self.pre_conv = Conv1d(in_channels=input_dim, out_channels=d_model)
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "audio_codes": NeuralType(('B', 'C', 'T_audio'), EncodedRepresentation()),
            "audio_lens": NeuralType(tuple('B'), LengthsType()),
        }

    @property
    def output_types(self):
        return {
            "context": NeuralType(('B', 'D', 'T'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, audio_codes, audio_lens):
        mask = get_mask_from_lengths(audio_lens)
        context = self.pre_conv(inputs=audio_codes, mask=mask)

        context = rearrange(context, 'B D T -> B T D')
        context = self.transformer(x=context, x_mask=mask)['output']
        context = rearrange(context, 'B T D -> B D T')

        return context


class AudioEncoderWithText(NeuralModule):
    def __init__(self, transformer):
        super(AudioEncoderWithText, self).__init__()
        self.transformer = transformer

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "text_enc": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, text_enc, text_mask):
        audio_enc = self.transformer(
            x=inputs,
            x_mask=audio_mask,
            cond=text_enc,
            cond_mask=text_mask,
        )['output']
        return audio_enc


class AudioEncoderWithContext(NeuralModule):
    def __init__(self, transformer):
        super(AudioEncoderWithContext, self).__init__()
        self.transformer = transformer
        num_layers = len(self.transformer.layers)
        self.multi_encoder_mapping = [0 if i % 2 == 0 else 1 for i in range(num_layers)]

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "context": NeuralType(('B', 'T_context', 'D'), EncodedRepresentation()),
            "context_mask": NeuralType(('B', 'T_context'), MaskType()),
            "text_enc": NeuralType(('B', 'T_text', 'D'), EncodedRepresentation()),
            "text_mask": NeuralType(('B', 'T_text'), MaskType()),
        }

    @property
    def output_types(self):
        return {
            "audio_enc": NeuralType(('B', 'T', 'D'), EncodedRepresentation()),
        }

    @typecheck()
    def forward(self, inputs, audio_mask, context, context_mask, text_enc, text_mask):
        cond = [text_enc, context]
        cond_mask = [text_mask, context_mask]
        audio_enc = self.transformer(
            x=inputs,
            x_mask=audio_mask,
            cond=cond,
            cond_mask=cond_mask,
            multi_encoder_mapping=self.multi_encoder_mapping,
        )['output']
        return audio_enc


class AudioDecoder(NeuralModule):

    def __init__(self, transformer, d_model, num_codebooks, codebook_size, codebook_dim):
        super(AudioDecoder, self).__init__()
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.transformer = transformer

        self.audio_hidden_layer = torch.nn.Linear(codebook_dim, d_model)
        self.audio_cond_layer = torch.nn.Linear(d_model, d_model)

        self.layer_norm = torch.nn.LayerNorm(d_model)
        self.audio_token_layer = torch.nn.Linear(d_model, self.num_logits)

        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, d_model]))

    @property
    def input_types(self):
        return {
            "inputs": NeuralType(('B', 'T_audio', 'D'), EncodedRepresentation()),
            "audio_mask": NeuralType(('B', 'T_audio'), MaskType()),
            "audio_codes": NeuralType(('B', 'T_audio', 'C'), EncodedRepresentation()),
            "audio_maskin": NeuralType(('B', 'T_audio'), MaskType(), optional=True),
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
    def forward(self, inputs, audio_mask, audio_codes, audio_maskin=None, temperature=None, topk=None):
        audio_mask_3d = rearrange(audio_mask, 'B T -> B T 1')

        audio_codes_shifted = audio_codes[:, :-1, :]
        audio_codes_shifted = torch.nn.functional.pad(audio_codes_shifted, pad=(0, 0, 1, 0))

        audio_res = self.audio_hidden_layer(audio_codes_shifted)
        audio_res = self.audio_cond_layer(audio_res)

        if audio_maskin is not None:
            audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
            audio_res = torch.where(audio_maskin_3d, audio_res, self.audio_mask_emb)

        dec_input = inputs + audio_res
        dec_input = dec_input * audio_mask_3d

        # [batch_size, audio_len, hidden_dim]
        dec_out = self.transformer(x=dec_input, x_mask=audio_mask)['output']
        dec_out = self.layer_norm(dec_out)

        # [batch_size, audio_len, num_codebook * codebook_size]
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

    def infer(
        self,
        inputs,
        audio_lens,
        frames_per_iter,
        vector_quantizer,
        temperature=None,
        topk=None,
        use_cache=True,
    ):
        self.transformer.reset_cache(use_cache=use_cache, frames_per_iter=frames_per_iter)

        batch_size = inputs.shape[0]
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens, pad_to_factor=frames_per_iter)

        audio_lens_padded = torch.ceil(audio_lens / frames_per_iter).int() * frames_per_iter
        max_len = audio_lens.max()
        max_len_padded = audio_lens_padded.max()
        inputs = torch.nn.functional.pad(inputs, (0, 0, 0, max_len_padded - max_len))

        # [B, T, C]
        audio_token_shape = [batch_size, max_len_padded, self.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [batch_size, max_len_padded, self.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(0, max_len_padded, frames_per_iter):
            inputs_i = inputs[:, :i + frames_per_iter, :]
            audio_codes_i = audio_codes[:, : i + frames_per_iter, :]
            audio_mask_i = audio_mask[:, : i + frames_per_iter]

            audio_maskin_i = audio_mask_i.clone()
            for j in range(1, frames_per_iter):
                audio_maskin_i[:, i + j] = False

            # [B, C, T], [B, C, W, T]
            audio_tokens_i, audio_logits = self.forward(
                inputs=inputs_i,
                audio_mask=audio_mask_i,
                audio_codes=audio_codes_i,
                audio_maskin=audio_maskin_i,
                temperature=temperature,
                topk=topk,
            )
            audio_tokens_i = rearrange(audio_tokens_i, 'B C T -> B T C')
            audio_tokens_rearrange_i = rearrange(audio_tokens_i, 'B T C -> C B T')
            # [B, D, T]
            audio_codes_pred_i = vector_quantizer.decode(indices=audio_tokens_rearrange_i, input_len=audio_lens)
            audio_codes_pred_i = rearrange(audio_codes_pred_i, 'B D T -> B T D')
            for j in range(frames_per_iter):
                audio_codes[:, i + j, :] = audio_codes_pred_i[:, i + j, :]
                audio_tokens[:, i + j, :] = audio_tokens_i[:, i + j, :]

        audio_tokens = audio_tokens[:, :max_len, :]
        audio_mask_unpadded = get_mask_from_lengths(audio_lens)
        audio_tokens = audio_tokens * audio_mask_unpadded.unsqueeze(2)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        self.transformer.reset_cache(use_cache=False)

        return audio_tokens

    def infer_diffusion(
        self,
        inputs,
        audio_lens,
        num_iters,
        vector_quantizer,
        temperature=None,
        topk=None,
    ):
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)
        num_tokens = inputs.shape[1]

        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        # [B, T]
        audio_maskin = torch.zeros_like(audio_mask, dtype=torch.bool)
        audio_maskin[:, 0] = True
        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1], self.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(num_iters):
            # [B, C, T], [B, C, W, T]
            audio_tokens_i, audio_logits = self(
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
            audio_codes_i = vector_quantizer.decode(
                indices=audio_tokens_rearrange_i, input_len=audio_lens
            )
            audio_codes_i = rearrange(audio_codes_i, 'B D T -> B T D')

            top_i = torch.clamp_max(index_shift + i, max=num_tokens - 1)
            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)

            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(audio_mask, maskin_i, False)
            maskin_3d_i = rearrange(maskin_i, 'B T -> B T 1')

            audio_tokens = torch.where(maskin_3d_i, audio_tokens_i, audio_tokens)
            audio_codes = torch.where(maskin_3d_i, audio_codes_i, audio_codes)

            next_i = torch.clamp_max(index_shift + i + 1, max=num_tokens - 1)
            # [B, T // num_iters, T]
            next_one_hot = torch.nn.functional.one_hot(next_i, num_classes=num_tokens)
            # [B, T]
            next_maskin = next_one_hot.sum(dim=1).bool()
            next_maskin = torch.where(audio_mask, next_maskin, False)
            audio_maskin = torch.logical_or(audio_maskin, next_maskin)

        audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
        audio_tokens = torch.where(audio_maskin_3d, audio_tokens, audio_tokens_i)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens


class AudioDiffusionDecoder(NeuralModule):

    def __init__(self, transformer, d_model, num_codebooks, codebook_size, codebook_dim):
        super(AudioDiffusionDecoder, self).__init__()
        self.d_model = d_model
        self.num_codebooks = num_codebooks
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.num_logits = self.num_codebooks * self.codebook_size

        self.audio_mask_emb = torch.nn.Parameter(torch.zeros([1, 1, self.d_model]))
        self.transformer = transformer

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
        dec_out = self.transformer(x=dec_input, x_mask=audio_mask)['output']

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

    def infer(
        self,
        inputs,
        audio_lens,
        num_iters,
        vector_quantizer,
        temperature=None,
        topk=None,
    ):
        # [B, T]
        audio_mask = get_mask_from_lengths(audio_lens)
        num_tokens = inputs.shape[1]

        # [T]
        index_shift = num_iters * torch.arange(0, math.ceil(num_tokens / num_iters), device=inputs.device)
        index_shift = rearrange(index_shift, 'T -> 1 T')

        # [B, T]
        audio_maskin = torch.zeros_like(audio_mask, dtype=torch.bool)
        # [B, T, C]
        audio_token_shape = [audio_mask.shape[0], audio_mask.shape[1], self.num_codebooks]
        audio_tokens = torch.zeros(audio_token_shape, dtype=torch.int, device=inputs.device)
        # [B, T, D]
        audio_code_shape = [audio_mask.shape[0], audio_mask.shape[1], self.codebook_dim]
        audio_codes = torch.zeros(audio_code_shape, dtype=torch.float, device=inputs.device)

        for i in range(num_iters):
            if i == 0:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self.forward_parallel(
                    inputs=inputs,
                    audio_mask=audio_mask,
                    temperature=temperature,
                    topk=topk,
                )
            else:
                # [B, C, T], [B, C, W, T]
                audio_tokens_i, audio_logits = self(
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
            audio_codes_i = vector_quantizer.decode(
                indices=audio_tokens_rearrange_i, input_len=audio_lens
            )
            audio_codes_i = rearrange(audio_codes_i, 'B D T -> B T D')

            top_i = torch.clamp_max(index_shift + i, max=num_tokens - 1)
            # [B, T // num_iters, T]
            one_hot = torch.nn.functional.one_hot(top_i, num_classes=num_tokens)

            # [B, T]
            maskin_i = one_hot.sum(dim=1).bool()
            maskin_i = torch.where(audio_mask, maskin_i, False)
            maskin_i = torch.where(audio_maskin, False, maskin_i)
            maskin_3d_i = rearrange(maskin_i, 'B T -> B T 1')

            audio_tokens = torch.where(maskin_3d_i, audio_tokens_i, audio_tokens)
            audio_codes = torch.where(maskin_3d_i, audio_codes_i, audio_codes)

            audio_maskin = torch.logical_or(audio_maskin, maskin_i)

        audio_maskin_3d = rearrange(audio_maskin, 'B T -> B T 1')
        audio_tokens = torch.where(audio_maskin_3d, audio_tokens, audio_tokens_i)
        audio_tokens = rearrange(audio_tokens, 'B T C -> B C T')

        return audio_tokens