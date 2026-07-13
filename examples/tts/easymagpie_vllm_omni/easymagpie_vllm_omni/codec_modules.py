# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
"""Standalone (NeMo-free) re-implementation of the EasyMagpie codec decode path.

These modules are a faithful, inference-only port of the decode-side pieces of
``nemo.collections.tts.modules.audio_codec_modules`` (and the activations in
``nemo.collections.common.parts.utils``). They let the Stage-1 code2wav worker
decode acoustic tokens to a waveform with **only vLLM / vLLM-Omni installed** --
no ``nemo`` import at serve time.

Scope (decode only):

* ``FiniteScalarQuantizer`` / ``GroupFiniteScalarQuantizer`` -- FSQ dequantize +
  ``codes_to_indices`` (both are parameter-free; fully defined by ``num_levels``).
* ``VectorQuantizerIndexConverter`` -- lossless remap between the talker's FSQ
  grouping and the codec's native FSQ grouping.
* ``ResNetDecoder`` (+ ``CausalConv1dNorm`` / ``CausalConvTranspose1dNorm`` /
  ``Conv1dNorm`` / ``ConvTranspose1dNorm`` / ``ResidualBlockV2``) -- the causal
  audio decoder.
* ``VendoredAudioCodec`` -- thin container exposing ``decode(tokens, tokens_len)``
  with the same signature as ``AudioCodecModel.decode`` so the existing
  ``_EasyMagpieCodecDecoder`` glue and CUDA-graph wrapper work unchanged.

Differences from NeMo that are intentional and safe for inference:

* No ``NeuralModule`` / ``typecheck`` / ``NeuralType`` -- plain ``nn.Module``.
* No ``weight_norm`` parametrization -- weights are exported *after*
  ``remove_weight_norm()`` at conversion time, so the convs hold plain
  ``conv.weight`` / ``conv.bias`` tensors.
* ``CausalConv1dNorm`` computes its pad amounts on the host (python ints derived
  from the static input length), so it is CUDA-graph-capture-safe by
  construction (no device->host ``.item()`` sync). This removes the need for the
  runtime ``_patch_codec_for_cudagraph`` monkey-patch on this path.
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers / activations (ported from nemo.collections.common.parts.utils)
# ---------------------------------------------------------------------------
def mask_sequence_tensor(tensor: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Zero out out-of-bound time steps given per-example ``lengths``.

    Supports tensors of shape ``(B, L)``, ``(B, D, L)`` and ``(B, D1, D2, L)``.
    """
    batch_size, *_, max_lengths = tensor.shape
    if tensor.dim() == 2:
        mask = torch.ones(batch_size, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= lengths.view(batch_size, 1)
    elif tensor.dim() == 3:
        mask = torch.ones(batch_size, 1, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= lengths.view(batch_size, 1, 1)
    elif tensor.dim() == 4:
        mask = torch.ones(batch_size, 1, 1, max_lengths, dtype=lengths.dtype, device=lengths.device).cumsum(dim=-1)
        mask = mask <= lengths.view(batch_size, 1, 1, 1)
    else:
        raise ValueError("Can only mask tensors of shape B x L, B x D x L and B x D1 x D2 x L")
    return tensor * mask


@torch.jit.script
def snake(x: torch.Tensor, alpha: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Snake activation: ``x + (alpha + eps)^-1 * sin(alpha * x)^2``."""
    shape = x.shape
    x = x.reshape(shape[0], shape[1], -1)
    x = x + (alpha + eps).reciprocal() * torch.sin(alpha * x).pow(2)
    x = x.reshape(shape)
    return x


class Snake(nn.Module):
    """Snake activation function (https://arxiv.org/abs/2006.08195)."""

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return snake(x, self.alpha)


class HalfSnake(nn.Module):
    """Snake on the first half of channels, leaky-relu on the second half."""

    def __init__(self, channels: int):
        super().__init__()
        self.snake_channels = channels // 2
        self.snake_act = Snake(self.snake_channels)
        self.lrelu = torch.nn.LeakyReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        snake_out = self.snake_act(x[:, : self.snake_channels, :])
        lrelu_out = self.lrelu(x[:, self.snake_channels :, :])
        return torch.cat([snake_out, lrelu_out], dim=1)


class ClampActivation(nn.Module):
    def __init__(self, min_value: float = -1.0, max_value: float = 1.0, clamp_training: bool = True):
        super().__init__()
        self.min_value = min_value
        self.max_value = max_value
        self.clamp_training = clamp_training

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and not self.clamp_training:
            return x
        return torch.clamp(x, min=self.min_value, max=self.max_value)


class CodecActivation(nn.Module):
    """Select an activation by name (matches NeMo's ``CodecActivation``)."""

    def __init__(self, activation: str = "elu", channels: int = 1):
        super().__init__()
        activation = activation.lower()
        if activation == "elu":
            self.activation = nn.ELU()
        elif activation == "lrelu":
            self.activation = torch.nn.LeakyReLU()
        elif activation == "snake":
            self.activation = Snake(channels)
        elif activation == "half_snake":
            self.activation = HalfSnake(channels)
        else:
            raise ValueError(f"Unknown activation {activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


def get_padding(kernel_size: int, dilation: int = 1) -> int:
    return (kernel_size * dilation - dilation) // 2


def get_up_sample_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    output_padding = (kernel_size - stride) % 2
    padding = (kernel_size - stride + 1) // 2
    return padding, output_padding


# ---------------------------------------------------------------------------
# convolution blocks (ported from audio_codec_modules, weight-norm-free)
# ---------------------------------------------------------------------------
class CausalConv1dNorm(nn.Module):
    """Causal Conv1d. Pad amounts are computed on the host (capture-safe)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        activation: Optional[str] = None,
        pad_mode: str = "zeros",
        extra_pad_mode: str = "constant",
        bias: bool = True,
    ):
        super().__init__()
        self.extra_pad_mode = extra_pad_mode
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=pad_mode,
        )
        self.activation = CodecActivation(activation=activation, channels=out_channels) if activation else nn.Identity()

        # Static conv geometry as python ints -> no device sync in forward.
        eff_kernel_size = (self.conv.kernel_size[0] - 1) * self.conv.dilation[0] + 1
        self.kernel_size = int(eff_kernel_size)
        self.stride = int(self.conv.stride[0])
        self.padding_total = int(eff_kernel_size - self.conv.stride[0])

    @staticmethod
    def _pad1d(hidden_states: torch.Tensor, paddings: Tuple[int, int], mode: str = "constant", value: float = 0.0):
        length = hidden_states.shape[-1]
        padding_left, padding_right = paddings
        if mode != "reflect":
            return F.pad(hidden_states, paddings, mode, value)
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            hidden_states = F.pad(hidden_states, (0, extra_pad))
        padded = F.pad(hidden_states, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        length = int(inputs.shape[-1])
        # ceil((L - k + p) / s) == NeMo's ceil((L - k + p)/s + 1) - 1
        n_frames = math.ceil((length - self.kernel_size + self.padding_total) / self.stride)
        ideal_length = n_frames * self.stride + self.kernel_size - self.padding_total
        extra_padding = ideal_length - length

        hidden_states = self._pad1d(inputs, (self.padding_total, extra_padding), mode=self.extra_pad_mode)
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = mask_sequence_tensor(hidden_states, input_len)
        return hidden_states


class CausalConvTranspose1dNorm(nn.Module):
    """Causal ConvTranspose1d with right-trimming."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = None,
        activation: Optional[str] = None,
        trim_right_ratio: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.trim_right_ratio = trim_right_ratio
        groups = out_channels if groups is None else groups
        self.conv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, groups=groups, bias=bias)
        self.activation = CodecActivation(activation=activation, channels=out_channels) if activation else nn.Identity()

        kernel_size = self.conv.kernel_size[0]
        stride = self.conv.stride[0]
        padding_total = kernel_size - stride
        self.padding_right = math.ceil(padding_total * self.trim_right_ratio)
        self.padding_left = padding_total - self.padding_right

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        hidden_states = self.conv(inputs)
        end = hidden_states.shape[-1] - self.padding_right
        hidden_states = hidden_states[..., self.padding_left : end]
        hidden_states = self.activation(hidden_states)
        hidden_states = mask_sequence_tensor(hidden_states, input_len)
        return hidden_states


class Conv1dNorm(nn.Module):
    """Non-causal Conv1d (kept for fidelity; unused when the codec is causal)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        padding: Optional[int] = None,
        pad_mode: str = "reflect",
        activation: Optional[str] = None,
    ):
        super().__init__()
        if not padding:
            padding = get_padding(kernel_size=kernel_size, dilation=dilation)
        self.conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            padding_mode=pad_mode,
        )
        self.activation = CodecActivation(activation=activation, channels=out_channels) if activation else nn.Identity()

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.activation(out)
        out = mask_sequence_tensor(out, input_len)
        return out


class ConvTranspose1dNorm(nn.Module):
    """Non-causal ConvTranspose1d (kept for fidelity; unused when causal)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        groups: int = 1,
        activation: Optional[str] = None,
    ):
        super().__init__()
        padding, output_padding = get_up_sample_padding(kernel_size, stride)
        self.conv = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            padding_mode="zeros",
            groups=groups,
        )
        self.activation = CodecActivation(activation=activation, channels=out_channels) if activation else nn.Identity()

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        out = self.conv(inputs)
        out = self.activation(out)
        out = mask_sequence_tensor(out, input_len)
        return out


class ResidualBlockV2(nn.Module):
    """Residual block applying the activation to the output (NeMo ``ResidualBlockV2``)."""

    def __init__(
        self,
        channels: int,
        filters: int,
        kernel_size: int = 3,
        activation: str = "lrelu",
        is_causal: bool = False,
        pad_mode: str = "reflect",
    ):
        super().__init__()
        if not is_causal:
            self.input_conv = Conv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.skip_conv = Conv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )
        else:
            self.input_conv = CausalConv1dNorm(
                in_channels=channels,
                out_channels=filters,
                kernel_size=kernel_size,
                activation=activation,
                pad_mode=pad_mode,
            )
            self.skip_conv = CausalConv1dNorm(
                in_channels=filters, out_channels=channels, kernel_size=kernel_size, pad_mode=pad_mode
            )
        self.output_activation = CodecActivation(activation=activation, channels=channels)

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        res = self.input_conv(inputs=inputs, input_len=input_len)
        res = self.skip_conv(inputs=res, input_len=input_len)
        out = inputs + res
        out = self.output_activation(out)
        out = mask_sequence_tensor(out, lengths=input_len)
        return out


class ResNetDecoder(nn.Module):
    """Low-latency residual decoder (NeMo ``ResNetDecoder``), inference-only."""

    def __init__(
        self,
        input_dim: int,
        input_filters: int,
        pre_up_sample_rates: List[int],
        pre_up_sample_filters: List[int],
        n_hidden_layers: int,
        hidden_filters: int,
        resblock_up_sample_rates: List[int],
        resblock_up_sample_filters: List[int],
        resblock_up_sample_kernel_size: int = 7,
        kernel_size: int = 3,
        activation: str = "half_snake",
        is_causal: bool = False,
        pad_mode: str = "replicate",
    ):
        super().__init__()
        assert len(pre_up_sample_rates) == len(pre_up_sample_filters)
        assert len(resblock_up_sample_rates) == len(resblock_up_sample_filters)

        conv_class = CausalConv1dNorm if is_causal else Conv1dNorm
        conv_transpose_class = CausalConvTranspose1dNorm if is_causal else ConvTranspose1dNorm

        self.pre_conv = conv_class(in_channels=input_dim, out_channels=input_filters, kernel_size=kernel_size)

        in_channels = input_filters
        self.pre_up_sample_rates = pre_up_sample_rates
        self.pre_resblocks = nn.ModuleList([])
        self.pre_up_sample_layers = nn.ModuleList([])
        for up_sample_rate, filters in zip(self.pre_up_sample_rates, pre_up_sample_filters):
            self.pre_resblocks.append(
                ResidualBlockV2(
                    channels=in_channels,
                    filters=(2 * in_channels),
                    kernel_size=kernel_size,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
            )
            self.pre_up_sample_layers.append(
                conv_transpose_class(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=(2 * up_sample_rate),
                    stride=up_sample_rate,
                    activation=activation,
                )
            )
            in_channels = filters

        self.conv_layers = nn.ModuleList(
            [
                ResidualBlockV2(
                    channels=in_channels,
                    filters=hidden_filters,
                    kernel_size=kernel_size,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
                for _ in range(n_hidden_layers)
            ]
        )

        self.resblock_up_sample_rates = resblock_up_sample_rates
        self.resblock_up_sample_layers = nn.ModuleList([])
        self.resblocks = nn.ModuleList([])
        for up_sample_rate, filters in zip(self.resblock_up_sample_rates, resblock_up_sample_filters):
            self.resblock_up_sample_layers.append(
                conv_transpose_class(
                    in_channels=in_channels,
                    out_channels=filters,
                    kernel_size=(2 * up_sample_rate),
                    stride=up_sample_rate,
                    activation=activation,
                )
            )
            self.resblocks.append(
                ResidualBlockV2(
                    channels=filters,
                    filters=(2 * filters),
                    kernel_size=resblock_up_sample_kernel_size,
                    activation=activation,
                    is_causal=is_causal,
                    pad_mode=pad_mode,
                )
            )
            in_channels = filters

        self.post_conv = conv_class(
            in_channels=in_channels, out_channels=1, kernel_size=resblock_up_sample_kernel_size, pad_mode=pad_mode
        )
        self.out_activation = ClampActivation(clamp_training=False)

    def forward(self, inputs: torch.Tensor, input_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out = self.pre_conv(inputs=inputs, input_len=input_len)

        audio_len = input_len
        for pre_up_sample_rate, pre_up_sample_layer, pre_resblock in zip(
            self.pre_up_sample_rates, self.pre_up_sample_layers, self.pre_resblocks
        ):
            out = pre_resblock(inputs=out, input_len=audio_len)
            audio_len = pre_up_sample_rate * audio_len
            out = pre_up_sample_layer(inputs=out, input_len=audio_len)

        for conv in self.conv_layers:
            out = conv(inputs=out, input_len=audio_len)

        for resblock_up_sample_rate, resblock_up_sample_layer, resblock in zip(
            self.resblock_up_sample_rates, self.resblock_up_sample_layers, self.resblocks
        ):
            audio_len = resblock_up_sample_rate * audio_len
            out = resblock_up_sample_layer(inputs=out, input_len=audio_len)
            out = resblock(inputs=out, input_len=audio_len)

        out = self.post_conv(inputs=out, input_len=audio_len)
        out = out.squeeze(1)  # (B, 1, T) -> (B, T)
        audio = self.out_activation(out)
        audio = mask_sequence_tensor(audio, audio_len)
        return audio, audio_len


# ---------------------------------------------------------------------------
# finite scalar quantizers (parameter-free; ported from audio_codec_modules)
# ---------------------------------------------------------------------------
class FiniteScalarQuantizer(nn.Module):
    """Finite Scalar Quantization (https://arxiv.org/abs/2309.15505), decode side."""

    def __init__(self, num_levels: List[int], eps: float = 1e-3):
        super().__init__()
        dim_base_index = torch.cumprod(torch.tensor([1] + list(num_levels[:-1])), dim=0, dtype=torch.int32)
        self.register_buffer("dim_base_index", dim_base_index.view(1, -1, 1))
        num_levels_t = torch.tensor(num_levels, dtype=torch.int32).view(1, -1, 1)
        self.register_buffer("num_levels", num_levels_t)
        self.eps = eps

    @property
    def num_codebooks(self) -> int:
        return 1

    @property
    def codebook_size(self) -> int:
        return int(self.num_levels.prod().item())

    @property
    def dim(self) -> int:
        return self.num_levels.numel()

    def codes_to_nonnegative(self, codes: torch.Tensor) -> torch.Tensor:
        scale = offset = self.num_levels // 2
        return scale * codes + offset

    def nonnegative_to_codes(self, codes_nonnegative: torch.Tensor) -> torch.Tensor:
        scale = offset = self.num_levels // 2
        return (codes_nonnegative - offset) / scale

    def codes_to_indices(self, codes: torch.Tensor) -> torch.Tensor:
        """Convert a code vector ``(B, D, T)`` to a single index ``(B, T)``."""
        if codes.size(1) != self.dim:
            raise RuntimeError(
                f"Input code dimension {codes.size(1)} != expected {self.dim}, codes shape {codes.shape}"
            )
        indices = self.codes_to_nonnegative(codes)
        indices = torch.sum(indices * self.dim_base_index, dim=1)
        return indices.to(torch.int32)

    def decode(self, indices: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert a single index ``(1, B, T)`` to a continuous code vector ``(B, D, T)``."""
        if indices.size(0) > 1:
            raise ValueError(f"Expected a single codebook, got {indices.size(0)} for indices {indices.shape}.")
        indices = indices.permute(1, 0, 2)  # (D=1, B, T) -> (B, 1, T)
        codes_nonnegative = (indices // self.dim_base_index) % self.num_levels
        dequantized = self.nonnegative_to_codes(codes_nonnegative)
        if input_len is not None:
            dequantized = mask_sequence_tensor(dequantized, input_len)
        return dequantized


class GroupFiniteScalarQuantizer(nn.Module):
    """Split the input into groups and apply FSQ per group (decode side)."""

    def __init__(self, num_groups: int, num_levels_per_group: List[int], **kwargs):
        super().__init__()
        self.num_groups = num_groups
        self.codebook_dim_per_group = len(num_levels_per_group)
        self.fsqs = nn.ModuleList(
            [FiniteScalarQuantizer(num_levels=num_levels_per_group, **kwargs) for _ in range(self.num_groups)]
        )

    @property
    def num_codebooks(self) -> int:
        return self.num_groups

    @property
    def codebook_size(self) -> int:
        return self.fsqs[0].codebook_size

    @property
    def codebook_dim(self) -> int:
        return self.codebook_dim_per_group * self.num_groups

    def decode(self, indices: torch.Tensor, input_len: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode ``(C, B, T)`` indices into ``(B, D, T)`` dequantized codes."""
        indices_grouped = indices.chunk(self.num_groups, dim=0)
        dequantized = [fsq.decode(indices=g, input_len=input_len) for g, fsq in zip(indices_grouped, self.fsqs)]
        return torch.cat(dequantized, dim=1)

    def codes_to_indices(self, codes: torch.Tensor, input_len: torch.Tensor) -> torch.Tensor:
        """Convert ``(B, D, T)`` codes into ``(B, C, T)`` per-group indices."""
        codes_rearrange = codes.permute(1, 0, 2)  # (B, D, T) -> (D, B, T)
        codes_grouped = codes_rearrange.chunk(self.num_groups, dim=0)
        indices = []
        for codes_group, fsq_group in zip(codes_grouped, self.fsqs):
            codes_group_rearrange = codes_group.permute(1, 0, 2)  # (D, B, T) -> (B, D, T)
            indices_group = fsq_group.codes_to_indices(codes=codes_group_rearrange)
            indices_group = mask_sequence_tensor(indices_group, input_len)
            indices.append(indices_group)
        return torch.stack(indices, dim=1)


class VectorQuantizerIndexConverter(nn.Module):
    """Losslessly remap indices between two FSQ definitions (decode side)."""

    def __init__(self, vector_quantizer_original: nn.Module, vector_quantizer_new: nn.Module):
        super().__init__()
        self.vector_quantizer_original = vector_quantizer_original
        self.vector_quantizer_new = vector_quantizer_new

    def convert_new_to_original(self, audio_tokens: torch.Tensor, audio_lens: torch.Tensor) -> torch.Tensor:
        """``(B, C_new, T)`` -> ``(B, C_original, T)``."""
        audio_tokens_rearrange = audio_tokens.permute(1, 0, 2)  # (B, C, T) -> (C, B, T)
        audio_codes = self.vector_quantizer_new.decode(indices=audio_tokens_rearrange, input_len=audio_lens)
        return self.vector_quantizer_original.codes_to_indices(codes=audio_codes, input_len=audio_lens)


class VendoredAudioCodec(nn.Module):
    """Minimal decode-only stand-in for NeMo's ``AudioCodecModel``.

    Exposes ``decode(tokens, tokens_len)`` with the same signature so the
    existing ``_EasyMagpieCodecDecoder`` glue works unchanged. ``tokens`` are the
    codec's *native* FSQ indices ``(B, C_original, T)``.
    """

    def __init__(self, vector_quantizer: nn.Module, audio_decoder: nn.Module, output_sample_rate: int):
        super().__init__()
        self.vector_quantizer = vector_quantizer
        self.audio_decoder = audio_decoder
        self.output_sample_rate = int(output_sample_rate)

    def dequantize(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> torch.Tensor:
        tokens = tokens.permute(1, 0, 2)  # (B, C, T) -> (C, B, T)
        return self.vector_quantizer.decode(indices=tokens, input_len=tokens_len)

    def decode(self, tokens: torch.Tensor, tokens_len: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dequantized = self.dequantize(tokens=tokens, tokens_len=tokens_len)
        # FSQ dequantize is an int/int true-division whose result dtype follows the
        # ambient ``torch.get_default_dtype()`` -- which vLLM sets to the model dtype
        # (e.g. float16) during init. Cast to the decoder's weight dtype so the conv
        # inputs always match its (fp32) weights, matching NeMo's decode() behaviour.
        dequantized = dequantized.to(self._decoder_dtype)
        audio, audio_len = self.audio_decoder(inputs=dequantized, input_len=tokens_len)
        return audio, audio_len

    @property
    def _decoder_dtype(self) -> torch.dtype:
        return next(self.audio_decoder.parameters()).dtype
