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
"""Compatibility shims for running vLLM-Omni against older vLLM builds."""

from __future__ import annotations

import builtins
import contextlib
import dataclasses
import importlib
import inspect
import logging
import os
import queue as stdlib_queue
import random
import sys
import types
from collections import defaultdict
from typing import Any, Literal

_INSTALLED = False
_IMPORT_PATCH_FLAG = "_easymagpie_vllm_compat_import_patch_installed"
logger = logging.getLogger(__name__)

try:
    import torch as _TORCH_MODULE
except Exception:
    _TORCH_MODULE = None

try:
    from vllm.attention.backends.flash_attn import FlashAttentionMetadata as _LEGACY_FLASH_ATTENTION_METADATA_CLS
except Exception:
    _LEGACY_FLASH_ATTENTION_METADATA_CLS = None

try:
    from vllm.attention.backends.xformers import XFormersMetadata as _XFORMERS_METADATA_CLS
except Exception:
    _XFORMERS_METADATA_CLS = None

try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata as _V1_FLASH_ATTENTION_METADATA_CLS
except Exception:
    _V1_FLASH_ATTENTION_METADATA_CLS = None

try:
    from vllm.v1.attention.backends.triton_attn import TritonAttentionMetadata as _V1_TRITON_ATTENTION_METADATA_CLS
except Exception:
    _V1_TRITON_ATTENTION_METADATA_CLS = None

try:
    from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata as _V1_MAMBA2_ATTENTION_METADATA_CLS
except Exception:
    _V1_MAMBA2_ATTENTION_METADATA_CLS = None

_SUPPORTED_KWARGS_CACHE: dict[int, set[str] | None] = {}


def _as_v1_kv_cache_config(cache_config: Any) -> Any:
    if cache_config is None or hasattr(cache_config, "kv_cache_groups"):
        return cache_config
    cached = getattr(cache_config, "_easymagpie_v1_kv_cache_config", None)
    if cached is None:
        try:
            from vllm.v1.kv_cache_interface import KVCacheConfig

            cached = KVCacheConfig(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
        except Exception:
            cached = types.SimpleNamespace(num_blocks=1, kv_cache_tensors=[], kv_cache_groups=[])
        try:
            cache_config._easymagpie_v1_kv_cache_config = cached
        except Exception:
            pass
    return cached


def _buffer_slice_to_int_list(buffer: Any, length: int) -> list[int] | None:
    if buffer is None:
        return None
    candidates = (
        getattr(buffer, "cpu", None),
        getattr(buffer, "np", None),
        getattr(buffer, "gpu", None),
        buffer,
    )
    for candidate in candidates:
        if candidate is None:
            continue
        try:
            values = candidate[:length]
            if hasattr(values, "detach"):
                values = values.detach().cpu().tolist()
            elif hasattr(values, "tolist"):
                values = values.tolist()
            return [int(v) for v in values]
        except Exception:
            continue
    return None


def _balanced_profile_lengths(num_tokens: int, num_reqs: int) -> list[int]:
    num_reqs = max(1, int(num_reqs))
    base = max(1, int(num_tokens) // num_reqs)
    remainder = max(0, int(num_tokens) - base * num_reqs)
    lengths = [base] * num_reqs
    lengths[-1] += remainder
    return lengths


def _cache_supported_kwargs(callable_obj: Any) -> set[str] | None:
    cache_key = id(callable_obj)
    if cache_key in _SUPPORTED_KWARGS_CACHE:
        return _SUPPORTED_KWARGS_CACHE[cache_key]
    try:
        signature = inspect.signature(callable_obj)
    except Exception:
        _SUPPORTED_KWARGS_CACHE[cache_key] = None
        return None
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        _SUPPORTED_KWARGS_CACHE[cache_key] = None
        return None
    supported = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    _SUPPORTED_KWARGS_CACHE[cache_key] = supported
    return supported


def _prime_supported_kwargs_cache() -> None:
    for callable_obj in (
        _LEGACY_FLASH_ATTENTION_METADATA_CLS,
        _XFORMERS_METADATA_CLS,
        _V1_FLASH_ATTENTION_METADATA_CLS,
        _V1_TRITON_ATTENTION_METADATA_CLS,
        _V1_MAMBA2_ATTENTION_METADATA_CLS,
    ):
        if callable_obj is not None:
            _cache_supported_kwargs(callable_obj)


def _call_with_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> Any:
    supported = _cache_supported_kwargs(callable_obj)
    if supported is None:
        return callable_obj(**kwargs)
    return callable_obj(**{key: value for key, value in kwargs.items() if key in supported})


_prime_supported_kwargs_cache()


def _max_int_sequence(values: Any) -> int:
    max_value = 0
    for value in values:
        int_value = int(value)
        if int_value > max_value:
            max_value = int_value
    return max_value


def _is_torch_dynamo_compiling() -> bool:
    try:
        torch_module = _TORCH_MODULE
        dynamo = getattr(torch_module, "_dynamo", None)
        is_compiling = getattr(dynamo, "is_compiling", None)
        return bool(is_compiling is not None and is_compiling())
    except Exception:
        return False


def _attach_profile_attention_metadata_attrs(
    metadata: Any,
    *,
    num_tokens: int,
    num_reqs: int,
    query_lens: list[int],
    seq_lens: list[int],
    seq_lens_tensor: Any,
    seq_start_loc: Any,
    query_start_loc: Any = None,
    context_lens_tensor: Any = None,
    block_tables: Any = None,
    slot_mapping: Any = None,
    max_query_len: int | None = None,
) -> Any:
    if query_start_loc is None:
        query_start_loc = seq_start_loc
    resolved_max_query_len = max(int(max_query_len or 0), _max_int_sequence(query_lens))
    for name, value in (
        ("num_prefills", int(num_reqs)),
        ("num_prefill_tokens", int(num_tokens)),
        ("num_decodes", 0),
        ("num_decode_tokens", 0),
        ("query_lens", list(query_lens)),
        ("seq_lens", list(seq_lens)),
        ("seq_lens_tensor", seq_lens_tensor),
        ("seq_start_loc", seq_start_loc),
        ("query_start_loc", query_start_loc),
        ("max_query_len", resolved_max_query_len),
        ("max_prefill_seq_len", _max_int_sequence(seq_lens)),
        ("max_decode_query_len", 0),
        ("max_decode_seq_len", 0),
        ("context_lens_tensor", context_lens_tensor),
        ("block_tables", block_tables),
        ("slot_mapping", slot_mapping),
        ("prefill_metadata", metadata),
        ("decode_metadata", None),
    ):
        try:
            setattr(metadata, name, value)
        except Exception:
            pass
    return metadata


def _is_mamba2_attention_metadata(value: Any) -> bool:
    common = all(
        hasattr(value, attr)
        for attr in (
            "num_prefills",
            "num_prefill_tokens",
            "num_decodes",
            "num_decode_tokens",
            "seq_lens",
            "chunk_size",
        )
    )
    if not common:
        return False
    legacy_state = all(hasattr(value, attr) for attr in ("query_start_loc", "state_indices_tensor"))
    split_state = all(
        hasattr(value, attr)
        for attr in ("query_start_loc_p", "state_indices_tensor_p", "query_start_loc_d", "state_indices_tensor_d")
    )
    return legacy_state or split_state


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _legacy_mamba2_attention_metadata_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return {
        "num_prefills": kwargs.get("num_prefills", 0),
        "num_prefill_tokens": kwargs.get("num_prefill_tokens", 0),
        "num_decodes": kwargs.get("num_decodes", 0),
        "num_decode_tokens": kwargs.get("num_decode_tokens", 0),
        "query_start_loc": _first_not_none(
            kwargs.get("query_start_loc"),
            kwargs.get("legacy_query_start_loc"),
            kwargs.get("query_start_loc_p"),
            kwargs.get("query_start_loc_d"),
        ),
        "seq_lens": kwargs["seq_lens"],
        "prep_initial_states": kwargs.get("prep_initial_states", False),
        "chunk_size": kwargs.get("chunk_size", 0),
        "has_initial_states_p": kwargs.get("has_initial_states_p"),
        "seq_idx_p": kwargs.get("seq_idx_p"),
        "chunk_indices_p": kwargs.get("chunk_indices_p"),
        "chunk_offsets_p": kwargs.get("chunk_offsets_p"),
        "state_indices_tensor": _first_not_none(
            kwargs.get("state_indices_tensor"),
            kwargs.get("legacy_state_indices_tensor"),
            kwargs.get("state_indices_tensor_p"),
            kwargs.get("state_indices_tensor_d"),
        ),
        "nums_dict": kwargs.get("nums_dict"),
        "cu_seqlen": kwargs.get("cu_seqlen"),
        "batch_ptr": kwargs.get("batch_ptr"),
        "token_chunk_offset_ptr": kwargs.get("token_chunk_offset_ptr"),
    }


def _filter_cached_supported_kwargs(metadata_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    supported = _cache_supported_kwargs(metadata_cls)
    if supported is None:
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in supported}


def _new_mamba2_attention_metadata(metadata_cls: Any, kwargs: dict[str, Any]) -> Any:
    constructor_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key not in {"legacy_query_start_loc", "legacy_state_indices_tensor"}
    }
    supported = _cache_supported_kwargs(metadata_cls)
    if supported is not None and "query_start_loc_p" not in supported and "query_start_loc" in supported:
        return metadata_cls(**_filter_cached_supported_kwargs(metadata_cls, _legacy_mamba2_attention_metadata_kwargs(kwargs)))
    try:
        return metadata_cls(**_filter_cached_supported_kwargs(metadata_cls, constructor_kwargs))
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" in message and "query_start_loc_p" in kwargs:
            legacy_kwargs = _legacy_mamba2_attention_metadata_kwargs(kwargs)
            try:
                return metadata_cls(**_filter_cached_supported_kwargs(metadata_cls, legacy_kwargs))
            except TypeError as legacy_exc:
                if "num_reqs" not in kwargs or "num_reqs" not in str(legacy_exc):
                    raise
                legacy_kwargs["num_reqs"] = kwargs["num_reqs"]
                return metadata_cls(**_filter_cached_supported_kwargs(metadata_cls, legacy_kwargs))
        if "num_reqs" not in kwargs or "num_reqs" not in message:
            raise
        fallback_kwargs = _filter_cached_supported_kwargs(metadata_cls, constructor_kwargs)
        fallback_kwargs.pop("num_reqs", None)
        return metadata_cls(**fallback_kwargs)


def _resize_mamba_state_indices(state_indices: Any, needed_state_indices: int, device: Any) -> Any:
    import torch

    if getattr(state_indices, "numel", lambda: 0)() > needed_state_indices:
        return state_indices.to(device=device, dtype=torch.int32)[:needed_state_indices]

    current_numel = int(getattr(state_indices, "numel", lambda: 0)())
    if current_numel > 0:
        base = state_indices.to(device=device, dtype=torch.int32).reshape(-1)
        repeats = (needed_state_indices + current_numel - 1) // current_numel
        return base.repeat(repeats)[:needed_state_indices]

    return torch.arange(needed_state_indices, dtype=torch.int32, device=device)


def _replace_mamba2_attention_metadata_fields(value: Any, replacements: dict[str, Any]) -> Any:
    if not replacements:
        return value
    if _is_torch_dynamo_compiling():
        return value
    try:
        if dataclasses.is_dataclass(value):
            supported_fields = getattr(type(value), "__dataclass_fields__", {})
            supported_replacements = {
                key: replacement
                for key, replacement in replacements.items()
                if not supported_fields or key in supported_fields
            }
            if supported_replacements:
                return dataclasses.replace(value, **supported_replacements)
    except Exception:
        pass
    fields = dict(getattr(value, "__dict__", {}))
    fields.update(replacements)
    return types.SimpleNamespace(**fields)


def _balanced_tensor_lengths(total: int, count: int, device: Any) -> Any:
    import torch

    count = max(0, int(count))
    if count <= 0:
        return torch.empty(0, dtype=torch.int32, device=device)
    base, extra = divmod(max(0, int(total)), count)
    lengths = torch.full((count,), base, dtype=torch.int32, device=device)
    if extra:
        lengths[:extra] += 1
    return lengths


def _infer_mamba2_prefill_query_start_loc(value: Any, device: Any) -> Any:
    import torch

    num_prefills = int(getattr(value, "num_prefills", 0) or 0)
    num_prefill_tokens = int(getattr(value, "num_prefill_tokens", 0) or 0)
    if num_prefills <= 0:
        return torch.zeros(1, dtype=torch.int32, device=device)

    candidates = []
    query_start_loc_p = getattr(value, "query_start_loc_p", None)
    if query_start_loc_p is not None and hasattr(query_start_loc_p, "numel"):
        candidates.append(query_start_loc_p)

    query_start_loc = getattr(value, "query_start_loc", None)
    if query_start_loc is not None and hasattr(query_start_loc, "numel"):
        query_start_loc = query_start_loc.to(device=device, dtype=torch.int64)
        if int(query_start_loc.numel()) >= num_prefills + 1:
            num_decode_tokens = int(getattr(value, "num_decode_tokens", 0) or 0)
            candidates.append(query_start_loc[-num_prefills - 1 :] - num_decode_tokens)
            candidates.append(query_start_loc[: num_prefills + 1])

    for candidate in candidates:
        try:
            candidate = candidate.to(device=device, dtype=torch.int64)
            if int(candidate.numel()) != num_prefills + 1:
                continue
            if _is_torch_dynamo_compiling():
                return candidate.to(dtype=torch.int32)
            if int(candidate[0].item()) != 0 or int(candidate[-1].item()) != num_prefill_tokens:
                continue
            lengths = candidate[1:] - candidate[:-1]
            if bool((lengths >= 0).all().item()):
                return candidate.to(dtype=torch.int32)
        except Exception:
            continue

    lengths = _balanced_tensor_lengths(num_prefill_tokens, num_prefills, device)
    query_start_loc_p = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
    if num_prefills > 0:
        query_start_loc_p[1:] = torch.cumsum(lengths, dim=0)
    return query_start_loc_p


def _uses_mamba2_varlen_chunk_metadata(value: Any) -> bool:
    return hasattr(value, "cu_chunk_seqlen_p") or hasattr(value, "last_chunk_indices_p")


def _compute_mamba2_varlen_chunk_metadata(query_start_loc: Any, chunk_size: int) -> tuple[Any, Any, Any]:
    import torch

    try:
        module = importlib.import_module("vllm.v1.attention.backends.mamba2_attn")
        compute = getattr(module, "compute_varlen_chunk_metadata", None)
        if compute is not None:
            return compute(query_start_loc, chunk_size)
    except Exception:
        pass

    query_start_loc = query_start_loc.to(dtype=torch.int64)
    device = query_start_loc.device
    if int(query_start_loc.numel()) == 0:
        zero = torch.tensor([0], dtype=torch.int32, device=device)
        return zero, torch.empty(0, dtype=torch.int32, device=device), torch.empty(0, dtype=torch.int32, device=device)

    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0:
        chunk_size = max(1, int(query_start_loc[-1].item()))

    starts = query_start_loc[:-1].detach().cpu().tolist()
    ends = query_start_loc[1:].detach().cpu().tolist()
    chunk_lens: list[int] = []
    seq_idx_chunks: list[int] = []
    last_chunk_indices: list[int] = [-1] * len(starts)

    for seq_idx, (start, end) in enumerate(zip(starts, ends)):
        pos = int(start)
        end = int(end)
        while pos < end:
            room = chunk_size - (pos % chunk_size)
            take = min(room, end - pos)
            if take <= 0:
                break
            chunk_lens.append(int(take))
            seq_idx_chunks.append(seq_idx)
            last_chunk_indices[seq_idx] = len(chunk_lens) - 1
            pos += take

    boundaries = [0]
    for chunk_len in chunk_lens:
        boundaries.append(boundaries[-1] + chunk_len)
    cu_chunk_seqlens = torch.tensor(boundaries, dtype=torch.int32, device=device)
    last_chunk_indices_t = torch.tensor(last_chunk_indices, dtype=torch.int32, device=device)
    seq_idx_chunks_t = torch.tensor(seq_idx_chunks, dtype=torch.int32, device=device)
    return cu_chunk_seqlens, last_chunk_indices_t, seq_idx_chunks_t


def _repair_mamba2_attention_metadata_prefill_seq_idx(value: Any) -> Any:
    if not _is_mamba2_attention_metadata(value):
        return value
    try:
        import torch
    except Exception:
        return value

    num_prefills = int(getattr(value, "num_prefills", 0) or 0)
    num_prefill_tokens = int(getattr(value, "num_prefill_tokens", 0) or 0)
    if num_prefills <= 0 or num_prefill_tokens <= 0:
        return value
    if _is_torch_dynamo_compiling():
        return value

    seq_idx_p = getattr(value, "seq_idx_p", None)
    query_start_loc = getattr(value, "query_start_loc", None)
    query_start_loc_p = getattr(value, "query_start_loc_p", None)
    if seq_idx_p is None and not hasattr(query_start_loc, "numel") and not hasattr(query_start_loc_p, "numel"):
        return value

    device = getattr(seq_idx_p, "device", None)
    if device is None:
        device = getattr(query_start_loc, "device", None)
    if device is None:
        device = getattr(query_start_loc_p, "device", None)
    if device is None:
        device = getattr(getattr(value, "state_indices_tensor", None), "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_start_loc_p = _infer_mamba2_prefill_query_start_loc(value, device)
    lengths = (query_start_loc_p[1:] - query_start_loc_p[:-1]).to(dtype=torch.int32)
    if int(lengths.numel()) != num_prefills or int(lengths.sum().item()) != num_prefill_tokens:
        lengths = _balanced_tensor_lengths(num_prefill_tokens, num_prefills, device)
        query_start_loc_p = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
        query_start_loc_p[1:] = torch.cumsum(lengths, dim=0)

    if _uses_mamba2_varlen_chunk_metadata(value):
        cu_chunk_seqlens, last_chunk_indices, fixed_seq_idx_p = _compute_mamba2_varlen_chunk_metadata(
            query_start_loc_p,
            int(getattr(value, "chunk_size", 0) or 0),
        )
        expected_shape = (int(cu_chunk_seqlens.numel()) - 1,)
        current_cu_chunk_seqlens = getattr(value, "cu_chunk_seqlen_p", None)
        current_last_chunk_indices = getattr(value, "last_chunk_indices_p", None)
        if (
            seq_idx_p is not None
            and hasattr(seq_idx_p, "shape")
            and tuple(seq_idx_p.shape) == expected_shape
            and current_cu_chunk_seqlens is not None
            and hasattr(current_cu_chunk_seqlens, "shape")
            and tuple(current_cu_chunk_seqlens.shape) == tuple(cu_chunk_seqlens.shape)
            and torch.equal(current_cu_chunk_seqlens.to(device=device, dtype=torch.int32), cu_chunk_seqlens)
            and current_last_chunk_indices is not None
            and hasattr(current_last_chunk_indices, "shape")
            and tuple(current_last_chunk_indices.shape) == tuple(last_chunk_indices.shape)
            and torch.equal(current_last_chunk_indices.to(device=device, dtype=torch.int32), last_chunk_indices)
            and torch.equal(seq_idx_p.to(device=device, dtype=torch.int32), fixed_seq_idx_p)
        ):
            return value
        return _replace_mamba2_attention_metadata_fields(
            value,
            {
                "seq_idx_p": fixed_seq_idx_p,
                "cu_chunk_seqlen_p": cu_chunk_seqlens,
                "last_chunk_indices_p": last_chunk_indices,
            },
        )

    expected_shape = (1, num_prefill_tokens)
    if (
        seq_idx_p is not None
        and hasattr(seq_idx_p, "shape")
        and tuple(seq_idx_p.shape) == expected_shape
    ):
        return value

    fixed_seq_idx_p = torch.repeat_interleave(
        torch.arange(num_prefills, dtype=torch.int32, device=device),
        lengths,
        output_size=num_prefill_tokens,
    ).unsqueeze(0)

    replacements: dict[str, Any] = {"seq_idx_p": fixed_seq_idx_p}
    if bool(getattr(value, "prep_initial_states", False)):
        try:
            module = importlib.import_module("vllm.v1.attention.backends.mamba2_attn")
            converter = getattr(module, "_query_start_loc_to_chunk_indices_offsets")
            chunk_indices, chunk_offsets = converter(
                query_start_loc_p,
                int(getattr(value, "chunk_size", 0) or 0),
                num_prefill_tokens,
            )
            replacements["chunk_indices_p"] = chunk_indices
            replacements["chunk_offsets_p"] = chunk_offsets
        except Exception:
            pass
    return _replace_mamba2_attention_metadata_fields(value, replacements)


def _compact_legacy_mamba_seq_idx(seq_idx: Any, device: Any) -> Any:
    import torch

    seq_idx_flat = seq_idx.detach().reshape(-1).to(device="cpu", dtype=torch.int64)
    valid = seq_idx_flat >= 0
    if not bool(valid.all().item()):
        seq_idx_flat = seq_idx_flat[valid]
    if seq_idx_flat.numel() == 0:
        return seq_idx_flat.to(device=device, dtype=torch.int32)

    _, compact = torch.unique(seq_idx_flat, sorted=True, return_inverse=True)
    return compact.to(device=device, dtype=torch.int32)


def _mamba2_attention_metadata_from_mapping(value: dict[str, Any]) -> Any | None:
    required_fields = (
        "num_prefills",
        "num_prefill_tokens",
        "num_decodes",
        "num_decode_tokens",
        "query_start_loc",
        "seq_lens",
        "prep_initial_states",
        "chunk_size",
        "has_initial_states_p",
        "seq_idx_p",
        "chunk_indices_p",
        "chunk_offsets_p",
        "state_indices_tensor",
    )
    if not all(field in value for field in required_fields):
        return None
    try:
        import torch
        from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadata

        kwargs = {field: value[field] for field in required_fields if field in value}
        needed_state_indices = int(value.get("num_prefills") or 0) + int(value.get("num_decode_tokens") or 0)
        kwargs["num_reqs"] = int(value.get("num_reqs") or needed_state_indices or 1)
        state_indices = value.get("state_indices_tensor")
        if (state_indices is None or hasattr(state_indices, "numel")) and (
            getattr(state_indices, "numel", lambda: 0)() != needed_state_indices
        ):
            device = getattr(state_indices, "device", None)
            if device is None:
                device = getattr(value.get("query_start_loc"), "device", None)
            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            state_indices = _resize_mamba_state_indices(state_indices, needed_state_indices, device)
            kwargs["state_indices_tensor"] = state_indices
        return _new_mamba2_attention_metadata(Mamba2AttentionMetadata, kwargs)
    except Exception:
        return types.SimpleNamespace(**{field: value.get(field) for field in required_fields})


def _repair_mamba2_attention_metadata_state_indices(value: Any) -> Any:
    if not _is_mamba2_attention_metadata(value):
        return value
    try:
        import torch
    except Exception:
        return value
    if _is_torch_dynamo_compiling():
        return value
    value = _repair_mamba2_attention_metadata_prefill_seq_idx(value)
    if hasattr(value, "state_indices_tensor_p") or hasattr(value, "state_indices_tensor_d"):
        device = None
        state_indices_p = getattr(value, "state_indices_tensor_p", None)
        state_indices_d = getattr(value, "state_indices_tensor_d", None)
        for state_indices in (state_indices_p, state_indices_d, getattr(value, "query_start_loc_p", None), getattr(value, "query_start_loc_d", None)):
            device = getattr(state_indices, "device", None)
            if device is not None:
                break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        replacements: dict[str, Any] = {}
        needed_prefill_indices = int(getattr(value, "num_prefills", 0) or 0)
        if needed_prefill_indices > 0 and (
            state_indices_p is None
            or not hasattr(state_indices_p, "numel")
            or int(state_indices_p.numel()) != needed_prefill_indices
        ):
            replacements["state_indices_tensor_p"] = _resize_mamba_state_indices(
                state_indices_p,
                needed_prefill_indices,
                device,
            )
        needed_decode_indices = int(getattr(value, "num_decode_tokens", 0) or 0)
        if needed_decode_indices > 0 and (
            state_indices_d is None
            or not hasattr(state_indices_d, "numel")
            or int(state_indices_d.numel()) != needed_decode_indices
        ):
            replacements["state_indices_tensor_d"] = _resize_mamba_state_indices(
                state_indices_d,
                needed_decode_indices,
                device,
            )
        if not replacements:
            return value
        try:
            if dataclasses.is_dataclass(value):
                return dataclasses.replace(value, **replacements)
        except Exception:
            pass
        fields = dict(getattr(value, "__dict__", {}))
        fields.update(replacements)
        return types.SimpleNamespace(**fields)

    needed_state_indices = int(getattr(value, "num_prefills", 0) or 0) + int(
        getattr(value, "num_decode_tokens", 0) or 0
    )
    state_indices = getattr(value, "state_indices_tensor", None)
    if state_indices is not None and not hasattr(state_indices, "numel"):
        return value
    if getattr(state_indices, "numel", lambda: 0)() == needed_state_indices:
        return value

    device = getattr(state_indices, "device", None)
    if device is None:
        device = getattr(getattr(value, "query_start_loc", None), "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fixed_state_indices = _resize_mamba_state_indices(state_indices, needed_state_indices, device)

    try:
        if dataclasses.is_dataclass(value):
            return dataclasses.replace(value, state_indices_tensor=fixed_state_indices)
    except Exception:
        pass
    fields = dict(getattr(value, "__dict__", {}))
    fields["state_indices_tensor"] = fixed_state_indices
    return types.SimpleNamespace(**fields)


def _repair_mamba_cache_params_state_indices(cache_params: Any, attn_metadata: Any) -> Any:
    if cache_params is None or not _is_mamba2_attention_metadata(attn_metadata):
        return cache_params
    try:
        import torch
    except Exception:
        return cache_params
    needed_state_indices = int(getattr(attn_metadata, "num_prefills", 0) or 0) + int(
        getattr(attn_metadata, "num_decode_tokens", 0) or 0
    )
    state_indices = getattr(cache_params, "state_indices_tensor", None)
    if state_indices is not None and not hasattr(state_indices, "numel"):
        return cache_params

    device = getattr(state_indices, "device", None)
    if device is None:
        device = getattr(getattr(attn_metadata, "query_start_loc", None), "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    metadata_state_indices = getattr(attn_metadata, "state_indices_tensor", None)
    if (
        metadata_state_indices is None
        or not hasattr(metadata_state_indices, "numel")
        or int(metadata_state_indices.numel()) != needed_state_indices
    ):
        split_indices: list[Any] = []
        prefill_indices = getattr(attn_metadata, "state_indices_tensor_p", None)
        decode_indices = getattr(attn_metadata, "state_indices_tensor_d", None)
        if prefill_indices is not None and hasattr(prefill_indices, "numel") and int(prefill_indices.numel()) > 0:
            split_indices.append(prefill_indices)
        if decode_indices is not None and hasattr(decode_indices, "numel") and int(decode_indices.numel()) > 0:
            split_indices.append(decode_indices)
        if split_indices:
            try:
                metadata_state_indices = torch.cat(
                    [item.detach().reshape(-1).to(device=device, dtype=torch.int32) for item in split_indices],
                    dim=0,
                )
            except Exception:
                metadata_state_indices = None

    if (
        metadata_state_indices is not None
        and hasattr(metadata_state_indices, "numel")
        and int(metadata_state_indices.numel()) == needed_state_indices
    ):
        fixed_state_indices = metadata_state_indices.detach().reshape(-1).to(device=device, dtype=torch.int32)
    else:
        fixed_state_indices = _resize_mamba_state_indices(state_indices, needed_state_indices, device)

    if getattr(state_indices, "numel", lambda: 0)() == needed_state_indices:
        try:
            current = state_indices.detach().reshape(-1).to(device=device, dtype=torch.int32)
            if torch.equal(current, fixed_state_indices):
                return cache_params
        except Exception:
            return cache_params

    try:
        if dataclasses.is_dataclass(cache_params):
            return dataclasses.replace(cache_params, state_indices_tensor=fixed_state_indices)
    except Exception:
        pass
    fields = dict(getattr(cache_params, "__dict__", {}))
    fields["state_indices_tensor"] = fixed_state_indices
    return types.SimpleNamespace(**fields)


def _select_mamba2_attention_metadata(value: Any) -> Any | None:
    if not isinstance(value, dict):
        return _repair_mamba2_attention_metadata_state_indices(value) if _is_mamba2_attention_metadata(value) else None
    converted = _mamba2_attention_metadata_from_mapping(value)
    if converted is not None:
        return _repair_mamba2_attention_metadata_state_indices(converted)
    for child in value.values():
        selected = _select_mamba2_attention_metadata(child)
        if selected is not None:
            return selected
    return None


def _normalize_mamba2_attention_metadata_groups(attn_metadata: Any) -> Any:
    if not isinstance(attn_metadata, dict):
        return attn_metadata
    normalized = {}
    changed = False
    for key, value in attn_metadata.items():
        selected = _select_mamba2_attention_metadata(value)
        if selected is not None and selected is not value:
            normalized[key] = selected
            changed = True
            continue
        normalized_value = _normalize_mamba2_attention_metadata_groups(value)
        normalized[key] = normalized_value
        changed = changed or normalized_value is not value
    return normalized if changed else attn_metadata


def _as_v1_mamba2_attention_metadata_dict(attn_metadata: Any, vllm_config: Any) -> Any:
    if isinstance(attn_metadata, dict) or not _looks_like_easymagpie_vllm_config(vllm_config):
        return attn_metadata
    selected = _select_mamba2_attention_metadata(attn_metadata)
    if selected is None:
        return attn_metadata
    selected = _repair_mamba2_attention_metadata_state_indices(selected)

    layer_metadata: dict[str, Any] = {}
    static_context = getattr(getattr(vllm_config, "compilation_config", None), "static_forward_context", None)
    if isinstance(static_context, dict):
        for layer_name in static_context:
            if layer_name.endswith(".mixer"):
                layer_metadata.setdefault(layer_name, selected)

    pattern = _get_easymagpie_hybrid_pattern(vllm_config)
    if pattern:
        for layer_idx, layer_type in enumerate(pattern):
            if layer_type == "M":
                layer_metadata.setdefault(f"backbone.layers.{layer_idx}.mixer", selected)

    return layer_metadata or attn_metadata


def _build_flash_attention_metadata_from_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    device: Any = None,
    slot_mapping: Any = None,
    query_start_values: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> Any | None:
    torch = _TORCH_MODULE
    FlashAttentionMetadata = _LEGACY_FLASH_ATTENTION_METADATA_CLS
    if torch is None or FlashAttentionMetadata is None:
        return None

    num_tokens = max(1, int(num_tokens))
    num_reqs = max(1, int(num_reqs))
    if query_start_values is not None and len(query_start_values) == num_reqs + 1:
        query_lens = [
            max(0, query_start_values[index + 1] - query_start_values[index])
            for index in range(num_reqs)
        ]
        if sum(query_lens) != num_tokens:
            query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
            query_start_values = None
    else:
        query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
        query_start_values = None

    if (
        seq_lens is None
        or len(seq_lens) != num_reqs
        or any(length <= 0 for length in seq_lens)
        or sum(seq_lens) != num_tokens
    ):
        seq_lens = list(query_lens)

    if query_start_values is None:
        query_start_values = [0]
        for length in query_lens:
            query_start_values.append(query_start_values[-1] + int(length))
    seq_start_values = [0]
    for length in seq_lens:
        seq_start_values.append(seq_start_values[-1] + int(length))

    if device is None:
        device = getattr(slot_mapping, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_start_loc = torch.tensor(query_start_values, dtype=torch.int32, device=device)
    seq_start_loc = torch.tensor(seq_start_values, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
    context_lens_tensor = torch.zeros(num_reqs, dtype=torch.int, device=device)
    if slot_mapping is None or not hasattr(slot_mapping, "shape"):
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        slot_mapping = slot_mapping[:num_tokens].to(device=device, dtype=torch.long)
    block_tables = torch.empty((num_reqs, 0), dtype=torch.int, device=device)

    return FlashAttentionMetadata(
        num_prefills=num_reqs,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        multi_modal_placeholder_index_maps={},
        enable_kv_scales_calculation=True,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_tensor,
        max_query_len=max(int(max_query_len), _max_int_sequence(query_lens)),
        max_prefill_seq_len=_max_int_sequence(seq_lens),
        max_decode_query_len=0,
        max_decode_seq_len=0,
        query_start_loc=query_start_loc,
        seq_start_loc=seq_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_tables,
        use_cuda_graph=False,
    )


def _requested_attention_backend() -> str:
    try:
        import os

        return str(os.environ.get("VLLM_ATTENTION_BACKEND") or "").upper()
    except Exception:
        return ""


def _profile_attention_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    query_start_values: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> tuple[list[int], list[int], list[int]]:
    num_tokens = max(1, int(num_tokens))
    num_reqs = max(1, int(num_reqs))
    if query_start_values is not None and len(query_start_values) == num_reqs + 1:
        query_lens = [
            max(0, query_start_values[index + 1] - query_start_values[index])
            for index in range(num_reqs)
        ]
        if sum(query_lens) != num_tokens:
            query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
            query_start_values = None
    else:
        query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
        query_start_values = None

    if (
        seq_lens is None
        or len(seq_lens) != num_reqs
        or any(length <= 0 for length in seq_lens)
        or sum(seq_lens) != num_tokens
    ):
        seq_lens = list(query_lens)

    if query_start_values is None:
        query_start_values = [0]
        for length in query_lens:
            query_start_values.append(query_start_values[-1] + int(length))
    return query_lens, query_start_values, seq_lens


def _build_v1_triton_attention_metadata_from_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    device: Any = None,
    slot_mapping: Any = None,
    query_start_values: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> Any | None:
    torch = _TORCH_MODULE
    TritonAttentionMetadata = _V1_TRITON_ATTENTION_METADATA_CLS
    if torch is None or TritonAttentionMetadata is None:
        return None

    query_lens, query_start_values, seq_lens = _profile_attention_lengths(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_start_values=query_start_values,
        seq_lens=seq_lens,
    )
    if device is None:
        device = getattr(slot_mapping, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tokens = max(1, int(num_tokens))
    num_reqs = max(1, int(num_reqs))
    query_start_loc = torch.tensor(query_start_values, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    if slot_mapping is None or not hasattr(slot_mapping, "shape"):
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        slot_mapping = slot_mapping[:num_tokens].to(device=device, dtype=torch.long)
    block_table = torch.empty((num_reqs, 0), dtype=torch.int32, device=device)
    empty_float = torch.empty((0,), dtype=torch.float32, device=device)
    context_lens_tensor = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    resolved_max_query_len = max(int(max_query_len), _max_int_sequence(query_lens))

    metadata = _call_with_supported_kwargs(
        TritonAttentionMetadata,
        {
            "num_actual_tokens": num_tokens,
            "max_query_len": resolved_max_query_len,
            "query_start_loc": query_start_loc,
            "max_seq_len": _max_int_sequence(seq_lens),
            "seq_lens": seq_lens_tensor,
            "block_table": block_table,
            "slot_mapping": slot_mapping,
            "seq_threshold_3D": 0,
            "num_par_softmax_segments": 0,
            "softmax_segm_output": empty_float,
            "softmax_segm_max": empty_float,
            "softmax_segm_expsum": empty_float,
            "use_cascade": False,
            "common_prefix_len": 0,
            "cu_prefix_query_lens": None,
            "prefix_kv_lens": None,
            "suffix_kv_lens": None,
            "scheduler_metadata": None,
            "prefix_scheduler_metadata": None,
            "mm_prefix_range": None,
            "mm_prefix_range_tensor": None,
        },
    )
    return _attach_profile_attention_metadata_attrs(
        metadata,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_lens=query_lens,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_tensor,
        seq_start_loc=query_start_loc,
        query_start_loc=query_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_table,
        slot_mapping=slot_mapping,
        max_query_len=resolved_max_query_len,
    )


def _build_v1_flash_attention_metadata_from_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    device: Any = None,
    slot_mapping: Any = None,
    query_start_values: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> Any | None:
    torch = _TORCH_MODULE
    FlashAttentionMetadata = _V1_FLASH_ATTENTION_METADATA_CLS
    if torch is None or FlashAttentionMetadata is None:
        return None

    query_lens, query_start_values, seq_lens = _profile_attention_lengths(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_start_values=query_start_values,
        seq_lens=seq_lens,
    )
    if device is None:
        device = getattr(slot_mapping, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_tokens = max(1, int(num_tokens))
    num_reqs = max(1, int(num_reqs))
    query_start_loc = torch.tensor(query_start_values, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int32, device=device)
    if slot_mapping is None or not hasattr(slot_mapping, "shape"):
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        slot_mapping = slot_mapping[:num_tokens].to(device=device, dtype=torch.long)
    block_table = torch.empty((num_reqs, 0), dtype=torch.int32, device=device)
    context_lens_tensor = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    resolved_max_query_len = max(int(max_query_len), _max_int_sequence(query_lens))

    metadata = _call_with_supported_kwargs(
        FlashAttentionMetadata,
        {
            "num_actual_tokens": num_tokens,
            "max_query_len": resolved_max_query_len,
            "query_start_loc": query_start_loc,
            "max_seq_len": _max_int_sequence(seq_lens),
            "seq_lens": seq_lens_tensor,
            "block_table": block_table,
            "slot_mapping": slot_mapping,
            "use_cascade": False,
            "common_prefix_len": 0,
            "cu_prefix_query_lens": None,
            "prefix_kv_lens": None,
            "suffix_kv_lens": None,
            "max_dcp_context_kv_len": None,
            "dcp_context_kv_lens": None,
            "scheduler_metadata": None,
            "prefix_scheduler_metadata": None,
            "max_num_splits": 0,
            "causal": True,
        },
    )
    return _attach_profile_attention_metadata_attrs(
        metadata,
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        query_lens=query_lens,
        seq_lens=seq_lens,
        seq_lens_tensor=seq_lens_tensor,
        seq_start_loc=query_start_loc,
        query_start_loc=query_start_loc,
        context_lens_tensor=context_lens_tensor,
        block_tables=block_table,
        slot_mapping=slot_mapping,
        max_query_len=resolved_max_query_len,
    )


def _build_vllm_attention_metadata_from_lengths(
    *,
    num_tokens: int,
    num_reqs: int,
    max_query_len: int,
    device: Any = None,
    slot_mapping: Any = None,
    query_start_values: list[int] | None = None,
    seq_lens: list[int] | None = None,
) -> Any:
    requested_backend = _requested_attention_backend()
    is_flashinfer_backend = requested_backend.startswith("FLASHINFER")
    is_flash_attention_backend = requested_backend.startswith("FLASH_ATTN") or requested_backend in {
        "FLASH",
        "FLASHATTN",
    }
    if is_flash_attention_backend:
        metadata = _build_v1_flash_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is not None:
            return metadata
    if not is_flashinfer_backend and ("TRITON" in requested_backend or requested_backend != "XFORMERS"):
        metadata = _build_v1_triton_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is not None:
            return metadata
        metadata = _build_v1_flash_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is not None:
            return metadata

    torch = _TORCH_MODULE
    XFormersMetadata = _XFORMERS_METADATA_CLS
    if torch is None or XFormersMetadata is None:
        metadata = _build_v1_flash_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is not None:
            return metadata
        metadata = _build_v1_triton_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is not None:
            return metadata
        metadata = _build_flash_attention_metadata_from_lengths(
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            max_query_len=max_query_len,
            device=device,
            slot_mapping=slot_mapping,
            query_start_values=query_start_values,
            seq_lens=seq_lens,
        )
        if metadata is None:
            return None
        for attr in ("attn_bias", "encoder_attn_bias", "cross_attn_bias"):
            if not hasattr(metadata, attr):
                setattr(metadata, attr, None)
        return metadata

    num_tokens = max(1, int(num_tokens))
    num_reqs = max(1, int(num_reqs))
    if query_start_values is not None and len(query_start_values) == num_reqs + 1:
        query_lens = [
            max(0, query_start_values[index + 1] - query_start_values[index])
            for index in range(num_reqs)
        ]
        if sum(query_lens) != num_tokens:
            query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
            query_start_values = None
    else:
        query_lens = _balanced_profile_lengths(num_tokens, num_reqs)
        query_start_values = None

    if (
        seq_lens is None
        or len(seq_lens) != num_reqs
        or any(length <= 0 for length in seq_lens)
        or sum(seq_lens) != num_tokens
    ):
        seq_lens = list(query_lens)

    if query_start_values is None:
        query_start_values = [0]
        for length in query_lens:
            query_start_values.append(query_start_values[-1] + int(length))
    seq_start_values = [0]
    for length in seq_lens:
        seq_start_values.append(seq_start_values[-1] + int(length))

    if device is None:
        device = getattr(slot_mapping, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    query_start_loc = torch.tensor(query_start_values, dtype=torch.int32, device=device)
    seq_start_loc = torch.tensor(seq_start_values, dtype=torch.int32, device=device)
    seq_lens_tensor = torch.tensor(seq_lens, dtype=torch.int, device=device)
    context_lens_tensor = torch.zeros(num_reqs, dtype=torch.int, device=device)
    if slot_mapping is None or not hasattr(slot_mapping, "shape"):
        slot_mapping = torch.arange(num_tokens, dtype=torch.long, device=device)
    else:
        slot_mapping = slot_mapping[:num_tokens].to(device=device, dtype=torch.long)
    block_tables = torch.empty((num_reqs, 0), dtype=torch.int, device=device)

    return XFormersMetadata(
        seq_lens_tensor=seq_lens_tensor,
        max_decode_seq_len=0,
        block_tables=block_tables,
        num_prefills=num_reqs,
        num_prefill_tokens=num_tokens,
        num_decode_tokens=0,
        slot_mapping=slot_mapping,
        multi_modal_placeholder_index_maps={},
        enable_kv_scales_calculation=True,
        max_prefill_seq_len=_max_int_sequence(seq_lens),
        use_cuda_graph=False,
        seq_lens=seq_lens,
        seq_start_loc=seq_start_loc,
        context_lens_tensor=context_lens_tensor,
        max_query_len=max(int(max_query_len), _max_int_sequence(query_lens)),
        max_decode_query_len=0,
        query_start_loc=query_start_loc,
    )


def _build_profile_flash_attention_metadata(self: Any, num_tokens: int, num_reqs: int, max_query_len: int) -> Any:
    query_start_values = _buffer_slice_to_int_list(getattr(self, "query_start_loc", None), int(num_reqs) + 1)
    seq_lens = _buffer_slice_to_int_list(getattr(self, "seq_lens", None), int(num_reqs))
    device = getattr(self, "device", None)
    if device is None:
        query_start_gpu = getattr(getattr(self, "query_start_loc", None), "gpu", None)
        device = getattr(query_start_gpu, "device", None)
    return _build_vllm_attention_metadata_from_lengths(
        num_tokens=num_tokens,
        num_reqs=num_reqs,
        max_query_len=max_query_len,
        device=device,
        query_start_values=query_start_values,
        seq_lens=seq_lens,
    )


def _get_easymagpie_hybrid_pattern(vllm_config: Any) -> str | None:
    model_config = getattr(vllm_config, "model_config", None)
    hf_config = getattr(model_config, "hf_config", None)
    for candidate in (hf_config, model_config, vllm_config):
        pattern = getattr(candidate, "hybrid_override_pattern", None)
        if isinstance(pattern, str) and pattern:
            return pattern
    return None


def _get_easymagpie_chunk_size(vllm_config: Any) -> int:
    model_config = getattr(vllm_config, "model_config", None)
    get_mamba_chunk_size = getattr(model_config, "get_mamba_chunk_size", None)
    if get_mamba_chunk_size is not None:
        try:
            chunk_size = get_mamba_chunk_size()
            if chunk_size is not None:
                return int(chunk_size)
        except Exception:
            pass
    hf_config = getattr(model_config, "hf_config", None)
    for candidate in (hf_config, model_config, vllm_config):
        chunk_size = getattr(candidate, "chunk_size", None)
        if chunk_size is not None:
            return int(chunk_size)
    return 128


def _build_profile_mamba2_attention_metadata(flash_metadata: Any, chunk_size: int) -> Any:
    torch = _TORCH_MODULE
    Mamba2AttentionMetadata = _V1_MAMBA2_ATTENTION_METADATA_CLS
    if torch is None or Mamba2AttentionMetadata is None or flash_metadata is None:
        return None

    query_start_loc = getattr(flash_metadata, "query_start_loc", None)
    if query_start_loc is None:
        device = getattr(getattr(flash_metadata, "slot_mapping", None), "device", None)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_prefills = int(getattr(flash_metadata, "num_prefills", 1) or 1)
        query_start_loc = torch.arange(num_prefills + 1, dtype=torch.int32, device=device)
    else:
        device = query_start_loc.device
        num_prefills = max(0, int(getattr(flash_metadata, "num_prefills", query_start_loc.numel() - 1)))

    num_prefill_tokens_value = getattr(flash_metadata, "num_prefill_tokens", None)
    if num_prefill_tokens_value is None:
        num_prefill_tokens_value = query_start_loc[-1].item()
    num_prefill_tokens = int(num_prefill_tokens_value)
    seq_lens = getattr(flash_metadata, "seq_lens_tensor", None)
    if seq_lens is None:
        seq_lens_values = getattr(flash_metadata, "seq_lens", None)
        if seq_lens_values is None:
            seq_lens = query_start_loc[1:] - query_start_loc[:-1]
        else:
            seq_lens = torch.tensor(seq_lens_values, dtype=torch.int, device=device)
    else:
        seq_lens = seq_lens.to(device=device)

    if num_prefills > 0:
        query_lens = _balanced_profile_lengths(num_prefill_tokens, num_prefills)
        query_start_values = [0]
        for query_len in query_lens:
            query_start_values.append(query_start_values[-1] + int(query_len))
        query_start_loc = torch.tensor(query_start_values, dtype=torch.int32, device=device)
        seq_lens = torch.tensor(query_lens, dtype=torch.int, device=device)
        query_start_loc_p = query_start_loc[: num_prefills + 1]
        seq_idx_p = torch.repeat_interleave(
            torch.arange(num_prefills, dtype=torch.int32, device=device),
            query_start_loc_p.diff(),
            output_size=num_prefill_tokens,
        )
        seq_idx_p.unsqueeze_(0)
        has_initial_states_p = torch.zeros(num_prefills, dtype=torch.bool, device=device)
    else:
        seq_idx_p = None
        has_initial_states_p = None

    state_indices_tensor_p = (
        torch.arange(max(1, num_prefills), dtype=torch.int32, device=device)[:num_prefills]
        if num_prefills > 0
        else None
    )
    num_computed_tokens_p = torch.zeros(num_prefills, dtype=torch.int32, device=device) if num_prefills > 0 else None
    kwargs = {
        "num_reqs": num_prefills,
        "num_prefills": num_prefills,
        "num_prefill_tokens": num_prefill_tokens,
        "num_decodes": 0,
        "num_decode_tokens": 0,
        "query_start_loc_p": query_start_loc_p if num_prefills > 0 else None,
        "query_start_loc_d": None,
        "seq_lens": seq_lens,
        "num_computed_tokens_p": num_computed_tokens_p,
        "prep_initial_states": False,
        "chunk_size": int(chunk_size),
        "has_initial_states_p": has_initial_states_p,
        "seq_idx_p": seq_idx_p,
        "state_indices_tensor_p": state_indices_tensor_p,
        "state_indices_tensor_d": None,
        "num_accepted_tokens": None,
        "block_idx_last_scheduled_token": None,
        "block_idx_first_scheduled_token_p": None,
        "block_idx_last_computed_token": None,
        "cu_chunk_seqlen_p": None,
        "last_chunk_indices_p": None,
        "nums_dict": None,
        "batch_ptr": None,
        "token_chunk_offset_ptr": None,
    }
    metadata = _new_mamba2_attention_metadata(Mamba2AttentionMetadata, kwargs)
    return metadata


def _build_mamba2_attention_metadata_from_legacy(
    legacy_metadata: Any,
    mamba_cache_params: Any,
    hidden_states: Any,
) -> Any | None:
    torch = _TORCH_MODULE
    Mamba2AttentionMetadata = _V1_MAMBA2_ATTENTION_METADATA_CLS
    if torch is None or Mamba2AttentionMetadata is None:
        return None

    if legacy_metadata is None:
        return None

    device = getattr(hidden_states, "device", None)
    if device is None:
        state_indices = getattr(mamba_cache_params, "state_indices_tensor", None)
        device = getattr(state_indices, "device", None)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    hidden_len = int(getattr(hidden_states, "shape", [1])[0] or 1)
    legacy_seq_idx_p = getattr(legacy_metadata, "seq_idx", None)
    chunk_indices_p = getattr(legacy_metadata, "chunk_indices", None)
    chunk_offsets_p = getattr(legacy_metadata, "chunk_offsets", None)
    chunk_size = int(getattr(legacy_metadata, "chunk_size", 128) or 128)

    if legacy_seq_idx_p is not None and getattr(legacy_seq_idx_p, "numel", lambda: 0)() > 0:
        seq_idx_flat = _compact_legacy_mamba_seq_idx(legacy_seq_idx_p, device)
        num_prefill_tokens = int(seq_idx_flat.numel())
        num_prefills = int(seq_idx_flat.max().item()) + 1 if num_prefill_tokens > 0 else 0
        counts = (
            torch.bincount(seq_idx_flat, minlength=num_prefills).to(dtype=torch.int32, device=device)
            if num_prefills > 0
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        query_start_loc_p = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
        if num_prefills > 0:
            query_start_loc_p[1:] = torch.cumsum(counts, dim=0)
        num_decode_tokens = max(0, hidden_len - num_prefill_tokens)
        num_decodes = num_decode_tokens
        decode_lens = (
            torch.ones(num_decode_tokens, dtype=torch.int32, device=device)
            if num_decode_tokens > 0
            else torch.empty(0, dtype=torch.int32, device=device)
        )
        seq_lens = torch.cat([counts, decode_lens], dim=0) if num_decode_tokens > 0 else counts
        legacy_query_start_loc = torch.zeros(num_prefills + num_decodes + 1, dtype=torch.int32, device=device)
        if seq_lens.numel() > 0:
            legacy_query_start_loc[1:] = torch.cumsum(seq_lens, dim=0)
        has_initial_states_p = getattr(legacy_metadata, "has_initial_states", None)
        if has_initial_states_p is None:
            has_initial_states_p = torch.zeros(num_prefills, dtype=torch.bool, device=device)
        else:
            has_initial_states_p = has_initial_states_p.to(device=device, dtype=torch.bool)[:num_prefills]
        seq_idx_p = seq_idx_flat.unsqueeze(0) if num_prefill_tokens > 0 else None
    else:
        num_prefills = 0
        num_prefill_tokens = 0
        num_decodes = max(1, hidden_len)
        num_decode_tokens = num_decodes
        query_start_loc_p = None
        legacy_query_start_loc = torch.arange(num_decodes + 1, dtype=torch.int32, device=device)
        seq_lens = torch.ones(num_decodes, dtype=torch.int32, device=device)
        has_initial_states_p = None
        seq_idx_p = None

    state_indices_tensor = getattr(mamba_cache_params, "state_indices_tensor", None)
    state_indices_tensor_p = getattr(mamba_cache_params, "state_indices_tensor_p", None)
    if num_prefills > 0 and (
        state_indices_tensor_p is None or getattr(state_indices_tensor_p, "numel", lambda: 0)() != num_prefills
    ):
        state_indices_tensor_p = _resize_mamba_state_indices(
            state_indices_tensor_p if state_indices_tensor_p is not None else state_indices_tensor,
            num_prefills,
            device,
        )
    elif num_prefills <= 0:
        state_indices_tensor_p = None

    state_indices_tensor_d = getattr(mamba_cache_params, "state_indices_tensor_d", None)
    if num_decode_tokens > 0 and (
        state_indices_tensor_d is None
        or getattr(state_indices_tensor_d, "numel", lambda: 0)() != num_decode_tokens
    ):
        state_indices_tensor_d = _resize_mamba_state_indices(
            state_indices_tensor_d if state_indices_tensor_d is not None else state_indices_tensor,
            num_decode_tokens,
            device,
        )
    elif num_decode_tokens <= 0:
        state_indices_tensor_d = None

    query_start_loc_d = (
        torch.arange(num_decodes + 1, dtype=torch.int32, device=device) if num_decode_tokens > 0 else None
    )
    num_computed_tokens_p = torch.zeros(num_prefills, dtype=torch.int32, device=device) if num_prefills > 0 else None
    legacy_state_indices_tensor = _resize_mamba_state_indices(
        state_indices_tensor,
        num_prefills + num_decode_tokens,
        device,
    )

    metadata = _new_mamba2_attention_metadata(
        Mamba2AttentionMetadata,
        {
            "num_reqs": num_prefills + num_decodes,
            "num_prefills": num_prefills,
            "num_prefill_tokens": num_prefill_tokens,
            "num_decodes": num_decodes,
            "num_decode_tokens": num_decode_tokens,
            "query_start_loc_p": query_start_loc_p,
            "query_start_loc_d": query_start_loc_d,
            "legacy_query_start_loc": legacy_query_start_loc,
            "seq_lens": seq_lens,
            "num_computed_tokens_p": num_computed_tokens_p,
            "prep_initial_states": bool(getattr(legacy_metadata, "prep_initial_states", False)),
            "chunk_size": chunk_size,
            "has_initial_states_p": has_initial_states_p,
            "seq_idx_p": seq_idx_p,
            "state_indices_tensor_p": state_indices_tensor_p,
            "state_indices_tensor_d": state_indices_tensor_d,
            "legacy_state_indices_tensor": legacy_state_indices_tensor,
            "num_accepted_tokens": None,
            "block_idx_last_scheduled_token": None,
            "block_idx_first_scheduled_token_p": None,
            "block_idx_last_computed_token": None,
            "cu_chunk_seqlen_p": None,
            "last_chunk_indices_p": None,
            "nums_dict": None,
            "batch_ptr": None,
            "token_chunk_offset_ptr": None,
        },
    )
    return _repair_mamba2_attention_metadata_prefill_seq_idx(metadata)


def _select_legacy_mamba2_metadata(value: Any) -> Any | None:
    if not isinstance(value, dict):
        if any(hasattr(value, attr) for attr in ("seq_idx", "has_initial_states", "chunk_indices", "chunk_offsets")):
            return value
        return None
    for child in value.values():
        selected = _select_legacy_mamba2_metadata(child)
        if selected is not None:
            return selected
    return None


def _build_hybrid_profile_attention_metadata(vllm_config: Any, flash_metadata: Any) -> Any:
    layer_metadata: dict[str, Any] = {}
    mamba_metadata = _build_profile_mamba2_attention_metadata(
        flash_metadata,
        _get_easymagpie_chunk_size(vllm_config),
    )
    static_context = getattr(getattr(vllm_config, "compilation_config", None), "static_forward_context", None)
    if isinstance(static_context, dict):
        for layer_name in static_context:
            if layer_name.endswith(".attn") and flash_metadata is not None:
                layer_metadata.setdefault(layer_name, flash_metadata)
            elif layer_name.endswith(".mixer") and mamba_metadata is not None:
                layer_metadata.setdefault(layer_name, mamba_metadata)

    pattern = _get_easymagpie_hybrid_pattern(vllm_config)
    if pattern:
        for layer_idx, layer_type in enumerate(pattern):
            if layer_type == "M" and mamba_metadata is not None:
                layer_metadata.setdefault(f"backbone.layers.{layer_idx}.mixer", mamba_metadata)
            elif layer_type == "*" and flash_metadata is not None:
                layer_metadata.setdefault(f"backbone.layers.{layer_idx}.mixer.attn", flash_metadata)
    return layer_metadata or flash_metadata


def _looks_like_easymagpie_vllm_config(vllm_config: Any) -> bool:
    candidates = [vllm_config, getattr(vllm_config, "model_config", None)]
    model_config = candidates[-1]
    candidates.append(getattr(model_config, "hf_config", None))
    for candidate in candidates:
        if candidate is None:
            continue
        class_name = candidate.__class__.__name__.lower()
        if "easymagpie" in class_name or "nemotronh" in class_name or "nemotron_h" in class_name:
            return True
        pattern = getattr(candidate, "hybrid_override_pattern", None)
        if isinstance(pattern, str) and "M" in pattern and hasattr(candidate, "chunk_size"):
            return True
        for attr_name in ("model", "served_model_name", "model_type"):
            value = getattr(candidate, attr_name, None)
            if value is not None and (
                "easymagpie" in str(value).lower()
                or "nemotronh" in str(value).lower()
                or "nemotron_h" in str(value).lower()
            ):
                return True
        architectures = getattr(candidate, "architectures", None) or ()
        if any(
            "easymagpie" in str(architecture).lower()
            or "nemotronh" in str(architecture).lower()
            or "nemotron_h" in str(architecture).lower()
            for architecture in architectures
        ):
            return True
    return False


def _is_generic_profile_attention_metadata(attn_metadata: Any) -> bool:
    if attn_metadata is None or isinstance(attn_metadata, dict):
        return False
    if hasattr(attn_metadata, "use_cascade"):
        return False
    return any(
        hasattr(attn_metadata, attr)
        for attr in (
            "num_prefills",
            "num_prefill_tokens",
            "query_start_loc",
            "seq_lens",
            "seq_lens_tensor",
        )
    )


def _is_empty_attention_kv_cache(kv_cache: Any) -> bool:
    if kv_cache is None:
        return True
    shape = getattr(kv_cache, "shape", None)
    if shape is not None:
        try:
            return int(shape[0]) < 2
        except Exception:
            pass
    try:
        return len(kv_cache) < 2
    except Exception:
        return False


def _filter_supported_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        parameters = inspect.signature(callable_obj).parameters
    except Exception:
        return dict(kwargs)
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in parameters}


def _synthetic_profile_attention_metadata_from_context(attn_metadata: Any, vllm_config: Any, kwargs: dict[str, Any]) -> Any:
    if not _looks_like_easymagpie_vllm_config(vllm_config):
        return attn_metadata
    if attn_metadata is not None and not _is_generic_profile_attention_metadata(attn_metadata):
        return attn_metadata

    num_tokens = None
    num_reqs = None
    max_query_len = None
    slot_mapping = kwargs.get("slot_mapping")
    query_start_values = None
    seq_lens = None
    if attn_metadata is not None:
        num_tokens = getattr(attn_metadata, "num_prefill_tokens", None)
        if num_tokens is None:
            num_tokens = getattr(attn_metadata, "num_actual_tokens", None)
        num_reqs = getattr(attn_metadata, "num_prefills", None)
        if num_reqs is None:
            num_reqs = getattr(attn_metadata, "num_reqs", None)
        max_query_len = getattr(attn_metadata, "max_query_len", None)
        metadata_slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if metadata_slot_mapping is not None:
            slot_mapping = metadata_slot_mapping
        if num_reqs is not None:
            query_start_values = _buffer_slice_to_int_list(
                getattr(attn_metadata, "query_start_loc", None),
                int(num_reqs) + 1,
            )
            seq_lens = _buffer_slice_to_int_list(
                _first_not_none(
                    getattr(attn_metadata, "seq_lens_tensor", None),
                    getattr(attn_metadata, "seq_lens", None),
                ),
                int(num_reqs),
            )

    if num_tokens is None:
        num_tokens = kwargs.get("num_tokens")
    if num_tokens is None:
        return attn_metadata
    if num_reqs is None:
        batch_descriptor = kwargs.get("batch_descriptor")
        num_reqs = getattr(batch_descriptor, "num_reqs", None)
    if num_reqs is None:
        num_reqs = min(max(1, int(num_tokens)), 1)
    if max_query_len is None:
        max_query_len = max(1, int(num_tokens) // max(1, int(num_reqs)))
    flash_metadata = _build_vllm_attention_metadata_from_lengths(
        num_tokens=int(num_tokens),
        num_reqs=int(num_reqs),
        max_query_len=int(max_query_len),
        slot_mapping=slot_mapping,
        query_start_values=query_start_values,
        seq_lens=seq_lens,
    )
    return _build_hybrid_profile_attention_metadata(vllm_config, flash_metadata)


def _install_mamba2_metadata_compat() -> None:
    try:
        module = importlib.import_module("vllm.model_executor.layers.mamba.mamba2_metadata")
    except Exception:
        return
    original = getattr(module, "prepare_mamba2_metadata", None)
    if original is None or getattr(original, "_easymagpie_compat", False):
        return

    def _empty_attn_metadata():
        return types.SimpleNamespace(
            num_prefills=0,
            num_prefill_tokens=0,
            context_lens_tensor=None,
            query_start_loc=None,
        )

    def _select_attn_metadata(attn_metadata: Any):
        if not isinstance(attn_metadata, dict):
            return attn_metadata if hasattr(attn_metadata, "num_prefills") else None
        for value in attn_metadata.values():
            selected = _select_attn_metadata(value)
            if selected is not None:
                return selected
        return None

    def _balanced_tensor_lengths(total: int, count: int, device: Any):
        import torch

        if count <= 0:
            return torch.empty(0, dtype=torch.int32, device=device)
        base, extra = divmod(max(0, int(total)), int(count))
        lengths = torch.full((count,), base, dtype=torch.int32, device=device)
        if extra:
            lengths[:extra] += 1
        return lengths

    def _build_safe_legacy_metadata(chunk_size: int, attn_metadata: Any, mamba2_metadata: Any = None):
        import torch

        num_prefills = int(getattr(attn_metadata, "num_prefills", 0) or 0)
        num_prefill_tokens = int(getattr(attn_metadata, "num_prefill_tokens", 0) or 0)
        query_start_loc = getattr(attn_metadata, "query_start_loc", None)
        if num_prefills <= 0 or num_prefill_tokens <= 0 or query_start_loc is None:
            return None

        query_start_loc = query_start_loc.to(dtype=torch.int64)
        if int(query_start_loc.numel()) < 2:
            return None
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        device = query_start_loc.device

        if int(query_start_loc.numel()) >= num_prefills + 1:
            first_prefill_lens = query_start_loc[: num_prefills + 1].diff()
            try:
                if int(first_prefill_lens.sum().item()) == num_prefill_tokens:
                    return None
            except Exception:
                pass

        prefill_lens = query_lens[query_lens > 1]
        try:
            prefill_lens_ok = (
                int(prefill_lens.numel()) == num_prefills
                and int(prefill_lens.sum().item()) == num_prefill_tokens
            )
        except Exception:
            prefill_lens_ok = False
        if not prefill_lens_ok:
            prefill_lens = _balanced_tensor_lengths(num_prefill_tokens, num_prefills, device)
        else:
            prefill_lens = prefill_lens.to(dtype=torch.int32)

        query_start_loc_p = torch.zeros(num_prefills + 1, dtype=torch.int32, device=device)
        if num_prefills > 0:
            query_start_loc_p[1:] = torch.cumsum(prefill_lens.to(dtype=torch.int32), dim=0)
        seq_idx = torch.repeat_interleave(
            torch.arange(num_prefills, dtype=torch.int32, device=device),
            prefill_lens.to(dtype=torch.int32),
        ).unsqueeze(0)

        has_initial_states = None
        prep_initial_states = False
        context_lens_tensor = getattr(attn_metadata, "context_lens_tensor", None)
        if context_lens_tensor is not None:
            try:
                context_lens_tensor = context_lens_tensor.to(device=device)
                if int(context_lens_tensor.numel()) == int(query_lens.numel()) and int(
                    (query_lens > 1).sum().item()
                ) == num_prefills:
                    context_lens_tensor = context_lens_tensor[query_lens > 1]
                else:
                    context_lens_tensor = context_lens_tensor[:num_prefills]
                has_initial_states = context_lens_tensor > 0
                prep_initial_states = bool(torch.any(has_initial_states).item())
            except Exception:
                has_initial_states = None
                prep_initial_states = False

        chunk_indices, chunk_offsets = None, None
        if prep_initial_states:
            try:
                converter = getattr(module, "_query_start_loc_to_chunk_indices_offsets")
                chunk_indices, chunk_offsets = converter(query_start_loc_p, chunk_size, num_prefill_tokens)
            except Exception:
                chunk_indices, chunk_offsets = None, None

        target = mamba2_metadata
        if target is None:
            metadata_cls = getattr(module, "Mamba2Metadata", None)
            if metadata_cls is None:
                return types.SimpleNamespace(
                    has_initial_states=has_initial_states,
                    prep_initial_states=prep_initial_states,
                    chunk_size=chunk_size,
                    seq_idx=seq_idx,
                    chunk_indices=chunk_indices,
                    chunk_offsets=chunk_offsets,
                    cu_seqlen=None,
                )
            return metadata_cls(
                has_initial_states=has_initial_states,
                prep_initial_states=prep_initial_states,
                chunk_size=chunk_size,
                seq_idx=seq_idx,
                chunk_indices=chunk_indices,
                chunk_offsets=chunk_offsets,
            )

        target.has_initial_states = has_initial_states
        target.prep_initial_states = prep_initial_states
        target.chunk_size = chunk_size
        target.seq_idx = seq_idx
        target.chunk_indices = chunk_indices
        target.chunk_offsets = chunk_offsets
        target.cu_seqlen = None
        return target

    def prepare_mamba2_metadata(chunk_size: int, attn_metadata: Any, mamba2_metadata: Any = None):
        if attn_metadata is None:
            attn_metadata = _empty_attn_metadata()
        elif isinstance(attn_metadata, dict):
            attn_metadata = _select_attn_metadata(attn_metadata) or _empty_attn_metadata()
        if not hasattr(attn_metadata, "num_prefills"):
            attn_metadata = types.SimpleNamespace(
                num_prefills=0,
                num_prefill_tokens=0,
                context_lens_tensor=None,
                query_start_loc=None,
            )
        safe_metadata = _build_safe_legacy_metadata(chunk_size, attn_metadata, mamba2_metadata)
        if safe_metadata is not None:
            return safe_metadata
        return original(chunk_size, attn_metadata, mamba2_metadata)

    prepare_mamba2_metadata._easymagpie_compat = True  # type: ignore[attr-defined]
    module.prepare_mamba2_metadata = prepare_mamba2_metadata
    for module_name in ("vllm.model_executor.models.nemotron_h",):
        loaded_module = sys.modules.get(module_name)
        if loaded_module is not None and getattr(loaded_module, "prepare_mamba2_metadata", None) is original:
            loaded_module.prepare_mamba2_metadata = prepare_mamba2_metadata


def _install_triton_attention_profile_metadata_compat() -> None:
    try:
        module = importlib.import_module("vllm.v1.attention.backends.triton_attn")
    except Exception:
        return
    impl_cls = getattr(module, "TritonAttentionImpl", None)
    original = getattr(impl_cls, "forward", None)
    if original is None or getattr(original, "_easymagpie_profile_metadata_compat", False):
        return

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if _is_generic_profile_attention_metadata(attn_metadata):
            return original(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                None,
                output,
                output_scale,
                output_block_scale,
            )
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            output_block_scale,
        )

    forward._easymagpie_profile_metadata_compat = True  # type: ignore[attr-defined]
    impl_cls.forward = forward


def _install_flash_attention_profile_metadata_compat() -> None:
    try:
        module = importlib.import_module("vllm.v1.attention.backends.flash_attn")
    except Exception:
        return
    impl_cls = getattr(module, "FlashAttentionImpl", None)
    original = getattr(impl_cls, "forward", None)
    if original is None or getattr(original, "_easymagpie_profile_metadata_compat", False):
        return

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        **kwargs,
    ):
        optional_kwargs = _filter_supported_kwargs(
            original,
            {"output_block_scale": output_block_scale, **kwargs},
        )
        if _is_generic_profile_attention_metadata(attn_metadata) or _is_empty_attention_kv_cache(kv_cache):
            return original(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                None,
                output,
                output_scale,
                **optional_kwargs,
            )
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            **optional_kwargs,
        )

    forward._easymagpie_profile_metadata_compat = True  # type: ignore[attr-defined]
    impl_cls.forward = forward


def _install_flashinfer_attention_profile_metadata_compat() -> None:
    try:
        module = importlib.import_module("vllm.v1.attention.backends.flashinfer")
    except Exception:
        return
    impl_cls = getattr(module, "FlashInferImpl", None)
    original = getattr(impl_cls, "forward", None)
    if original is None or getattr(original, "_easymagpie_profile_metadata_compat", False):
        return

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=None,
        output_scale=None,
        output_block_scale=None,
        **kwargs,
    ):
        optional_kwargs = _filter_supported_kwargs(
            original,
            {"output_block_scale": output_block_scale, **kwargs},
        )
        if _is_generic_profile_attention_metadata(attn_metadata) or _is_empty_attention_kv_cache(kv_cache):
            return original(
                self,
                layer,
                query,
                key,
                value,
                kv_cache,
                None,
                output,
                output_scale,
                **optional_kwargs,
            )
        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale,
            **optional_kwargs,
        )

    forward._easymagpie_profile_metadata_compat = True  # type: ignore[attr-defined]
    impl_cls.forward = forward


def _install_mamba2_profile_no_kv_cache_compat() -> None:
    try:
        from vllm import envs
        from vllm.forward_context import get_forward_context

        module = importlib.import_module("vllm.model_executor.layers.mamba.mamba_mixer2")
    except Exception:
        return
    mixer_cls = getattr(module, "MambaMixer2", None)
    original = getattr(mixer_cls, "forward_cuda", None)
    if mixer_cls is None:
        return

    def _has_usable_mamba_kv_cache_for_conv(self: Any) -> bool:
        kv_cache = getattr(self, "kv_cache", None)
        try:
            if kv_cache is None:
                return False
            if len(kv_cache) == 1 and isinstance(kv_cache[0], (list, tuple)):
                kv_cache = kv_cache[0]
            if len(kv_cache) < 2:
                return False
            conv_cache = kv_cache[0]
            ssm_cache = kv_cache[1]
            return (
                getattr(conv_cache, "ndim", 0) >= 2
                and getattr(ssm_cache, "ndim", 0) >= 2
            )
        except Exception:
            return False

    def _is_profile_only_mamba_metadata(attn_metadata: Any) -> bool:
        if attn_metadata is None:
            return False
        if getattr(attn_metadata, "batch_ptr", None) is not None:
            return False
        if getattr(attn_metadata, "token_chunk_offset_ptr", None) is not None:
            return False
        num_decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
        num_prefill_tokens = int(getattr(attn_metadata, "num_prefill_tokens", 0) or 0)
        return num_decode_tokens == 0 and num_prefill_tokens > 0

    def _with_v1_mamba_metadata_dict(
        self,
        callback,
        *args,
        allow_profile_metadata_fallback: bool = False,
        **kwargs,
    ):
        try:
            use_v1 = bool(getattr(envs, "VLLM_USE_V1", False))
        except Exception:
            use_v1 = False
        if not use_v1:
            return callback(*args, **kwargs)

        try:
            forward_context = get_forward_context()
        except Exception:
            forward_context = None
        attn_metadata = getattr(forward_context, "attn_metadata", None)
        prefix = getattr(self, "prefix", None)
        selected_mamba_metadata = _select_mamba2_attention_metadata(attn_metadata)
        if selected_mamba_metadata is None:
            projected_states = kwargs.get("projected_states")
            if projected_states is None and args:
                projected_states = args[0]
            try:
                num_tokens = max(1, int(getattr(projected_states, "shape", [1])[0] or 1))
                device = getattr(projected_states, "device", None)
                if attn_metadata is not None and hasattr(attn_metadata, "num_prefills"):
                    flash_metadata = attn_metadata
                else:
                    flash_metadata = _build_vllm_attention_metadata_from_lengths(
                        num_tokens=num_tokens,
                        num_reqs=1,
                        max_query_len=num_tokens,
                        device=device,
                    )
                chunk_size = int(getattr(self, "chunk_size", None) or getattr(self, "chunk_size_padded", None) or 256)
                selected_mamba_metadata = _build_profile_mamba2_attention_metadata(flash_metadata, chunk_size)
            except Exception:
                selected_mamba_metadata = None
        if prefix is None or selected_mamba_metadata is None or forward_context is None:
            return callback(*args, **kwargs)

        if allow_profile_metadata_fallback and (
            not _has_usable_mamba_kv_cache_for_conv(self)
            or _is_profile_only_mamba_metadata(selected_mamba_metadata)
        ):
            previous_attn_metadata = forward_context.attn_metadata
            forward_context.attn_metadata = None
            try:
                return callback(*args, **kwargs)
            finally:
                forward_context.attn_metadata = previous_attn_metadata

        selected_mamba_metadata = _repair_mamba2_attention_metadata_state_indices(selected_mamba_metadata)
        replacement_attn_metadata = None
        if isinstance(attn_metadata, dict):
            layer_metadata = attn_metadata.get(prefix)
            if layer_metadata is not selected_mamba_metadata:
                replacement_attn_metadata = dict(attn_metadata)
                replacement_attn_metadata[prefix] = selected_mamba_metadata
        else:
            replacement_attn_metadata = {prefix: selected_mamba_metadata}

        if replacement_attn_metadata is None:
            return callback(*args, **kwargs)

        previous_attn_metadata = forward_context.attn_metadata
        forward_context.attn_metadata = replacement_attn_metadata
        try:
            return callback(*args, **kwargs)
        finally:
            forward_context.attn_metadata = previous_attn_metadata

    original_conv_ssm_forward = getattr(mixer_cls, "conv_ssm_forward", None)
    if original_conv_ssm_forward is not None and not getattr(
        original_conv_ssm_forward, "_easymagpie_v1_attn_metadata_compat", False
    ):

        def conv_ssm_forward(self, *args, **kwargs):
            return _with_v1_mamba_metadata_dict(
                self,
                lambda *call_args, **call_kwargs: original_conv_ssm_forward(self, *call_args, **call_kwargs),
                *args,
                allow_profile_metadata_fallback=True,
                **kwargs,
            )

        conv_ssm_forward._easymagpie_v1_attn_metadata_compat = True  # type: ignore[attr-defined]
        mixer_cls.conv_ssm_forward = conv_ssm_forward

    original_forward = getattr(mixer_cls, "forward", None)
    if original_forward is not None and not getattr(original_forward, "_easymagpie_v1_attn_metadata_compat", False):

        def forward(self, *args, **kwargs):
            return _with_v1_mamba_metadata_dict(
                self,
                lambda *call_args, **call_kwargs: original_forward(self, *call_args, **call_kwargs),
                *args,
                **kwargs,
            )

        forward._easymagpie_v1_attn_metadata_compat = True  # type: ignore[attr-defined]
        mixer_cls.forward = forward

    if original is None or getattr(original, "_easymagpie_no_kv_profile_compat", False):
        return

    try:
        original_signature = inspect.signature(original)
    except Exception:
        original_signature = None

    def has_explicit_cache_args(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> bool:
        positional_cache_args = (
            (len(args) > 2 and args[2] is not None)
            or (len(args) > 3 and args[3] is not None)
        )
        keyword_cache_args = (
            kwargs.get("mamba_cache_params") is not None
            or kwargs.get("mamba2_metadata") is not None
        )
        if original_signature is None:
            return positional_cache_args or keyword_cache_args
        try:
            bound = original_signature.bind_partial(self, *args, **kwargs)
        except Exception:
            return positional_cache_args or keyword_cache_args
        return (
            bound.arguments.get("mamba_cache_params") is not None
            or bound.arguments.get("mamba2_metadata") is not None
            or positional_cache_args
            or keyword_cache_args
        )

    def bound_forward_argument(self: Any, args: tuple[Any, ...], kwargs: dict[str, Any], name: str) -> Any:
        if original_signature is not None:
            try:
                bound = original_signature.bind_partial(self, *args, **kwargs)
                value = bound.arguments.get(name)
                if value is not None:
                    return value
            except Exception:
                pass
        positional_index = {
            "hidden_states": 0,
            "output": 1,
            "mamba_cache_params": 2,
            "mamba2_metadata": 3,
            "mup_vector": 4,
        }.get(name)
        if positional_index is not None and len(args) > positional_index:
            return args[positional_index]
        return kwargs.get(name)

    def replace_forward_argument(
        args: tuple[Any, ...], kwargs: dict[str, Any], name: str, value: Any
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        positional_index = {
            "hidden_states": 0,
            "output": 1,
            "mamba_cache_params": 2,
            "mamba2_metadata": 3,
            "mup_vector": 4,
        }.get(name)
        if positional_index is not None and len(args) > positional_index:
            new_args = list(args)
            new_args[positional_index] = value
            return tuple(new_args), kwargs
        new_kwargs = dict(kwargs)
        new_kwargs[name] = value
        return args, new_kwargs

    def forward_cuda(self, *args, **kwargs):
        try:
            use_v1 = bool(getattr(envs, "VLLM_USE_V1", False))
        except Exception:
            use_v1 = False

        forward_context = None
        original_attn_metadata = None
        selected_mamba_metadata = None
        restore_attn_metadata = False
        try:
            forward_context = get_forward_context()
        except Exception:
            forward_context = None
        original_attn_metadata = getattr(forward_context, "attn_metadata", None)
        normalized_attn_metadata = _normalize_mamba2_attention_metadata_groups(original_attn_metadata)
        prefix = getattr(self, "prefix", None)
        force_v0_mamba_forward = False
        selected_from_legacy_mapping = False
        if isinstance(normalized_attn_metadata, dict) and prefix is not None:
            layer_metadata = normalized_attn_metadata.get(prefix)
            selected_mamba_metadata = _select_mamba2_attention_metadata(layer_metadata)
            if selected_mamba_metadata is None:
                legacy_metadata = bound_forward_argument(self, args, kwargs, "mamba2_metadata")
                if legacy_metadata is None:
                    legacy_metadata = _select_legacy_mamba2_metadata(layer_metadata)
                if legacy_metadata is None:
                    legacy_metadata = layer_metadata
                selected_mamba_metadata = _build_mamba2_attention_metadata_from_legacy(
                    legacy_metadata,
                    bound_forward_argument(self, args, kwargs, "mamba_cache_params"),
                    bound_forward_argument(self, args, kwargs, "hidden_states"),
                )
                force_v0_mamba_forward = selected_mamba_metadata is None and isinstance(layer_metadata, dict)
                selected_from_legacy_mapping = selected_mamba_metadata is not None and isinstance(layer_metadata, dict)
            if selected_mamba_metadata is not None:
                selected_mamba_metadata = _repair_mamba2_attention_metadata_state_indices(selected_mamba_metadata)
            if selected_mamba_metadata is not None and selected_mamba_metadata is not layer_metadata:
                normalized_attn_metadata = dict(normalized_attn_metadata)
                normalized_attn_metadata[prefix] = selected_mamba_metadata
        elif _is_mamba2_attention_metadata(normalized_attn_metadata):
            selected_mamba_metadata = _repair_mamba2_attention_metadata_state_indices(normalized_attn_metadata)
            normalized_attn_metadata = selected_mamba_metadata
        if forward_context is not None:
            replacement_attn_metadata = normalized_attn_metadata
            if use_v1 and selected_mamba_metadata is not None and not isinstance(replacement_attn_metadata, dict):
                prefix = getattr(self, "prefix", None)
                if prefix is not None:
                    replacement_attn_metadata = {prefix: selected_mamba_metadata}
            elif not use_v1 and selected_mamba_metadata is not None:
                replacement_attn_metadata = selected_mamba_metadata
            if replacement_attn_metadata is not original_attn_metadata:
                forward_context.attn_metadata = replacement_attn_metadata
                restore_attn_metadata = True

        call_args = args
        call_kwargs = kwargs
        if selected_mamba_metadata is not None:
            cache_params = bound_forward_argument(self, args, kwargs, "mamba_cache_params")
            repaired_cache_params = _repair_mamba_cache_params_state_indices(cache_params, selected_mamba_metadata)
            if repaired_cache_params is not cache_params:
                call_args, call_kwargs = replace_forward_argument(
                    args, kwargs, "mamba_cache_params", repaired_cache_params
                )

        explicit_cache_args = has_explicit_cache_args(self, call_args, call_kwargs)
        try:
            if (
                use_v1
                and forward_context is not None
                and selected_mamba_metadata is not None
                and not explicit_cache_args
                and _is_profile_only_mamba_metadata(selected_mamba_metadata)
            ):
                previous_attn_metadata = getattr(forward_context, "attn_metadata", None)
                forward_context.attn_metadata = None
                try:
                    return original(self, *call_args, **call_kwargs)
                finally:
                    forward_context.attn_metadata = previous_attn_metadata
            if (
                use_v1
                and forward_context is not None
                and (explicit_cache_args or force_v0_mamba_forward)
                and (selected_mamba_metadata is not None or force_v0_mamba_forward)
            ):
                previous_use_v1 = getattr(envs, "VLLM_USE_V1", None)
                previous_attn_metadata = getattr(forward_context, "attn_metadata", None)
                envs.VLLM_USE_V1 = False
                if selected_mamba_metadata is not None:
                    if selected_from_legacy_mapping and prefix is not None:
                        forward_context.attn_metadata = {prefix: selected_mamba_metadata}
                    else:
                        forward_context.attn_metadata = selected_mamba_metadata
                try:
                    return original(self, *call_args, **call_kwargs)
                finally:
                    envs.VLLM_USE_V1 = previous_use_v1
                    forward_context.attn_metadata = previous_attn_metadata
            if (
                selected_mamba_metadata is not None
                and forward_context is not None
                and hasattr(self, "kv_cache")
                and not explicit_cache_args
            ):
                previous_use_v1 = getattr(envs, "VLLM_USE_V1", None)
                previous_attn_metadata = getattr(forward_context, "attn_metadata", None)
                prefix = getattr(self, "prefix", None)
                if isinstance(normalized_attn_metadata, dict):
                    v1_attn_metadata = normalized_attn_metadata
                elif prefix is not None:
                    v1_attn_metadata = {prefix: selected_mamba_metadata}
                else:
                    v1_attn_metadata = previous_attn_metadata
                envs.VLLM_USE_V1 = True
                forward_context.attn_metadata = v1_attn_metadata
                try:
                    return original(self, *call_args, **call_kwargs)
                finally:
                    envs.VLLM_USE_V1 = previous_use_v1
                    forward_context.attn_metadata = previous_attn_metadata
            return original(self, *call_args, **call_kwargs)
        finally:
            if restore_attn_metadata and forward_context is not None:
                try:
                    forward_context.attn_metadata = original_attn_metadata
                except Exception:
                    pass

    forward_cuda._easymagpie_no_kv_profile_compat = True  # type: ignore[attr-defined]
    mixer_cls.forward_cuda = forward_cuda


def _install_lora_config_alias() -> None:
    try:
        from vllm.config import LoRAConfig
    except Exception:
        return
    module = types.ModuleType("vllm.config.lora")
    module.LoRAConfig = LoRAConfig
    module.MaxLoRARanks = Literal[1, 8, 16, 32, 64, 128, 256, 320, 512]
    sys.modules.setdefault("vllm.config.lora", module)


def _install_config_decorator_compat() -> None:
    try:
        import vllm.config.utils as config_utils
    except Exception:
        return
    original = getattr(config_utils, "config", None)
    if original is None or getattr(original, "_easymagpie_compat", False):
        return

    def compat_config(cls=None, **kwargs):
        def _decorate(real_cls):
            try:
                return original(real_cls, **kwargs)
            except TypeError:
                if not kwargs:
                    raise
                return original(real_cls)

        if cls is None:
            return _decorate
        return _decorate(cls)

    compat_config._easymagpie_compat = True  # type: ignore[attr-defined]
    config_utils.config = compat_config


def _install_model_arch_convertor() -> None:
    if "vllm.transformers_utils.model_arch_config_convertor" in sys.modules:
        return
    module = types.ModuleType("vllm.transformers_utils.model_arch_config_convertor")

    class ModelArchConfigConvertorBase:
        def __init__(self, hf_config: Any, hf_text_config: Any) -> None:
            self.hf_config = hf_config
            self.hf_text_config = hf_text_config

        def _normalize_quantization_config(self, config: Any) -> Any:
            if config is None:
                return None
            if isinstance(config, dict):
                return config.get("quantization_config")
            return getattr(config, "quantization_config", None)

        def get_quantization_config(self) -> Any:
            return self._normalize_quantization_config(self.hf_text_config) or self._normalize_quantization_config(
                self.hf_config
            )

        def convert(self) -> dict[str, Any]:
            quant_config = self.get_quantization_config()
            return {"quantization_config": quant_config} if quant_config is not None else {}

    module.ModelArchConfigConvertorBase = ModelArchConfigConvertorBase
    sys.modules["vllm.transformers_utils.model_arch_config_convertor"] = module


def _install_io_processor_stub() -> None:
    if "vllm.plugins.io_processors" in sys.modules:
        return
    module = types.ModuleType("vllm.plugins.io_processors")

    def get_io_processor(*_args: Any, **_kwargs: Any) -> None:
        return None

    module.get_io_processor = get_io_processor
    sys.modules["vllm.plugins.io_processors"] = module


def _install_tokenizer_aliases() -> None:
    try:
        import vllm.transformers_utils.tokenizer as tokenizer_module
    except Exception:
        return
    if not hasattr(tokenizer_module, "TokenizerLike"):
        tokenizer_module.TokenizerLike = getattr(tokenizer_module, "AnyTokenizer", object)
    sys.modules.setdefault("vllm.tokenizers", tokenizer_module)
    try:
        import vllm.transformers_utils.tokenizers.mistral as mistral_module
    except Exception:
        return
    sys.modules.setdefault("vllm.tokenizers.mistral", mistral_module)


def _install_repo_utils_alias() -> None:
    if "vllm.transformers_utils.repo_utils" in sys.modules:
        return
    try:
        from vllm.transformers_utils.config import file_or_path_exists
    except Exception:
        return
    module = types.ModuleType("vllm.transformers_utils.repo_utils")
    module.file_or_path_exists = file_or_path_exists
    sys.modules["vllm.transformers_utils.repo_utils"] = module


def _install_config_parser_aliases() -> None:
    try:
        import vllm.transformers_utils.config as config_module
    except Exception:
        return

    if not hasattr(config_module, "MistralConfigParser"):

        class MistralConfigParser:
            pass

        config_module.MistralConfigParser = MistralConfigParser

    if not hasattr(config_module, "register_config_parser"):

        def register_config_parser(_name: str):
            def decorator(cls):
                return cls

            return decorator

        config_module.register_config_parser = register_config_parser


def _install_nemotron_h_auto_config() -> None:
    try:
        from transformers import AutoConfig
        from vllm.transformers_utils.configs import NemotronHConfig
    except Exception:
        return
    try:
        AutoConfig.register("nemotron_h", NemotronHConfig)
    except ValueError:
        pass


def _register_easy_magpie_plugin() -> None:
    try:
        from vllm_plugin_easymagpie_omni import register
    except Exception:
        return
    register()


def _install_vllm_inputs_data_alias() -> None:
    if "vllm.inputs.data" in sys.modules:
        return
    try:
        import vllm.inputs as inputs
    except Exception:
        return
    try:
        if importlib.util.find_spec("vllm.inputs.data") is not None:
            return
    except Exception:
        pass

    module = types.ModuleType("vllm.inputs.data")
    module.__doc__ = "Compatibility aliases for vLLM input types exported from vllm.inputs."
    module.__package__ = "vllm.inputs"
    for name in dir(inputs):
        if not name.startswith("__"):
            setattr(module, name, getattr(inputs, name))
    for old_name, new_name in {
        "TokenInputs": "TokensInput",
        "EmbedsInputs": "EmbedsInput",
        "SingletonInputs": "SingletonInput",
    }.items():
        if not hasattr(module, old_name) and hasattr(inputs, new_name):
            setattr(module, old_name, getattr(inputs, new_name))

    sys.modules["vllm.inputs.data"] = module
    sys.modules.setdefault("vllm.inputs.parse", module)
    if not hasattr(inputs, "data"):
        inputs.data = module
    if not hasattr(inputs, "parse"):
        inputs.parse = sys.modules["vllm.inputs.parse"]


def _install_vllm_multimodal_inputs_alias() -> None:
    """Expose vLLM-Omni's legacy plural multimodal input alias on vLLM 0.21+."""

    try:
        multimodal_inputs = importlib.import_module("vllm.multimodal.inputs")
    except Exception:
        return
    if hasattr(multimodal_inputs, "MultiModalInputs"):
        return

    candidate = None
    for name in ("MultiModalInputsV2", "MultiModalInput"):
        candidate = getattr(multimodal_inputs, name, None)
        if candidate is not None:
            break
    if candidate is None:
        try:
            inputs = importlib.import_module("vllm.inputs")
            candidate = getattr(inputs, "MultiModalInput", None)
        except Exception:
            candidate = None
    if candidate is None:
        candidate = dict
    multimodal_inputs.MultiModalInputs = candidate


def _install_renderer_aliases() -> None:
    if "vllm.renderers" not in sys.modules:
        renderers = types.ModuleType("vllm.renderers")
        renderers.__path__ = []  # type: ignore[attr-defined]

        class BaseRenderer:
            pass

        def merge_kwargs(*items: Any) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for item in items:
                if item:
                    merged.update(dict(item))
            return merged

        def renderer_from_config(*_args: Any, **_kwargs: Any) -> None:
            return None

        renderers.BaseRenderer = BaseRenderer
        renderers.merge_kwargs = merge_kwargs
        renderers.renderer_from_config = renderer_from_config
        sys.modules["vllm.renderers"] = renderers
    else:
        renderers = sys.modules["vllm.renderers"]
        if not hasattr(renderers, "__path__"):
            renderers.__path__ = []  # type: ignore[attr-defined]
    inputs = sys.modules.get("vllm.renderers.inputs")
    if inputs is None:
        inputs = types.ModuleType("vllm.renderers.inputs")
        sys.modules["vllm.renderers.inputs"] = inputs
    inputs.__package__ = "vllm.renderers.inputs"
    inputs.__path__ = []  # type: ignore[attr-defined]
    try:
        from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
    except Exception:
        try:
            from vllm.inputs.data import EmbedsPrompt, TextPrompt, TokensPrompt
        except Exception:
            EmbedsPrompt = dict  # type: ignore[assignment]
            TextPrompt = dict  # type: ignore[assignment]
            TokensPrompt = dict  # type: ignore[assignment]

    try:
        from typing import TypedDict

        class EncoderDecoderDictPrompt(TypedDict):
            encoder_prompt: Any
            decoder_prompt: Any | None

        class EncoderDecoderTokPrompt(TypedDict):
            encoder_prompt: Any
            decoder_prompt: Any | None

    except Exception:
        EncoderDecoderDictPrompt = dict  # type: ignore[assignment]
        EncoderDecoderTokPrompt = dict  # type: ignore[assignment]

    for name, value in {
        "EmbedsPrompt": EmbedsPrompt,
        "TextPrompt": TextPrompt,
        "TokensPrompt": TokensPrompt,
        "DecoderDictPrompt": TextPrompt | TokensPrompt,
        "DecoderOnlyDictPrompt": TextPrompt | TokensPrompt | EmbedsPrompt,
        "DecoderOnlyTokPrompt": TokensPrompt | EmbedsPrompt,
        "DecoderTokPrompt": TokensPrompt,
        "DictPrompt": dict,
        "EncoderDictPrompt": TextPrompt | TokensPrompt,
        "EncoderTokPrompt": TokensPrompt,
        "SingletonDictPrompt": dict,
        "SingletonTokPrompt": dict,
        "TokPrompt": dict,
        "EncoderDecoderDictPrompt": EncoderDecoderDictPrompt,
        "EncoderDecoderTokPrompt": EncoderDecoderTokPrompt,
    }.items():
        if not hasattr(inputs, name):
            setattr(inputs, name, value)

    preprocess = sys.modules.get("vllm.renderers.inputs.preprocess")
    if preprocess is None:
        preprocess = types.ModuleType("vllm.renderers.inputs.preprocess")
        preprocess.__package__ = "vllm.renderers.inputs"
        sys.modules["vllm.renderers.inputs.preprocess"] = preprocess
        inputs.preprocess = preprocess

        def _is_list_of(value: object, item_type: type) -> bool:
            return isinstance(value, list) and all(isinstance(item, item_type) for item in value)

        def _validate_prompt_dict(prompt: Any) -> None:
            if "prompt" not in prompt or "prompt_token_ids" in prompt or "prompt_embeds" in prompt:
                return
            if not isinstance(prompt["prompt"], str):
                raise TypeError("Prompt text should be a string")

        def _parse_singleton_prompt(prompt: object, *, allow_embeds: bool = True) -> Any:
            if isinstance(prompt, str):
                return TextPrompt(prompt=prompt)
            if _is_list_of(prompt, int):
                return TokensPrompt(prompt_token_ids=prompt)
            if isinstance(prompt, dict):
                if "encoder_prompt" in prompt:
                    raise TypeError("Cannot pass encoder-decoder prompt to a singleton prompt parser")
                _validate_prompt_dict(prompt)
                if "prompt" in prompt or "prompt_token_ids" in prompt or (allow_embeds and "prompt_embeds" in prompt):
                    return prompt
                expected = "text, tokens, or embeddings" if allow_embeds else "text or tokens"
                raise TypeError(f"Prompt dictionary must contain {expected}")
            raise TypeError("Prompt should be a string, list of tokens, or dictionary")

        def parse_dec_only_prompt(prompt: object) -> Any:
            if isinstance(prompt, dict) and "encoder_prompt" in prompt:
                raise TypeError("Cannot pass encoder-decoder prompt to decoder-only models")
            return _parse_singleton_prompt(prompt)

        def parse_enc_dec_prompt(prompt: object) -> Any:
            if isinstance(prompt, dict) and "encoder_prompt" in prompt:
                enc_prompt = prompt["encoder_prompt"]
                dec_prompt = prompt.get("decoder_prompt")
            else:
                enc_prompt = prompt
                dec_prompt = None
            return EncoderDecoderDictPrompt(
                encoder_prompt=_parse_singleton_prompt(enc_prompt, allow_embeds=False),
                decoder_prompt=None if dec_prompt is None else _parse_singleton_prompt(dec_prompt, allow_embeds=False),
            )

        for name in (
            "DecoderDictPrompt",
            "DecoderOnlyDictPrompt",
            "DictPrompt",
            "EncoderDictPrompt",
            "EncoderDecoderDictPrompt",
            "SingletonDictPrompt",
            "TextPrompt",
            "TokensPrompt",
            "EmbedsPrompt",
        ):
            setattr(preprocess, name, getattr(inputs, name))
        preprocess.parse_dec_only_prompt = parse_dec_only_prompt
        preprocess.parse_enc_dec_prompt = parse_enc_dec_prompt
    elif not hasattr(inputs, "preprocess"):
        inputs.preprocess = preprocess

    tokenize = sys.modules.get("vllm.renderers.inputs.tokenize")
    if tokenize is None:
        tokenize = types.ModuleType("vllm.renderers.inputs.tokenize")
        tokenize.__package__ = "vllm.renderers.inputs"
        sys.modules["vllm.renderers.inputs.tokenize"] = tokenize
        inputs.tokenize = tokenize
        for name in (
            "DecoderOnlyTokPrompt",
            "DecoderTokPrompt",
            "EncoderTokPrompt",
            "EncoderDecoderTokPrompt",
            "SingletonTokPrompt",
            "TokPrompt",
            "TokensPrompt",
            "EmbedsPrompt",
        ):
            setattr(tokenize, name, getattr(inputs, name))
    elif not hasattr(inputs, "tokenize"):
        inputs.tokenize = tokenize


def _install_input_processor_alias() -> None:
    try:
        from vllm.inputs.preprocess import InputPreprocessor
        from vllm.transformers_utils.tokenizer import cached_tokenizer_from_config
        from vllm.v1.engine.processor import Processor
    except Exception:
        return

    original_init = InputPreprocessor.__init__
    if not getattr(original_init, "_easymagpie_compat", False):

        def compat_init(self, *args: Any, **kwargs: Any) -> None:
            renderer = kwargs.pop("renderer", None)
            if "vllm_config" in kwargs:
                vllm_config = kwargs.pop("vllm_config")
                tokenizer = kwargs.pop("tokenizer", None)
                if tokenizer is None and not getattr(vllm_config.model_config, "skip_tokenizer_init", False):
                    tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)
                original_init(self, vllm_config.model_config, tokenizer, *args, **kwargs)
                self.renderer = renderer
                return
            original_init(self, *args, **kwargs)
            self.renderer = getattr(self, "renderer", renderer)

        compat_init._easymagpie_compat = True  # type: ignore[attr-defined]
        InputPreprocessor.__init__ = compat_init

    if "vllm.v1.engine.input_processor" in sys.modules:
        return

    module = types.ModuleType("vllm.v1.engine.input_processor")

    class InputProcessor(Processor):
        def __init__(self, *, vllm_config: Any, tokenizer: Any = None, mm_registry: Any = None, **_kwargs: Any) -> None:
            if tokenizer is None and not getattr(vllm_config.model_config, "skip_tokenizer_init", False):
                tokenizer = cached_tokenizer_from_config(model_config=vllm_config.model_config)
            if mm_registry is None:
                super().__init__(vllm_config, tokenizer)
            else:
                super().__init__(vllm_config, tokenizer, mm_registry)
            self.renderer = getattr(self.input_preprocessor, "renderer", None)

    module.InputProcessor = InputProcessor
    sys.modules["vllm.v1.engine.input_processor"] = module


def _install_processor_process_inputs_compat() -> None:
    try:
        from vllm.v1.engine.processor import Processor
    except Exception:
        return

    original_process_inputs = Processor.process_inputs
    if getattr(original_process_inputs, "_easymagpie_compat", False):
        return

    try:
        signature = inspect.signature(original_process_inputs)
        parameters = signature.parameters.values()
        accepts_supported_tasks = "supported_tasks" in signature.parameters or any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters
        )
    except Exception:
        accepts_supported_tasks = False

    if accepts_supported_tasks:
        return

    def compat_process_inputs(self, *args: Any, **kwargs: Any):
        has_omni_supported_tasks = "supported_tasks" in kwargs
        kwargs.pop("supported_tasks", None)
        result = original_process_inputs(self, *args, **kwargs)
        if has_omni_supported_tasks and isinstance(result, tuple) and len(result) == 2:
            return result[1]
        return result

    compat_process_inputs._easymagpie_compat = True  # type: ignore[attr-defined]
    Processor.process_inputs = compat_process_inputs


def _install_omni_input_preprocessor_signature_compat() -> None:
    try:
        from vllm_omni.inputs.preprocess import OmniInputPreprocessor
    except Exception:
        return

    original_prompt_to_inputs = OmniInputPreprocessor._prompt_to_llm_inputs
    if getattr(original_prompt_to_inputs, "_easymagpie_compat", False):
        return

    try:
        signature = inspect.signature(original_prompt_to_inputs)
        parameters = signature.parameters
        accepts_kwargs = any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in parameters.values())
    except Exception:
        signature = None
        parameters = {}
        accepts_kwargs = False

    if accepts_kwargs or {"lora_request", "return_mm_hashes"}.issubset(parameters):
        return

    accepted_kwargs = {
        name
        for name, parameter in parameters.items()
        if name != "self" and parameter.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }

    def compat_prompt_to_inputs(self, *args: Any, **kwargs: Any):
        if signature is not None:
            kwargs = {key: value for key, value in kwargs.items() if key in accepted_kwargs}
        else:
            kwargs.pop("lora_request", None)
            kwargs.pop("return_mm_hashes", None)
        return original_prompt_to_inputs(self, *args, **kwargs)

    compat_prompt_to_inputs._easymagpie_compat = True  # type: ignore[attr-defined]
    OmniInputPreprocessor._prompt_to_llm_inputs = compat_prompt_to_inputs


def _install_input_preprocessor_truncate_compat() -> None:
    try:
        from vllm.inputs.preprocess import InputPreprocessor
    except Exception:
        return

    if hasattr(InputPreprocessor, "_truncate_inputs"):
        return

    def _truncate_inputs(self, prompt_token_ids: Any, tokenization_kwargs: dict[str, Any] | None = None):
        limit = None
        if tokenization_kwargs:
            limit = tokenization_kwargs.get("max_length", tokenization_kwargs.get("truncate_prompt_tokens"))
        if limit is None:
            return prompt_token_ids
        try:
            limit_int = int(limit)
        except Exception:
            return prompt_token_ids
        if limit_int <= 0:
            return prompt_token_ids
        return prompt_token_ids[-limit_int:]

    InputPreprocessor._truncate_inputs = _truncate_inputs


def _install_omni_engine_request_compat() -> None:
    try:
        from vllm.v1.engine import EngineCoreRequest
        from vllm_omni.engine import OmniEngineCoreRequest
    except Exception:
        return

    for request_cls in (EngineCoreRequest, OmniEngineCoreRequest):
        if not hasattr(request_cls, "prompt_embeds"):
            request_cls.prompt_embeds = None

    try:
        async_omni_engine = importlib.import_module("vllm_omni.engine.async_omni_engine")
        from vllm_omni.engine.serialization import serialize_additional_information
    except Exception:
        return

    try:
        request_signature = inspect.signature(OmniEngineCoreRequest)
    except Exception:
        request_signature = None

    if not getattr(async_omni_engine._upgrade_to_omni_request, "_easymagpie_compat", False):

        def compat_upgrade_to_omni_request(request: Any, raw_prompt: Any) -> Any:
            prompt_embeds = getattr(request, "prompt_embeds", None)
            additional_information = None

            if isinstance(raw_prompt, dict):
                if prompt_embeds is None:
                    raw_prompt_embeds = raw_prompt.get("prompt_embeds")
                    try:
                        import torch

                        if isinstance(raw_prompt_embeds, torch.Tensor):
                            prompt_embeds = raw_prompt_embeds
                    except Exception:
                        pass
                additional_information = serialize_additional_information(
                    raw_prompt.get("additional_information"),
                    log_prefix="AsyncOmniEngine",
                )

            if prompt_embeds is None and additional_information is None:
                return request

            if request_signature is None:
                return request

            values: dict[str, Any] = {}
            for name, parameter in request_signature.parameters.items():
                if name == "prompt_embeds":
                    values[name] = prompt_embeds
                elif name == "additional_information":
                    values[name] = additional_information
                elif hasattr(request, name):
                    values[name] = getattr(request, name)
                elif parameter.default is inspect.Parameter.empty:
                    values[name] = None

            return OmniEngineCoreRequest(**values)

        compat_upgrade_to_omni_request._easymagpie_compat = True  # type: ignore[attr-defined]
        async_omni_engine._upgrade_to_omni_request = compat_upgrade_to_omni_request

    async_engine_cls = getattr(async_omni_engine, "AsyncOmniEngine", None)
    original_build = getattr(async_engine_cls, "_build_add_request_message", None)
    if original_build is None or getattr(original_build, "_easymagpie_external_req_compat", False):
        return
    try:
        original_build_signature = inspect.signature(original_build)
    except Exception:
        original_build_signature = None
    if original_build_signature is not None:
        original_build_params = original_build_signature.parameters
        if "prompt_text" in original_build_params and "message_type" in original_build_params:
            # Newer vLLM-Omni already handles request construction, prompt text,
            # final-stage metadata, and orchestrator admission correctly. The
            # patched _upgrade_to_omni_request above is enough for EasyMagpie's
            # prompt embeddings/additional-information fields.
            return

    engine_core_request_cls = EngineCoreRequest
    inject_global_id = getattr(async_omni_engine, "_inject_global_id", None)

    def compat_build_add_request_message(
        self: Any,
        request_id: str,
        prompt: Any,
        sampling_params_list: Any = None,
        final_stage_id: int = 0,
        arrival_time: float | None = None,
    ) -> dict[str, Any]:
        effective_sampling_params_list = (
            list(sampling_params_list) if sampling_params_list is not None else list(self.default_sampling_params_list)
        )
        if not effective_sampling_params_list:
            raise ValueError(
                f"Missing sampling params for stage 0. Got {len(effective_sampling_params_list)} stage params."
            )
        params = effective_sampling_params_list[0]
        original_prompt = prompt

        stage_type = self.stage_metadata[0].get("stage_type")
        if stage_type != "diffusion" and not isinstance(prompt, engine_core_request_cls):
            if inject_global_id is not None:
                if isinstance(prompt, dict):
                    inject_global_id(prompt, request_id)
                elif isinstance(prompt, list):
                    for item in prompt:
                        inject_global_id(item, request_id)

            request = self.input_processor.process_inputs(
                request_id=request_id,
                prompt=prompt,
                params=params,
                supported_tasks=self.supported_tasks,
                arrival_time=arrival_time,
            )
            request = async_omni_engine._upgrade_to_omni_request(request, prompt)
            try:
                request.external_req_id = request_id
            except AttributeError:
                pass

            self.output_processors[0].add_request(
                request=request,
                prompt=prompt,
                parent_req=None,
                request_index=0,
                queue=None,
            )
            prompt = request

        return {
            "type": "add_request",
            "request_id": request_id,
            "prompt": prompt,
            "original_prompt": original_prompt,
            "sampling_params_list": effective_sampling_params_list,
            "final_stage_id": final_stage_id,
        }

    compat_build_add_request_message._easymagpie_external_req_compat = True  # type: ignore[attr-defined]
    async_engine_cls._build_add_request_message = compat_build_add_request_message


def _install_omni_request_compat() -> None:
    try:
        from vllm.v1.request import StructuredOutputRequest
        import vllm_omni.request as omni_request_module
    except Exception:
        return

    OmniRequest = getattr(omni_request_module, "OmniRequest", None)
    if OmniRequest is None:
        return
    try:
        BaseRequest = OmniRequest.__mro__[1]
    except Exception:
        return

    original_init = getattr(OmniRequest, "__init__", None)
    if original_init is not None and not getattr(original_init, "_easymagpie_request_init_compat", False):
        base_init = BaseRequest.__init__
        try:
            base_signature = inspect.signature(base_init)
        except Exception:
            base_signature = None

        def compat_omni_request_init(
            self: Any,
            prompt_embeds: Any = None,
            external_req_id: str | None = None,
            additional_information: Any = None,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            prompt_embeds_tensor = self._maybe_decode_prompt_embeds(prompt_embeds)
            if base_signature is None:
                base_init(self, *args, **kwargs)
            else:
                accepts_kwargs = any(
                    parameter.kind is inspect.Parameter.VAR_KEYWORD
                    for parameter in base_signature.parameters.values()
                )
                base_kwargs = dict(kwargs)
                if "prompt_embeds" in base_signature.parameters:
                    base_kwargs["prompt_embeds"] = prompt_embeds_tensor
                if not accepts_kwargs:
                    base_kwargs = {
                        key: value
                        for key, value in base_kwargs.items()
                        if key in base_signature.parameters and key != "self"
                    }
                base_init(self, *args, **base_kwargs)

            self.prompt_embeds = prompt_embeds_tensor
            self.prompt_embeds_payload = prompt_embeds if prompt_embeds is not prompt_embeds_tensor else None
            self.external_req_id = external_req_id
            self.additional_information = additional_information

        compat_omni_request_init._easymagpie_request_init_compat = True  # type: ignore[attr-defined]
        OmniRequest.__init__ = compat_omni_request_init

    original_from_request = getattr(OmniRequest, "from_engine_core_request", None)
    raw_from_request = getattr(original_from_request, "__func__", original_from_request)
    if raw_from_request is None or getattr(raw_from_request, "_easymagpie_request_convert_compat", False):
        return

    def compat_from_engine_core_request(cls: Any, request: Any, block_hasher: Any) -> Any:
        mm_kwargs = getattr(request, "mm_kwargs", None)
        if mm_kwargs is not None:
            mm_kwargs = list(mm_kwargs)
        elif getattr(request, "mm_features", None) is not None:
            mm_kwargs = list(getattr(request, "mm_features"))

        sampling_params = getattr(request, "sampling_params", None)
        structured_output_request = None
        if sampling_params is not None:
            from_sampling_params = getattr(StructuredOutputRequest, "from_sampling_params", None)
            if callable(from_sampling_params):
                structured_output_request = from_sampling_params(sampling_params)
            else:
                try:
                    structured_output_request = StructuredOutputRequest(sampling_params=sampling_params)
                except TypeError:
                    structured_outputs = getattr(sampling_params, "structured_outputs", None)
                    if structured_outputs and not structured_outputs.all_constraints_none():
                        structured_output_request = StructuredOutputRequest(params=structured_outputs)

        return cls(
            request_id=request.request_id,
            external_req_id=getattr(request, "external_req_id", None) or request.request_id,
            client_index=getattr(request, "client_index", 0),
            prompt_token_ids=request.prompt_token_ids,
            prompt_embeds=getattr(request, "prompt_embeds", None),
            multi_modal_kwargs=mm_kwargs,
            multi_modal_hashes=getattr(request, "mm_hashes", None),
            multi_modal_placeholders=getattr(request, "mm_placeholders", None),
            sampling_params=sampling_params,
            pooling_params=getattr(request, "pooling_params", None),
            eos_token_id=getattr(request, "eos_token_id", None),
            arrival_time=getattr(request, "arrival_time", None),
            lora_request=getattr(request, "lora_request", None),
            structured_output_request=structured_output_request,
            cache_salt=getattr(request, "cache_salt", None),
            priority=getattr(request, "priority", 0),
            block_hasher=block_hasher,
            additional_information=getattr(request, "additional_information", None),
        )

    compat_from_engine_core_request._easymagpie_request_convert_compat = True  # type: ignore[attr-defined]
    OmniRequest.from_engine_core_request = classmethod(compat_from_engine_core_request)


def _install_engine_utils_compat() -> None:
    try:
        import vllm.v1.engine.utils as engine_utils
        from vllm.v1.utils import get_engine_client_zmq_addr
    except Exception:
        return

    if not hasattr(engine_utils, "get_engine_zmq_addresses"):

        def get_engine_zmq_addresses(vllm_config: Any, num_api_servers: int = 1):
            parallel_config = vllm_config.parallel_config
            dp_size = parallel_config.data_parallel_size
            local_engine_count = parallel_config.data_parallel_size_local
            local_start_index = parallel_config.data_parallel_rank_local
            host = parallel_config.data_parallel_master_ip
            local_only = (
                local_start_index is not None
                or parallel_config.data_parallel_hybrid_lb
                or parallel_config.data_parallel_external_lb
                or local_engine_count == dp_size
            )
            return engine_utils.EngineZmqAddresses(
                inputs=[get_engine_client_zmq_addr(local_only, host) for _ in range(num_api_servers)],
                outputs=[get_engine_client_zmq_addr(local_only, host) for _ in range(num_api_servers)],
            )

        engine_utils.get_engine_zmq_addresses = get_engine_zmq_addresses

    launch_core_engines = getattr(engine_utils, "launch_core_engines", None)
    if launch_core_engines is not None and not getattr(launch_core_engines, "_easymagpie_compat", False):
        signature = inspect.signature(launch_core_engines)
        parameters = signature.parameters
        accepts_addresses = "addresses" in parameters
        positional_parameters = [
            name
            for name, parameter in parameters.items()
            if parameter.kind
            in {
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            }
        ]
        addresses_pos = positional_parameters.index("addresses") if "addresses" in positional_parameters else None
        num_api_servers_pos = (
            positional_parameters.index("num_api_servers") if "num_api_servers" in positional_parameters else None
        )

        class LaunchCoreEnginesContextCompat:
            def __init__(self, context_manager: Any) -> None:
                self._context_manager = context_manager

            def __enter__(self) -> Any:
                value = self._context_manager.__enter__()
                if isinstance(value, tuple) and len(value) > 3:
                    return value[:3]
                return value

            def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Any:
                return self._context_manager.__exit__(exc_type, exc, tb)

        def compat_launch_core_engines(*args: Any, addresses: Any = None, **kwargs: Any):
            if not accepts_addresses:
                return launch_core_engines(*args, **kwargs)

            call_kwargs = dict(kwargs)
            address_supplied_positionally = addresses_pos is not None and len(args) > addresses_pos
            if not address_supplied_positionally and "addresses" not in call_kwargs:
                if addresses is None:
                    vllm_config = call_kwargs.get("vllm_config")
                    if vllm_config is None and args:
                        vllm_config = args[0]
                    num_api_servers = call_kwargs.get("num_api_servers")
                    if num_api_servers is None and num_api_servers_pos is not None and len(args) > num_api_servers_pos:
                        num_api_servers = args[num_api_servers_pos]
                    if num_api_servers is None:
                        num_api_servers_parameter = parameters.get("num_api_servers")
                        if num_api_servers_parameter is not None:
                            num_api_servers = num_api_servers_parameter.default
                    if num_api_servers is inspect.Parameter.empty:
                        num_api_servers = 1
                    addresses = engine_utils.get_engine_zmq_addresses(vllm_config, int(num_api_servers or 1))
                call_kwargs["addresses"] = addresses
            launch_context = launch_core_engines(*args, **call_kwargs)
            if hasattr(launch_context, "__enter__") and hasattr(launch_context, "__exit__"):
                return LaunchCoreEnginesContextCompat(launch_context)
            return launch_context

        compat_launch_core_engines._easymagpie_compat = True  # type: ignore[attr-defined]
        engine_utils.launch_core_engines = compat_launch_core_engines


def _install_import_utils_alias() -> None:
    if "vllm.utils.import_utils" in sys.modules:
        return
    module = types.ModuleType("vllm.utils.import_utils")

    class LazyLoader(types.ModuleType):
        def __init__(self, local_name: str, parent_globals: dict[str, Any], name: str) -> None:
            super().__init__(local_name)
            self._local_name = local_name
            self._parent_globals = parent_globals
            self._module_name = name
            self._module = None

        def _load(self):
            if self._module is None:
                self._module = importlib.import_module(self._module_name)
                self._parent_globals[self._local_name] = self._module
            return self._module

        def __getattr__(self, item: str) -> Any:
            return getattr(self._load(), item)

    def resolve_obj_by_qualname(qualname: str) -> Any:
        module_name, _, attr = qualname.replace(":", ".").rpartition(".")
        if not module_name:
            raise ValueError(f"Invalid qualified name: {qualname}")
        obj = importlib.import_module(module_name)
        for part in attr.split("."):
            obj = getattr(obj, part)
        return obj

    try:
        from vllm.utils import import_pynvml
    except Exception:

        def import_pynvml():
            return importlib.import_module("pynvml")

    module.LazyLoader = LazyLoader
    module.resolve_obj_by_qualname = resolve_obj_by_qualname
    module.import_pynvml = import_pynvml
    sys.modules["vllm.utils.import_utils"] = module


def _install_torch_utils_alias() -> None:
    if "vllm.utils.torch_utils" in sys.modules:
        return
    module = types.ModuleType("vllm.utils.torch_utils")

    def set_random_seed(seed: int | None) -> None:
        import numpy as np
        import torch

        seed = 0 if seed is None else int(seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    @contextlib.contextmanager
    def set_default_torch_dtype(dtype):
        import torch

        old_dtype = torch.get_default_dtype()
        torch.set_default_dtype(dtype)
        try:
            yield
        finally:
            torch.set_default_dtype(old_dtype)

    try:
        from vllm.utils import supports_xccl
    except Exception:

        def supports_xccl() -> bool:
            return False

    module.set_random_seed = set_random_seed
    module.set_default_torch_dtype = set_default_torch_dtype
    module.supports_xccl = supports_xccl
    sys.modules["vllm.utils.torch_utils"] = module


def _install_math_utils_alias() -> None:
    if "vllm.utils.math_utils" in sys.modules:
        return
    try:
        from vllm.utils import cdiv
    except Exception:

        def cdiv(a: int, b: int) -> int:
            return -(a // -b)

    module = types.ModuleType("vllm.utils.math_utils")
    module.cdiv = cdiv
    sys.modules["vllm.utils.math_utils"] = module


def _install_mem_utils_alias() -> None:
    if "vllm.utils.mem_utils" in sys.modules:
        return
    try:
        import vllm.utils as vllm_utils
    except Exception:
        return

    module = types.ModuleType("vllm.utils.mem_utils")

    class MemorySnapshot(vllm_utils.MemorySnapshot):
        def __init__(self, *args, device=None, **kwargs) -> None:
            super().__init__(*args, **kwargs)

    def format_gib(num_bytes: int | float) -> str:
        return f"{float(num_bytes) / float(vllm_utils.GiB_bytes):.2f}"

    module.MemorySnapshot = MemorySnapshot
    module.memory_profiling = vllm_utils.memory_profiling
    module.format_gib = format_gib
    sys.modules["vllm.utils.mem_utils"] = module


def _install_vllm_config_profiler_default() -> None:
    try:
        from vllm.config import VllmConfig
    except Exception:
        return
    if hasattr(VllmConfig, "profiler_config"):
        return

    def get_profiler_config(self):
        stored = getattr(self, "_easymagpie_profiler_config", None)
        if stored is not None:
            return stored
        additional_config = getattr(self, "additional_config", None)
        if isinstance(additional_config, dict):
            return additional_config.get("profiler_config")
        return getattr(additional_config, "profiler_config", None)

    def set_profiler_config(self, value) -> None:
        object.__setattr__(self, "_easymagpie_profiler_config", value)

    VllmConfig.profiler_config = property(get_profiler_config, set_profiler_config)


def _install_parallel_config_defaults() -> None:
    try:
        from vllm.config import ParallelConfig
    except Exception:
        return

    missing = object()

    def install_property(name: str, default):
        if hasattr(ParallelConfig, name):
            return

        storage_name = f"_easymagpie_{name}"

        def getter(self):
            stored = getattr(self, storage_name, missing)
            if stored is not missing:
                return stored
            return default(self) if callable(default) else default

        def setter(self, value) -> None:
            object.__setattr__(self, storage_name, value)

        setattr(ParallelConfig, name, property(getter, setter))

    install_property(
        "nnodes_within_dp",
        lambda self: max(
            1,
            int(getattr(self, "data_parallel_size", 1)) // int(getattr(self, "data_parallel_size_local", 1)),
        ),
    )
    install_property("data_parallel_index", lambda self: getattr(self, "data_parallel_rank", 0))
    install_property(
        "local_world_size",
        lambda self: (
            int(getattr(self, "pipeline_parallel_size", 1))
            * int(getattr(self, "tensor_parallel_size", 1))
            * int(getattr(self, "data_parallel_size_local", 1))
        ),
    )
    install_property("enable_dbo", False)
    install_property("num_ubatches", 1)
    install_property("use_ubatching", False)


def _install_cache_config_defaults() -> None:
    try:
        from vllm.config import CacheConfig
    except Exception:
        return
    if hasattr(CacheConfig, "kv_cache_memory_bytes"):
        return

    missing = object()
    storage_name = "_easymagpie_kv_cache_memory_bytes"

    def get_kv_cache_memory_bytes(self):
        stored = getattr(self, storage_name, missing)
        if stored is not missing:
            return stored
        return None

    def set_kv_cache_memory_bytes(self, value) -> None:
        object.__setattr__(self, storage_name, value)

    CacheConfig.kv_cache_memory_bytes = property(get_kv_cache_memory_bytes, set_kv_cache_memory_bytes)


def _install_platform_dtype_check() -> None:
    try:
        from vllm.platforms import current_platform
    except Exception:
        return

    dtype_check = getattr(current_platform, "check_if_supports_dtype", None)
    if callable(dtype_check):
        return

    def check_if_supports_dtype(_dtype) -> None:
        return None

    try:
        current_platform.check_if_supports_dtype = check_if_supports_dtype
    except Exception:
        pass
    try:
        setattr(type(current_platform), "check_if_supports_dtype", staticmethod(check_if_supports_dtype))
    except Exception:
        pass


def _install_torch_accelerator_compat() -> None:
    try:
        import torch
    except Exception:
        return

    accelerator = getattr(torch, "accelerator", None)
    if accelerator is None or hasattr(accelerator, "empty_cache"):
        return
    empty_cache = getattr(getattr(torch, "cuda", None), "empty_cache", None)
    if not callable(empty_cache):
        return
    accelerator.empty_cache = empty_cache


def _install_logger_info_once_scope_compat() -> None:
    try:
        import vllm.logger as vllm_logger
    except Exception:
        vllm_logger = None

    if vllm_logger is not None:
        original_print = getattr(vllm_logger, "_print_info_once", None)
        if original_print is not None and not getattr(original_print, "_easymagpie_compat", False):

            def _print_info_once(logger, msg, *args, **kwargs):
                kwargs.pop("scope", None)
                return original_print(logger, msg, *args, **kwargs)

            _print_info_once._easymagpie_compat = True  # type: ignore[attr-defined]
            vllm_logger._print_info_once = _print_info_once
            methods_to_patch = getattr(vllm_logger, "_METHODS_TO_PATCH", None)
            if isinstance(methods_to_patch, dict):
                methods_to_patch["info_once"] = _print_info_once

    original = getattr(logging.Logger, "info_once", None)
    if original is None or getattr(original, "_easymagpie_compat", False):
        return

    def info_once(self, msg, *args, **kwargs):
        kwargs.pop("scope", None)
        return original(self, msg, *args, **kwargs)

    info_once._easymagpie_compat = True  # type: ignore[attr-defined]
    logging.Logger.info_once = info_once


def _install_v1_serial_utils_dense_tensor_compat() -> None:
    try:
        import msgspec.msgpack as msgpack
        import torch
        import vllm.v1.serial_utils as serial_utils
    except Exception:
        return

    encoder_cls = getattr(serial_utils, "MsgpackEncoder", None)
    if encoder_cls is None:
        return
    original = getattr(encoder_cls, "_encode_tensor", None)
    if original is None or getattr(original, "_easymagpie_dense_tensor_compat", False):
        return

    custom_type_raw_view = getattr(serial_utils, "CUSTOM_TYPE_RAW_VIEW", None)
    if custom_type_raw_view is None:
        return

    def _encode_tensor_dense(
        self: Any,
        obj: torch.Tensor,
    ) -> tuple[str, tuple[int, ...], Any]:
        assert self.aux_buffers is not None
        tensor = obj.detach()
        if tensor.device.type != "cpu":
            tensor = tensor.cpu()
        flat = tensor.reshape(-1)
        dense = torch.empty((int(flat.numel()),), dtype=flat.dtype, device="cpu")
        dense.copy_(flat)
        arr = dense.view(torch.uint8).numpy()
        if obj.nbytes < self.size_threshold:
            data = msgpack.Ext(custom_type_raw_view, arr.data)
        else:
            data = len(self.aux_buffers)
            self.aux_buffers.append(arr.data)
        dtype = str(obj.dtype).removeprefix("torch.")
        return dtype, obj.shape, data

    _encode_tensor_dense._easymagpie_dense_tensor_compat = True  # type: ignore[attr-defined]
    _encode_tensor_dense._easymagpie_original = original  # type: ignore[attr-defined]
    encoder_cls._encode_tensor = _encode_tensor_dense

    decoder_cls = getattr(serial_utils, "MsgpackDecoder", None)
    if decoder_cls is None:
        return
    original_decode = getattr(decoder_cls, "decode", None)
    if original_decode is None or getattr(original_decode, "_easymagpie_easy_refit_tensor_decode_compat", False):
        return

    easy_magpie_refit_methods = {
        "easymagpie_load_weights",
        "easymagpie_load_non_text_weights",
        "easymagpie_update_text_embedding_rows",
    }

    def _is_encoded_tensor_triple(obj: Any) -> bool:
        if not isinstance(obj, (list, tuple)) or len(obj) != 3:
            return False
        dtype, shape, data = obj
        if not isinstance(dtype, str):
            return False
        torch_dtype = getattr(torch, dtype, None)
        if not isinstance(torch_dtype, torch.dtype):
            return False
        if not isinstance(shape, (list, tuple)) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
            return False
        if isinstance(data, int):
            return 0 <= data < len(getattr(_decode_easy_magpie_refit_tensors, "_active_aux_buffers", ()) or ())
        return isinstance(data, (bytes, bytearray, memoryview))

    def _decode_easy_magpie_refit_tensors(self: Any, obj: Any) -> Any:
        if _is_encoded_tensor_triple(obj):
            return self._decode_tensor(obj)
        if isinstance(obj, list):
            return [_decode_easy_magpie_refit_tensors(self, item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(_decode_easy_magpie_refit_tensors(self, item) for item in obj)
        if isinstance(obj, dict):
            return {key: _decode_easy_magpie_refit_tensors(self, value) for key, value in obj.items()}
        return obj

    def _is_easy_magpie_refit_utility_request(obj: Any) -> bool:
        if not isinstance(obj, (list, tuple)) or len(obj) < 4:
            return False
        utility_method = obj[2]
        utility_args = obj[3]
        if utility_method in easy_magpie_refit_methods:
            return True
        if utility_method != "collective_rpc" or not isinstance(utility_args, (list, tuple)) or not utility_args:
            return False
        return utility_args[0] in easy_magpie_refit_methods

    def decode_with_easy_magpie_refit_tensors(self: Any, bufs: Any) -> Any:
        bytestr_types = (bytes, bytearray, memoryview)
        zmq_module = getattr(serial_utils, "zmq", None)
        frame_cls = getattr(zmq_module, "Frame", None)
        if frame_cls is not None:
            bytestr_types = (*bytestr_types, frame_cls)
        if isinstance(bufs, bytestr_types):
            decoded = self.decoder.decode(bufs)
            if _is_easy_magpie_refit_utility_request(decoded):
                decoded = _decode_easy_magpie_refit_tensors(self, decoded)
            return decoded

        self.aux_buffers = bufs
        _decode_easy_magpie_refit_tensors._active_aux_buffers = bufs  # type: ignore[attr-defined]
        try:
            decoded = self.decoder.decode(bufs[0])
            if _is_easy_magpie_refit_utility_request(decoded):
                decoded = _decode_easy_magpie_refit_tensors(self, decoded)
            return decoded
        finally:
            self.aux_buffers = ()
            _decode_easy_magpie_refit_tensors._active_aux_buffers = ()  # type: ignore[attr-defined]

    decode_with_easy_magpie_refit_tensors._easymagpie_easy_refit_tensor_decode_compat = (  # type: ignore[attr-defined]
        True
    )
    decode_with_easy_magpie_refit_tensors._easymagpie_original = original_decode  # type: ignore[attr-defined]
    decoder_cls.decode = decode_with_easy_magpie_refit_tensors


def _install_cudagraph_mode_compat() -> None:
    try:
        from vllm.config import CUDAGraphMode
    except Exception:
        return
    if hasattr(CUDAGraphMode.NONE, "valid_runtime_modes"):
        return

    def valid_runtime_modes(self):
        return self in (CUDAGraphMode.NONE, CUDAGraphMode.PIECEWISE, CUDAGraphMode.FULL)

    CUDAGraphMode.valid_runtime_modes = valid_runtime_modes  # type: ignore[attr-defined]


def _install_cuda_graph_stat_alias() -> None:
    try:
        import vllm.compilation.cuda_graph as cuda_graph
    except Exception:
        return
    if hasattr(cuda_graph, "CUDAGraphStat"):
        return

    class CUDAGraphStat:
        pass

    cuda_graph.CUDAGraphStat = CUDAGraphStat


def _install_cuda_piecewise_no_sym_shape_compat() -> None:
    try:
        import vllm.compilation.cuda_piecewise_backend as cuda_piecewise_backend
    except Exception:
        return
    backend_cls = getattr(cuda_piecewise_backend, "PiecewiseBackend", None)
    if backend_cls is None:
        return
    original = getattr(backend_cls, "__call__", None)
    if original is None or getattr(original, "_easymagpie_no_sym_shape_compat", False):
        return

    def _runtime_shape_from_args(args: tuple[Any, ...]) -> int | None:
        for arg in args:
            shape = getattr(arg, "shape", None)
            if shape is None or len(shape) == 0:
                continue
            try:
                return int(shape[0])
            except Exception:
                continue
        return None

    def _signature_from_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
        signature: list[Any] = []
        for arg in args:
            shape = getattr(arg, "shape", None)
            stride = getattr(arg, "stride", None)
            dtype = getattr(arg, "dtype", None)
            device = getattr(arg, "device", None)
            if shape is None:
                signature.append((type(arg).__qualname__, repr(arg)))
                continue
            try:
                stride_value = tuple(int(x) for x in stride()) if callable(stride) else None
            except Exception:
                stride_value = None
            try:
                shape_value = tuple(int(x) for x in shape)
            except Exception:
                shape_value = tuple(shape)
            signature.append(
                (
                    "tensor",
                    shape_value,
                    stride_value,
                    str(dtype),
                    str(device),
                )
            )
        return tuple(signature)

    def _compile_no_sym_shape(self, args: tuple[Any, ...], runtime_shape: int | None):
        compile_fn = getattr(getattr(self, "vllm_backend", None), "compiler_manager", None)
        compile_fn = getattr(compile_fn, "compile", None)
        if not callable(compile_fn):
            return None
        return compile_fn(
            self.graph,
            args,
            self.compilation_config.inductor_compile_config,
            self.compilation_config,
            graph_index=self.piecewise_compile_index,
            num_graphs=self.total_piecewise_compiles,
            runtime_shape=runtime_shape,
        )

    def __call__(self, *args):
        if getattr(self, "sym_shape_indices", None):
            return original(self, *args)
        signature = _signature_from_args(args)
        runtime_shape = _runtime_shape_from_args(args)
        runnables = getattr(self, "_easymagpie_no_sym_shape_runnables", None)
        if runnables is None:
            runnables = {}
            self._easymagpie_no_sym_shape_runnables = runnables
        if not getattr(self, "first_run_finished", False):
            self.first_run_finished = True
            runnables[signature] = self.compiled_graph_for_general_shape
            check_for_ending_compilation = getattr(self, "check_for_ending_compilation", None)
            if callable(check_for_ending_compilation):
                check_for_ending_compilation()
            self._easymagpie_no_sym_shape_compile_count = 0
            self._easymagpie_no_sym_shape_last_runtime_shape = runtime_shape
            self._easymagpie_no_sym_shape_last_signature = signature
            return self.compiled_graph_for_general_shape(*args)
        runnable = runnables.get(signature)
        if runnable is None:
            runnable = _compile_no_sym_shape(self, args, runtime_shape)
            if runnable is None:
                runnable = self.compiled_graph_for_general_shape
            runnables[signature] = runnable
            self._easymagpie_no_sym_shape_compile_count = int(
                getattr(self, "_easymagpie_no_sym_shape_compile_count", 0)
            ) + 1
        self._easymagpie_no_sym_shape_last_runtime_shape = runtime_shape
        self._easymagpie_no_sym_shape_last_signature = signature
        return runnable(*args)

    __call__._easymagpie_no_sym_shape_compat = True  # type: ignore[attr-defined]
    __call__._easymagpie_original = original  # type: ignore[attr-defined]
    backend_cls.__call__ = __call__


def _install_kv_connector_stats_alias() -> None:
    module_name = "vllm.distributed.kv_transfer.kv_connector.v1.metrics"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class KVConnectorStats:
        def aggregate(self, _other):
            return self

    module.KVConnectorStats = KVConnectorStats
    sys.modules[module_name] = module


def _install_perf_stats_alias() -> None:
    module_name = "vllm.v1.metrics.perf"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class PerfStats:
        pass

    module.PerfStats = PerfStats
    sys.modules[module_name] = module


def _install_scheduler_make_stats_compat() -> None:
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
    except Exception:
        return
    original = getattr(Scheduler, "make_stats", None)
    if original is None or getattr(original, "_easymagpie_compat", False):
        return

    def make_stats(self, spec_decoding_stats=None, *_args, **_kwargs):
        return original(self, spec_decoding_stats)

    make_stats._easymagpie_compat = True  # type: ignore[attr-defined]
    Scheduler.make_stats = make_stats


def _install_sched_utils_aliases() -> None:
    try:
        import vllm.v1.core.sched.utils as sched_utils
    except Exception:
        return
    if not hasattr(sched_utils, "remove_all"):

        def remove_all(items, removed):
            removed_set = set(removed)
            return [item for item in items if item not in removed_set]

        sched_utils.remove_all = remove_all


def _install_sched_interface_aliases() -> None:
    try:
        import vllm.v1.core.sched.interface as sched_interface
    except Exception:
        return
    if hasattr(sched_interface, "PauseState"):
        return

    from enum import Enum

    class PauseState(Enum):
        UNPAUSED = 0
        PAUSED_ALL = 1

    sched_interface.PauseState = PauseState


def _install_output_processor_signature_compat() -> None:
    try:
        from vllm.v1.engine.output_processor import OutputProcessor, RequestState
    except Exception:
        return

    original_init = getattr(OutputProcessor, "__init__", None)
    if original_init is not None and not getattr(original_init, "_easymagpie_compat", False):

        def __init__(self, tokenizer, log_stats: bool, stream_interval: int = 1, tracing_enabled: bool = False):
            original_init(self, tokenizer=tokenizer, log_stats=log_stats)
            self.stream_interval = int(stream_interval)
            self.tracing_enabled = bool(tracing_enabled)
            if not hasattr(self, "external_req_ids"):
                self.external_req_ids = defaultdict(list)

        __init__._easymagpie_compat = True  # type: ignore[attr-defined]
        OutputProcessor.__init__ = __init__

    original_from_new_descriptor = RequestState.__dict__.get("from_new_request")
    original_from_new = (
        original_from_new_descriptor.__func__
        if isinstance(original_from_new_descriptor, classmethod)
        else original_from_new_descriptor
    )
    if original_from_new is None or getattr(original_from_new, "_easymagpie_compat", False):
        return
    try:
        original_from_new_signature = inspect.signature(original_from_new)
    except Exception:
        original_from_new_signature = None

    def from_new_request(cls, *args, stream_interval: int = 1, **kwargs):
        if original_from_new_signature is not None:
            param_names = list(original_from_new_signature.parameters)
            if "stream_interval" in param_names and "stream_interval" not in kwargs:
                stream_interval_arg_index = param_names.index("stream_interval") - 1
                if len(args) <= stream_interval_arg_index:
                    kwargs["stream_interval"] = stream_interval
        state = original_from_new(cls, *args, **kwargs)
        if not hasattr(state, "stream_interval"):
            state.stream_interval = int(stream_interval)
        if not hasattr(state, "sent_tokens_offset"):
            state.sent_tokens_offset = 0
        request = kwargs.get("request")
        if request is None:
            for arg in args:
                if hasattr(arg, "request_id") and hasattr(arg, "prompt_token_ids"):
                    request = arg
                    break
        external_req_id = getattr(request, "external_req_id", None) or getattr(request, "request_id", None)
        if external_req_id is not None and not hasattr(state, "external_req_id"):
            state.external_req_id = external_req_id
        return state

    from_new_request._easymagpie_compat = True  # type: ignore[attr-defined]
    RequestState.from_new_request = classmethod(from_new_request)


def _install_flash_attention_builder_compat() -> None:
    try:
        import vllm.v1.attention.backends.flash_attn as v1_flash_attn
        from vllm.attention.backends.flash_attn import FlashAttentionBackend as LegacyBackend
        from vllm.attention.backends.flash_attn import FlashAttentionMetadataBuilder as LegacyBuilder
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder as V1Builder
    except Exception:
        return

    original_get_sliding_window_configs = getattr(v1_flash_attn, "_get_sliding_window_configs", None)
    if original_get_sliding_window_configs is not None and not getattr(
        original_get_sliding_window_configs, "_easymagpie_compat", False
    ):

        def _get_sliding_window_configs(vllm_config):
            sliding_window_configs = set()
            layers = v1_flash_attn.get_layers_from_vllm_config(vllm_config, v1_flash_attn.Attention)
            for layer in layers.values():
                impl = getattr(layer, "impl", None)
                sliding_window_configs.add(getattr(impl, "sliding_window", None))
            return sliding_window_configs

        _get_sliding_window_configs._easymagpie_compat = True  # type: ignore[attr-defined]
        v1_flash_attn._get_sliding_window_configs = _get_sliding_window_configs

    original_get_builder_cls = getattr(LegacyBackend, "get_builder_cls", None)
    if original_get_builder_cls is None or getattr(original_get_builder_cls, "_easymagpie_compat", False):
        return
    try:
        current_builder = LegacyBackend.get_builder_cls()
    except Exception:
        current_builder = None
    if current_builder is not LegacyBuilder:
        return

    def get_builder_cls():
        return V1Builder

    get_builder_cls._easymagpie_compat = True  # type: ignore[attr-defined]
    LegacyBackend.get_builder_cls = staticmethod(get_builder_cls)


def _install_worker_utils_compat() -> None:
    try:
        import vllm.v1.worker.utils as worker_utils
        from vllm.utils import GiB_bytes
    except Exception:
        return
    if hasattr(worker_utils, "request_memory"):
        request_memory = worker_utils.request_memory
    else:

        def request_memory(init_snapshot, cache_config):
            requested_memory = init_snapshot.total_memory * cache_config.gpu_memory_utilization
            if init_snapshot.free_memory < requested_memory:
                gib = lambda value: round(value / GiB_bytes, 2)
                raise ValueError(
                    f"Free memory on device ({gib(init_snapshot.free_memory)}/"
                    f"{gib(init_snapshot.total_memory)} GiB) on startup is less "
                    f"than desired GPU memory utilization ({cache_config.gpu_memory_utilization}, "
                    f"{gib(requested_memory)} GiB). Decrease GPU memory utilization "
                    "or reduce GPU memory used by other processes."
                )
            return requested_memory

        worker_utils.request_memory = request_memory

    if not hasattr(worker_utils, "is_residual_scattered_for_sp"):

        def is_residual_scattered_for_sp(*_args, **_kwargs) -> bool:
            return False

        worker_utils.is_residual_scattered_for_sp = is_residual_scattered_for_sp


def _install_worker_workspace_alias() -> None:
    if "vllm.v1.worker.workspace" in sys.modules:
        return
    module = types.ModuleType("vllm.v1.worker.workspace")

    def init_workspace_manager(*_args, **_kwargs):
        return None

    module.init_workspace_manager = init_workspace_manager
    sys.modules["vllm.v1.worker.workspace"] = module


def _install_gpu_ar_worker_default_attrs() -> None:
    try:
        module = importlib.import_module("vllm_omni.worker.gpu_ar_worker")
    except Exception:
        return
    worker_cls = getattr(module, "GPUARWorker", None)
    if worker_cls is None or hasattr(worker_cls, "use_v2_model_runner"):
        return
    # vLLM-Omni still relies on the v1 runner hooks for EasyMagpie. Newer vLLM
    # worker bases no longer create this flag, so default the class lookup to
    # the same v1 path the worker forces when the flag is present.
    worker_cls.use_v2_model_runner = False


def _filter_easy_magpie_refit_weights_for_model(
    model: Any,
    weights: list[tuple[str, Any]],
) -> tuple[list[tuple[str, Any]], list[str]]:
    """Drop converted optional weights that the live vLLM model does not define."""
    try:
        model_param_names = {str(name) for name, _ in model.named_parameters()}
    except Exception:
        model_param_names = set()
    if "context_text_embedding.weight" in model_param_names:
        return weights, []

    filtered: list[tuple[str, Any]] = []
    dropped: list[str] = []
    for name, tensor in weights:
        if name == "context_text_embedding.weight":
            dropped.append(name)
            continue
        filtered.append((name, tensor))
    return filtered, dropped


def _safe_tensor_cache_key(tensor: Any) -> tuple[Any, ...] | None:
    try:
        return (
            str(getattr(tensor, "device", "")),
            int(tensor.data_ptr()),
            tuple(int(dim) for dim in tensor.shape),
            tuple(int(stride) for stride in tensor.stride()),
            int(tensor.storage_offset()),
        )
    except Exception:
        return None


def _zero_cache_tensors(
    value: Any,
    path: str,
    seen: set[tuple[Any, ...]],
) -> tuple[int, int, list[dict[str, Any]], list[str]]:
    try:
        import torch
    except Exception as exc:
        return 0, 0, [], [f"{path}: import torch failed: {type(exc).__name__}: {exc}"]

    if isinstance(value, torch.Tensor):
        cache_key = _safe_tensor_cache_key(value)
        if cache_key is not None and cache_key in seen:
            return 0, 0, [], []
        if cache_key is not None:
            seen.add(cache_key)
        try:
            numel = int(value.numel())
            value.zero_()
            return (
                1,
                numel,
                [
                    {
                        "path": path,
                        "shape": [int(dim) for dim in value.shape],
                        "dtype": str(value.dtype).replace("torch.", ""),
                        "device": str(value.device),
                        "numel": numel,
                    }
                ],
                [],
            )
        except Exception as exc:
            return 0, 0, [], [f"{path}: {type(exc).__name__}: {exc}"]

    if isinstance(value, dict):
        total_tensors = 0
        total_numel = 0
        items: list[dict[str, Any]] = []
        errors: list[str] = []
        for key, item in value.items():
            sub_tensors, sub_numel, sub_items, sub_errors = _zero_cache_tensors(
                item,
                f"{path}.{key}",
                seen,
            )
            total_tensors += sub_tensors
            total_numel += sub_numel
            items.extend(sub_items)
            errors.extend(sub_errors)
        return total_tensors, total_numel, items, errors

    if isinstance(value, (list, tuple)):
        total_tensors = 0
        total_numel = 0
        items = []
        errors = []
        for index, item in enumerate(value):
            sub_tensors, sub_numel, sub_items, sub_errors = _zero_cache_tensors(
                item,
                f"{path}[{index}]",
                seen,
            )
            total_tensors += sub_tensors
            total_numel += sub_numel
            items.extend(sub_items)
            errors.extend(sub_errors)
        return total_tensors, total_numel, items, errors

    return 0, 0, [], []


def _reset_easy_magpie_cache_tensors_after_refit(model_runner: Any) -> dict[str, Any]:
    seen: set[tuple[Any, ...]] = set()
    reset_attrs: list[str] = []
    items: list[dict[str, Any]] = []
    errors: list[str] = []
    num_tensors = 0
    numel = 0

    def reset_value(value: Any, path: str) -> None:
        nonlocal num_tensors, numel
        sub_tensors, sub_numel, sub_items, sub_errors = _zero_cache_tensors(value, path, seen)
        if sub_tensors:
            reset_attrs.append(path)
        num_tensors += sub_tensors
        numel += sub_numel
        items.extend(sub_items)
        errors.extend(sub_errors)

    for attr_name in ("kv_caches", "kv_cache", "gpu_cache", "cache_tensors"):
        if hasattr(model_runner, attr_name):
            reset_value(getattr(model_runner, attr_name), f"model_runner.{attr_name}")

    model = getattr(model_runner, "model", None)
    if model is not None:
        model_mamba_cache = getattr(model, "mamba_cache", None)
        if model_mamba_cache is not None:
            reset_value(model_mamba_cache, "model.mamba_cache")
            try:
                setattr(model, "mamba_cache", None)
                reset_attrs.append("model.mamba_cache=None")
            except Exception as exc:
                errors.append(f"model.mamba_cache=None: {type(exc).__name__}: {exc}")

        modules = getattr(model, "modules", None)
        if callable(modules):
            try:
                for module_index, module in enumerate(modules()):
                    if hasattr(module, "kv_cache"):
                        reset_value(getattr(module, "kv_cache"), f"model.modules[{module_index}].kv_cache")
            except Exception as exc:
                errors.append(f"model.modules: {type(exc).__name__}: {exc}")

    vllm_config = getattr(model_runner, "vllm_config", None)
    static_context = getattr(getattr(vllm_config, "compilation_config", None), "static_forward_context", None)
    if isinstance(static_context, dict):
        for layer_name, layer in static_context.items():
            if hasattr(layer, "kv_cache"):
                reset_value(getattr(layer, "kv_cache"), f"static_forward_context.{layer_name}.kv_cache")

    return {
        "num_tensors": num_tensors,
        "numel": numel,
        "reset_attrs": reset_attrs,
        "items_head": items[:16],
        "items_tail": items[-16:],
        "errors": errors,
    }


def _scrub_empty_easy_magpie_input_batch(input_batch: Any) -> dict[str, Any]:
    if input_batch is None:
        return {"present": False}

    remaining_req_ids = [
        str(req_id)
        for req_id in list(getattr(input_batch, "req_ids", []) or [])
        if req_id is not None
    ]
    result: dict[str, Any] = {
        "present": True,
        "applied": False,
        "remaining_request_ids": len(remaining_req_ids),
        "cleared_fields": [],
        "errors": [],
    }
    if remaining_req_ids:
        result["request_ids_head"] = remaining_req_ids[:8]
        return result

    result["applied"] = True
    for attr_name in ("_req_ids", "req_output_token_ids"):
        value = getattr(input_batch, attr_name, None)
        if isinstance(value, list):
            try:
                size_before = len(value)
                value.clear()
                result["cleared_fields"].append({"name": attr_name, "size_before": size_before})
            except Exception as exc:
                result["errors"].append(f"{attr_name}: {type(exc).__name__}: {exc}")

    for attr_name in (
        "req_id_to_index",
        "generators",
        "num_logprobs",
        "num_prompt_logprobs",
        "in_progress_prompt_logprobs_cpu",
        "bad_words_token_ids",
        "pooling_params",
    ):
        value = getattr(input_batch, attr_name, None)
        if isinstance(value, dict):
            try:
                size_before = len(value)
                value.clear()
                result["cleared_fields"].append({"name": attr_name, "size_before": size_before})
            except Exception as exc:
                result["errors"].append(f"{attr_name}: {type(exc).__name__}: {exc}")

    for attr_name in (
        "greedy_reqs",
        "random_reqs",
        "top_p_reqs",
        "top_k_reqs",
        "spec_decode_unsupported_reqs",
        "frequency_penalties_reqs",
        "presence_penalties_reqs",
        "repetition_penalties_reqs",
        "has_allowed_token_ids",
    ):
        value = getattr(input_batch, attr_name, None)
        if isinstance(value, set):
            try:
                size_before = len(value)
                value.clear()
                result["cleared_fields"].append({"name": attr_name, "size_before": size_before})
            except Exception as exc:
                result["errors"].append(f"{attr_name}: {type(exc).__name__}: {exc}")

    for attr_name in (
        "token_ids_cpu_tensor",
        "num_computed_tokens_cpu_tensor",
        "allowed_token_ids_mask_cpu_tensor",
    ):
        value = getattr(input_batch, attr_name, None)
        if hasattr(value, "zero_"):
            try:
                value.zero_()
                result["cleared_fields"].append({"name": attr_name, "zeroed": True})
            except Exception as exc:
                result["errors"].append(f"{attr_name}.zero_: {type(exc).__name__}: {exc}")

    for attr_name in (
        "token_ids_cpu",
        "num_tokens",
        "num_tokens_no_spec",
        "num_prompt_tokens",
        "num_computed_tokens_cpu",
        "temperature_cpu",
        "top_p_cpu",
        "top_k_cpu",
        "frequency_penalties_cpu",
        "presence_penalties_cpu",
        "repetition_penalties_cpu",
    ):
        value = getattr(input_batch, attr_name, None)
        if hasattr(value, "fill"):
            try:
                value.fill(0)
                result["cleared_fields"].append({"name": attr_name, "filled": 0})
            except Exception as exc:
                result["errors"].append(f"{attr_name}.fill: {type(exc).__name__}: {exc}")

    block_table = getattr(input_batch, "block_table", None)
    clear = getattr(block_table, "clear", None)
    if callable(clear):
        try:
            clear()
            result["cleared_fields"].append({"name": "block_table", "cleared": True})
        except Exception as exc:
            result["errors"].append(f"block_table.clear: {type(exc).__name__}: {exc}")

    batch_update_builder = getattr(input_batch, "batch_update_builder", None)
    get_and_reset = getattr(batch_update_builder, "get_and_reset", None)
    if callable(get_and_reset):
        try:
            get_and_reset(0)
            result["cleared_fields"].append({"name": "batch_update_builder", "reset": True})
        except Exception as exc:
            result["errors"].append(f"batch_update_builder.get_and_reset: {type(exc).__name__}: {exc}")

    return result


def _reset_easy_magpie_runner_state_after_refit(model_runner: Any) -> dict[str, Any]:
    """Clear runner-side request/cache state after live refit.

    EasyMagpie refit is a between-rollout operation. Finished or aborted
    request ids left in the model runner can still influence the next Mamba
    cache update, so remove them before admitting a post-refit request.
    """

    if model_runner is None:
        return {
            "cleared_attrs": [],
            "missing_attrs": [],
            "cleared_mappings": [],
            "input_batch_reset": {"present": False},
            "input_batch_scrub": {"present": False},
            "cache_tensors_reset": {"num_tensors": 0, "numel": 0, "reset_attrs": [], "errors": []},
            "errors": ["model_runner is None"],
        }

    cleared_attrs: list[str] = []
    missing_attrs: list[str] = []
    cleared_mappings: list[dict[str, Any]] = []
    errors: list[str] = []
    for name in (
        "_easymagpie_active_mamba_request_ids",
        "_easymagpie_mamba_request_cache_kwargs",
        "_easymagpie_mamba_request_cache_state",
        "_easymagpie_mamba_cache_batch_size",
    ):
        if not hasattr(model_runner, name):
            missing_attrs.append(name)
            continue
        try:
            delattr(model_runner, name)
            cleared_attrs.append(name)
        except Exception as exc:
            errors.append(f"{name}: {type(exc).__name__}: {exc}")

    for name in ("requests", "encoder_cache"):
        mapping = getattr(model_runner, name, None)
        if not isinstance(mapping, dict):
            continue
        size_before = len(mapping)
        try:
            mapping.clear()
            cleared_mappings.append({"name": name, "size_before": size_before})
        except Exception as exc:
            errors.append(f"{name}.clear: {type(exc).__name__}: {exc}")

    input_batch_reset: dict[str, Any] = {"present": False}
    input_batch = getattr(model_runner, "input_batch", None)
    if input_batch is not None:
        req_ids = [
            str(req_id)
            for req_id in list(getattr(input_batch, "req_ids", []) or [])
            if req_id is not None
        ]
        removed: list[str] = []
        missing: list[str] = []
        input_batch_errors: list[str] = []
        remove_request = getattr(input_batch, "remove_request", None)
        if callable(remove_request):
            for req_id in req_ids:
                try:
                    removed_index = remove_request(req_id)
                except Exception as exc:
                    input_batch_errors.append(f"{req_id}: {type(exc).__name__}: {exc}")
                    continue
                if removed_index is None:
                    missing.append(req_id)
                else:
                    removed.append(req_id)
            condense = getattr(input_batch, "condense", None)
            if callable(condense):
                try:
                    condense()
                except Exception as exc:
                    input_batch_errors.append(f"condense: {type(exc).__name__}: {exc}")
        else:
            raw_req_ids = getattr(input_batch, "_req_ids", None)
            req_id_to_index = getattr(input_batch, "req_id_to_index", None)
            if isinstance(raw_req_ids, list) and isinstance(req_id_to_index, dict):
                raw_req_ids.clear()
                req_id_to_index.clear()
                removed = req_ids
            else:
                input_batch_errors.append("input_batch has no remove_request method")
        errors.extend(f"input_batch.{error}" for error in input_batch_errors)
        input_batch_reset = {
            "present": True,
            "num_request_ids_before": len(req_ids),
            "num_removed": len(removed),
            "num_missing": len(missing),
            "request_ids_before_head": req_ids[:8],
            "request_ids_before_tail": req_ids[-8:],
            "errors": input_batch_errors,
        }

    input_batch_scrub = _scrub_empty_easy_magpie_input_batch(input_batch)
    cache_tensors_reset = _reset_easy_magpie_cache_tensors_after_refit(model_runner)
    errors.extend(f"input_batch_scrub.{error}" for error in input_batch_scrub.get("errors", []))
    errors.extend(f"cache_tensors_reset.{error}" for error in cache_tensors_reset.get("errors", []))

    return {
        "cleared_attrs": cleared_attrs,
        "missing_attrs": missing_attrs,
        "cleared_mappings": cleared_mappings,
        "input_batch_reset": input_batch_reset,
        "input_batch_scrub": input_batch_scrub,
        "cache_tensors_reset": cache_tensors_reset,
        "errors": errors,
    }


_EASYMAGPIE_REFIT_RPC_COMPAT_VERSION = 5
_EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION = 1


def _install_easy_magpie_refit_rpc_compat() -> None:
    """Expose a tiny worker RPC for loading EasyMagpie refit weights."""

    def _looks_like_refit_pair(item: Any) -> bool:
        return isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], str)

    def _materialize_refit_weights(weights: Any) -> list[tuple[str, Any]]:
        if isinstance(weights, (str, bytes, os.PathLike)):
            try:
                import torch

                loaded = torch.load(os.fspath(weights), map_location="cpu", weights_only=True)
            except TypeError:
                import torch

                loaded = torch.load(os.fspath(weights), map_location="cpu")
            if isinstance(loaded, dict) and "weights" in loaded:
                loaded = loaded["weights"]
            return _materialize_refit_weights(loaded)

        if isinstance(weights, dict) and "path" in weights:
            return _materialize_refit_weights(weights["path"])

        if isinstance(weights, dict):
            raw_items = list(weights.items())
        else:
            raw_items = list(weights)

        # Some collective_rpc paths preserve the outer argument tuple, so the
        # worker receives ``([("name", tensor), ...],)`` instead of the list.
        if (
            len(raw_items) == 1
            and isinstance(raw_items[0], (list, tuple))
            and not _looks_like_refit_pair(raw_items[0])
            and all(_looks_like_refit_pair(item) for item in raw_items[0])
        ):
            raw_items = list(raw_items[0])

        materialized: list[tuple[str, Any]] = []
        for item in raw_items:
            if not _looks_like_refit_pair(item):
                raise TypeError(f"invalid EasyMagpie refit weight entry: {type(item).__name__}")
            name, tensor = item
            # A few RPC transports normalize tuple payloads to nested lists,
            # e.g. ["decoder.foo", [tensor]]. Keep the model loader strict but
            # undo that harmless wrapper here.
            if isinstance(tensor, (list, tuple)) and len(tensor) == 1 and hasattr(tensor[0], "shape"):
                tensor = tensor[0]
            if not hasattr(tensor, "shape"):
                try:
                    import torch

                    tensor = torch.as_tensor(tensor)
                except Exception as exc:
                    raise TypeError(f"invalid tensor for EasyMagpie refit weight {name!r}") from exc
            materialized.append((name, tensor))
        return materialized

    def _materialize_text_row_payload(payload: Any) -> tuple[Any, list[tuple[str, Any]]]:
        if isinstance(payload, (str, bytes, os.PathLike)):
            try:
                import torch

                loaded = torch.load(os.fspath(payload), map_location="cpu", weights_only=True)
            except TypeError:
                import torch

                loaded = torch.load(os.fspath(payload), map_location="cpu")
            return _materialize_text_row_payload(loaded)
        if isinstance(payload, dict) and "path" in payload:
            return _materialize_text_row_payload(payload["path"])
        if not isinstance(payload, dict):
            raise TypeError(f"invalid EasyMagpie text-row refit payload: {type(payload).__name__}")
        row_ids = payload.get("row_ids")
        rows = payload.get("weights", payload.get("rows"))
        if row_ids is None or rows is None:
            raise ValueError("EasyMagpie text-row refit payload needs row_ids and weights")
        return row_ids, _materialize_refit_weights(rows)

    def _sample_text_row_copy_check(
        *,
        name: str,
        row_ids: Any,
        source_rows: Any,
        target_weight: Any,
    ) -> dict[str, Any]:
        import torch

        ids = torch.as_tensor(row_ids, dtype=torch.long, device=target_weight.device).reshape(-1)
        src = source_rows.to(device=target_weight.device, dtype=target_weight.dtype)
        sampled = target_weight.index_select(0, ids)
        if src.numel() == 0:
            max_diff = 0.0
            matched = True
        else:
            diff = (sampled.float() - src.float()).abs()
            max_diff = float(diff.max().item()) if diff.numel() else 0.0
            matched = bool(torch.allclose(sampled.float(), src.float(), atol=5e-3, rtol=5e-3))
        return {
            "name": name,
            "checked": True,
            "matched": matched,
            "num_rows": int(ids.numel()),
            "row_ids_head": [int(x) for x in ids[:8].detach().cpu().tolist()],
            "row_ids_tail": [int(x) for x in ids[-8:].detach().cpu().tolist()],
            "source_shape": list(source_rows.shape),
            "target_shape": list(target_weight.shape),
            "max_sample_abs_diff": max_diff,
        }

    def _apply_text_row_update(model: Any, row_ids: Any, rows: list[tuple[str, Any]]) -> dict[str, Any]:
        import torch

        ids_cpu = torch.as_tensor(row_ids, dtype=torch.long, device="cpu").reshape(-1)
        if ids_cpu.numel() <= 0:
            raise ValueError("EasyMagpie text-row refit needs at least one row")
        if len(torch.unique(ids_cpu)) != ids_cpu.numel():
            raise ValueError("EasyMagpie text-row refit row_ids must be unique")
        own_params = dict(model.named_parameters())
        updated: list[str] = []
        dropped: list[str] = []
        copy_checks: list[dict[str, Any]] = []
        for name, tensor in rows:
            name = str(name)
            if name not in {"text_embedding.weight", "context_text_embedding.weight"}:
                raise ValueError(f"unsupported EasyMagpie text-row refit weight: {name!r}")
            target = own_params.get(name)
            if target is None:
                if name == "context_text_embedding.weight" and getattr(model, "context_text_embedding", None) is getattr(model, "text_embedding", None):
                    dropped.append(name)
                    continue
                raise KeyError(f"EasyMagpie text-row refit target not found: {name}")
            values = tensor if hasattr(tensor, "shape") else torch.as_tensor(tensor)
            if values.ndim != 2:
                raise ValueError(f"{name} row tensor must be 2-D, got shape {tuple(values.shape)}")
            if int(values.shape[0]) != int(ids_cpu.numel()) or int(values.shape[1]) != int(target.shape[1]):
                raise ValueError(
                    f"{name} row tensor shape {tuple(values.shape)} does not match "
                    f"row_ids={int(ids_cpu.numel())}, hidden={int(target.shape[1])}"
                )
            min_id = int(ids_cpu.min().item())
            max_id = int(ids_cpu.max().item())
            if min_id < 0 or max_id >= int(target.shape[0]):
                raise ValueError(
                    f"{name} row ids must be in [0, {int(target.shape[0]) - 1}], got [{min_id}, {max_id}]"
                )
            ids = ids_cpu.to(device=target.device)
            with torch.no_grad():
                target.data.index_copy_(0, ids, values.to(device=target.device, dtype=target.dtype))
            updated.append(name)
            copy_checks.append(
                _sample_text_row_copy_check(
                    name=name,
                    row_ids=ids,
                    source_rows=values,
                    target_weight=target,
                )
            )
        failed = [item for item in copy_checks if not bool(item.get("matched"))]
        return {
            "ok": not failed and bool(updated),
            "num_rows": int(ids_cpu.numel()),
            "row_ids_head": [int(x) for x in ids_cpu[:8].tolist()],
            "row_ids_tail": [int(x) for x in ids_cpu[-8:].tolist()],
            "updated": updated,
            "dropped_unsupported": dropped,
            "copy_check": {
                "ok": not failed,
                "num_checked": len(copy_checks),
                "num_failed": len(failed),
                "failed_head": failed[:8],
                "items": copy_checks[:8],
            },
        }

    _MISSING_ATTR = object()

    def _iter_easy_magpie_refit_flag_targets(model: Any):
        seen: set[int] = set()
        stack = [model]
        while stack:
            target = stack.pop()
            if target is None:
                continue
            target_id = id(target)
            if target_id in seen:
                continue
            seen.add(target_id)
            if target is model or callable(getattr(target, "load_weights", None)) or hasattr(target, "text_embedding"):
                yield target

            modules = getattr(target, "modules", None)
            if callable(modules):
                try:
                    stack.extend(child for child in modules() if child is not target)
                except Exception:
                    pass
            for attr_name in ("model", "module", "_module"):
                try:
                    child = getattr(target, attr_name, None)
                except Exception:
                    child = None
                if child is not None and child is not target:
                    stack.append(child)

    def _load_easy_magpie_weights_for_refit(
        model: Any,
        *,
        weights: list[tuple[str, Any]],
        allow_missing_text_tables: bool,
    ):
        flag_targets = list(_iter_easy_magpie_refit_flag_targets(model))
        previous_refit_active: list[tuple[Any, Any]] = []
        previous_allow_missing: list[tuple[Any, Any]] = []
        for target in flag_targets:
            previous_value = getattr(target, "_easymagpie_refit_rpc_active", _MISSING_ATTR)
            try:
                target._easymagpie_refit_rpc_active = True
            except Exception:
                continue
            previous_refit_active.append((target, previous_value))
        if allow_missing_text_tables:
            for target in flag_targets:
                previous_value = getattr(target, "_easymagpie_allow_missing_text_tables_refit", _MISSING_ATTR)
                try:
                    target._easymagpie_allow_missing_text_tables_refit = True
                except Exception:
                    continue
                previous_allow_missing.append((target, previous_value))
        try:
            return model.load_weights(weights=weights)
        finally:
            for target, previous_value in reversed(previous_allow_missing):
                if previous_value is _MISSING_ATTR:
                    try:
                        delattr(target, "_easymagpie_allow_missing_text_tables_refit")
                    except Exception:
                        pass
                else:
                    try:
                        target._easymagpie_allow_missing_text_tables_refit = previous_value
                    except Exception:
                        pass
            for target, previous_value in reversed(previous_refit_active):
                if previous_value is _MISSING_ATTR:
                    try:
                        delattr(target, "_easymagpie_refit_rpc_active")
                    except Exception:
                        pass
                else:
                    try:
                        target._easymagpie_refit_rpc_active = previous_value
                    except Exception:
                        pass
            try:
                flag_targets.clear()
            except Exception:
                pass

    worker_classes = []
    for module_name, class_name in (
        ("vllm_omni.worker.gpu_ar_worker", "GPUARWorker"),
        ("vllm_omni.worker.gpu_generation_worker", "GPUGenerationWorker"),
    ):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        worker_cls = getattr(module, class_name, None)
        if worker_cls is not None:
            worker_classes.append(worker_cls)

    for worker_cls in worker_classes:
        installed_version = int(getattr(worker_cls, "_easymagpie_refit_rpc_compat_version", 0) or 0)
        if (
            getattr(worker_cls, "_easymagpie_refit_rpc_compat", False)
            and installed_version >= _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION
        ):
            continue

        def _easymagpie_load_weights_impl(self, weights, *, allow_missing_text_tables: bool):
            model_runner = getattr(self, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if model is None or not hasattr(model, "load_weights"):
                return {
                    "ok": False,
                    "refit_rpc_compat_version": _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION,
                    "error": "EasyMagpie refit RPC could not find model_runner.model.load_weights",
                }
            materialized = []
            try:
                materialized = _materialize_refit_weights(weights)
                materialized, dropped = _filter_easy_magpie_refit_weights_for_model(model, materialized)
                loaded = _load_easy_magpie_weights_for_refit(
                    model,
                    weights=materialized,
                    allow_missing_text_tables=allow_missing_text_tables,
                )
                runtime_state_reset = _reset_easy_magpie_runner_state_after_refit(model_runner)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                input_names = [str(name) for name, _ in materialized]
                loaded_names = sorted(str(item) for item in loaded) if loaded is not None else []
                num_loaded = len(loaded_names) if loaded is not None else None
                load_summary = getattr(model, "_last_easy_magpie_load_weights_summary", None)
                if isinstance(load_summary, dict):
                    load_ok = bool(load_summary.get("ok", True))
                    reset_ok = not runtime_state_reset.get("errors")
                    result = {
                        "ok": bool(load_ok and reset_ok),
                        "num_input_tensors": int(load_summary.get("num_input_tensors", len(materialized))),
                        "num_loaded": int(load_summary.get("num_loaded_targets", num_loaded or 0)),
                        "num_loaded_targets": int(load_summary.get("num_loaded_targets", num_loaded or 0)),
                        "input_head": load_summary.get("input_head", input_names[:16]),
                        "input_tail": load_summary.get("input_tail", input_names[-16:]),
                        "loaded_head": load_summary.get("loaded_head", loaded_names[:16]),
                        "loaded_tail": load_summary.get("loaded_tail", loaded_names[-16:]),
                        "dropped_unsupported": dropped,
                        "allow_missing_text_tables": bool(allow_missing_text_tables),
                        "load_summary": load_summary,
                        "runtime_state_reset": runtime_state_reset,
                        "refit_rpc_compat_version": _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION,
                    }
                    if not load_ok:
                        result["error"] = "EasyMagpie refit load summary reported incomplete update"
                    elif not reset_ok:
                        result["error"] = "EasyMagpie refit loaded weights but failed to reset runner runtime state"
                    return result
                if num_loaded is not None and num_loaded != len(materialized):
                    return {
                        "ok": False,
                        "error": (
                            "EasyMagpie refit loaded tensor count mismatch: "
                            f"loaded {num_loaded} of {len(materialized)} input tensors"
                        ),
                        "num_input_tensors": len(materialized),
                        "num_loaded": num_loaded,
                        "input_head": input_names[:16],
                        "input_tail": input_names[-16:],
                        "loaded_head": loaded_names[:16],
                        "loaded_tail": loaded_names[-16:],
                        "dropped_unsupported": dropped,
                        "allow_missing_text_tables": bool(allow_missing_text_tables),
                        "runtime_state_reset": runtime_state_reset,
                        "refit_rpc_compat_version": _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION,
                    }
                reset_ok = not runtime_state_reset.get("errors")
                return {
                    "ok": bool(reset_ok),
                    "num_input_tensors": len(materialized),
                    "num_loaded": num_loaded,
                    "input_head": input_names[:16],
                    "input_tail": input_names[-16:],
                    "loaded_head": loaded_names[:16],
                    "loaded_tail": loaded_names[-16:],
                    "dropped_unsupported": dropped,
                    "allow_missing_text_tables": bool(allow_missing_text_tables),
                    "runtime_state_reset": runtime_state_reset,
                    "refit_rpc_compat_version": _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION,
                    **(
                        {"error": "EasyMagpie refit loaded weights but failed to reset runner runtime state"}
                        if not reset_ok
                        else {}
                    ),
                }
            except Exception as exc:
                logger.exception("EasyMagpie refit RPC failed")
                try:
                    num_input_tensors = len(weights)
                except Exception:
                    num_input_tensors = None
                return {
                    "ok": False,
                    "refit_rpc_compat_version": _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                    "num_input_tensors": num_input_tensors,
                }

        def easymagpie_load_weights(self, weights):
            return _easymagpie_load_weights_impl(
                self,
                weights,
                allow_missing_text_tables=False,
            )

        def easymagpie_load_non_text_weights(self, weights):
            result = _easymagpie_load_weights_impl(
                self,
                weights,
                allow_missing_text_tables=True,
            )
            if isinstance(result, dict):
                result["refit_mode"] = "non_text_weights"
            return result

        def easymagpie_update_text_embedding_rows(self, payload):
            model_runner = getattr(self, "model_runner", None)
            model = getattr(model_runner, "model", None)
            if model is None or not hasattr(model, "named_parameters"):
                return {
                    "ok": False,
                    "text_row_refit_rpc_compat_version": _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION,
                    "error": "EasyMagpie text-row refit RPC could not find model_runner.model.named_parameters",
                }
            try:
                row_ids, rows = _materialize_text_row_payload(payload)
                update_summary = _apply_text_row_update(model, row_ids, rows)
                runtime_state_reset = _reset_easy_magpie_runner_state_after_refit(model_runner)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                reset_ok = not runtime_state_reset.get("errors")
                result = {
                    **update_summary,
                    "runtime_state_reset": runtime_state_reset,
                    "text_row_refit_rpc_compat_version": _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION,
                }
                result["ok"] = bool(update_summary.get("ok")) and reset_ok
                if not result["ok"] and not reset_ok:
                    result["error"] = "EasyMagpie text-row refit loaded rows but failed to reset runner runtime state"
                return result
            except Exception as exc:
                logger.exception("EasyMagpie text-row refit RPC failed")
                return {
                    "ok": False,
                    "text_row_refit_rpc_compat_version": _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        easymagpie_load_weights._easymagpie_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_load_weights._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        easymagpie_load_non_text_weights._easymagpie_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_load_non_text_weights._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        easymagpie_update_text_embedding_rows._easymagpie_text_row_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_update_text_embedding_rows._easymagpie_text_row_refit_rpc_compat_version = _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        worker_cls.easymagpie_load_weights = easymagpie_load_weights
        worker_cls.easymagpie_load_non_text_weights = easymagpie_load_non_text_weights
        worker_cls.easymagpie_update_text_embedding_rows = easymagpie_update_text_embedding_rows
        worker_cls._easymagpie_refit_rpc_compat = True
        worker_cls._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION
        worker_cls._easymagpie_text_row_refit_rpc_compat = True
        worker_cls._easymagpie_text_row_refit_rpc_compat_version = _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION


def install_easy_magpie_refit_rpc_compat() -> None:
    """Install only the EasyMagpie refit worker RPC, leaving generation untouched."""

    _install_ray_objectref_aliases()
    _install_ray_placement_group_aliases()
    _install_vllm_inputs_data_alias()
    _install_vllm_multimodal_inputs_alias()
    _install_engine_utils_compat()
    _install_easy_magpie_refit_rpc_compat()
    _install_async_omni_client_compat()


def install_easy_magpie_runtime_compat() -> None:
    """Install focused shims needed by EasyMagpie RL rollout workers.

    This intentionally avoids the broad vLLM-Omni import/signature shims in
    ``install_vllm_omni_compat``. The RL path needs the refit RPC plus a dense
    tensor serializer for vLLM V1 outputs, but the broader compatibility layer
    can perturb newer vLLM/Pydantic stacks.
    """

    _install_ray_objectref_aliases()
    _install_ray_placement_group_aliases()
    _install_vllm_inputs_data_alias()
    _install_vllm_multimodal_inputs_alias()
    _install_engine_utils_compat()
    _install_easy_magpie_refit_rpc_compat()
    _install_v1_serial_utils_dense_tensor_compat()
    _install_async_omni_client_compat()


def _install_forward_context_kwargs_compat() -> None:
    try:
        import vllm.forward_context as forward_context
    except Exception:
        return
    original = getattr(forward_context, "set_forward_context", None)
    if original is None or getattr(original, "_easymagpie_compat", False):
        return
    original_signature = inspect.signature(original)
    supported_kwargs = set(original_signature.parameters)

    def set_forward_context(*args, **kwargs):
        context_kwargs = dict(kwargs)
        filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported_kwargs}
        bound = original_signature.bind_partial(*args, **filtered_kwargs)
        if "attn_metadata" in bound.arguments:
            for key, value in bound.arguments.items():
                if key not in ("attn_metadata", "vllm_config"):
                    context_kwargs.setdefault(key, value)
            attn_metadata = _normalize_mamba2_attention_metadata_groups(bound.arguments["attn_metadata"])
            attn_metadata = _synthetic_profile_attention_metadata_from_context(
                attn_metadata,
                bound.arguments.get("vllm_config"),
                context_kwargs,
            )
            bound.arguments["attn_metadata"] = _as_v1_mamba2_attention_metadata_dict(
                attn_metadata,
                bound.arguments.get("vllm_config"),
            )
        return original(*bound.args, **bound.kwargs)

    set_forward_context._easymagpie_compat = True  # type: ignore[attr-defined]
    forward_context.set_forward_context = set_forward_context
    for module_name in (
        "vllm_omni.worker.gpu_model_runner",
        "vllm_omni.worker.gpu_ar_model_runner",
        "vllm_omni.worker.gpu_generation_model_runner",
    ):
        module = sys.modules.get(module_name)
        if module is not None and getattr(module, "set_forward_context", None) is original:
            module.set_forward_context = set_forward_context


def _install_omni_gpu_model_runner_method_compat() -> None:
    try:
        from vllm.config import CUDAGraphMode
        module = importlib.import_module("vllm_omni.worker.gpu_model_runner")
    except Exception:
        return
    _install_cudagraph_mode_compat()
    runner_cls = getattr(module, "OmniGPUModelRunner", None)
    if runner_cls is None:
        return
    if not hasattr(runner_cls, "enable_prompt_embeds"):
        runner_cls.enable_prompt_embeds = False
    if not hasattr(runner_cls, "uses_xdrope_dim"):
        runner_cls.uses_xdrope_dim = 0
    if not hasattr(runner_cls, "kv_cache_config"):

        def kv_cache_config(self):
            return _as_v1_kv_cache_config(getattr(self, "_kv_cache_config", getattr(self, "cache_config", None)))

        def set_kv_cache_config(self, value):
            self._kv_cache_config = value

        def del_kv_cache_config(self):
            if hasattr(self, "_kv_cache_config"):
                delattr(self, "_kv_cache_config")

        runner_cls.kv_cache_config = property(kv_cache_config, set_kv_cache_config, del_kv_cache_config)

    if not hasattr(runner_cls, "_determine_batch_execution_and_padding"):

        class _BatchDescriptor:
            def __init__(self, *, num_tokens: int, num_reqs: int, uniform_decode: bool = False) -> None:
                self.num_tokens = num_tokens
                self.num_reqs = num_reqs
                self.uniform_decode = uniform_decode

        def _determine_batch_execution_and_padding(self, **kwargs):
            num_tokens = int(kwargs["num_tokens"])
            num_reqs = int(kwargs["num_reqs"])
            force_uniform_decode = bool(kwargs.get("force_uniform_decode", False))
            if hasattr(self, "get_dp_padding"):
                num_pad, num_tokens_across_dp = self.get_dp_padding(num_tokens)
                num_tokens += int(num_pad)
            else:
                num_tokens_across_dp = None
            batch_desc = _BatchDescriptor(
                num_tokens=num_tokens,
                num_reqs=num_reqs,
                uniform_decode=force_uniform_decode,
            )
            return CUDAGraphMode.NONE, batch_desc, False, num_tokens_across_dp, None

        runner_cls._determine_batch_execution_and_padding = _determine_batch_execution_and_padding

    if not hasattr(runner_cls, "synchronize_input_prep"):

        def synchronize_input_prep(self):
            return contextlib.nullcontext()

        runner_cls.synchronize_input_prep = synchronize_input_prep

    if not hasattr(runner_cls, "_register_layerwise_nvtx_hooks"):

        def _register_layerwise_nvtx_hooks(self):
            return None

        runner_cls._register_layerwise_nvtx_hooks = _register_layerwise_nvtx_hooks

    original_build_attention_metadata = getattr(runner_cls, "_build_attention_metadata", None)
    if original_build_attention_metadata is not None and not getattr(
        original_build_attention_metadata, "_easymagpie_compat", False
    ):
        original_build_attention_metadata_signature = inspect.signature(original_build_attention_metadata)

        def _build_attention_metadata(self, *args, **kwargs):
            bound = original_build_attention_metadata_signature.bind_partial(self, *args, **kwargs)
            result = original_build_attention_metadata(*bound.args, **bound.kwargs)
            attn_metadata = result[0] if isinstance(result, tuple) and result else result
            if isinstance(attn_metadata, dict) and not attn_metadata:
                num_reqs = bound.arguments.get("num_reqs")
                num_tokens = bound.arguments.get("num_tokens", 0)
                max_query_len = bound.arguments.get("max_query_len", num_tokens)
                if num_reqs is not None:
                    num_reqs = int(num_reqs)
                    self._easymagpie_mamba_cache_batch_size = int(num_reqs)
                    attn_metadata = _build_profile_flash_attention_metadata(
                        self,
                        num_tokens=int(num_tokens),
                        num_reqs=num_reqs,
                        max_query_len=int(max_query_len),
                    )
                    if isinstance(result, tuple):
                        result = (attn_metadata, *result[1:])
                    else:
                        result = attn_metadata
            else:
                normalized_attn_metadata = _normalize_mamba2_attention_metadata_groups(attn_metadata)
                if normalized_attn_metadata is not attn_metadata:
                    attn_metadata = normalized_attn_metadata
                    if isinstance(result, tuple):
                        result = (attn_metadata, *result[1:])
                    else:
                        result = attn_metadata
                num_reqs = bound.arguments.get("num_reqs")
                if num_reqs is not None:
                    self._easymagpie_mamba_cache_batch_size = int(num_reqs)
                elif hasattr(self, "_easymagpie_mamba_cache_batch_size"):
                    delattr(self, "_easymagpie_mamba_cache_batch_size")
            return result

        _build_attention_metadata._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls._build_attention_metadata = _build_attention_metadata

    original_init_model_kwargs = getattr(runner_cls, "_init_model_kwargs", None)
    if original_init_model_kwargs is not None and not getattr(
        original_init_model_kwargs, "_easymagpie_compat", False
    ):
        try:
            signature = inspect.signature(original_init_model_kwargs)
            parameters = list(signature.parameters.values())
            accepts_num_tokens = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in parameters
            ) or (len(parameters) >= 2)
        except Exception:
            accepts_num_tokens = True

        def _call_base_init_model_kwargs(self, num_tokens):
            for base_cls in getattr(runner_cls, "__mro__", ())[1:]:
                base_init_model_kwargs = base_cls.__dict__.get("_init_model_kwargs")
                if base_init_model_kwargs is None:
                    continue
                if getattr(base_init_model_kwargs, "_easymagpie_compat", False):
                    continue
                if base_init_model_kwargs is original_init_model_kwargs:
                    continue
                try:
                    base_signature = inspect.signature(base_init_model_kwargs)
                    base_parameters = list(base_signature.parameters.values())
                    base_accepts_num_tokens = any(
                        parameter.kind == inspect.Parameter.VAR_POSITIONAL for parameter in base_parameters
                    ) or (len(base_parameters) >= 2)
                except Exception:
                    base_accepts_num_tokens = True
                if base_accepts_num_tokens:
                    try:
                        return base_init_model_kwargs(self, num_tokens)
                    except TypeError:
                        pass
                return base_init_model_kwargs(self)
            return None

        def _init_model_kwargs(self, num_tokens=None):
            if num_tokens is None:
                num_tokens = int(getattr(self, "max_num_tokens", 0) or 0)
            if accepts_num_tokens:
                try:
                    model_kwargs = original_init_model_kwargs(self, num_tokens)
                except TypeError:
                    model_kwargs = _call_base_init_model_kwargs(self, num_tokens)
                    if model_kwargs is None:
                        model_kwargs = original_init_model_kwargs(self)
            else:
                try:
                    model_kwargs = original_init_model_kwargs(self)
                except TypeError:
                    model_kwargs = _call_base_init_model_kwargs(self, num_tokens)
                    if model_kwargs is None:
                        raise
            cache_batch_size = getattr(self, "_easymagpie_mamba_cache_batch_size", None)
            if cache_batch_size is not None:
                model_kwargs = dict(model_kwargs)
                model_kwargs.setdefault("easymagpie_mamba_cache_batch_size", int(cache_batch_size))
            request_cache_kwargs = _easymagpie_materialize_request_cache_kwargs(
                self,
                getattr(self, "_easymagpie_mamba_request_cache_state", None)
                or getattr(self, "_easymagpie_mamba_request_cache_kwargs", None),
            )
            if request_cache_kwargs:
                model_kwargs = dict(model_kwargs)
                model_kwargs.update(request_cache_kwargs)
            return model_kwargs

        _init_model_kwargs._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls._init_model_kwargs = _init_model_kwargs

    original_get_cumsum_and_arange = getattr(runner_cls, "_get_cumsum_and_arange", None)
    if original_get_cumsum_and_arange is not None and not getattr(
        original_get_cumsum_and_arange, "_easymagpie_compat", False
    ):
        try:
            cumsum_signature = inspect.signature(original_get_cumsum_and_arange)
            cumsum_parameters = list(cumsum_signature.parameters.values())
            requires_arange_out = any(
                parameter.name == "arange_out" and parameter.default is inspect.Parameter.empty
                for parameter in cumsum_parameters
            )
        except Exception:
            requires_arange_out = False

        def _new_arange_out(self, num_tokens):
            import numpy as np

            total_num_tokens = int(np.asarray(num_tokens).sum())
            arange_np = getattr(self, "arange_np", None)
            dtype = getattr(arange_np, "dtype", None) or getattr(num_tokens, "dtype", None) or np.int64
            return np.empty(total_num_tokens, dtype=dtype)

        def _get_cumsum_and_arange(self, num_tokens, arange_out=None, *args, **kwargs):
            if not requires_arange_out:
                if arange_out is None:
                    return original_get_cumsum_and_arange(self, num_tokens, *args, **kwargs)
                return original_get_cumsum_and_arange(self, num_tokens, arange_out, *args, **kwargs)

            if arange_out is not None:
                return original_get_cumsum_and_arange(self, num_tokens, arange_out, *args, **kwargs)

            generated_arange_out = _new_arange_out(self, num_tokens)
            result = original_get_cumsum_and_arange(
                self,
                num_tokens,
                generated_arange_out,
                *args,
                **kwargs,
            )
            if isinstance(result, tuple):
                return result
            try:
                total_num_tokens = int(result[-1])
            except Exception:
                total_num_tokens = int(generated_arange_out.shape[0])
            return result, generated_arange_out[:total_num_tokens]

        _get_cumsum_and_arange._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls._get_cumsum_and_arange = _get_cumsum_and_arange

    def _easymagpie_input_batch_req_ids(self) -> list[str]:
        return [
            str(req_id)
            for req_id in getattr(getattr(self, "input_batch", None), "req_ids", [])
            if req_id is not None
        ]

    def _easymagpie_request_ids_to_seq_ids(self, request_ids: list[str]) -> dict[str, list[int]]:
        input_batch = getattr(self, "input_batch", None)
        req_id_to_index = getattr(input_batch, "req_id_to_index", None)
        used_seq_ids: set[int] = set()
        next_fallback = 0
        mapped: dict[str, list[int]] = {}

        def fallback_seq_id() -> int:
            nonlocal next_fallback
            while next_fallback in used_seq_ids:
                next_fallback += 1
            seq_idx = next_fallback
            used_seq_ids.add(seq_idx)
            next_fallback += 1
            return seq_idx

        if isinstance(req_id_to_index, dict):
            missing_or_duplicate: list[str] = []
            seq_by_req_id: dict[str, int] = {}
            for req_id in request_ids:
                raw_idx = req_id_to_index.get(req_id)
                if raw_idx is None:
                    raw_idx = req_id_to_index.get(str(req_id))
                try:
                    seq_idx = int(raw_idx)
                except (TypeError, ValueError):
                    missing_or_duplicate.append(str(req_id))
                else:
                    if seq_idx in used_seq_ids:
                        missing_or_duplicate.append(str(req_id))
                    else:
                        used_seq_ids.add(seq_idx)
                        seq_by_req_id[str(req_id)] = seq_idx
            for req_id in missing_or_duplicate:
                seq_by_req_id[req_id] = fallback_seq_id()
            for req_id in request_ids:
                mapped[str(req_id)] = [seq_by_req_id[str(req_id)]]
            return mapped
        for req_id in request_ids:
            mapped[str(req_id)] = [fallback_seq_id()]
        return mapped

    def _easymagpie_request_cache_state(self, scheduler_output: Any) -> dict[str, Any]:
        num_scheduled_tokens = getattr(scheduler_output, "num_scheduled_tokens", None)
        if not num_scheduled_tokens:
            return {}

        scheduled_counts = {
            str(req_id): int(count or 0)
            for req_id, count in num_scheduled_tokens.items()
            if int(count or 0) > 0
        }
        if not scheduled_counts:
            return {}

        finished_requests_ids = getattr(scheduler_output, "finished_req_ids", None)
        if finished_requests_ids is None:
            finished_requests_ids = getattr(scheduler_output, "finished_requests_ids", None)
        return {
            "scheduled_request_ids": list(scheduled_counts),
            "scheduled_counts": scheduled_counts,
            "finished_requests_ids": (
                None
                if finished_requests_ids is None
                else sorted(str(req_id) for req_id in finished_requests_ids)
            ),
            "previous_active_request_ids": sorted(
                str(req_id)
                for req_id in getattr(self, "_easymagpie_active_mamba_request_ids", set())
            ),
            "pre_update_request_ids": _easymagpie_input_batch_req_ids(self),
        }

    def _easymagpie_materialize_request_cache_kwargs(self, state: Any) -> dict[str, Any]:
        if not state:
            return {}
        if "request_ids_to_seq_ids" in state:
            return state

        scheduled_counts = {
            str(req_id): int(count or 0)
            for req_id, count in dict(state.get("scheduled_counts", {})).items()
            if int(count or 0) > 0
        }
        if not scheduled_counts:
            return {}

        ordered_request_ids: list[str] = []
        seen: set[str] = set()
        for req_id in _easymagpie_input_batch_req_ids(self):
            if req_id in scheduled_counts and req_id not in seen:
                ordered_request_ids.append(req_id)
                seen.add(req_id)
        for req_id in state.get("scheduled_request_ids", scheduled_counts):
            req_id = str(req_id)
            if req_id in scheduled_counts and req_id not in seen:
                ordered_request_ids.append(req_id)
                seen.add(req_id)
        if not ordered_request_ids:
            return {}

        current_active = set(str(req_id) for req_id in state.get("pre_update_request_ids", []))
        current_active.update(_easymagpie_input_batch_req_ids(self))
        current_active.update(ordered_request_ids)
        previous_active = set(str(req_id) for req_id in state.get("previous_active_request_ids", []))
        implicit_finished = previous_active - current_active
        finished_requests_ids = state.get("finished_requests_ids")
        if finished_requests_ids is None:
            finished_requests_ids = sorted(implicit_finished)
        else:
            finished_requests_ids = sorted(
                set(str(req_id) for req_id in finished_requests_ids) | implicit_finished
            )
        self._easymagpie_active_mamba_request_ids = current_active
        return {
            "request_ids_to_seq_ids": _easymagpie_request_ids_to_seq_ids(self, ordered_request_ids),
            "finished_requests_ids": finished_requests_ids,
        }

    original_model_forward = getattr(runner_cls, "_model_forward", None)
    if original_model_forward is not None and not getattr(original_model_forward, "_easymagpie_compat", False):

        def _model_forward(self, *args, **kwargs):
            request_cache_kwargs = _easymagpie_materialize_request_cache_kwargs(
                self,
                getattr(self, "_easymagpie_mamba_request_cache_state", None)
                or getattr(self, "_easymagpie_mamba_request_cache_kwargs", None),
            )
            if request_cache_kwargs:
                kwargs = dict(kwargs)
                for key, value in request_cache_kwargs.items():
                    kwargs.setdefault(key, value)
            return original_model_forward(self, *args, **kwargs)

        _model_forward._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls._model_forward = _model_forward

    def _install_execute_model_request_cache_compat(target_cls: type[Any]) -> None:
        original_execute_model = getattr(target_cls, "execute_model", None)
        if original_execute_model is None or getattr(original_execute_model, "_easymagpie_compat", False):
            return
        original_execute_model_signature = inspect.signature(original_execute_model)

        def execute_model(self, *args, **kwargs):
            bound = original_execute_model_signature.bind_partial(self, *args, **kwargs)
            scheduler_output = bound.arguments.get("scheduler_output")
            request_cache_state = (
                _easymagpie_request_cache_state(self, scheduler_output)
                if scheduler_output is not None
                else {}
            )
            if request_cache_state:
                self._easymagpie_mamba_request_cache_state = request_cache_state
            elif hasattr(self, "_easymagpie_mamba_request_cache_state"):
                delattr(self, "_easymagpie_mamba_request_cache_state")
            try:
                return original_execute_model(*bound.args, **bound.kwargs)
            finally:
                for name in (
                    "_easymagpie_mamba_request_cache_state",
                    "_easymagpie_mamba_request_cache_kwargs",
                ):
                    if hasattr(self, name):
                        delattr(self, name)

        execute_model._easymagpie_compat = True  # type: ignore[attr-defined]
        target_cls.execute_model = execute_model

    _install_execute_model_request_cache_compat(runner_cls)
    for module_name, class_names in (
        ("vllm_omni.worker.gpu_ar_model_runner", ("GPUARModelRunner",)),
        ("vllm_omni.worker.gpu_generation_model_runner", ("GPUGenerationModelRunner",)),
    ):
        extra_module = sys.modules.get(module_name)
        if extra_module is None:
            continue
        for class_name in class_names:
            extra_runner_cls = getattr(extra_module, class_name, None)
            if extra_runner_cls is not None:
                _install_execute_model_request_cache_compat(extra_runner_cls)

    original_lora_context = getattr(runner_cls, "maybe_dummy_run_with_lora", None)
    if original_lora_context is not None and not getattr(original_lora_context, "_easymagpie_compat", False):

        def maybe_dummy_run_with_lora(self, lora_config, num_scheduled_tokens, *_args, **_kwargs):
            if lora_config is None:
                return contextlib.nullcontext()
            return original_lora_context(self, lora_config, num_scheduled_tokens)

        maybe_dummy_run_with_lora._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls.maybe_dummy_run_with_lora = maybe_dummy_run_with_lora

    original_randomize_context = getattr(runner_cls, "maybe_randomize_inputs", None)
    if original_randomize_context is not None and not getattr(
        original_randomize_context, "_easymagpie_compat", False
    ):

        def maybe_randomize_inputs(self, input_ids, inputs_embeds=None):
            if input_ids is None:
                return contextlib.nullcontext()
            try:
                return original_randomize_context(self, input_ids, inputs_embeds)
            except TypeError:
                return original_randomize_context(self, input_ids)

        maybe_randomize_inputs._easymagpie_compat = True  # type: ignore[attr-defined]
        runner_cls.maybe_randomize_inputs = maybe_randomize_inputs

    original_dummy_run = getattr(runner_cls, "_dummy_run", None)
    if original_dummy_run is not None and not getattr(original_dummy_run, "_easymagpie_force_attn_compat", False):
        original_dummy_run_signature = inspect.signature(original_dummy_run)
        original_dummy_run_parameters = original_dummy_run_signature.parameters
        original_dummy_run_has_cudagraph_mode = "cudagraph_runtime_mode" in original_dummy_run_parameters
        original_dummy_run_var_keyword = next(
            (
                name
                for name, parameter in original_dummy_run_parameters.items()
                if parameter.kind == inspect.Parameter.VAR_KEYWORD
            ),
            None,
        )

        def get_bound_keyword(bound, name: str, default: Any = None):
            if name in bound.arguments:
                return bound.arguments[name]
            if original_dummy_run_var_keyword is not None:
                kwargs = bound.arguments.get(original_dummy_run_var_keyword, {})
                if isinstance(kwargs, dict) and name in kwargs:
                    return kwargs[name]
            return default

        def set_bound_keyword(bound, name: str, value: Any) -> None:
            parameter = original_dummy_run_parameters.get(name)
            if parameter is not None and parameter.kind != inspect.Parameter.VAR_KEYWORD:
                bound.arguments[name] = value
                return
            if original_dummy_run_var_keyword is not None:
                kwargs = dict(bound.arguments.get(original_dummy_run_var_keyword, {}))
                kwargs[name] = value
                bound.arguments[original_dummy_run_var_keyword] = kwargs
                return
            bound.arguments[name] = value

        def needs_mamba_attention_metadata(self) -> bool:
            candidates = [getattr(self, "model", None)]
            seen: set[int] = set()
            while candidates:
                candidate = candidates.pop()
                if candidate is None:
                    continue
                candidate_id = id(candidate)
                if candidate_id in seen:
                    continue
                seen.add(candidate_id)
                class_name = candidate.__class__.__name__.lower()
                config = getattr(candidate, "config", None)
                pattern = getattr(config, "hybrid_override_pattern", None)
                architectures = getattr(config, "architectures", None) or ()
                if "easymagpie" in class_name or "easy_magpie" in class_name:
                    return True
                if "nemotronh" in class_name or "nemotron_h" in class_name:
                    return True
                if isinstance(pattern, str) and "M" in pattern and hasattr(config, "chunk_size"):
                    return True
                if any(
                    "easymagpie" in str(architecture).lower()
                    or "nemotronh" in str(architecture).lower()
                    or "nemotron_h" in str(architecture).lower()
                    for architecture in architectures
                ):
                    return True
                for attr_name in ("backbone", "model", "language_model", "llm", "decoder"):
                    child = getattr(candidate, attr_name, None)
                    if child is not None:
                        candidates.append(child)
            return False

        def ensure_easy_mamba_no_compile_layers(self) -> None:
            vllm_config = getattr(self, "vllm_config", None)
            compilation_config = getattr(vllm_config, "compilation_config", None)
            static_context = getattr(compilation_config, "static_forward_context", None)
            if not isinstance(static_context, dict):
                return
            backbone = getattr(getattr(self, "model", None), "backbone", None)
            layers = getattr(backbone, "layers", None)
            if layers is None:
                return
            for layer_idx, layer in enumerate(layers):
                mixer = getattr(layer, "mixer", None)
                if mixer is not None:
                    static_context.setdefault(f"backbone.layers.{layer_idx}.mixer", mixer)

        def _dummy_run(self, *args, **kwargs):
            bound = original_dummy_run_signature.bind_partial(self, *args, **kwargs)
            force_v1_env = False
            if needs_mamba_attention_metadata(self):
                ensure_easy_mamba_no_compile_layers(self)
                num_tokens = get_bound_keyword(bound, "num_tokens")
                max_num_reqs = getattr(getattr(self, "scheduler_config", None), "max_num_seqs", None)
                if num_tokens is not None and max_num_reqs is not None:
                    self._easymagpie_mamba_cache_batch_size = min(max(1, int(num_tokens)), max(1, int(max_num_reqs)))
                use_v1_dummy_run = original_dummy_run_has_cudagraph_mode
                if not use_v1_dummy_run and original_dummy_run_var_keyword is not None:
                    try:
                        from vllm import envs

                        use_v1_dummy_run = bool(getattr(envs, "VLLM_USE_V1", False))
                    except Exception:
                        use_v1_dummy_run = False
                if use_v1_dummy_run:
                    set_bound_keyword(bound, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
                    set_bound_keyword(bound, "force_attention", False)
                    force_v1_env = True
                elif not get_bound_keyword(bound, "force_attention", False):
                    set_bound_keyword(bound, "force_attention", True)
            if not force_v1_env:
                return original_dummy_run(*bound.args, **bound.kwargs)
            try:
                from vllm import envs
            except Exception:
                return original_dummy_run(*bound.args, **bound.kwargs)
            previous_use_v1 = getattr(envs, "VLLM_USE_V1", None)
            envs.VLLM_USE_V1 = True
            try:
                return original_dummy_run(*bound.args, **bound.kwargs)
            finally:
                envs.VLLM_USE_V1 = previous_use_v1

        _dummy_run._easymagpie_force_attn_compat = True  # type: ignore[attr-defined]
        runner_cls._dummy_run = _dummy_run


def _install_omni_worker_import_patch() -> None:
    if getattr(builtins, _IMPORT_PATCH_FLAG, False):
        return
    original_import = builtins.__import__

    def import_with_omni_worker_patch(name, globals=None, locals=None, fromlist=(), level=0):
        should_patch_preprocess = (
            name == "vllm_omni.inputs.preprocess"
            or name.startswith("vllm_omni.inputs.preprocess.")
            or (name == "vllm_omni.inputs" and "preprocess" in fromlist)
        )
        if should_patch_preprocess:
            _install_vllm_inputs_data_alias()
            _install_vllm_multimodal_inputs_alias()
        module = original_import(name, globals, locals, fromlist, level)
        should_patch = (
            name == "vllm_omni.worker.gpu_model_runner"
            or name == "vllm_omni.worker.gpu_ar_model_runner"
            or name == "vllm_omni.worker.gpu_generation_model_runner"
            or name.startswith("vllm_omni.worker.gpu_model_runner.")
            or name.startswith("vllm_omni.worker.gpu_ar_model_runner.")
            or name.startswith("vllm_omni.worker.gpu_generation_model_runner.")
            or (name == "vllm_omni.worker" and "gpu_model_runner" in fromlist)
            or (
                name == "vllm_omni.worker"
                and any(item in fromlist for item in ("gpu_ar_model_runner", "gpu_generation_model_runner"))
            )
        )
        if should_patch:
            _install_forward_context_kwargs_compat()
            _install_omni_gpu_model_runner_method_compat()
        should_patch_worker = (
            name == "vllm_omni.worker.gpu_ar_worker"
            or name == "vllm_omni.worker.gpu_generation_worker"
            or name.startswith("vllm_omni.worker.gpu_ar_worker.")
            or name.startswith("vllm_omni.worker.gpu_generation_worker.")
            or (
                name == "vllm_omni.worker"
                and any(item in fromlist for item in ("gpu_ar_worker", "gpu_generation_worker"))
            )
        )
        if should_patch_worker:
            _install_easy_magpie_refit_rpc_compat()
        if should_patch_preprocess:
            _install_omni_input_preprocessor_signature_compat()
        should_patch_async_engine = (
            name == "vllm_omni.engine.async_omni_engine"
            or name.startswith("vllm_omni.engine.async_omni_engine.")
            or (name == "vllm_omni.engine" and "async_omni_engine" in fromlist)
        )
        if should_patch_async_engine:
            _install_omni_engine_request_compat()
        should_patch_request = name == "vllm_omni.request" or name.startswith("vllm_omni.request.")
        if should_patch_request:
            _install_omni_request_compat()
        return module

    builtins.__import__ = import_with_omni_worker_patch
    setattr(builtins, _IMPORT_PATCH_FLAG, True)


def _install_ec_transfer_alias() -> None:
    if "vllm.distributed.ec_transfer" in sys.modules:
        return
    module = types.ModuleType("vllm.distributed.ec_transfer")

    class _DisabledECTransfer:
        is_consumer = False
        is_producer = False

    def has_ec_transfer() -> bool:
        return False

    def get_ec_transfer():
        return _DisabledECTransfer()

    module.has_ec_transfer = has_ec_transfer
    module.get_ec_transfer = get_ec_transfer
    sys.modules["vllm.distributed.ec_transfer"] = module


def _install_routed_experts_capturer_alias() -> None:
    module_name = "vllm.model_executor.layers.fused_moe.routed_experts_capturer"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    class RoutedExpertsCapturer:
        @classmethod
        def get_instance(cls):
            return None

        def clear_buffer(self) -> None:
            return None

    module.RoutedExpertsCapturer = RoutedExpertsCapturer
    sys.modules[module_name] = module


def _install_structured_output_aliases() -> None:
    try:
        import vllm.v1.core.sched.output as sched_output
    except Exception:
        sched_output = None
    if sched_output is not None and not hasattr(sched_output, "GrammarOutput"):
        sched_output.GrammarOutput = object

    try:
        import vllm.v1.structured_output.utils as structured_utils
    except Exception:
        return
    if hasattr(structured_utils, "apply_grammar_bitmask"):
        return

    def apply_grammar_bitmask(*_args, **_kwargs):
        return None

    structured_utils.apply_grammar_bitmask = apply_grammar_bitmask


def _install_outputs_aliases() -> None:
    try:
        import vllm.v1.outputs as outputs
    except Exception:
        return
    if not hasattr(outputs, "AsyncModelRunnerOutput") and hasattr(outputs, "ModelRunnerOutput"):
        outputs.AsyncModelRunnerOutput = outputs.ModelRunnerOutput
    if not hasattr(outputs, "make_empty_encoder_model_runner_output"):

        def make_empty_encoder_model_runner_output(*_args, **_kwargs):
            return outputs.EMPTY_MODEL_RUNNER_OUTPUT

        outputs.make_empty_encoder_model_runner_output = make_empty_encoder_model_runner_output


def _install_spec_decode_aliases() -> None:
    class _UnsupportedSpecDecodeProposer:
        def __init__(self, *args, **kwargs) -> None:
            raise NotImplementedError("This vLLM build does not expose the legacy spec-decode proposer.")

    aliases = {
        "vllm.v1.spec_decode.draft_model": ("DraftModelProposer", _UnsupportedSpecDecodeProposer),
        "vllm.v1.spec_decode.extract_hidden_states": (
            "ExtractHiddenStatesProposer",
            _UnsupportedSpecDecodeProposer,
        ),
    }
    for module_name, (class_name, cls) in aliases.items():
        if module_name in sys.modules:
            continue
        module = types.ModuleType(module_name)
        setattr(module, class_name, cls)
        sys.modules[module_name] = module


def _install_v1_utils_aliases() -> None:
    try:
        import vllm.v1.utils as v1_utils
    except Exception:
        return
    if hasattr(v1_utils, "record_function_or_nullcontext"):
        return

    def record_function_or_nullcontext(name: str):
        try:
            import torch

            return torch.profiler.record_function(name)
        except Exception:
            return contextlib.nullcontext()

    v1_utils.record_function_or_nullcontext = record_function_or_nullcontext


def _install_gpu_model_runner_aliases() -> None:
    try:
        import vllm.v1.worker.gpu_model_runner as gpu_model_runner
    except Exception:
        return
    if not hasattr(gpu_model_runner, "PerLayerAttnMetadata"):
        gpu_model_runner.PerLayerAttnMetadata = dict
    if hasattr(gpu_model_runner, "AsyncGPUModelRunnerOutput"):
        return

    class AsyncGPUModelRunnerOutput:
        def __init__(
            self,
            model_runner_output=None,
            sampled_token_ids=None,
            logprobs_tensors=None,
            invalid_req_indices=None,
            async_output_copy_stream=None,
            vocab_size=None,
        ) -> None:
            self.model_runner_output = model_runner_output
            self.sampled_token_ids = sampled_token_ids
            self.logprobs_tensors = logprobs_tensors
            self.invalid_req_indices = invalid_req_indices
            self.async_output_copy_stream = async_output_copy_stream
            self.vocab_size = vocab_size
            self.sampled_token_ids_cpu = sampled_token_ids
            self.async_copy_ready_event = None

    gpu_model_runner.AsyncGPUModelRunnerOutput = AsyncGPUModelRunnerOutput


def _install_model_interface_aliases() -> None:
    try:
        import vllm.model_executor.models.interfaces as interfaces
    except Exception:
        return
    if hasattr(interfaces, "supports_mrope"):
        return

    def supports_mrope(*_args, **_kwargs) -> bool:
        return False

    interfaces.supports_mrope = supports_mrope


def _install_ubatch_utils_alias() -> None:
    module_name = "vllm.v1.worker.ubatch_utils"
    if module_name in sys.modules:
        return
    module = types.ModuleType(module_name)

    def maybe_create_ubatch_slices(*_args, **_kwargs):
        return None, None

    module.maybe_create_ubatch_slices = maybe_create_ubatch_slices
    sys.modules[module_name] = module


def _install_tracing_instrument_alias() -> None:
    try:
        import vllm.tracing as tracing
    except Exception:
        return
    if hasattr(tracing, "instrument"):
        return

    def instrument(func=None, **_kwargs):
        def decorator(inner):
            return inner

        if callable(func):
            return decorator(func)
        return decorator

    tracing.instrument = instrument


def _install_executor_alias() -> None:
    try:
        import vllm.v1.executor as executor_pkg
        from vllm.v1.executor.abstract import Executor
    except Exception:
        return
    if not hasattr(executor_pkg, "Executor"):
        executor_pkg.Executor = Executor


def _install_async_omni_client_compat() -> None:
    try:
        async_omni_module = importlib.import_module("vllm_omni.entrypoints.async_omni")
    except Exception:
        return

    AsyncOmni = getattr(async_omni_module, "AsyncOmni", None)
    if AsyncOmni is None:
        return

    if not hasattr(AsyncOmni, "get_model_config") or getattr(
        AsyncOmni.get_model_config, "__isabstractmethod__", False
    ):

        async def get_model_config(self):
            return self.model_config

        AsyncOmni.get_model_config = get_model_config

    if not hasattr(AsyncOmni, "get_decoding_config") or getattr(
        AsyncOmni.get_decoding_config, "__isabstractmethod__", False
    ):

        async def get_decoding_config(self):
            vllm_config = self.vllm_config
            decoding_config = getattr(vllm_config, "decoding_config", None)
            if decoding_config is not None:
                return decoding_config
            model_config = self.model_config
            return getattr(model_config, "decoding_config", None)

        AsyncOmni.get_decoding_config = get_decoding_config

    if not hasattr(AsyncOmni, "notify_kv_transfer_request_rejected") or getattr(
        AsyncOmni.notify_kv_transfer_request_rejected, "__isabstractmethod__", False
    ):

        def notify_kv_transfer_request_rejected(self, *args, **kwargs):
            return None

        AsyncOmni.notify_kv_transfer_request_rejected = notify_kv_transfer_request_rejected

    abstract_methods = getattr(AsyncOmni, "__abstractmethods__", frozenset())
    if abstract_methods:
        AsyncOmni.__abstractmethods__ = frozenset(
            name
            for name in abstract_methods
            if name
            not in {
                "get_decoding_config",
                "get_model_config",
                "notify_kv_transfer_request_rejected",
            }
        )


def _install_ray_objectref_aliases() -> None:
    """Restore Ray type aliases expected by NeMo-RL annotation imports."""

    try:
        import ray
    except Exception:
        return
    try:
        import ray._raylet as raylet
    except Exception:
        raylet = None

    class _EasyMagpieRayObjectRef:
        pass

    class _EasyMagpieRayObjectRefGenerator:
        pass

    for name, fallback in (
        ("ObjectRef", _EasyMagpieRayObjectRef),
        ("ObjectRefGenerator", _EasyMagpieRayObjectRefGenerator),
    ):
        if hasattr(ray, name):
            continue
        alias = getattr(raylet, name, None) if raylet is not None else None
        if alias is None:
            alias = fallback
        try:
            setattr(ray, name, alias)
        except Exception:
            pass


def _install_ray_placement_group_aliases() -> None:
    """Restore Ray placement-group imports used by NeMo-RL type imports."""

    try:
        import ray
    except Exception:
        return

    if not hasattr(ray, "is_initialized"):
        ray.is_initialized = lambda: False

    if not hasattr(ray, "remote"):

        class _UnavailableRemote:
            def __init__(self, target, options=None):
                self._target = target
                self._options = dict(options or {})
                self.__name__ = getattr(target, "__name__", type(target).__name__)
                self.__qualname__ = getattr(target, "__qualname__", self.__name__)
                self.__doc__ = getattr(target, "__doc__", None)

            def options(self, **kwargs):
                merged = dict(self._options)
                merged.update(kwargs)
                return type(self)(self._target, merged)

            def remote(self, *_args, **_kwargs):
                raise RuntimeError(
                    "ray.remote is unavailable in this Ray build. The EasyMagpie "
                    "compatibility shim only restores this symbol so NeMo-RL modules "
                    "can be imported in non-Ray train loops."
                )

        def remote(*args, **kwargs):
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return _UnavailableRemote(args[0])

            def decorate(target):
                return _UnavailableRemote(target, kwargs)

            return decorate

        ray.remote = remote

    actor_module_name = "ray.actor"
    try:
        actor_module = importlib.import_module(actor_module_name)
    except Exception:
        actor_module = types.ModuleType(actor_module_name)
        sys.modules[actor_module_name] = actor_module

    if not hasattr(actor_module, "ActorHandle"):

        class ActorHandle:
            pass

        actor_module.ActorHandle = ActorHandle

    if not hasattr(ray, "actor"):
        try:
            setattr(ray, "actor", actor_module)
        except Exception:
            pass

    try:
        import ray.util as ray_util
    except Exception:
        ray_util = types.ModuleType("ray.util")
        sys.modules["ray.util"] = ray_util
        try:
            setattr(ray, "util", ray_util)
        except Exception:
            pass

    if not hasattr(ray_util, "__path__"):
        try:
            ray_util.__path__ = []  # type: ignore[attr-defined]
        except Exception:
            pass

    module_name = "ray.util.placement_group"
    try:
        module = importlib.import_module(module_name)
    except Exception:
        module = types.ModuleType(module_name)
        sys.modules[module_name] = module

    if not hasattr(module, "PlacementGroup"):

        class PlacementGroup:
            pass

        module.PlacementGroup = PlacementGroup

    if not hasattr(module, "placement_group"):

        def placement_group(*_args, **_kwargs):
            raise RuntimeError(
                "ray.util.placement_group.placement_group is unavailable in this Ray build. "
                "The EasyMagpie compatibility shim only restores this symbol so NeMo-RL "
                "modules can be imported in non-Ray train loops."
            )

        module.placement_group = placement_group

    if not hasattr(ray_util, "placement_group"):
        try:
            setattr(ray_util, "placement_group", module)
        except Exception:
            pass

    if not hasattr(module, "placement_group_table"):

        def placement_group_table(*_args, **_kwargs):
            return {}

        module.placement_group_table = placement_group_table

    if not hasattr(module, "remove_placement_group"):

        def remove_placement_group(*_args, **_kwargs):
            return None

        module.remove_placement_group = remove_placement_group

    if not hasattr(ray_util, "placement_group_table"):
        ray_util.placement_group_table = module.placement_group_table

    strategies_module_name = "ray.util.scheduling_strategies"
    try:
        strategies_module = importlib.import_module(strategies_module_name)
    except Exception:
        strategies_module = types.ModuleType(strategies_module_name)
        sys.modules[strategies_module_name] = strategies_module

    if not hasattr(strategies_module, "PlacementGroupSchedulingStrategy"):

        class PlacementGroupSchedulingStrategy:
            def __init__(
                self,
                placement_group=None,
                placement_group_bundle_index: int | None = None,
                placement_group_capture_child_tasks: bool | None = None,
                **kwargs,
            ):
                self.placement_group = placement_group
                self.placement_group_bundle_index = placement_group_bundle_index
                self.placement_group_capture_child_tasks = placement_group_capture_child_tasks
                self.kwargs = kwargs

        strategies_module.PlacementGroupSchedulingStrategy = PlacementGroupSchedulingStrategy

    if not hasattr(strategies_module, "NodeAffinitySchedulingStrategy"):

        class NodeAffinitySchedulingStrategy:
            def __init__(self, node_id=None, soft: bool | None = None, **kwargs):
                self.node_id = node_id
                self.soft = soft
                self.kwargs = kwargs

        strategies_module.NodeAffinitySchedulingStrategy = NodeAffinitySchedulingStrategy

    if not hasattr(ray_util, "scheduling_strategies"):
        try:
            setattr(ray_util, "scheduling_strategies", strategies_module)
        except Exception:
            pass

    queue_module_name = "ray.util.queue"
    try:
        queue_module = importlib.import_module(queue_module_name)
    except Exception:
        queue_module = types.ModuleType(queue_module_name)
        sys.modules[queue_module_name] = queue_module

    if not hasattr(queue_module, "Queue"):

        class Queue:
            def __init__(self, maxsize: int = 0, *args, **kwargs):
                del args, kwargs
                self._queue = stdlib_queue.Queue(maxsize=maxsize)

            def put(self, item, block: bool = True, timeout: float | None = None):
                return self._queue.put(item, block=block, timeout=timeout)

            def get(self, block: bool = True, timeout: float | None = None):
                return self._queue.get(block=block, timeout=timeout)

            def empty(self) -> bool:
                return self._queue.empty()

            def qsize(self) -> int:
                return self._queue.qsize()

        queue_module.Queue = Queue

    if not hasattr(ray_util, "queue"):
        try:
            setattr(ray_util, "queue", queue_module)
        except Exception:
            pass


def _install_omni_model_config_field_compat() -> None:
    try:
        from vllm_omni.config.model import OmniModelConfig
    except Exception:
        return
    if getattr(OmniModelConfig, "_easymagpie_field_compat", False):
        return

    omni_defaults = {
        "stage_id": 0,
        "async_chunk": False,
        "model_stage": "thinker",
        "model_arch": None,
        "worker_type": None,
        "engine_output_type": None,
        "hf_config_name": None,
        "custom_process_next_stage_input_func": None,
        "stage_connector_config": lambda: {"name": "SharedMemoryConnector", "extra": {}},
        "omni_kv_config": None,
        "codec_frame_rate_hz": None,
        "task_type": None,
        "io_processor_plugin": None,
        "has_sampling_extra_args": False,
        "subtalker_sampling_params": None,
    }

    def add_defaults_to_omni_kwargs(cls, omni_kwargs):
        for key, default in omni_defaults.items():
            if key not in omni_kwargs:
                omni_kwargs[key] = default() if callable(default) else default

    def _validate_omni_fields(cls, **omni_kwargs):
        unexpected = set(omni_kwargs) - set(omni_defaults)
        if unexpected:
            raise ValueError(f"Unexpected omni kwarg: {sorted(unexpected)}")

    OmniModelConfig.add_defaults_to_omni_kwargs = classmethod(add_defaults_to_omni_kwargs)
    OmniModelConfig._validate_omni_fields = classmethod(_validate_omni_fields)
    if not hasattr(OmniModelConfig, "io_processor_plugin"):
        OmniModelConfig.io_processor_plugin = None
    if not hasattr(OmniModelConfig, "has_sampling_extra_args"):
        OmniModelConfig.has_sampling_extra_args = False
    if not hasattr(OmniModelConfig, "subtalker_sampling_params"):
        OmniModelConfig.subtalker_sampling_params = None
    OmniModelConfig._easymagpie_field_compat = True


def install_vllm_omni_compat() -> None:
    """Install import/signature shims required by vLLM-Omni on older vLLM."""

    global _INSTALLED
    if _INSTALLED:
        _install_ray_objectref_aliases()
        _install_ray_placement_group_aliases()
        _install_vllm_inputs_data_alias()
        _install_vllm_multimodal_inputs_alias()
        return
    _install_config_decorator_compat()
    _install_mamba2_metadata_compat()
    _install_triton_attention_profile_metadata_compat()
    _install_flash_attention_profile_metadata_compat()
    _install_flashinfer_attention_profile_metadata_compat()
    _install_mamba2_profile_no_kv_cache_compat()
    _install_lora_config_alias()
    _install_model_arch_convertor()
    _install_io_processor_stub()
    _install_tokenizer_aliases()
    _install_repo_utils_alias()
    _install_config_parser_aliases()
    _install_nemotron_h_auto_config()
    _install_vllm_inputs_data_alias()
    _install_vllm_multimodal_inputs_alias()
    _install_renderer_aliases()
    _install_input_processor_alias()
    _install_input_preprocessor_truncate_compat()
    _install_omni_input_preprocessor_signature_compat()
    _install_omni_engine_request_compat()
    _install_omni_request_compat()
    _install_processor_process_inputs_compat()
    _install_engine_utils_compat()
    _install_import_utils_alias()
    _install_torch_utils_alias()
    _install_math_utils_alias()
    _install_mem_utils_alias()
    _install_vllm_config_profiler_default()
    _install_parallel_config_defaults()
    _install_cache_config_defaults()
    _install_platform_dtype_check()
    _install_torch_accelerator_compat()
    _install_logger_info_once_scope_compat()
    _install_v1_serial_utils_dense_tensor_compat()
    _install_cudagraph_mode_compat()
    _install_cuda_graph_stat_alias()
    _install_cuda_piecewise_no_sym_shape_compat()
    _install_kv_connector_stats_alias()
    _install_perf_stats_alias()
    _install_scheduler_make_stats_compat()
    _install_sched_utils_aliases()
    _install_sched_interface_aliases()
    _install_output_processor_signature_compat()
    _install_flash_attention_builder_compat()
    _install_worker_utils_compat()
    _install_worker_workspace_alias()
    _install_gpu_ar_worker_default_attrs()
    _install_easy_magpie_refit_rpc_compat()
    _install_forward_context_kwargs_compat()
    _install_omni_gpu_model_runner_method_compat()
    _install_omni_worker_import_patch()
    _install_ec_transfer_alias()
    _install_routed_experts_capturer_alias()
    _install_structured_output_aliases()
    _install_outputs_aliases()
    _install_spec_decode_aliases()
    _install_v1_utils_aliases()
    _install_model_interface_aliases()
    _install_gpu_model_runner_aliases()
    _install_ubatch_utils_alias()
    # vLLM-Omni's runner import may fail until the late aliases above exist.
    _install_forward_context_kwargs_compat()
    _install_omni_gpu_model_runner_method_compat()
    _install_tracing_instrument_alias()
    _install_executor_alias()
    _install_ray_objectref_aliases()
    _install_ray_placement_group_aliases()
    _install_omni_model_config_field_compat()
    _install_async_omni_client_compat()
    _register_easy_magpie_plugin()
    _INSTALLED = True


__all__ = [
    "install_easy_magpie_refit_rpc_compat",
    "install_easy_magpie_runtime_compat",
    "install_vllm_omni_compat",
]
