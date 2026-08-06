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
"""Backbone-side patches applied at model ``__init__``.

Runtime fixes for the constructed ``NemotronHModel`` backbone. They live with
the model because they're inherent to running EasyMagpie SmallMamba
(``mlp_hidden_act=silu``) on vLLM's NemotronH implementation. Mirrors the
EasyMagpie vLLM *sidecar* (``easymagpie_vllm/backbone_patches.py``).
"""
from __future__ import annotations

import inspect
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import vllm.v1.attention.backends.mamba_attn as _mamba_attn
except ModuleNotFoundError:
    import vllm.v1.attention.backends.mamba2_attn as _mamba_attn
from vllm.logger import init_logger

logger = init_logger(__name__)


_EASYMAGPIE_MOE_OP_REGISTERED = False


def _easymagpie_moe_forward(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    from vllm.forward_context import get_forward_context

    forward_context = get_forward_context()
    layer = forward_context.no_compile_layers[layer_name]
    return layer.forward_impl(hidden_states, router_logits)


def _easymagpie_moe_forward_fake(
    hidden_states: torch.Tensor,
    router_logits: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    del router_logits, layer_name
    return torch.empty_like(hidden_states)


def _register_easymagpie_moe_custom_op() -> None:
    global _EASYMAGPIE_MOE_OP_REGISTERED
    if _EASYMAGPIE_MOE_OP_REGISTERED:
        return
    from vllm.platforms import current_platform
    from vllm.utils import direct_register_custom_op

    try:
        direct_register_custom_op(
            op_name="easymagpie_moe_forward",
            op_func=_easymagpie_moe_forward,
            mutates_args=[],
            fake_impl=_easymagpie_moe_forward_fake,
            dispatch_key=current_platform.dispatch_key,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "easymagpie_moe_forward" not in message and "already" not in message.lower():
            raise
    _EASYMAGPIE_MOE_OP_REGISTERED = True


def _has_vllm_forward_context() -> bool:
    try:
        import vllm.forward_context as forward_context
    except Exception:
        return False
    return getattr(forward_context, "_forward_context", None) is not None


def _env_flag(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def _repeat_kv_heads_for_gqa(kv: torch.Tensor, num_query_heads: int) -> torch.Tensor:
    if kv.shape[1] == num_query_heads:
        return kv
    repeats = num_query_heads // kv.shape[1]
    return kv.repeat_interleave(repeats, dim=1)


def _causal_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float,
    is_causal: bool,
    query_start_pos: int = 0,
) -> torch.Tensor:
    out_dtype = q.dtype
    num_query_heads = q.shape[1]
    k = _repeat_kv_heads_for_gqa(k, num_query_heads)
    v = _repeat_kv_heads_for_gqa(v, num_query_heads)

    qf = torch.nan_to_num(q.float())
    kf = torch.nan_to_num(k.float())
    vf = torch.nan_to_num(v.float())
    scores = torch.matmul(qf.transpose(0, 1), kf.transpose(0, 1).transpose(-2, -1)) * float(scale)
    if is_causal:
        q_pos = query_start_pos + torch.arange(q.shape[0], device=q.device)
        k_pos = torch.arange(k.shape[0], device=q.device)
        attn_mask = k_pos.unsqueeze(0) <= q_pos.unsqueeze(1)
        scores = scores.masked_fill(~attn_mask.unsqueeze(0), float("-inf"))
    attn = torch.softmax(scores, dim=-1)
    attn = torch.nan_to_num(attn)
    out = torch.matmul(attn, vf.transpose(0, 1))
    return out.transpose(0, 1).to(out_dtype).contiguous()


def _cache_block_size(key_cache: torch.Tensor) -> int:
    return int(key_cache.shape[1]) if key_cache.ndim >= 4 else 1


def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:
    return cache.reshape(-1, cache.shape[-2], cache.shape[-1])


def _gather_cache_slots(cache: torch.Tensor, slots: torch.Tensor) -> torch.Tensor:
    slots = slots.to(device=cache.device, dtype=torch.long)
    if cache.ndim >= 4:
        block_size = _cache_block_size(cache)
        blocks = slots // block_size
        offsets = slots % block_size
        return cache[blocks, offsets]
    return cache[slots]


def _cache_slots_for_sequence(
    block_tables: torch.Tensor,
    row: int,
    seq_len: int,
    *,
    block_size: int,
    device: torch.device,
) -> torch.Tensor:
    positions = torch.arange(seq_len, dtype=torch.long, device=device)
    if block_tables is None or block_tables.numel() == 0:
        return positions
    table = block_tables[row].to(device=device, dtype=torch.long)
    blocks = table[positions // block_size]
    return blocks * block_size + (positions % block_size)


def _slice_start_loc(start_loc: torch.Tensor | None, fallback_lens: torch.Tensor) -> torch.Tensor:
    if start_loc is not None:
        return start_loc.to(dtype=torch.long)
    out = torch.zeros(fallback_lens.numel() + 1, dtype=torch.long, device=fallback_lens.device)
    out[1:] = torch.cumsum(fallback_lens.to(dtype=torch.long), dim=0)
    return out


def _torch_flash_attention_forward(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    if kv_cache.numel() > 0 and key is not None and value is not None:
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[0],
            kv_cache[1],
            attn_metadata.slot_mapping.flatten(),
            impl.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    num_prefill_tokens = int(getattr(attn_metadata, "num_prefill_tokens", 0) or 0)
    decode_tokens = int(getattr(attn_metadata, "num_decode_tokens", 0) or 0)
    output.zero_()

    prefill_meta = getattr(attn_metadata, "prefill_metadata", None)
    if prefill_meta is not None and num_prefill_tokens > 0:
        seq_lens_tensor = getattr(prefill_meta, "seq_lens_tensor", None)
        if seq_lens_tensor is None:
            seq_lens_tensor = torch.tensor(
                list(getattr(prefill_meta, "seq_lens", []) or []),
                dtype=torch.long,
                device=query.device,
            )
        else:
            seq_lens_tensor = seq_lens_tensor.to(device=query.device, dtype=torch.long)
        query_start_loc = _slice_start_loc(getattr(prefill_meta, "query_start_loc", None), seq_lens_tensor)
        seq_start_loc = _slice_start_loc(getattr(prefill_meta, "seq_start_loc", None), seq_lens_tensor)
        block_tables = getattr(prefill_meta, "block_tables", None)
        use_cache_for_prefill = (
            kv_cache.numel() > 0
            and block_tables is not None
            and getattr(block_tables, "numel", lambda: 0)() > 0
        )
        if use_cache_for_prefill:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            block_size = _cache_block_size(kv_cache[0])
        for row in range(int(seq_lens_tensor.numel())):
            qs = int(query_start_loc[row].item())
            qe = int(query_start_loc[row + 1].item())
            ks = int(seq_start_loc[row].item())
            ke = int(seq_start_loc[row + 1].item())
            if qe <= qs or ke <= ks:
                continue
            q_row = query[qs:qe]
            if use_cache_for_prefill:
                slots = _cache_slots_for_sequence(
                    block_tables,
                    row,
                    int(seq_lens_tensor[row].item()),
                    block_size=block_size,
                    device=query.device,
                )
                k_row = _gather_cache_slots(key_cache, slots)
                v_row = _gather_cache_slots(value_cache, slots)
            else:
                k_row = key[ks:ke]
                v_row = value[ks:ke]
            query_start_pos = max(0, int(seq_lens_tensor[row].item()) - int(q_row.shape[0]))
            output[qs:qe].copy_(
                _causal_sdpa(
                    q_row,
                    k_row,
                    v_row,
                    scale=float(impl.scale),
                    is_causal=True,
                    query_start_pos=query_start_pos,
                )
            )

    decode_meta = getattr(attn_metadata, "decode_metadata", None)
    if decode_meta is not None and decode_tokens > 0:
        if kv_cache.numel() == 0:
            start = num_prefill_tokens
            output[start : start + decode_tokens].copy_(
                _causal_sdpa(
                    query[start : start + decode_tokens],
                    key[start : start + decode_tokens],
                    value[start : start + decode_tokens],
                    scale=float(impl.scale),
                    is_causal=True,
                )
            )
            return output
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        block_size = _cache_block_size(kv_cache[0])
        seq_lens = getattr(decode_meta, "seq_lens_tensor", None)
        if seq_lens is None:
            seq_lens = torch.tensor(
                list(getattr(decode_meta, "seq_lens", []) or []),
                dtype=torch.long,
                device=query.device,
            )
        else:
            seq_lens = seq_lens.to(device=query.device, dtype=torch.long)
        block_tables = getattr(decode_meta, "block_tables", None)
        query_start_loc = getattr(decode_meta, "query_start_loc", None)
        if query_start_loc is None:
            query_lens = torch.ones(seq_lens.numel(), dtype=torch.long, device=query.device)
            query_start_loc = _slice_start_loc(None, query_lens)
        else:
            query_start_loc = query_start_loc.to(device=query.device, dtype=torch.long)
        decode_query_offset = num_prefill_tokens
        for row in range(int(seq_lens.numel())):
            qs = decode_query_offset + int(query_start_loc[row].item())
            qe = decode_query_offset + int(query_start_loc[row + 1].item())
            if qe <= qs:
                continue
            seq_len = int(seq_lens[row].item())
            slots = _cache_slots_for_sequence(
                block_tables,
                row,
                seq_len,
                block_size=block_size,
                device=query.device,
            )
            output[qs:qe].copy_(
                _causal_sdpa(
                    query[qs:qe],
                    _gather_cache_slots(key_cache, slots),
                    _gather_cache_slots(value_cache, slots),
                    scale=float(impl.scale),
                    is_causal=(qe - qs) > 1,
                    query_start_pos=max(0, seq_len - (qe - qs)),
                )
            )
    return output


def _torch_v1_flash_attention_forward(
    impl,
    layer,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
) -> torch.Tensor:
    if attn_metadata is None:
        return output

    if (
        kv_cache.numel() > 0
        and key is not None
        and value is not None
        and getattr(impl, "kv_sharing_target_layer_name", None) is None
    ):
        slot_mapping = getattr(attn_metadata, "slot_mapping", None)
        if slot_mapping is None:
            slot_mapping = torch.arange(int(key.shape[0]), dtype=torch.long, device=key.device)
        else:
            slot_mapping = slot_mapping.to(device=key.device, dtype=torch.long).flatten()
        torch.ops._C_cache_ops.reshape_and_cache_flash(
            key,
            value,
            kv_cache[0],
            kv_cache[1],
            slot_mapping,
            impl.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    num_actual_tokens_value = getattr(attn_metadata, "num_actual_tokens", None)
    if num_actual_tokens_value is None:
        inferred_tokens = int(getattr(attn_metadata, "num_prefill_tokens", 0) or 0) + int(
            getattr(attn_metadata, "num_decode_tokens", 0) or 0
        )
        num_actual_tokens = inferred_tokens if inferred_tokens > 0 else int(query.shape[0])
    else:
        num_actual_tokens = int(num_actual_tokens_value or 0)
    num_actual_tokens = min(num_actual_tokens, int(query.shape[0]))
    if num_actual_tokens <= 0:
        return output

    output.zero_()
    query_start_loc = getattr(attn_metadata, "query_start_loc", None)
    if query_start_loc is None:
        query_start_loc = torch.tensor([0, num_actual_tokens], dtype=torch.long, device=query.device)
    else:
        query_start_loc = query_start_loc.to(device=query.device, dtype=torch.long)
    seq_lens = getattr(attn_metadata, "seq_lens", None)
    if seq_lens is None:
        seq_lens = torch.diff(query_start_loc)
    else:
        seq_lens = seq_lens.to(device=query.device, dtype=torch.long)

    block_table = getattr(attn_metadata, "block_table", None)
    use_cache = (
        kv_cache.numel() > 0
        and block_table is not None
        and getattr(block_table, "numel", lambda: 0)() > 0
    )
    if use_cache:
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        block_size = _cache_block_size(kv_cache[0])
    else:
        key_cache = value_cache = None
        block_size = 1
        block_table = None

    causal = bool(getattr(attn_metadata, "causal", True))
    for row in range(int(seq_lens.numel())):
        qs = int(query_start_loc[row].item())
        qe = int(query_start_loc[row + 1].item())
        q_len = qe - qs
        seq_len = int(seq_lens[row].item())
        if q_len <= 0 or seq_len <= 0:
            continue
        if use_cache:
            slots = _cache_slots_for_sequence(
                block_table,
                row,
                seq_len,
                block_size=block_size,
                device=query.device,
            )
            k_row = _gather_cache_slots(key_cache, slots)
            v_row = _gather_cache_slots(value_cache, slots)
        else:
            k_row = key[qs : qs + seq_len]
            v_row = value[qs : qs + seq_len]
        query_start_pos = max(0, seq_len - q_len)
        output[qs:qe].copy_(
            _causal_sdpa(
                query[qs:qe],
                k_row,
                v_row,
                scale=float(impl.scale),
                is_causal=causal,
                query_start_pos=query_start_pos,
            )
        )
    return output


def _scale_is_effectively_one(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, torch.Tensor):
        return float(value) == 1.0
    if value.numel() == 0:
        return True
    return bool(torch.all(value.detach().float() == 1.0).item())


def _window_is_unrestricted(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return True
        return bool(torch.all(value.detach().cpu() == -1).item())
    try:
        items = list(value)
    except TypeError:
        return False
    return len(items) > 0 and all(int(item) == -1 for item in items)


def _softcap_is_zero(value) -> bool:
    if value is None:
        return True
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return True
        return bool(torch.all(value.detach().float().cpu() == 0).item())
    return float(value) == 0.0


def _torch_flash_attn_varlen_forward(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    seqused_k: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
) -> torch.Tensor:
    del max_seqlen_q, max_seqlen_k
    scale = float(softmax_scale) if softmax_scale is not None else float(q.shape[-1] ** -0.5)
    cu_seqlens_q = cu_seqlens_q.to(device=q.device, dtype=torch.long)
    num_rows = int(cu_seqlens_q.numel()) - 1
    if seqused_k is not None:
        seq_lens = seqused_k.to(device=q.device, dtype=torch.long)
    elif cu_seqlens_k is not None:
        seq_lens = torch.diff(cu_seqlens_k.to(device=q.device, dtype=torch.long))
    else:
        seq_lens = torch.diff(cu_seqlens_q)

    use_block_cache = block_table is not None and getattr(block_table, "numel", lambda: 0)() > 0 and k.ndim >= 4
    if use_block_cache:
        key_cache = k
        value_cache = v
        block_size = _cache_block_size(k)
    else:
        key_cache = value_cache = None
        block_size = 1

    out.zero_()
    for row in range(num_rows):
        qs = int(cu_seqlens_q[row].item())
        qe = int(cu_seqlens_q[row + 1].item())
        q_len = qe - qs
        seq_len = int(seq_lens[row].item())
        if q_len <= 0 or seq_len <= 0:
            continue
        if use_block_cache:
            slots = _cache_slots_for_sequence(
                block_table,
                row,
                seq_len,
                block_size=block_size,
                device=q.device,
            )
            k_row = _gather_cache_slots(key_cache, slots)
            v_row = _gather_cache_slots(value_cache, slots)
        else:
            if cu_seqlens_k is not None:
                ks = int(cu_seqlens_k[row].item())
            else:
                ks = qs
            k_row = k[ks : ks + seq_len]
            v_row = v[ks : ks + seq_len]
        query_start_pos = max(0, seq_len - q_len)
        out[qs:qe].copy_(
            _causal_sdpa(
                q[qs:qe],
                k_row,
                v_row,
                scale=scale,
                is_causal=causal,
                query_start_pos=query_start_pos,
            )
        )
    return out


def _patch_v1_flash_attn_varlen_func() -> bool:
    try:
        import vllm.v1.attention.backends.flash_attn as v1_flash_attn
    except Exception:
        return False
    original = getattr(v1_flash_attn, "flash_attn_varlen_func", None)
    if original is None or getattr(original, "_easymagpie_torch_fallback", False):
        return False

    def flash_attn_varlen_func(q, k, v, *args, **kwargs):
        enabled = os.environ.get("EASYMAGPIE_TORCH_FLASH_ATTN_FALLBACK", "1").lower()
        unsupported = (
            args
            or enabled in {"0", "false", "no", "off"}
            or kwargs.get("out") is None
            or kwargs.get("q_v") is not None
            or float(kwargs.get("dropout_p", 0.0) or 0.0) != 0.0
            or not _window_is_unrestricted(kwargs.get("window_size"))
            or kwargs.get("alibi_slopes") is not None
            or not _softcap_is_zero(kwargs.get("softcap"))
            or bool(kwargs.get("return_attn_probs", False))
            or bool(kwargs.get("return_softmax_lse", False))
        )
        if unsupported:
            if not getattr(flash_attn_varlen_func, "_easymagpie_warned_unsupported", False):
                logger.warning(
                    "EasyMagpie torch-SDPA varlen fallback bypassed unsupported FlashAttention call: "
                    "args=%s window_size=%s alibi=%s softcap=%s q_v=%s out=%s",
                    bool(args),
                    kwargs.get("window_size"),
                    kwargs.get("alibi_slopes") is not None,
                    kwargs.get("softcap"),
                    kwargs.get("q_v") is not None,
                    kwargs.get("out") is not None,
                )
                flash_attn_varlen_func._easymagpie_warned_unsupported = True  # type: ignore[attr-defined]
            return original(q, k, v, *args, **kwargs)
        if not getattr(flash_attn_varlen_func, "_easymagpie_warned_active", False):
            logger.warning("EasyMagpie torch-SDPA varlen fallback active for vLLM v1 FlashAttention")
            flash_attn_varlen_func._easymagpie_warned_active = True  # type: ignore[attr-defined]
        return _torch_flash_attn_varlen_forward(
            q=q,
            k=k,
            v=v,
            out=kwargs["out"],
            cu_seqlens_q=kwargs["cu_seqlens_q"],
            cu_seqlens_k=kwargs.get("cu_seqlens_k"),
            seqused_k=kwargs.get("seqused_k"),
            max_seqlen_q=int(kwargs["max_seqlen_q"]),
            max_seqlen_k=int(kwargs["max_seqlen_k"]),
            block_table=kwargs.get("block_table"),
            softmax_scale=kwargs.get("softmax_scale"),
            causal=bool(kwargs.get("causal", False)),
        )

    flash_attn_varlen_func._easymagpie_torch_fallback = True  # type: ignore[attr-defined]
    v1_flash_attn.flash_attn_varlen_func = flash_attn_varlen_func
    return True


def _patch_flash_attention_impl(FlashAttentionImpl, *, is_v1: bool) -> bool:
    original = getattr(FlashAttentionImpl, "forward", None)
    if original is None or getattr(FlashAttentionImpl, "_easymagpie_torch_fallback_installed", False):
        return False

    original_signature = None
    try:
        original_signature = inspect.signature(original)
        original_accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD for param in original_signature.parameters.values()
        )
        original_accepts_output_block_scale = "output_block_scale" in original_signature.parameters
    except Exception:
        original_accepts_kwargs = False
        original_accepts_output_block_scale = True

    def _call_original(
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
        extra_kwargs,
    ):
        base_args = (self, layer, query, key, value, kv_cache, attn_metadata)
        if original_accepts_kwargs:
            return original(
                *base_args,
                output=output,
                output_scale=output_scale,
                output_block_scale=output_block_scale,
                **extra_kwargs,
            )
        if extra_kwargs:
            if original_signature is None:
                return original(*base_args, output, output_scale, output_block_scale)
            return original(
                *base_args,
                output=output,
                output_scale=output_scale,
                **{key: value for key, value in extra_kwargs.items() if key in original_signature.parameters},
            )
        if original_accepts_output_block_scale:
            return original(*base_args, output, output_scale, output_block_scale)
        return original(*base_args, output, output_scale)

    def _metadata_ok_for_fallback(self, key, value, kv_cache, attn_metadata) -> bool:
        del self, key, value, kv_cache
        return (not is_v1) or (attn_metadata is not None)

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
        enabled = os.environ.get("EASYMAGPIE_TORCH_FLASH_ATTN_FALLBACK", "1").lower()
        window_ok = _window_is_unrestricted(getattr(self, "sliding_window", None))
        softcap_ok = _softcap_is_zero(getattr(self, "logits_soft_cap", None))
        metadata_ok = _metadata_ok_for_fallback(self, key, value, kv_cache, attn_metadata)
        should_fallback = (
            enabled not in {"0", "false", "no", "off"}
            and output is not None
            and output_scale is None
            and output_block_scale is None
            and not kwargs
            and metadata_ok
            and window_ok
            and getattr(self, "alibi_slopes", None) is None
            and softcap_ok
            and not str(getattr(self, "kv_cache_dtype", "auto")).startswith("fp8")
        )
        if not should_fallback:
            if enabled not in {"0", "false", "no", "off"} and not getattr(
                forward, "_easymagpie_warned_bypassed", False
            ):
                logger.warning(
                    "EasyMagpie torch-SDPA direct fallback bypassed for %s FlashAttentionImpl: "
                    "output=%s output_scale=%s output_block_scale=%s extra_kwargs=%s "
                    "metadata_ok=%s window=%s window_ok=%s alibi=%s softcap=%s "
                    "softcap_ok=%s kv_cache_dtype=%s",
                    "v1" if is_v1 else "legacy",
                    output is not None,
                    output_scale is not None,
                    output_block_scale is not None,
                    sorted(kwargs),
                    metadata_ok,
                    getattr(self, "sliding_window", None),
                    window_ok,
                    getattr(self, "alibi_slopes", None) is not None,
                    getattr(self, "logits_soft_cap", None),
                    softcap_ok,
                    getattr(self, "kv_cache_dtype", None),
                )
                forward._easymagpie_warned_bypassed = True  # type: ignore[attr-defined]
            return _call_original(
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
                kwargs,
            )
        if not getattr(forward, "_easymagpie_warned_active", False):
            logger.warning(
                "EasyMagpie torch-SDPA direct fallback active for %s FlashAttentionImpl: "
                "window=%s softcap=%s kv_cache_dtype=%s",
                "v1" if is_v1 else "legacy",
                getattr(self, "sliding_window", None),
                getattr(self, "logits_soft_cap", None),
                getattr(self, "kv_cache_dtype", None),
            )
            forward._easymagpie_warned_active = True  # type: ignore[attr-defined]
        if is_v1:
            use_cascade = bool(getattr(attn_metadata, "use_cascade", False)) if attn_metadata is not None else False
            if use_cascade:
                if not getattr(forward, "_easymagpie_warned_cascade", False):
                    logger.warning("EasyMagpie torch-SDPA direct fallback bypassed v1 cascade attention")
                    forward._easymagpie_warned_cascade = True  # type: ignore[attr-defined]
                return _call_original(
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
                    kwargs,
                )
            return _torch_v1_flash_attention_forward(self, layer, query, key, value, kv_cache, attn_metadata, output)
        return _torch_flash_attention_forward(self, layer, query, key, value, kv_cache, attn_metadata, output)

    forward._easymagpie_torch_fallback = True  # type: ignore[attr-defined]
    FlashAttentionImpl.forward = forward
    FlashAttentionImpl._easymagpie_torch_fallback_installed = True
    return True


def patch_mamba_streaming_decode() -> None:
    """Treat 1-token streaming extends as decodes so FULL decode cudagraphs work.

    EasyMagpie's streaming-input path keeps extending each request's prompt with
    every chunk, so ``num_computed_tokens < num_prompt_tokens`` (the engine's
    ``is_prefilling`` flag) stays True for the whole stream. vLLM's Mamba2
    metadata builder calls
    :func:`vllm.v1.attention.backends.utils.split_decodes_and_prefills` with
    ``treat_short_extends_as_decodes=False``, so every single-token decode step
    is classified as a *prefill* (``num_prefills>0``).

    That collides with the cudagraph dispatcher, which keys only on query length:
    a uniform ``query_len==1`` batch dispatches the **FULL decode** graph
    regardless of ``is_prefilling``. Two failures result:

    * the replayed decode graph runs the decode Mamba kernels while the metadata
      says prefill, and
    * because ``num_prefills>0``, ``_update_metadata_for_cudagraph_capture``
      never refreshes the persistent ``state_indices_tensor_d`` buffer, so the
      captured kernel reads the capture-time dummy slot (0) instead of the
      request's real Mamba-cache slot -> garbage hidden states.

    Forcing ``treat_short_extends_as_decodes=True`` on vLLM versions that expose
    that flag makes single-token extends classify as decodes
    (``num_prefills==0``), which both matches the dispatched FULL decode graph
    and re-enables the per-step ``state_indices_tensor_d`` refresh. Older vLLM
    builds expose only ``decode_threshold`` and already classify
    ``query_len <= decode_threshold`` as decode, so this wrapper must be
    signature-aware and only forward kwargs the installed helper supports.
    Multi-token context prefills (``query_len>1``) still classify as prefills,
    so this is safe for mixed batches. Advancing Mamba state by one token via
    the decode kernels is semantically identical to a 1-token prefill chunk (it
    reads the slot's state and writes the advanced state back in place), so no
    state update is lost — the only requirement is exactly one new token per
    streamed step (``SamplingParams(max_tokens=1)``).

    Idempotent and process-global; the EasyMagpie plugin only ever serves this
    model so the global patch is acceptable.
    """
    orig = _mamba_attn.split_decodes_and_prefills
    if getattr(orig, "_easymagpie_patched", False):
        return
    try:
        orig_params = inspect.signature(orig).parameters
    except (TypeError, ValueError):
        orig_params = {}
    supports_require_uniform = "require_uniform" in orig_params
    supports_treat_short_extends = "treat_short_extends_as_decodes" in orig_params

    def patched(
        common_attn_metadata,
        decode_threshold: int = 1,
        require_uniform: bool = False,
        treat_short_extends_as_decodes: bool = True,
    ):
        kwargs = {"decode_threshold": decode_threshold}
        if supports_require_uniform:
            kwargs["require_uniform"] = require_uniform
        if supports_treat_short_extends:
            kwargs["treat_short_extends_as_decodes"] = True
        return orig(common_attn_metadata, **kwargs)

    patched._easymagpie_patched = True
    _mamba_attn.split_decodes_and_prefills = patched
    logger.info("Mamba streaming-decode classification patch installed")


class _SiluActivation(nn.Module):
    """``nn.Module`` wrapper around ``F.silu`` (so vLLM's NemotronHMLP can hold it)."""

    def forward(self, x):
        return F.silu(x)


def patch_nemotron_h_moe_layer() -> bool:
    """Register an ``E`` MoE layer for vLLM Nemotron-H builds that omit it.

    Some vLLM versions parse Nemotron-H but only ship Mamba, attention, and dense
    MLP layer classes. EasyMagpie SmallMamba checkpoints keep routed expert
    weights under ``layers.*.mixer.experts.*`` and need a real ``E`` layer to
    load. This lightweight fallback mirrors the checkpoint layout directly
    (router + two-matrix routed experts + two-matrix shared expert). It is slower
    than vLLM's fused MoE path, but it preserves the model contract and keeps the
    rollout backend usable on those builds.
    """
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.model_executor.models import nemotron_h as nh

    existing = nh.ALL_DECODER_LAYER_TYPES.get("E")
    if existing is not None and getattr(existing, "_easymagpie_moe_layer", False):
        return False
    if existing is not None and existing.__name__ != "NemotronHMLPDecoderLayer":
        return False

    class _NemotronHTwoMatrixMLP(nn.Module):
        def __init__(self, hidden_size: int, intermediate_size: int) -> None:
            super().__init__()
            self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
            self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
            self.act_fn = _SiluActivation()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(self.act_fn(self.up_proj(x)))

    class NemotronHMoE(nn.Module):
        def __init__(self, config, prefix: str = "") -> None:
            super().__init__()
            self.n_routed_experts = int(config.n_routed_experts)
            self.num_experts_per_tok = int(config.num_experts_per_tok)
            self.routed_scaling_factor = float(getattr(config, "routed_scaling_factor", 1.0))
            self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", True))
            self.route_indices_on_cpu = _env_flag("EASYMAGPIE_MOE_CPU_ROUTING", True)
            self.layer_name = prefix
            self.use_custom_op = False

            hidden_size = int(config.hidden_size)
            routed_intermediate = int(getattr(config, "moe_intermediate_size", config.intermediate_size))
            shared_intermediate = int(
                getattr(config, "moe_shared_expert_intermediate_size", routed_intermediate)
            )

            self.gate = nn.Linear(hidden_size, self.n_routed_experts, bias=False)
            self.gate.e_score_correction_bias = nn.Parameter(
                torch.empty(self.n_routed_experts), requires_grad=False
            )
            self.experts = nn.ModuleList(
                [_NemotronHTwoMatrixMLP(hidden_size, routed_intermediate) for _ in range(self.n_routed_experts)]
            )
            self.shared_experts = _NemotronHTwoMatrixMLP(hidden_size, shared_intermediate)

            if prefix:
                try:
                    from vllm import envs
                    from vllm.config import get_current_vllm_config

                    if bool(getattr(envs, "VLLM_USE_V1", False)):
                        compilation_config = get_current_vllm_config().compilation_config
                        static_context = compilation_config.static_forward_context
                        if prefix in static_context and static_context[prefix] is not self:
                            raise ValueError(f"Duplicate layer name: {prefix}")
                        static_context[prefix] = self
                        _register_easymagpie_moe_custom_op()
                        self.use_custom_op = True
                except Exception:
                    logger.debug("Could not register EasyMagpie MoE custom op", exc_info=True)

        def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            original_shape = hidden_states.shape
            hidden_states = hidden_states.reshape(-1, original_shape[-1])
            router_logits = self.gate(hidden_states).float()
            if self.use_custom_op and hidden_states.is_cuda and _has_vllm_forward_context():
                output = torch.ops.vllm.easymagpie_moe_forward(
                    hidden_states,
                    router_logits,
                    self.layer_name,
                )
            else:
                output = self.forward_impl(hidden_states, router_logits)
            return output.reshape(original_shape)

        def forward_impl(
            self,
            hidden_states: torch.Tensor,
            router_logits: torch.Tensor,
        ) -> torch.Tensor:
            bias = getattr(self.gate, "e_score_correction_bias", None)
            if bias is not None:
                router_logits = router_logits + bias.float().unsqueeze(0)
            scores = torch.softmax(router_logits, dim=-1)
            topk_weights, topk_ids = torch.topk(scores, k=self.num_experts_per_tok, dim=-1)
            if self.norm_topk_prob:
                topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp_min(1e-20)

            routed = torch.zeros_like(hidden_states)
            route_topk_ids = topk_ids
            route_on_cpu = self.route_indices_on_cpu and topk_ids.is_cuda
            if route_on_cpu:
                route_topk_ids = topk_ids.detach().to("cpu")
            for expert_id, expert in enumerate(self.experts):
                token_ids, slot_ids = torch.where(route_topk_ids == expert_id)
                if token_ids.numel() == 0:
                    continue
                if route_on_cpu:
                    token_ids = token_ids.to(device=hidden_states.device, dtype=torch.long)
                    slot_ids = slot_ids.to(device=hidden_states.device, dtype=torch.long)
                expert_out = expert(hidden_states[token_ids])
                weights = topk_weights[token_ids, slot_ids].to(expert_out.dtype).unsqueeze(-1)
                routed[token_ids] += expert_out * weights

            return routed * self.routed_scaling_factor + self.shared_experts(hidden_states)

    class NemotronHMoEDecoderLayer(nn.Module):
        _easymagpie_moe_layer = True

        def __init__(
            self,
            config,
            layer_idx: int,
            model_config=None,
            cache_config=None,
            quant_config=None,
            prefix: str = "",
        ) -> None:
            super().__init__()
            del layer_idx, model_config, cache_config, quant_config
            self.mixer = NemotronHMoE(config, prefix=f"{prefix}.mixer")
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        def forward(self, hidden_states: torch.Tensor, residual, **kwargs):
            del kwargs
            if residual is None:
                residual = hidden_states
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, residual = self.norm(hidden_states, residual)
            hidden_states = self.mixer(hidden_states)
            return hidden_states, residual

    nh.ALL_DECODER_LAYER_TYPES["E"] = NemotronHMoEDecoderLayer
    logger.info("EasyMagpie Nemotron-H MoE fallback layer registered")
    return True


def patch_silu_shared_experts(backbone) -> int:
    """Replace ``shared_experts.act_fn`` with SiLU on every NemotronHMoE layer.

    vLLM's ``NemotronHMLP`` hard-codes ReLU² for ``shared_experts`` (ignoring
    ``config.mlp_hidden_act``). SmallMamba trained with SiLU, so the mismatch
    blows up shared-expert norms ~5× and the per-layer cosine drops to ≈-0.7 by
    layer 30. Patching only ``act_fn`` (not the whole forward) keeps
    ``NemotronHMLP.forward`` in charge so torch.compile / CUDA-graph capture
    continue to wrap it unchanged.

    Args:
        backbone: the ``NemotronHModel`` instance.

    Returns:
        Number of layers patched.
    """
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        se = getattr(mixer, "shared_experts", None)
        if se is None:
            continue
        se.act_fn = _SiluActivation()
        patched += 1
    logger.info("SiLU shared_experts fix installed on %d layers", patched)
    return patched


def patch_moe_routed_scale(backbone) -> int:
    """Restore ``routed_scaling_factor`` on the NemotronHMoE output in FP16.

    vLLM's ``FusedMoE`` uses an FP16 overflow trick: with
    ``apply_routed_scale_to_output=True`` it does **not** multiply the routed
    output by ``s`` (=routed_scaling_factor); in FP16 it instead divides the
    *shared* output by ``s`` and relies on the decoder layer to keep the whole
    residual stream scaled by ``1/s`` (see ``DeepseekV2DecoderLayer.forward``).
    NemotronH's decoder layer never applies that compensation, so in FP16 the
    MoE block emits ``routed_raw + shared/s == (s*routed + shared)/s`` — the
    correct value divided by ``s``. The MoE contribution to the residual ends up
    ``s``× too small and the error accumulates across the MoE layers.

    We re-multiply each MoE mixer's output by ``s`` in FP16::

        s * (routed_raw + shared/s) = s*routed_raw + shared

    which matches the NeMo reference. FP32/BF16 already take the correct
    ``fused_output *= s`` branch, so the hook is a no-op there.

    Args:
        backbone: the ``NemotronHModel`` instance.

    Returns:
        Number of layers patched.
    """
    patched = 0
    for layer in backbone.layers:
        mixer = getattr(layer, "mixer", None)
        if mixer is None or mixer.__class__.__name__ != "NemotronHMoE":
            continue
        scale = float(getattr(mixer, "routed_scaling_factor", 1.0))
        if scale == 1.0:
            continue

        def _scale_output(_mod, _inp, out, _scale=scale):
            # FusedMoE only defers the scale in FP16; leave other dtypes alone.
            if isinstance(out, torch.Tensor) and out.dtype == torch.float16:
                return out * _scale
            return out

        mixer.register_forward_hook(_scale_output)
        patched += 1
    logger.info("FP16 MoE routed-scale fix installed on %d layers", patched)
    return patched
