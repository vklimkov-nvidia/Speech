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
"""Focused vLLM/vLLM-Omni shims for EasyMagpie NeMo-RL rollouts.

This review branch intentionally keeps only the compatibility surface used by
EasyMagpie RL serving: worker refit RPCs, dense tensor serialization for refit
payloads, light input/import aliases, and AsyncOmni abstract-method fill-ins.
The older broad ``full/all`` vLLM-Omni compatibility layer is not part of the
reviewable RL patch series.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import os
import queue as stdlib_queue
import sys
import types
from typing import Any

logger = logging.getLogger(__name__)


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

def _install_omni_gpu_model_runner_init_kwargs_compat() -> None:
    """Bridge vLLM-Omni's optional ``num_tokens`` arg across vLLM versions."""

    try:
        module = importlib.import_module("vllm_omni.worker.gpu_model_runner")
    except Exception:
        return

    runner_cls = getattr(module, "OmniGPUModelRunner", None)
    if runner_cls is None:
        return

    current = getattr(runner_cls, "_init_model_kwargs", None)
    if current is None or getattr(current, "_easymagpie_init_model_kwargs_compat", False):
        return

    def _base_init_model_kwargs(self: Any, num_tokens: int | None = None) -> Any:
        if num_tokens is None:
            num_tokens = int(getattr(self, "max_num_tokens", 0) or 0)
        mro = type(self).__mro__
        try:
            base_classes = mro[mro.index(runner_cls) + 1 :]
        except ValueError:
            base_classes = mro[1:]
        for base_cls in base_classes:
            base_method = base_cls.__dict__.get("_init_model_kwargs")
            if base_method is None:
                continue
            if getattr(base_method, "_easymagpie_init_model_kwargs_compat", False):
                continue
            try:
                signature = inspect.signature(base_method)
            except (TypeError, ValueError):
                return base_method(self, num_tokens)
            positional = [
                parameter
                for parameter in signature.parameters.values()
                if parameter.kind
                in {
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                }
            ]
            accepts_varargs = any(
                parameter.kind == inspect.Parameter.VAR_POSITIONAL
                for parameter in signature.parameters.values()
            )
            if accepts_varargs or len(positional) >= 2:
                return base_method(self, num_tokens)
            return base_method(self)
        return {}

    _base_init_model_kwargs._easymagpie_init_model_kwargs_compat = True  # type: ignore[attr-defined]
    _base_init_model_kwargs._easymagpie_original = current  # type: ignore[attr-defined]
    runner_cls._init_model_kwargs = _base_init_model_kwargs


def _install_omni_batch_execution_padding_compat() -> None:
    """Let vLLM 0.21 CUDA-graph capture use Omni's fallback batch descriptor.

    The vLLM-Omni branch used by EasyMagpie includes a compatibility fallback
    for ``_determine_batch_execution_and_padding`` that always returns
    ``CUDAGraphMode.NONE``. Newer vLLM V1 passes the runtime capture mode
    explicitly into ``_dummy_run`` and asserts it matches the returned mode.
    Preserve Omni's simple descriptor while reporting the configured runtime
    graph mode when the call is not forced eager.
    """

    try:
        module = importlib.import_module("vllm_omni.worker.gpu_model_runner")
        from vllm.config import CUDAGraphMode
    except Exception:
        return

    runner_cls = getattr(module, "OmniGPUModelRunner", None)
    if runner_cls is None:
        return

    current = getattr(runner_cls, "_determine_batch_execution_and_padding", None)
    if current is None or getattr(current, "_easymagpie_batch_execution_padding_compat", False):
        return

    def _runtime_mode_from_config(self: Any, *, force_uniform_decode: bool) -> Any:
        compilation_config = getattr(getattr(self, "vllm_config", None), "compilation_config", None)
        configured_mode = getattr(compilation_config, "cudagraph_mode", None)
        if configured_mode is None:
            return CUDAGraphMode.NONE
        if force_uniform_decode and hasattr(configured_mode, "decode_mode"):
            return configured_mode.decode_mode()
        if hasattr(configured_mode, "mixed_mode"):
            return configured_mode.mixed_mode()
        return configured_mode

    def _determine_batch_execution_and_padding(self: Any, *args: Any, **kwargs: Any) -> Any:
        result = current(self, *args, **kwargs)
        if not isinstance(result, tuple) or len(result) < 5:
            return result
        cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats = result[:5]
        if bool(kwargs.get("force_eager", False)) or cudagraph_mode != CUDAGraphMode.NONE:
            return result
        runtime_mode = _runtime_mode_from_config(
            self,
            force_uniform_decode=bool(kwargs.get("force_uniform_decode", False)),
        )
        if runtime_mode == CUDAGraphMode.NONE:
            return result
        patched = (runtime_mode, batch_desc, should_ubatch, num_tokens_across_dp, cudagraph_stats)
        if len(result) > 5:
            patched += result[5:]
        return patched

    _determine_batch_execution_and_padding._easymagpie_batch_execution_padding_compat = True  # type: ignore[attr-defined]
    _determine_batch_execution_and_padding._easymagpie_original = current  # type: ignore[attr-defined]
    runner_cls._determine_batch_execution_and_padding = _determine_batch_execution_and_padding


def _install_omni_structured_output_request_compat() -> None:
    """Bridge vLLM-Omni's legacy ``StructuredOutputRequest`` construction.

    The EasyMagpie vLLM-Omni branch builds structured-output metadata with
    ``StructuredOutputRequest(sampling_params=...)`` when every request is
    converted inside the engine core. Some vLLM 0.21 builds accept that keyword
    while others only accept positional construction or no construction for
    requests without guided/structured output constraints. Keep EasyMagpie audio
    requests moving across both variants without changing vLLM-Omni source.
    """

    try:
        request_module = importlib.import_module("vllm_omni.request")
    except Exception:
        return

    original = getattr(request_module, "StructuredOutputRequest", None)
    if original is None or getattr(original, "_easymagpie_structured_output_request_compat", False):
        return

    def structured_output_request_compat(*args: Any, **kwargs: Any) -> Any:
        if "sampling_params" not in kwargs:
            return original(*args, **kwargs)

        sampling_params = kwargs["sampling_params"]
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if "sampling_params" not in str(exc) and "unexpected keyword" not in str(exc):
                raise

        positional_kwargs = dict(kwargs)
        positional_kwargs.pop("sampling_params", None)
        try:
            return original(sampling_params, *args, **positional_kwargs)
        except TypeError:
            return None

    structured_output_request_compat._easymagpie_structured_output_request_compat = True  # type: ignore[attr-defined]
    structured_output_request_compat._easymagpie_original = original  # type: ignore[attr-defined]
    request_module.StructuredOutputRequest = structured_output_request_compat


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
_EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION = 1


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

        def easymagpie_reset_runtime_state(self):
            model_runner = getattr(self, "model_runner", None)
            if model_runner is None:
                return {
                    "ok": False,
                    "runtime_reset_rpc_compat_version": _EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION,
                    "error": "EasyMagpie runtime reset RPC could not find worker.model_runner",
                }
            try:
                runtime_state_reset = _reset_easy_magpie_runner_state_after_refit(model_runner)
                try:
                    import torch

                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                except Exception:
                    pass
                reset_ok = not runtime_state_reset.get("errors")
                return {
                    "ok": bool(reset_ok),
                    "runtime_state_reset": runtime_state_reset,
                    "runtime_reset_rpc_compat_version": _EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION,
                    **(
                        {"error": "EasyMagpie runtime-state reset reported errors"}
                        if not reset_ok
                        else {}
                    ),
                }
            except Exception as exc:
                logger.exception("EasyMagpie runtime-state reset RPC failed")
                return {
                    "ok": False,
                    "runtime_reset_rpc_compat_version": _EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION,
                    "error": f"{type(exc).__name__}: {exc}",
                }

        easymagpie_load_weights._easymagpie_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_load_weights._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        easymagpie_load_non_text_weights._easymagpie_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_load_non_text_weights._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        easymagpie_update_text_embedding_rows._easymagpie_text_row_refit_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_update_text_embedding_rows._easymagpie_text_row_refit_rpc_compat_version = _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        easymagpie_reset_runtime_state._easymagpie_runtime_reset_rpc_compat = True  # type: ignore[attr-defined]
        easymagpie_reset_runtime_state._easymagpie_runtime_reset_rpc_compat_version = _EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION  # type: ignore[attr-defined]
        worker_cls.easymagpie_load_weights = easymagpie_load_weights
        worker_cls.easymagpie_load_non_text_weights = easymagpie_load_non_text_weights
        worker_cls.easymagpie_update_text_embedding_rows = easymagpie_update_text_embedding_rows
        worker_cls.easymagpie_reset_runtime_state = easymagpie_reset_runtime_state
        worker_cls._easymagpie_refit_rpc_compat = True
        worker_cls._easymagpie_refit_rpc_compat_version = _EASYMAGPIE_REFIT_RPC_COMPAT_VERSION
        worker_cls._easymagpie_text_row_refit_rpc_compat = True
        worker_cls._easymagpie_text_row_refit_rpc_compat_version = _EASYMAGPIE_TEXT_ROW_REFIT_RPC_COMPAT_VERSION
        worker_cls._easymagpie_runtime_reset_rpc_compat = True
        worker_cls._easymagpie_runtime_reset_rpc_compat_version = _EASYMAGPIE_RUNTIME_RESET_RPC_COMPAT_VERSION


def install_easy_magpie_refit_rpc_compat() -> None:
    """Install only the EasyMagpie refit worker RPC, leaving generation untouched."""

    _install_ray_objectref_aliases()
    _install_ray_placement_group_aliases()
    _install_vllm_inputs_data_alias()
    _install_vllm_multimodal_inputs_alias()
    _install_engine_utils_compat()
    _install_omni_gpu_model_runner_init_kwargs_compat()
    _install_omni_batch_execution_padding_compat()
    _install_omni_structured_output_request_compat()
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
    _install_omni_gpu_model_runner_init_kwargs_compat()
    _install_omni_batch_execution_padding_compat()
    _install_omni_structured_output_request_compat()
    _install_easy_magpie_refit_rpc_compat()
    _install_v1_serial_utils_dense_tensor_compat()
    _install_async_omni_client_compat()

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



def install_vllm_omni_compat() -> None:
    """Reject the legacy broad compatibility mode in the review branch."""

    raise RuntimeError(
        "EASYMAGPIE_VLLM_COMPAT_MODE=full/all is not included in the "
        "review-friendly EasyMagpie RL branch. Use refit, runtime, serial, "
        "or rl mode for the NeMo-RL rollout/refit path."
    )


__all__ = [
    "install_easy_magpie_refit_rpc_compat",
    "install_easy_magpie_runtime_compat",
    "install_vllm_omni_compat",
]
