import contextlib
import inspect
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from vllm.compilation.cuda_graph import CUDAGraphWrapper as _OriginalCUDAGraphWrapper
from vllm.config import CUDAGraphMode
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context as _vllm_set_forward_context
from vllm.logger import init_logger
try:
    from vllm.model_executor.models.interfaces import supports_mrope
except ImportError:
    def supports_mrope(model) -> bool:
        return bool(getattr(model, "supports_mrope", False))
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.sampling_params import SamplingType
try:
    from vllm.tracing import instrument
except ImportError:
    def instrument(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from vllm.utils import LazyLoader
from vllm.utils import cdiv
try:
    from vllm.v1.spec_decode.draft_model import DraftModelProposer
except ImportError:
    class DraftModelProposer:
        pass
try:
    from vllm.v1.spec_decode.eagle import EagleProposer
except ImportError:
    class EagleProposer:
        pass
try:
    from vllm.v1.spec_decode.extract_hidden_states import ExtractHiddenStatesProposer
except ImportError:
    class ExtractHiddenStatesProposer:
        pass
from vllm.v1.worker.gpu_input_batch import CachedRequestState
try:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner, IntermediateTensors, PerLayerAttnMetadata
except ImportError:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner, IntermediateTensors
    PerLayerAttnMetadata = Any
try:
    from vllm.v1.worker.ubatch_utils import maybe_create_ubatch_slices
except ImportError:
    def maybe_create_ubatch_slices(*args, **kwargs):
        return None, None

from vllm_omni.engine.serialization import deserialize_additional_information
from vllm_omni.model_executor.layers.rotary_embedding.mrope import OmniMRotaryEmbedding as MRotaryEmbedding
from vllm_omni.model_executor.models.output_templates import OmniOutput

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")
    xgr_torch_compile = LazyLoader(
        "xgr_torch_compile",
        globals(),
        "xgrammar.kernels.apply_token_bitmask_inplace_torch_compile",
    )

logger = init_logger(__name__)


_EASYMAGPIE_VLLM_ARCH = "EasyMagpieTTSForConditionalGeneration"
_EASYMAGPIE_VLLM_TARGET = "easymagpie_vllm_omni.easymagpie:EasyMagpieTTSForConditionalGeneration"


def _iter_easymagpie_vllm_registries(model_config=None):
    try:
        from vllm.model_executor import models as model_executor_models

        yield model_executor_models.ModelRegistry
    except Exception as exc:
        logger.warning("Unable to inspect vLLM model_executor.models registry: %s", exc)
    try:
        from vllm.model_executor.models import registry as registry_module

        yield registry_module.ModelRegistry
    except Exception as exc:
        logger.warning("Unable to inspect vLLM registry module: %s", exc)
    if model_config is not None:
        registry = getattr(model_config, "registry", None)
        if registry is not None:
            yield registry


def _register_easymagpie_model_in_vllm(model_config=None, context: str = "worker") -> None:
    registered = 0
    seen: set[int] = set()
    errors: list[str] = []
    for registry in _iter_easymagpie_vllm_registries(model_config):
        registry_id = id(registry)
        if registry_id in seen:
            continue
        seen.add(registry_id)
        try:
            supported_archs = registry.get_supported_archs()
            if _EASYMAGPIE_VLLM_ARCH not in supported_archs:
                registry.register_model(_EASYMAGPIE_VLLM_ARCH, _EASYMAGPIE_VLLM_TARGET)
            if _EASYMAGPIE_VLLM_ARCH in registry.get_supported_archs():
                registered += 1
            else:
                errors.append(f"{type(registry).__name__} missing after register")
        except Exception as exc:
            errors.append(f"{type(registry).__name__}: {exc}")
    if registered == 0:
        raise RuntimeError(
            "Unable to register EasyMagpie model in vLLM worker process "
            f"({context}): {'; '.join(errors) or 'no registry candidates'}"
        )
    logger.warning(
        "EasyMagpie vLLM registry ready in %s across %d registry object(s)",
        context,
        registered,
    )


_EASYMAGPIE_MAMBA_COMMON_BY_BUILDER_ID: dict[int, Any] = {}
_EASYMAGPIE_MAMBA_CPU_FIELDS_BY_BUILDER_ID: dict[int, dict[str, torch.Tensor]] = {}
_EASYMAGPIE_MAMBA_ACTIVE_COMMON: Any | None = None
_EASYMAGPIE_MAMBA_ACTIVE_CPU_FIELDS: dict[str, torch.Tensor] = {}


def set_forward_context(*args, **kwargs):
    kwargs.pop("ubatch_slices", None)
    kwargs.pop("slot_mapping", None)
    return _vllm_set_forward_context(*args, **kwargs)


class _CompatCpuGpuBuffer:
    """Small compatibility wrapper for vLLM-Omni buffer-style fields."""

    def __init__(self, gpu: torch.Tensor, cpu: torch.Tensor | None = None) -> None:
        self.gpu = gpu
        self.cpu = cpu
        self.np = cpu.numpy() if cpu is not None else None

    def copy_to_gpu(self) -> None:
        if self.cpu is None:
            return
        self.gpu.copy_(self.cpu.to(device=self.gpu.device), non_blocking=True)

    def copy_(self, *args, **kwargs):
        return self.gpu.copy_(*args, **kwargs)

    def fill_(self, *args, **kwargs):
        return self.gpu.fill_(*args, **kwargs)

    def __getitem__(self, item):
        return self.gpu[item]

    def __setitem__(self, item, value) -> None:
        self.gpu[item] = value

    def __getattr__(self, name: str):
        return getattr(self.gpu, name)


def _as_compat_buffer(gpu: object, cpu: object | None = None) -> object:
    if isinstance(gpu, _CompatCpuGpuBuffer):
        return gpu
    if isinstance(gpu, torch.Tensor):
        cpu_tensor = cpu if isinstance(cpu, torch.Tensor) else None
        if cpu_tensor is None:
            cpu_tensor = torch.empty(gpu.shape, dtype=gpu.dtype, device="cpu")
        return _CompatCpuGpuBuffer(gpu, cpu_tensor)
    return gpu


def _get_block_table_device_tensor(blk_table: Any, num_reqs: int) -> torch.Tensor:
    try:
        return blk_table.get_device_tensor(num_reqs)
    except TypeError:
        return blk_table.get_device_tensor()


def _cpu_tensor_slice_from_buffer(buffer: Any, length: int) -> torch.Tensor | None:
    cpu_value = getattr(buffer, "cpu", None)
    if isinstance(cpu_value, torch.Tensor):
        return cpu_value[:length]
    if callable(cpu_value):
        try:
            cpu_tensor = cpu_value()
        except Exception:
            cpu_tensor = None
        if isinstance(cpu_tensor, torch.Tensor):
            return cpu_tensor[:length]
    gpu_value = getattr(buffer, "gpu", buffer)
    if isinstance(gpu_value, torch.Tensor):
        return gpu_value[:length].detach().to(device="cpu")
    return None


def _cpu_tensor_from_value(value: Any, length: int | None = None) -> torch.Tensor | None:
    if not isinstance(value, torch.Tensor):
        return None
    tensor = value.detach()
    if tensor.device.type != "cpu":
        tensor = tensor.to(device="cpu")
    if length is not None and tensor.ndim > 0:
        tensor = tensor[:length]
    return tensor


def _common_metadata_num_reqs(common_attn_metadata: Any) -> int | None:
    try:
        num_reqs = getattr(common_attn_metadata, "num_reqs")
    except Exception:
        return None
    try:
        if isinstance(num_reqs, torch.Tensor):
            return int(num_reqs.item())
        return int(num_reqs)
    except Exception:
        return None


def _common_metadata_cpu_field(common_attn_metadata: Any, field_name: str) -> torch.Tensor | None:
    num_reqs = _common_metadata_num_reqs(common_attn_metadata)
    if field_name == "query_start_loc_cpu":
        length = None if num_reqs is None else num_reqs + 1
    elif field_name in (
        "seq_lens_cpu",
        "_seq_lens_cpu",
        "seq_lens_cpu_upper_bound",
        "num_computed_tokens_cpu",
        "_num_computed_tokens_cpu",
    ):
        length = num_reqs
    else:
        length = None

    candidate_fields = {
        "query_start_loc_cpu": ("query_start_loc_cpu", "query_start_loc"),
        "seq_lens_cpu": ("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens"),
        "_seq_lens_cpu": ("_seq_lens_cpu", "seq_lens_cpu", "seq_lens_cpu_upper_bound", "seq_lens"),
        "seq_lens_cpu_upper_bound": ("seq_lens_cpu_upper_bound", "seq_lens_cpu", "_seq_lens_cpu", "seq_lens"),
        "num_computed_tokens_cpu": ("num_computed_tokens_cpu", "_num_computed_tokens_cpu"),
        "_num_computed_tokens_cpu": ("_num_computed_tokens_cpu", "num_computed_tokens_cpu"),
    }.get(field_name, (field_name,))

    for candidate in candidate_fields:
        try:
            value = getattr(common_attn_metadata, candidate)
        except Exception:
            value = None
        tensor = _cpu_tensor_from_value(value, length)
        if tensor is not None:
            return tensor

    if field_name in ("num_computed_tokens_cpu", "_num_computed_tokens_cpu"):
        seq_lens_cpu = _common_metadata_cpu_field(common_attn_metadata, "seq_lens_cpu")
        query_start_loc_cpu = _common_metadata_cpu_field(common_attn_metadata, "query_start_loc_cpu")
        if (
            seq_lens_cpu is not None
            and query_start_loc_cpu is not None
            and query_start_loc_cpu.ndim > 0
            and seq_lens_cpu.ndim > 0
            and query_start_loc_cpu.numel() >= seq_lens_cpu.numel() + 1
        ):
            query_lens_cpu = query_start_loc_cpu[1 : seq_lens_cpu.numel() + 1] - query_start_loc_cpu[: seq_lens_cpu.numel()]
            return seq_lens_cpu - query_lens_cpu

    return None


def _expanded_cpu_field_aliases(cpu_fields: dict[str, Any]) -> dict[str, Any]:
    def _first_present(*keys: str) -> Any:
        for key in keys:
            value = cpu_fields.get(key)
            if value is not None:
                return value
        return None

    expanded = dict(cpu_fields)
    seq_lens_cpu = _first_present("seq_lens_cpu", "_seq_lens_cpu", "seq_lens_cpu_upper_bound")
    if seq_lens_cpu is not None:
        expanded.setdefault("seq_lens_cpu", seq_lens_cpu)
        expanded.setdefault("_seq_lens_cpu", seq_lens_cpu)
        expanded.setdefault("seq_lens_cpu_upper_bound", seq_lens_cpu)
    num_computed_tokens_cpu = _first_present("num_computed_tokens_cpu", "_num_computed_tokens_cpu")
    if num_computed_tokens_cpu is not None:
        expanded.setdefault("num_computed_tokens_cpu", num_computed_tokens_cpu)
        expanded.setdefault("_num_computed_tokens_cpu", num_computed_tokens_cpu)
    return expanded


def _ensure_common_metadata_cpu_kwargs(kwargs: dict[str, Any]) -> None:
    for cpu_key, tensor_key in (
        ("query_start_loc_cpu", "query_start_loc"),
        ("seq_lens_cpu", "seq_lens"),
    ):
        if kwargs.get(cpu_key) is not None:
            continue
        tensor = kwargs.get(tensor_key)
        if isinstance(tensor, torch.Tensor):
            kwargs[cpu_key] = tensor.detach().to(device="cpu")
    if kwargs.get("num_computed_tokens_cpu") is None:
        seq_lens_cpu = kwargs.get("seq_lens_cpu")
        query_start_loc_cpu = kwargs.get("query_start_loc_cpu")
        if (
            isinstance(seq_lens_cpu, torch.Tensor)
            and isinstance(query_start_loc_cpu, torch.Tensor)
            and query_start_loc_cpu.ndim > 0
            and seq_lens_cpu.ndim > 0
            and query_start_loc_cpu.numel() >= seq_lens_cpu.numel() + 1
        ):
            query_lens_cpu = query_start_loc_cpu[1 : seq_lens_cpu.numel() + 1] - query_start_loc_cpu[: seq_lens_cpu.numel()]
            kwargs["num_computed_tokens_cpu"] = seq_lens_cpu - query_lens_cpu
    cpu_aliases = _expanded_cpu_field_aliases(
        {
            key: kwargs.get(key)
            for key in (
                "seq_lens_cpu",
                "_seq_lens_cpu",
                "seq_lens_cpu_upper_bound",
                "num_computed_tokens_cpu",
                "_num_computed_tokens_cpu",
            )
        }
    )
    for key, value in cpu_aliases.items():
        if value is not None and kwargs.get(key) is None:
            kwargs[key] = value


class _CommonAttentionMetadataCompatProxy:
    def __init__(self, metadata: Any, extras: dict[str, Any]) -> None:
        object.__setattr__(self, "_metadata", metadata)
        object.__setattr__(self, "_extras", extras)

    def __getattr__(self, key: str) -> Any:
        extras = object.__getattribute__(self, "_extras")
        if key in extras:
            return extras[key]
        return getattr(object.__getattribute__(self, "_metadata"), key)

    def __setattr__(self, key: str, value: Any) -> None:
        extras = object.__getattribute__(self, "_extras")
        if key in extras:
            extras[key] = value
            return
        setattr(object.__getattribute__(self, "_metadata"), key, value)


def _metadata_with_cpu_fields(metadata: Any, cpu_fields: dict[str, torch.Tensor]) -> Any:
    cpu_fields = _expanded_cpu_field_aliases(cpu_fields)
    if metadata is None or not cpu_fields:
        return metadata
    unresolved: dict[str, Any] = {}
    for key, value in cpu_fields.items():
        if value is None:
            continue
        try:
            if not hasattr(metadata, key) or getattr(metadata, key) is None:
                setattr(metadata, key, value)
        except Exception:
            pass
        try:
            resolved = getattr(metadata, key)
        except Exception:
            resolved = None
        if resolved is None:
            unresolved[key] = value
    if unresolved:
        return _CommonAttentionMetadataCompatProxy(metadata, unresolved)
    return metadata


def _make_common_attention_metadata(metadata_cls: Any, **kwargs: Any) -> Any:
    _ensure_common_metadata_cpu_kwargs(kwargs)
    try:
        parameters = inspect.signature(metadata_cls).parameters
    except (TypeError, ValueError):
        return metadata_cls(**kwargs)
    metadata_kwargs = kwargs
    if not any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()):
        metadata_kwargs = {key: value for key, value in kwargs.items() if key in parameters}
    metadata = metadata_cls(**metadata_kwargs)
    unresolved: dict[str, Any] = {}
    for key, value in kwargs.items():
        if key in metadata_kwargs:
            continue
        try:
            if not hasattr(metadata, key) or getattr(metadata, key) is None:
                setattr(metadata, key, value)
        except Exception:
            pass
        try:
            resolved = getattr(metadata, key)
        except Exception:
            resolved = None
        if resolved is None and value is not None:
            unresolved[key] = value
    if unresolved:
        return _CommonAttentionMetadataCompatProxy(metadata, unresolved)
    return metadata


def _patch_mamba_chunk_metadata_cpu_fields(builder: Any) -> None:
    builder_cls = builder.__class__
    if getattr(builder_cls, "_easymagpie_mamba_chunk_cpu_fields_compat", False):
        return
    original = getattr(builder_cls, "_build_chunk_metadata_tensors", None)
    if original is None:
        return
    try:
        signature = inspect.signature(original)
    except (TypeError, ValueError):
        return
    parameters = tuple(signature.parameters)
    accepts_var_keyword = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    field_names = (
        "seq_lens_cpu",
        "_seq_lens_cpu",
        "seq_lens_cpu_upper_bound",
        "query_start_loc_cpu",
        "num_computed_tokens_cpu",
        "_num_computed_tokens_cpu",
    )

    def _build_chunk_metadata_tensors(self: Any, *args: Any, **kwargs: Any) -> Any:
        common = (
            getattr(self, "_easymagpie_common_attn_metadata", None)
            or _EASYMAGPIE_MAMBA_COMMON_BY_BUILDER_ID.get(id(self))
            or _EASYMAGPIE_MAMBA_ACTIVE_COMMON
        )
        if common is None:
            return original(self, *args, **kwargs)
        explicit_fields = (
            getattr(self, "_easymagpie_mamba_cpu_field_values", None)
            or _EASYMAGPIE_MAMBA_CPU_FIELDS_BY_BUILDER_ID.get(id(self))
            or _EASYMAGPIE_MAMBA_ACTIVE_CPU_FIELDS
            or {}
        )
        args_list = [self, *args]
        kwargs_copy = dict(kwargs)
        changed = False
        for field_name in field_names:
            replacement = explicit_fields.get(field_name)
            if replacement is None:
                replacement = _common_metadata_cpu_field(common, field_name)
            if replacement is None:
                continue
            if field_name in parameters:
                parameter_index = parameters.index(field_name)
                if parameter_index < len(args_list):
                    if args_list[parameter_index] is None:
                        args_list[parameter_index] = replacement
                        changed = True
                    continue
                if kwargs_copy.get(field_name) is None:
                    kwargs_copy[field_name] = replacement
                    changed = True
                continue
            if accepts_var_keyword and kwargs_copy.get(field_name) is None:
                kwargs_copy[field_name] = replacement
                changed = True
        for metadata_param_name in ("common", "common_attn_metadata"):
            if metadata_param_name not in parameters:
                continue
            parameter_index = parameters.index(metadata_param_name)
            if parameter_index < len(args_list):
                args_list[parameter_index] = _metadata_with_cpu_fields(args_list[parameter_index], explicit_fields)
                changed = True
                continue
            if metadata_param_name in kwargs_copy:
                kwargs_copy[metadata_param_name] = _metadata_with_cpu_fields(
                    kwargs_copy[metadata_param_name],
                    explicit_fields,
                )
                changed = True
        if changed:
            return original(*args_list, **kwargs_copy)
        print(
            "EASYMAGPIE_MAMBA_CPU_FIELD_MISS "
            f"builder={self.__class__.__module__}.{self.__class__.__name__} "
            f"params={parameters} accepts_var_keyword={accepts_var_keyword} "
            f"explicit_keys={tuple(explicit_fields)} "
            f"common={type(common).__name__} kwargs={tuple(kwargs)} "
            f"args_len={len(args)}",
            file=sys.stderr,
            flush=True,
        )
        return original(self, *args, **kwargs)

    _build_chunk_metadata_tensors._easymagpie_original = original  # type: ignore[attr-defined]
    setattr(builder_cls, "_build_chunk_metadata_tensors", _build_chunk_metadata_tensors)
    setattr(builder_cls, "_easymagpie_mamba_chunk_cpu_fields_compat", True)


def _prepare_attention_metadata_builder(
    builder: Any,
    common_attn_metadata: Any,
    cpu_field_values: dict[str, torch.Tensor] | None = None,
) -> None:
    global _EASYMAGPIE_MAMBA_ACTIVE_COMMON, _EASYMAGPIE_MAMBA_ACTIVE_CPU_FIELDS
    module_name = getattr(builder.__class__, "__module__", "")
    class_name = getattr(builder.__class__, "__name__", "")
    if "mamba" not in module_name.lower() and "mamba" not in class_name.lower():
        return
    _patch_mamba_chunk_metadata_cpu_fields(builder)
    cpu_fields = cpu_field_values or {}
    _EASYMAGPIE_MAMBA_COMMON_BY_BUILDER_ID[id(builder)] = common_attn_metadata
    _EASYMAGPIE_MAMBA_CPU_FIELDS_BY_BUILDER_ID[id(builder)] = cpu_fields
    _EASYMAGPIE_MAMBA_ACTIVE_COMMON = common_attn_metadata
    _EASYMAGPIE_MAMBA_ACTIVE_CPU_FIELDS = cpu_fields
    try:
        setattr(builder, "_easymagpie_common_attn_metadata", common_attn_metadata)
    except Exception:
        pass
    try:
        setattr(builder, "_easymagpie_mamba_cpu_field_values", cpu_fields)
    except Exception:
        pass


def _adapt_v1_flash_attention_metadata_for_legacy_backend(attn_metadata: Any, common_attn_metadata: Any) -> Any:
    if not hasattr(attn_metadata, "num_actual_tokens") or hasattr(attn_metadata, "num_prefill_tokens"):
        return attn_metadata

    try:
        from vllm.v1.attention.backends.utils import split_decodes_and_prefills

        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = split_decodes_and_prefills(
            common_attn_metadata
        )
    except Exception:
        num_prefills = int(getattr(common_attn_metadata, "num_reqs", 0) or 0)
        num_prefill_tokens = int(getattr(common_attn_metadata, "num_actual_tokens", 0) or 0)
        num_decodes = 0
        num_decode_tokens = 0

    max_seq_len = int(getattr(attn_metadata, "max_seq_len", 0) or 0)
    max_query_len = int(getattr(attn_metadata, "max_query_len", 0) or 0)
    for name, value in (
        ("num_prefills", int(num_prefills)),
        ("num_prefill_tokens", int(num_prefill_tokens)),
        ("num_decode_tokens", int(num_decode_tokens)),
        ("seq_lens_tensor", getattr(attn_metadata, "seq_lens", None)),
        ("block_tables", getattr(attn_metadata, "block_table", None)),
        ("seq_start_loc", getattr(attn_metadata, "query_start_loc", None)),
        ("context_lens_tensor", None),
        ("use_cuda_graph", False),
        ("max_prefill_seq_len", max_seq_len if num_prefill_tokens else 0),
        ("max_decode_seq_len", max_seq_len if num_decode_tokens else 0),
        ("max_decode_query_len", max_query_len if num_decode_tokens else 0),
        ("multi_modal_placeholder_index_maps", None),
        ("enable_kv_scales_calculation", False),
        ("num_encoder_tokens", None),
        ("encoder_seq_lens", None),
        ("encoder_seq_lens_tensor", None),
        ("encoder_seq_start_loc", None),
        ("max_encoder_seq_len", None),
        ("cross_slot_mapping", None),
        ("cross_block_tables", None),
        ("prefill_metadata", attn_metadata if num_prefill_tokens else None),
        ("decode_metadata", attn_metadata if num_decode_tokens else None),
    ):
        try:
            setattr(attn_metadata, name, value)
        except Exception:
            pass

    if num_decodes and not num_decode_tokens:
        try:
            attn_metadata.num_decode_tokens = int(num_decodes)
        except Exception:
            pass
    return attn_metadata


class CUDAGraphWrapper(_OriginalCUDAGraphWrapper):
    def __getattr__(self, key: str) -> Any:
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        raise AttributeError(f"Attribute {key} not exists in the runnable of cudagraph wrapper")


# Patch vLLM's CUDAGraphWrapper with our optimized version
for _module_name, _module in sys.modules.items():
    if "vllm" not in _module_name:
        continue
    if hasattr(_module, "CUDAGraphWrapper") and _module.CUDAGraphWrapper is _OriginalCUDAGraphWrapper:
        _module.CUDAGraphWrapper = CUDAGraphWrapper


class OmniGPUModelRunner(GPUModelRunner):
    enable_prompt_embeds = False
    uses_xdrope_dim = 0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not hasattr(self, "enable_prompt_embeds"):
            self.enable_prompt_embeds = False
        if not hasattr(self, "uses_xdrope_dim"):
            self.uses_xdrope_dim = 0
        if not hasattr(self, "routed_experts_initialized"):
            self.routed_experts_initialized = False
        if not hasattr(self, "calculate_kv_scales"):
            self.calculate_kv_scales = False
        if not hasattr(self, "broadcast_pp_output"):
            self.broadcast_pp_output = False
        if not hasattr(self, "use_async_scheduling"):
            self.use_async_scheduling = False
        if not hasattr(self, "num_prompt_logprobs"):
            self.num_prompt_logprobs = {}
        self.input_ids = _as_compat_buffer(self.input_ids, getattr(self, "input_ids_cpu", None))
        self.positions = _as_compat_buffer(self.positions, getattr(self, "positions_cpu", None))
        self.query_start_loc = _as_compat_buffer(self.query_start_loc, getattr(self, "query_start_loc_cpu", None))
        self.seq_lens = _as_compat_buffer(self.seq_lens, getattr(self, "seq_lens_cpu", None))
        if hasattr(self, "mrope_positions"):
            self.mrope_positions = _as_compat_buffer(
                self.mrope_positions,
                getattr(self, "mrope_positions_cpu", None),
            )
        if hasattr(self, "xdrope_positions"):
            self.xdrope_positions = _as_compat_buffer(
                self.xdrope_positions,
                getattr(self, "xdrope_positions_cpu", None),
            )
        self.model_intermediate_buffer: dict[str, dict[str, Any]] = {}
        self._omni_num_scheduled_tokens_np: np.ndarray | None = None
        self._omni_last_model_output: object | None = None

    def _make_buffer(self, *size, dtype, numpy=True):
        device = getattr(self, "device", None)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        gpu = torch.empty(size, dtype=dtype, device=device)
        if numpy:
            cpu = torch.empty(size, dtype=dtype, device="cpu")
            return _CompatCpuGpuBuffer(gpu, cpu)
        return _CompatCpuGpuBuffer(gpu)

    def synchronize_input_prep(self):
        return contextlib.nullcontext()

    def maybe_dummy_run_with_lora(self, lora_config, num_scheduled_tokens, *_args, **_kwargs):
        if lora_config is None:
            return contextlib.nullcontext()
        return super().maybe_dummy_run_with_lora(lora_config, num_scheduled_tokens)

    def _init_model_kwargs(self, num_tokens=None):
        if num_tokens is None:
            num_tokens = int(getattr(self, "max_num_tokens", 0) or 0)
        return super()._init_model_kwargs(num_tokens)

    def maybe_randomize_inputs(self, input_ids, inputs_embeds=None):
        if input_ids is None:
            return contextlib.nullcontext()
        try:
            return super().maybe_randomize_inputs(input_ids, inputs_embeds)
        except TypeError:
            return super().maybe_randomize_inputs(input_ids)

    def maybe_get_kv_connector_output(self, scheduler_output, *args, **kwargs):
        kwargs.pop("defer_finalize", None)
        return super().maybe_get_kv_connector_output(
            scheduler_output, *args, **kwargs
        )

    def _determine_batch_execution_and_padding(self, **kwargs):
        """Compatibility fallback for vLLM-Omni against newer vLLM runners."""

        class _BatchDescriptor:
            def __init__(self, *, num_tokens: int, num_reqs: int, uniform_decode: bool = False) -> None:
                self.num_tokens = num_tokens
                self.num_reqs = num_reqs
                self.uniform_decode = uniform_decode

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

    def _get_slot_mappings(
        self,
        num_tokens_padded: int,
        num_reqs_padded: int,
        num_tokens_unpadded: int,
        ubatch_slices=None,
    ):
        if not (
            hasattr(self, "kv_cache_config")
            and self.kv_cache_config is not None
            and len(self.kv_cache_config.kv_cache_groups) > 0
        ):
            return None, None

        try:
            from vllm.v1.kv_cache_interface import EncoderOnlyAttentionSpec
        except Exception:
            EncoderOnlyAttentionSpec = ()  # type: ignore[assignment]

        def _mapping_for_group(kv_cache_gid: int):
            kv_cache_spec = self.kv_cache_config.kv_cache_groups[kv_cache_gid].kv_cache_spec
            if EncoderOnlyAttentionSpec and isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                slot_mapping = torch.zeros(
                    (num_tokens_padded,),
                    dtype=torch.int64,
                    device=self.device,
                )
            else:
                blk_table = self.input_batch.block_table[kv_cache_gid]
                raw_slot_mapping = blk_table.slot_mapping
                slot_mapping_tensor = getattr(raw_slot_mapping, "gpu", raw_slot_mapping)
                slot_mapping = slot_mapping_tensor[:num_tokens_padded]
            slot_mapping[num_tokens_unpadded:num_tokens_padded].fill_(-1)
            return slot_mapping

        slot_mappings_by_gid = {
            gid: _mapping_for_group(gid)
            for gid, _ in enumerate(self.kv_cache_config.kv_cache_groups)
        }
        slot_mappings_by_layer = {}
        for gid, kv_cache_group in enumerate(self.kv_cache_config.kv_cache_groups):
            slot_mapping = slot_mappings_by_gid[gid]
            for layer_name in kv_cache_group.layer_names:
                slot_mappings_by_layer[layer_name] = slot_mapping
        if ubatch_slices is not None:
            return slot_mappings_by_gid, [
                {
                    layer_name: slot_mapping[ubatch.token_slice]
                    for layer_name, slot_mapping in slot_mappings_by_layer.items()
                }
                for ubatch in ubatch_slices
            ]
        return slot_mappings_by_gid, slot_mappings_by_layer

    def _prepare_inputs(self, scheduler_output, num_scheduled_tokens_np=None):
        if num_scheduled_tokens_np is None:
            return super()._prepare_inputs(scheduler_output)

        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        self.input_batch.block_table.commit_block_table(num_reqs)
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens_np)
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens_np)

        positions_buffer_np = getattr(self.positions, "np", None)
        if positions_buffer_np is None:
            positions_np = self.input_batch.num_computed_tokens_cpu[req_indices] + arange[:total_num_scheduled_tokens]
        else:
            positions_np = positions_buffer_np[:total_num_scheduled_tokens]
            np.add(
                self.input_batch.num_computed_tokens_cpu[req_indices],
                arange,
                out=positions_np,
            )

        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        token_indices = positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            torch.from_numpy(token_indices),
            out=self.input_ids.cpu[:total_num_scheduled_tokens],
        )

        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        self.seq_lens.np[:num_reqs] = self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens_np

        positions_gpu = getattr(self.positions, "gpu", self.positions)
        if positions_buffer_np is None:
            positions_gpu[:total_num_scheduled_tokens].copy_(
                torch.from_numpy(positions_np).to(device=positions_gpu.device, dtype=positions_gpu.dtype),
                non_blocking=True,
            )
        else:
            self.positions.copy_to_gpu()
        self.query_start_loc.copy_to_gpu()

        try:
            self.input_batch.block_table.compute_slot_mapping(
                num_reqs,
                self.query_start_loc.gpu[: num_reqs + 1],
                positions_gpu[:total_num_scheduled_tokens],
            )
        except TypeError:
            self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
            self.input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        self.input_ids.copy_to_gpu()
        if self.uses_mrope:
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True,
            )
        self.query_start_loc.copy_to_gpu()
        self.seq_lens.copy_to_gpu()

        self.seq_lens.gpu[num_reqs:].fill_(0)
        self.query_start_loc.gpu[num_reqs + 1 :].fill_(int(cu_num_tokens[-1]))

        if len(scheduler_output.scheduled_spec_decode_tokens) == 0:
            logits_indices = self.query_start_loc.gpu[1 : num_reqs + 1] - 1
            spec_decode_metadata = None
        else:
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices

        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens_np)

        return logits_indices, spec_decode_metadata

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        ubatch_slices=None,
        logits_indices=None,
        use_spec_decode: bool = False,
        for_cudagraph_capture: bool = False,
        num_scheduled_tokens=None,
        cascade_attn_prefix_lens=None,
        slot_mappings=None,
    ):
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None
        if ubatch_slices is not None:
            raise NotImplementedError("EasyMagpie vLLM-Omni compatibility shim does not support ubatching")

        from vllm.v1.attention.backends.utils import CommonAttentionMetadata

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs
        attn_metadata = {}
        spec_decode_common_attn_metadata = None

        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_config.kv_cache_groups):
            blk_table = self.input_batch.block_table[kv_cache_group_id]
            block_table_tensor = _get_block_table_device_tensor(blk_table, num_reqs_padded)[:num_reqs_padded]
            if num_reqs_padded > num_reqs:
                block_table_tensor[num_reqs:num_reqs_padded].fill_(0)
            if slot_mappings is not None:
                slot_mapping = slot_mappings[kv_cache_group_id]
            else:
                raw_slot_mapping = blk_table.slot_mapping
                slot_mapping_tensor = getattr(raw_slot_mapping, "gpu", raw_slot_mapping)
                slot_mapping = slot_mapping_tensor[:num_tokens_padded]
                slot_mapping[num_tokens:num_tokens_padded].fill_(-1)

            seq_lens_cpu = _cpu_tensor_slice_from_buffer(self.seq_lens, num_reqs)
            if seq_lens_cpu is None:
                raise RuntimeError("EasyMagpie vLLM-Omni compatibility shim could not build seq_lens_cpu")
            query_start_loc_cpu = _cpu_tensor_slice_from_buffer(self.query_start_loc, num_reqs + 1)
            if query_start_loc_cpu is None:
                raise RuntimeError("EasyMagpie vLLM-Omni compatibility shim could not build query_start_loc_cpu")
            num_computed_tokens_cpu = self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs]
            num_prompt_tokens_cpu = getattr(self.input_batch, "num_prompt_tokens_cpu_tensor", None)
            is_prefilling = None
            if isinstance(num_prompt_tokens_cpu, torch.Tensor):
                is_prefilling = num_computed_tokens_cpu < num_prompt_tokens_cpu[:num_reqs]
            max_seq_len = int(seq_lens_cpu.max().item()) if num_reqs > 0 else 0
            common_attn_metadata = _make_common_attention_metadata(
                CommonAttentionMetadata,
                query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=self.seq_lens.gpu[:num_reqs],
                seq_lens_cpu=seq_lens_cpu,
                _seq_lens_cpu=seq_lens_cpu,
                seq_lens_cpu_upper_bound=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                _num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=num_tokens,
                max_query_len=max_query_len,
                max_seq_len=max_seq_len,
                block_table_tensor=block_table_tensor,
                slot_mapping=slot_mapping,
                causal=True,
                is_prefilling=is_prefilling,
                positions=self.positions.gpu[:num_tokens],
            )
            if use_spec_decode and spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                builder = getattr(attn_group, "metadata_builder", None)
                if builder is None:
                    builder = attn_group.get_metadata_builder(0)
                common_prefix_len = 0
                if cascade_attn_prefix_lens is not None:
                    common_prefix_len = cascade_attn_prefix_lens[kv_cache_group_id][0]
                _prepare_attention_metadata_builder(
                    builder,
                    common_attn_metadata,
                    {
                        "seq_lens_cpu": seq_lens_cpu,
                        "_seq_lens_cpu": seq_lens_cpu,
                        "seq_lens_cpu_upper_bound": seq_lens_cpu,
                        "query_start_loc_cpu": query_start_loc_cpu,
                        "num_computed_tokens_cpu": num_computed_tokens_cpu,
                        "_num_computed_tokens_cpu": num_computed_tokens_cpu,
                    },
                )
                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                )
                attn_metadata_i = _adapt_v1_flash_attention_metadata_for_legacy_backend(
                    attn_metadata_i, common_attn_metadata
                )
                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        return attn_metadata, spec_decode_common_attn_metadata

    def initialize_metadata_builders(self, kv_cache_config, kernel_block_sizes):
        """Override to fix scheduler_metadata buffer size for FA3 + CUDA graph.

        The upstream FlashAttentionMetadataBuilder pre-allocates
        scheduler_metadata with (max_num_seqs + 1) entries, but FA3's
        get_scheduler_metadata() can return up to
        (max_num_seqs * max_num_splits + 1) entries, causing a RuntimeError
        during CUDA graph capture.  After calling the parent implementation
        we resize any too-small buffers.
        """
        super().initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        for kv_cache_group in self.attn_groups:
            for attn_group in kv_cache_group:
                for builder in attn_group.metadata_builders:
                    sm = getattr(builder, "scheduler_metadata", None)
                    max_num_splits = getattr(builder, "max_num_splits", 0)
                    if sm is not None and max_num_splits > 1:
                        required = self.scheduler_config.max_num_seqs * max_num_splits + 1
                        if sm.shape[0] < required:
                            builder.scheduler_metadata = torch.zeros(
                                required,
                                dtype=sm.dtype,
                                device=sm.device,
                            )

    @instrument(span_name="Loading (GPU)")
    def load_model(self, *args, **kwargs) -> None:
        _register_easymagpie_model_in_vllm(
            getattr(self, "model_config", None),
            "gpu_model_runner.load_model.model_config",
        )
        _register_easymagpie_model_in_vllm(
            getattr(getattr(self, "vllm_config", None), "model_config", None),
            "gpu_model_runner.load_model.vllm_config",
        )
        super().load_model(*args, **kwargs)

        # TODO move this model specific logic to a separate class
        # TTS model IS the talker (no .talker sub-attr); use getattr to support both Omni and TTS.
        self.has_talker_mtp = False
        talker_mtp = getattr(self.model, "talker_mtp", None)
        if talker_mtp is not None:
            self.talker_mtp = talker_mtp  # type: ignore[assignment]
            self.has_talker_mtp = True
            cudagraph_mode = self.compilation_config.cudagraph_mode
            assert cudagraph_mode is not None
            # Only wrap talker_mtp in CUDAGraphWrapper for Omni models that
            # have a separate .talker sub-module.  TTS models' code predictor
            # has internal AR loops / torch.multinomial — not graph-safe.
            has_separate_talker = getattr(self.model, "talker", None) is not None
            if cudagraph_mode.has_full_cudagraphs() and has_separate_talker:
                self.talker_mtp = CUDAGraphWrapper(talker_mtp, self.vllm_config, runtime_mode=CUDAGraphMode.FULL)
            # TTS exposes mtp_hidden_size; Omni uses hf_text_config.hidden_size.
            hidden_size = int(
                getattr(self.model, "mtp_hidden_size", 0) or getattr(self.model_config.hf_text_config, "hidden_size")
            )
            max_batch_size = max(self.max_num_reqs, self.compilation_config.max_cudagraph_capture_size)
            self.talker_mtp_input_ids = self._make_buffer(max_batch_size, dtype=torch.int32)
            self.talker_mtp_inputs_embeds = self._make_buffer(
                max_batch_size, hidden_size, dtype=self.dtype, numpy=False
            )
            self.last_talker_hidden = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)
            self.text_step = self._make_buffer(max_batch_size, hidden_size, dtype=self.dtype, numpy=False)

    def _init_mrope_positions(self, req_state: CachedRequestState):
        """Initialize M-RoPE positions for multimodal inputs.

        Extracts multimodal feature metadata (image grids, video grids,
        audio features) and computes M-RoPE positions for proper positional
        encoding of multimodal tokens.

        Args:
            req_state: Cached request state containing multimodal features

        Raises:
            AssertionError: If the model does not support M-RoPE
        """
        image_grid_thw = []
        video_grid_thw = []
        second_per_grid_ts = []
        audio_feature_lengths = []
        use_audio_in_video = False
        for mm_feature in req_state.mm_features:
            mm_item = mm_feature.data
            if mm_item is None:
                continue
            mm_input = mm_item.get_data()
            if (t := mm_input.get("image_grid_thw")) is not None:
                image_grid_thw.append(t.tolist())
            if (t := mm_input.get("video_grid_thw")) is not None:
                video_grid_thw.append(t.tolist())
            if (t := mm_input.get("second_per_grid_ts")) is not None:
                second_per_grid_ts.append(t)
            if (t := mm_input.get("audio_feature_lengths")) is not None:
                audio_feature_lengths.append(t)
            # Check for use_audio_in_video
            use_audio_in_video_value = mm_input.get("use_audio_in_video")
            if use_audio_in_video_value is not None:
                use_audio_in_video = bool(use_audio_in_video_value.item())

        if supports_mrope(self.get_model()):
            # Model implements SupportsMRoPE interface
            # Pass all extracted metadata; models use what they need via **kwargs
            req_state.mrope_positions, req_state.mrope_position_delta = self.model.get_mrope_input_positions(
                req_state.prompt_token_ids,
                mm_features=req_state.mm_features,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )
        else:
            req_state.mrope_positions, req_state.mrope_position_delta = MRotaryEmbedding.get_input_positions_tensor(
                req_state.prompt_token_ids,
                hf_config=self.model_config.hf_config,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                audio_feature_lengths=audio_feature_lengths,
                use_audio_in_video=use_audio_in_video,
            )

    def _calc_mrope_positions(self, scheduler_output: "SchedulerOutput"):
        """Calculate M-RoPE positions for scheduled tokens.

        Delegates to the upstream implementation first, then applies a fixup
        pass for models that pre-compute 2D spatial decode positions (e.g.
        GLM-Image).  This avoids duplicating the full upstream method while
        still supporting non-linear decode position patterns.

        Models opt-in by declaring ``precomputed_mrope_decode = True`` as a
        class attribute.  When set, ``get_mrope_input_positions`` is expected
        to return positions covering **both** prefill and decode tokens.
        """
        # Run upstream logic (handles prompt positions + linear decode fallback)
        super()._calc_mrope_positions(scheduler_output)

        # Only run the fixup if the model pre-computes decode M-RoPE positions
        if not getattr(self.get_model(), "precomputed_mrope_decode", False):
            return

        self._fixup_precomputed_mrope_decode_positions(scheduler_output)

    def _fixup_precomputed_mrope_decode_positions(self, scheduler_output: "SchedulerOutput") -> None:
        """Overwrite linear decode M-RoPE positions with pre-computed ones.

        For image-generation models (like GLM-Image) that output tokens in 2D
        grid order, ``get_mrope_input_positions`` returns positions for the
        full sequence (prefill + decode).  The upstream runner only uses the
        prefill portion and falls back to linear increments for decode.  This
        method patches the decode slice with the correct pre-computed values.
        """
        from vllm.utils import length_from_prompt_token_ids_or_embeds

        mrope_pos_ptr = 0
        for index, req_id in enumerate(self.input_batch.req_ids):
            req = self.requests[req_id]
            assert req.mrope_positions is not None

            num_computed_tokens = self.input_batch.num_computed_tokens_cpu[index]
            num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_prompt_tokens = length_from_prompt_token_ids_or_embeds(req.prompt_token_ids, req.prompt_embeds)

            if num_computed_tokens + num_scheduled_tokens > num_prompt_tokens:
                prompt_part_len = max(0, num_prompt_tokens - num_computed_tokens)
                completion_part_len = max(0, num_scheduled_tokens - prompt_part_len)
            else:
                prompt_part_len = num_scheduled_tokens
                completion_part_len = 0

            mrope_pos_ptr += prompt_part_len

            if completion_part_len > 0:
                dst_start = mrope_pos_ptr
                decode_start = num_computed_tokens + prompt_part_len
                decode_end = decode_start + completion_part_len
                total_precomputed = req.mrope_positions.shape[1]

                if decode_end <= total_precomputed:
                    # Overwrite the linear positions written by upstream with
                    # the correct pre-computed 2D spatial positions.
                    self.mrope_positions.cpu[:, dst_start : dst_start + completion_part_len] = req.mrope_positions[
                        :, decode_start:decode_end
                    ]

                mrope_pos_ptr += completion_part_len

    def _update_req_spec_token_ids_compat(
        self,
        req_state: CachedRequestState,
        scheduled_spec_tokens: dict[str, Any],
    ) -> None:
        update_req_spec_token_ids = getattr(
            self.input_batch, "update_req_spec_token_ids", None
        )
        if update_req_spec_token_ids is not None:
            update_req_spec_token_ids(req_state, scheduled_spec_tokens)
            return

        spec_token_ids = (
            scheduled_spec_tokens.get(req_state.req_id, ())
            if scheduled_spec_tokens
            else ()
        )
        if not spec_token_ids:
            return
        req_index = self.input_batch.req_id_to_index.get(req_state.req_id)
        if req_index is None:
            return
        num_spec_tokens = len(spec_token_ids)
        start_index = self.input_batch.num_tokens_no_spec[req_index]
        end_token_index = start_index + num_spec_tokens
        self.input_batch.token_ids_cpu[
            req_index, start_index:end_token_index
        ] = spec_token_ids
        num_tokens = getattr(self.input_batch, "num_tokens", None)
        if num_tokens is not None:
            num_tokens[req_index] += num_spec_tokens

    def _compute_cascade_attn_prefix_lens(
        self,
        num_scheduled_tokens: np.ndarray,
        num_computed_tokens: np.ndarray,
        num_common_prefix_blocks: list[int],
    ) -> list[list[int]] | None:
        compute_prefix_len = getattr(
            super(), "_compute_cascade_attn_prefix_len", None
        )
        kv_cache_group_specs = getattr(self, "kv_cache_group_specs", None)
        attn_groups = getattr(self, "attn_groups", None)
        if (
            compute_prefix_len is None
            or kv_cache_group_specs is None
            or attn_groups is None
        ):
            return None

        prefix_lens: list[list[int]] = []
        for group_idx, kv_cache_spec in enumerate(kv_cache_group_specs):
            attn_metadata_builder = attn_groups[group_idx][0]
            common_blocks = (
                num_common_prefix_blocks[group_idx]
                if isinstance(num_common_prefix_blocks, list)
                and group_idx < len(num_common_prefix_blocks)
                else 0
            )
            common_prefix_len = compute_prefix_len(
                num_scheduled_tokens,
                common_blocks,
                kv_cache_spec,
                attn_metadata_builder,
            )
            prefix_lens.append([common_prefix_len])
        return prefix_lens if any(group[0] for group in prefix_lens) else None

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.model_intermediate_buffer.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        if hasattr(self, "late_interaction_runner"):
            self.late_interaction_runner.on_requests_finished(scheduler_output.finished_req_ids)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Zero GPU memory for freshly allocated cache blocks to prevent
        # stale NaN/data from corrupting attention or SSM computation.
        if hasattr(scheduler_output, "new_block_ids_to_zero") and scheduler_output.new_block_ids_to_zero:
            self._zero_block_ids(scheduler_output.new_block_ids_to_zero)

        # Free the cached encoder outputs.
        if hasattr(scheduler_output, "free_encoder_mm_hashes"):
            for mm_hash in scheduler_output.free_encoder_mm_hashes:
                self.encoder_cache.pop(mm_hash, None)
        else:
            for req_id, input_id in getattr(scheduler_output, "free_encoder_input_ids", []):
                encoder_outputs = self.encoder_cache.get(req_id)
                if encoder_outputs is not None:
                    encoder_outputs.pop(input_id, None)
                    if not encoder_outputs:
                        self.encoder_cache.pop(req_id, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = getattr(scheduler_output.scheduled_cached_reqs, "resumed_req_ids", set())
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint,
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

            mm_kwargs = list(getattr(new_req_data, "mm_kwargs", None) or getattr(new_req_data, "mm_features", None) or [])
            cached_request_fields = set(getattr(CachedRequestState, "__dataclass_fields__", {}).keys())
            cached_request_kwargs = {
                "req_id": req_id,
                "prompt_token_ids": new_req_data.prompt_token_ids,
                "prompt_embeds": getattr(new_req_data, "prompt_embeds", None),
                "mm_features": mm_kwargs,
                "mm_kwargs": mm_kwargs,
                "mm_positions": list(getattr(new_req_data, "mm_positions", None) or []),
                "sampling_params": sampling_params,
                "pooling_params": pooling_params,
                "generator": generator,
                "block_ids": new_req_data.block_ids,
                "num_computed_tokens": new_req_data.num_computed_tokens,
                "output_token_ids": [],
                "lora_request": new_req_data.lora_request,
            }
            req_state = CachedRequestState(
                **{key: value for key, value in cached_request_kwargs.items() if key in cached_request_fields}
            )
            # Keep Omni's historical attribute names available for model code.
            req_state.prompt_embeds = getattr(new_req_data, "prompt_embeds", None)
            req_state.mm_features = mm_kwargs
            req_state.prev_num_draft_len = 0
            self.requests[req_id] = req_state
            if hasattr(self, "late_interaction_runner"):
                self.late_interaction_runner.register_request(req_id, pooling_params)

            # If prompt embeddings are provided, decode and attach to inter_data
            try:
                if getattr(new_req_data, "prompt_embeds", None) is not None:
                    payload = new_req_data.prompt_embeds
                    dtype = getattr(np, payload.dtype)
                    arr = np.frombuffer(payload.data, dtype=dtype)
                    arr = arr.reshape(payload.shape)
                    pe_cpu = torch.from_numpy(arr)
                    setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
                    try:
                        new_req_data.prompt_embeds = pe_cpu  # type: ignore[assignment]
                    except Exception:
                        pass
            except Exception as e:
                logger.error(f"Error decoding prompt embeds: {e}")
            # Decode additional_information payloads (dictionary)
            try:
                if getattr(new_req_data, "additional_information", None) is not None:
                    logger.warning_once(
                        "additional_information on request data is deprecated, use model_intermediate_buffer"
                    )
                    payload_info = new_req_data.additional_information
                    info_dict = deserialize_additional_information(payload_info)
                    if info_dict:
                        self.model_intermediate_buffer[req_id] = info_dict
                        setattr(
                            self.requests[req_id],
                            "additional_information_cpu",
                            info_dict,
                        )
            except Exception as e:
                logger.error(f"Error decoding additional information: {e}")

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            if self.uses_xdrope_dim > 0:
                self._init_xdrope_positions(req_state)

            reqs_to_add.append(self.requests[req_id])

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        # Older Omni runner code eagerly fetched the async speculative decode
        # sampled-token count here. Current vLLM no longer exposes that helper
        # on the base runner, and the value is only needed when an existing
        # request has pending draft tokens.
        needs_valid_sampled_token_count = any(
            getattr(self.requests.get(req_id), "prev_num_draft_len", 0)
            for req_id in req_data.req_ids
        )
        get_valid_sampled_token_count = getattr(
            self, "_get_valid_sampled_token_count", None
        )
        valid_sampled_token_count = (
            get_valid_sampled_token_count()
            if needs_valid_sampled_token_count
            and get_valid_sampled_token_count is not None
            else None
        )

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_req_ids = getattr(req_data, "resumed_req_ids", set())
            resumed_flags = getattr(req_data, "resumed_from_preemption", [])
            resumed_from_preemption = req_id in resumed_req_ids or (
                i < len(resumed_flags) and bool(resumed_flags[i])
            )
            num_output_tokens_data = getattr(req_data, "num_output_tokens", None)
            num_output_tokens = (
                num_output_tokens_data[i]
                if num_output_tokens_data is not None and i < len(num_output_tokens_data)
                else len(req_state.output_token_ids)
            )
            req_index = self.input_batch.req_id_to_index.get(req_id)

            if getattr(req_state, "prev_num_draft_len", 0) and self.use_async_scheduling:
                # prev_num_draft_len is used in async scheduling mode with
                # spec decode. it indicates if need to update num_computed_tokens
                # of the request. for example:
                # first step: num_computed_tokens = 0, spec_tokens = [],
                # prev_num_draft_len = 0.
                # second step: num_computed_tokens = 100(prompt length),
                # spec_tokens = [a,b], prev_num_draft_len = 0.
                # third step: num_computed_tokens = 100 + 2, spec_tokens = [c,d],
                # prev_num_draft_len = 2.
                # num_computed_tokens in first step and second step doesn't contain
                # the spec tokens length, but in third step it contains the
                # spec tokens length. we only need to update num_computed_tokens
                # when prev_num_draft_len > 0.
                if valid_sampled_token_count is None:
                    req_state.prev_num_draft_len = 0
                elif req_index is None:
                    req_state.prev_num_draft_len = 0
                else:
                    assert self.input_batch.prev_req_id_to_index is not None
                    prev_req_index = self.input_batch.prev_req_id_to_index[req_id]
                    num_accepted = valid_sampled_token_count[prev_req_index] - 1
                    num_rejected = req_state.prev_num_draft_len - num_accepted
                    num_computed_tokens -= num_rejected
                    req_state.output_token_ids.extend([-1] * num_accepted)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                if not req_data.new_token_ids:
                    new_token_ids: list[int] = []
                else:
                    new_token_ids = req_data.new_token_ids[i]
                    num_new_tokens = num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                    if num_new_tokens == 1:
                        req_state.output_token_ids.append(new_token_ids[-1])
                    elif num_new_tokens > 0:
                        req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = self.input_batch.num_prompt_tokens[req_index] + num_output_tokens
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.

                if self.use_async_scheduling and num_output_tokens > 0:
                    # We must recover the output token ids for resumed requests in the
                    # async scheduling case, so that correct input_ids are obtained.
                    resumed_token_ids = req_data.all_token_ids[req_id]
                    req_state.output_token_ids = resumed_token_ids[-num_output_tokens:]

                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[req_index, start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self._update_req_spec_token_ids_compat(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self._update_req_spec_token_ids_compat(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    @torch.inference_mode()
    def extract_multimodal_outputs(self, hidden_states: torch.Tensor | list[torch.Tensor] | OmniOutput) -> dict:
        if (
            hasattr(self.model, "have_multimodal_outputs")
            and self.model.have_multimodal_outputs
            and isinstance(hidden_states, OmniOutput)
        ):
            text_hidden_states = hidden_states.text_hidden_states
            multimodal_outputs = hidden_states.multimodal_outputs

        elif isinstance(hidden_states, torch.Tensor):
            text_hidden_states = hidden_states
            multimodal_outputs = {}
        elif isinstance(hidden_states, list) or isinstance(hidden_states, tuple):
            text_hidden_states = hidden_states[0]
            multimodal_outputs = {}
        else:
            raise ValueError(f"Invalid hidden states type: {type(hidden_states)}")
        return text_hidden_states, multimodal_outputs

    @torch.inference_mode()
    def _dummy_run(
        self,
        num_tokens: int,
        cudagraph_runtime_mode: CUDAGraphMode | None = None,
        force_attention: bool = False,
        uniform_decode: bool = False,
        allow_microbatching: bool = True,
        skip_eplb: bool = False,
        is_profile: bool = False,
        create_mixed_batch: bool = False,
        remove_lora: bool = True,
        is_graph_capturing: bool = False,
        num_active_loras: int = 0,
        profile_seq_lens: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run a dummy forward pass to warm up/profile run or capture the
        CUDA graph for the model.

        Args:
            num_tokens: Number of tokens to run the dummy forward pass.
            cudagraph_runtime_mode: used to control the behavior.
                - if not set will determine the cudagraph mode based on using
                    the self.cudagraph_dispatcher.
                - CUDAGraphMode.NONE: No cudagraph, for warm up and profile run
                - CUDAGraphMode.PIECEWISE: Piecewise cudagraph.
                - CUDAGraphMode.FULL: Full cudagraph, attention metadata is
                    needed.
            force_attention: If True, always create attention metadata. Used to
                warm up attention backend when mode is NONE.
            uniform_decode: If True, the batch is a uniform decode batch.
            skip_eplb: If True, skip EPLB state update.
            is_profile: If True, this is a profile run.
            create_mixed_batch: If True, create a mixed batch with both decode
                (1 token) and prefill (multiple tokens) requests.
            remove_lora: If False, dummy LoRAs are not destroyed after the run
            num_active_loras: Number of distinct active LoRAs to capture for.
                LoRA is activated when num_active_loras > 0.
            profile_seq_lens: If provided, use this value for seq_lens instead
                of max_query_len. Used to profile attention workspace that
                scales with context length.
        """
        mm_config = self.vllm_config.model_config.multimodal_config
        if mm_config and mm_config.mm_encoder_only:
            # The current dummy run only covers LM execution, so we can skip it.
            # mm encoder dummy run may need to add in the future.
            return torch.tensor([]), torch.tensor([])

        assert cudagraph_runtime_mode is None or cudagraph_runtime_mode.valid_runtime_modes()

        # If cudagraph_mode.decode_mode() == FULL and
        # cudagraph_mode.separate_routine(). This means that we are using
        # different graphs and/or modes for mixed prefill-decode batches vs.
        # uniform decode batches. A uniform decode batch means that all
        # requests have identical query length, except a potential virtual
        # request (shorter) in the batch account for padding.
        # Uniform decode batch could either be common pure decode, where
        # max_query_len == 1, or speculative decode, where
        # max_query_len == 1 + num_spec_decode_tokens.

        # When setting max_query_len = 1, we switch to and capture the optimized
        # routine of FA2 for pure decode, i.e., Flashdecode + an optimization
        # for GQA/MQA.
        max_query_len = self.uniform_decode_query_len if uniform_decode else num_tokens

        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.max_num_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        if create_mixed_batch:
            assert not uniform_decode
            # Create mixed batch:
            # first half decode tokens, second half one prefill
            num_decode_tokens = min(max_num_reqs - 1, num_tokens // 2)
            num_prefill_tokens = num_tokens - num_decode_tokens
            num_reqs = num_decode_tokens + 1

            # Create decode requests (1 token each) followed by prefill request
            num_scheduled_tokens_list = [1] * num_decode_tokens + [num_prefill_tokens]
            # Note: Overriding max_query_len to be the prefill tokens
            max_query_len = num_prefill_tokens
        elif uniform_decode:
            assert not create_mixed_batch
            num_reqs = min(max_num_reqs, cdiv(num_tokens, max_query_len))
            num_scheduled_tokens_list = [max_query_len] * num_reqs
            if num_tokens % max_query_len != 0:
                num_scheduled_tokens_list[-1] = num_tokens % max_query_len
        else:
            num_reqs = min(num_tokens, max_num_reqs)
            min_tokens_per_req = num_tokens // num_reqs
            num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
            num_scheduled_tokens_list[-1] += num_tokens % num_reqs

        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)

        _cudagraph_mode, batch_desc, should_ubatch, num_tokens_across_dp, _ = (
            self._determine_batch_execution_and_padding(
                num_tokens=num_tokens_unpadded,
                num_reqs=num_reqs,
                num_scheduled_tokens_np=num_scheduled_tokens,
                max_num_scheduled_tokens=max_query_len,
                use_cascade_attn=False,
                allow_microbatching=allow_microbatching,
                force_eager=is_profile or (cudagraph_runtime_mode == CUDAGraphMode.NONE),
                # `force_uniform_decode` is used for cudagraph capture; because for
                # capturing mixed prefill-decode batches, we sometimes use
                # num_tokens == num_reqs which looks like a uniform decode batch to the
                # dispatcher; but we actually want to capture a piecewise cudagraph
                force_uniform_decode=uniform_decode,
                # `force_has_lora` is used for cudagraph capture; because LoRA is
                # activated later in the context manager, but we need to know the
                # LoRA state when determining the batch descriptor for capture
                force_has_lora=num_active_loras > 0,
                # `force_num_active_loras` is used for cudagraph capture; because we
                # need to capture graphs for specific num_active_loras counts
                force_num_active_loras=num_active_loras,
            )
        )

        if cudagraph_runtime_mode is None:
            cudagraph_runtime_mode = _cudagraph_mode
        else:
            assert cudagraph_runtime_mode == _cudagraph_mode, (
                f"Cudagraph runtime mode mismatch in dummy_run. "
                f"Expected {_cudagraph_mode}, but got {cudagraph_runtime_mode}."
            )

        num_tokens_padded = batch_desc.num_tokens
        num_reqs_padded = batch_desc.num_reqs if batch_desc.num_reqs is not None else num_reqs
        ubatch_slices, ubatch_slices_padded = maybe_create_ubatch_slices(
            should_ubatch,
            num_scheduled_tokens,
            num_tokens_padded,
            num_reqs_padded,
            getattr(self.vllm_config.parallel_config, "num_ubatches", 1),
        )
        logger.debug(
            "ubatch_slices: %s, ubatch_slices_padded: %s",
            ubatch_slices,
            ubatch_slices_padded,
        )

        attn_metadata: PerLayerAttnMetadata | None = None

        if (
            is_profile
            and getattr(getattr(self, "model", None), "has_preprocess", False)
            and getattr(self, "kv_cache_config", None) is not None
        ):
            force_attention = True

        slot_mappings_by_group, slot_mappings = self._get_slot_mappings(
            num_tokens_padded=num_tokens,
            num_reqs_padded=num_reqs_padded,
            num_tokens_unpadded=num_tokens_unpadded,
            ubatch_slices=ubatch_slices_padded,
        )
        if slot_mappings_by_group is not None:
            for slot_mapping in slot_mappings_by_group.values():
                slot_mapping.fill_(-1)

        with self.synchronize_input_prep():
            # If force_attention is True, we always capture attention.
            # Otherwise, it only happens for cudagraph_runtime_mode=FULL.
            if force_attention or cudagraph_runtime_mode == CUDAGraphMode.FULL:
                if profile_seq_lens is not None:
                    seq_lens = profile_seq_lens  # type: ignore[assignment]
                elif create_mixed_batch:
                    # In the mixed batch mode (used for FI warmup), we use
                    # shorter sequence lengths to run faster.
                    # TODO(luka) better system for describing dummy batches
                    seq_lens = [1] * num_decode_tokens + [num_prefill_tokens + 1]  # type: ignore[assignment]
                else:
                    seq_lens = max_query_len  # type: ignore[assignment]
                self.seq_lens.np[:num_reqs] = seq_lens
                self.seq_lens.np[num_reqs:] = 0
                self.seq_lens.copy_to_gpu()

                cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
                self.query_start_loc.np[1 : num_reqs + 1] = cum_num_tokens
                self.query_start_loc.copy_to_gpu()

                pad_attn = cudagraph_runtime_mode == CUDAGraphMode.FULL
                attn_metadata, _ = self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded if pad_attn else None,
                    num_reqs=num_reqs_padded,
                    max_query_len=max_query_len,
                    ubatch_slices=(ubatch_slices_padded if pad_attn else ubatch_slices),
                    for_cudagraph_capture=is_graph_capturing,
                    slot_mappings=slot_mappings_by_group,
                    use_spec_decode=self.speculative_config is not None,
                )

        with self.maybe_dummy_run_with_lora(
            self.lora_config,
            num_scheduled_tokens,
            num_sampled_tokens,
            remove_lora,
            num_active_loras,
        ):
            # Make sure padding doesn't exceed max_num_tokens
            assert num_tokens_padded <= self.max_num_tokens
            model_kwargs = self._init_model_kwargs()
            if self.supports_mm_inputs and not self.model_config.is_encoder_decoder:
                input_ids, inputs_embeds = self._prepare_mm_inputs(num_tokens_padded)

                model_kwargs = {
                    **model_kwargs,
                    **self._dummy_mm_kwargs(num_reqs),
                }
            elif self.enable_prompt_embeds:
                input_ids = None
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
                model_kwargs = self._init_model_kwargs()
            elif getattr(getattr(self, "model", None), "has_preprocess", False):
                # Capture CUDA graph with inputs_embeds path so replay reads
                # from the same buffer that _preprocess writes into.
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = self.inputs_embeds.gpu[:num_tokens_padded]
            else:
                input_ids = self.input_ids.gpu[:num_tokens_padded]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions.gpu[:, :num_tokens_padded]
            elif self.uses_xdrope_dim > 0:
                positions = self.xdrope_positions.gpu[:, :num_tokens_padded]
            else:
                positions = self.positions.gpu[:num_tokens_padded]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = self.model.make_empty_intermediate_tensors(
                        batch_size=self.max_num_tokens,
                        dtype=self.model_config.dtype,
                        device=self.device,
                    )

                intermediate_tensors = self.sync_and_slice_intermediate_tensors(num_tokens_padded, None, False)

            if ubatch_slices_padded is not None:
                # Adjust values to reflect a single ubatch.
                # TODO(sage,lucas): this is cruft that should be addressed in
                #  the padding refactor.
                num_tokens_padded = ubatch_slices_padded[0].num_tokens
                if num_tokens_across_dp is not None:
                    num_tokens_across_dp[:] = num_tokens_padded

            with (
                self.maybe_randomize_inputs(input_ids, inputs_embeds),
                set_forward_context(
                    attn_metadata,
                    self.vllm_config,
                    num_tokens=num_tokens_padded,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    batch_descriptor=batch_desc,
                    ubatch_slices=ubatch_slices_padded,
                    slot_mapping=slot_mappings,
                ),
            ):
                if getattr(self.model, "talker", None) is not None and self.has_talker_mtp:
                    num_tokens_padded_talker_mtp = num_tokens_padded
                    if num_tokens_padded_talker_mtp == self.max_num_tokens:
                        num_tokens_padded_talker_mtp = self.talker_mtp_input_ids.gpu.shape[0]
                    outputs = self.talker_mtp(
                        self.talker_mtp_input_ids.gpu[:num_tokens_padded_talker_mtp],
                        self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded_talker_mtp],
                        self.last_talker_hidden.gpu[:num_tokens_padded_talker_mtp],
                        self.text_step.gpu[:num_tokens_padded_talker_mtp],
                    )
                    self.compilation_config.cache_dir = None
                outputs = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs,
                )

            if self.use_aux_hidden_state_outputs:
                hidden_states, _ = outputs
            else:
                hidden_states = outputs
            hidden_states, multimodal_outputs = self.extract_multimodal_outputs(hidden_states)
            if self.speculative_config and (
                self.speculative_config.use_eagle()
                or self.speculative_config.uses_draft_model()
                or self.speculative_config.uses_extract_hidden_states()
            ):
                assert isinstance(
                    self.drafter,
                    EagleProposer | DraftModelProposer | ExtractHiddenStatesProposer,
                )
                assert self.speculative_config is not None
                # Eagle currently only supports PIECEWISE cudagraphs.
                # Therefore only use cudagraphs if the main model uses PIECEWISE
                # NOTE(lucas): this is a hack, need to clean up.
                use_cudagraphs = (
                    (is_graph_capturing and cudagraph_runtime_mode == CUDAGraphMode.PIECEWISE)
                    or (not is_graph_capturing and cudagraph_runtime_mode != CUDAGraphMode.NONE)
                ) and not self.speculative_config.enforce_eager

                # Note(gnovack) - We need to disable cudagraphs for one of the two
                # lora cases when cudagraph_specialize_lora is enabled. This is a
                # short term mitigation for issue mentioned in
                # https://github.com/vllm-project/vllm/issues/28334
                if self.compilation_config.cudagraph_specialize_lora and num_active_loras > 0:
                    use_cudagraphs = False

                self.drafter.dummy_run(
                    num_tokens,
                    use_cudagraphs=use_cudagraphs,
                    is_graph_capturing=is_graph_capturing,
                    slot_mappings=slot_mappings,
                )

        # We register layerwise NVTX hooks here after the first dynamo tracing is
        # done to avoid nvtx operations in hook functions being traced by
        # torch dynamo and causing graph breaks.
        # Note that for DYNAMO_ONCE and VLLM_COMPILE mode,
        # compiled model's dynamo tracing is only done once and the compiled model's
        # __call__ function is replaced by calling the compiled function.
        # So it's safe to register hooks here. Hooks will be registered to
        # both compiled and uncompiled models but they will never
        # be called on the compiled model execution path.
        register_layerwise_nvtx_hooks = getattr(self, "_register_layerwise_nvtx_hooks", None)
        if callable(register_layerwise_nvtx_hooks):
            register_layerwise_nvtx_hooks()

        # This is necessary to avoid blocking DP.
        # For dummy runs, we typically skip EPLB since we don't have any real
        # requests to process.
        # However, in DP settings, there may be cases when some DP ranks do
        # not have any requests to process, so they're executing dummy batches.
        # In such cases, we still have to trigger EPLB to make sure
        # ranks execute the rearrangement in synchronization.
        if not skip_eplb:
            self.eplb_step(is_dummy=True, is_profile=is_profile)

        logit_indices = np.cumsum(num_scheduled_tokens) - 1
        logit_indices_device = torch.from_numpy(logit_indices).to(self.device, non_blocking=True)
        return hidden_states, hidden_states[logit_indices_device]

    # ------------------------------------------------------------------
    # Payload decoding helpers (torch.Tensor passthrough + legacy
    # PromptEmbedsPayload / AdditionalInformationPayload support)
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_prompt_embeds_cpu(
        pe: "torch.Tensor | object | None",
    ) -> torch.Tensor | None:
        """Convert *prompt_embeds* to a contiguous CPU tensor.

        Accepts:
        - ``torch.Tensor`` – moved to CPU as-is (the normal path after
          upstream added ``prompt_embeds`` to ``EngineCoreRequest``).
        - Legacy ``PromptEmbedsPayload`` (or any duck-typed object with
          ``.data``, ``.shape``, ``.dtype``) – decoded via numpy.
        - ``None`` – returns ``None``.
        """
        if pe is None:
            return None
        try:
            if isinstance(pe, torch.Tensor):
                return pe.detach().cpu().contiguous()
            data = getattr(pe, "data", None)
            shape = getattr(pe, "shape", None)
            if data is not None and shape is not None:
                dt = np.dtype(getattr(pe, "dtype", "float32"))
                arr = np.frombuffer(data, dtype=dt).reshape(shape)
                return torch.from_numpy(arr.copy())
        except Exception:
            logger.exception("Failed to decode prompt_embeds payload")
        return None

    def _decode_and_store_request_payloads(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Decode per-request prompt_embeds and additional_information for
        newly scheduled requests and store them on CPU in the request state.
        """
        new_reqs = getattr(scheduler_output, "scheduled_new_reqs", [])
        if not new_reqs:
            return
        for nr in new_reqs:
            req_id = getattr(nr, "req_id", None) or getattr(nr, "request_id", None)
            if req_id is None or req_id not in self.requests:
                continue
            pe_cpu = self._resolve_prompt_embeds_cpu(getattr(nr, "prompt_embeds", None))
            if pe_cpu is not None:
                setattr(self.requests[req_id], "prompt_embeds_cpu", pe_cpu)
            info_payload = getattr(nr, "additional_information", None)
            if info_payload is not None:
                logger.warning_once(
                    "additional_information on request data is deprecated, use model_intermediate_buffer"
                )
            info_dict = deserialize_additional_information(info_payload)
            if info_dict:
                self.model_intermediate_buffer[req_id] = info_dict
                setattr(self.requests[req_id], "additional_information_cpu", info_dict)

    def _gather_runtime_additional_information(self) -> list[dict]:
        """Gather per-request model_intermediate_buffer in batch order."""
        per_req_runtime_info = []
        for req_id in self.input_batch.req_ids:
            req_state = self.requests.get(req_id)
            # MammothModa2 AR grid constraint: the model must emit a special
            # end-of-line (EOL) token at the end of each image row.  To determine
            # whether the current decoding step falls on a row boundary, the
            # constraint logic (see MammothModa2ARForConditionalGeneration.
            # _apply_t2i_token_constraints) computes:
            #   column_id = generated_len % (ar_width + 1)
            # and forces the EOL token when column_id == ar_width.
            generated_len = len(req_state.output_token_ids) if req_state is not None else 0
            info = self.model_intermediate_buffer.get(req_id, {})
            if info:
                info["generated_len"] = generated_len
                per_req_runtime_info.append(info)
                if "thinker_reply_part_per_request" in info:
                    q = info["thinker_reply_part_per_request"]
                    if hasattr(q, "shape"):
                        logger.debug(f"[OMNI] req={req_id} has thinker_reply_part_per_request queue shape: {q.shape}")
            else:
                per_req_runtime_info.append({})
        return per_req_runtime_info

    def _compute_request_token_spans(self, num_scheduled_tokens_np) -> list[tuple[int, int]]:
        """Compute (start, end) token spans for each request within the flattened step sequence."""
        req_token_spans: list[tuple[int, int]] = []
        for req_index in range(len(self.input_batch.req_ids)):
            start_offset = int(self.query_start_loc.cpu[req_index])
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            req_token_spans.append((start_offset, start_offset + sched_tokens))
        return req_token_spans

    def _build_model_kwargs_extra(self) -> dict:
        """Build extra keyword arguments passed to the model for this step."""
        model_kwargs_extra: dict[str, object] = {}
        try:
            buffer_map = self._gather_runtime_additional_information()
            model_kwargs_extra["model_intermediate_buffer"] = buffer_map
            # Backward compatible: also emit old name
            model_kwargs_extra["runtime_additional_information"] = buffer_map
        except Exception as e:
            logger.error(f"[OMNI DEBUG] Error building model_kwargs_extra: {e}")
            import traceback

            traceback.print_exc()
        return model_kwargs_extra

    def _process_additional_information_updates(
        self,
        hidden_states: torch.Tensor,
        multimodal_outputs: object,
        num_scheduled_tokens_np: np.ndarray,
        scheduler_output: "SchedulerOutput",
    ) -> None:
        """Process model-provided per-request updates and merge into model_intermediate_buffer."""
        try:
            # execute the custom postprocess function
            # TODO(Peiqi): do we have a more elegant way to do this?
            if hasattr(self.model, "has_postprocess") and self.model.has_postprocess:
                for req_index, req_id in enumerate(self.input_batch.req_ids):
                    req_infos = self.model_intermediate_buffer.get(req_id, {})
                    start_offset = int(self.query_start_loc.cpu[req_index])
                    sched_tokens = int(num_scheduled_tokens_np[req_index])
                    s, e = start_offset, start_offset + sched_tokens
                    # only consider to store data into update dict.
                    hidden_states_slice = hidden_states[s:e]
                    update_dict = self.model.postprocess(
                        hidden_states_slice, multimodal_outputs=multimodal_outputs, **req_infos
                    )
                    self._update_intermediate_buffer(req_id, update_dict)
        except Exception as e:
            logger.error(
                f"Error merging for requests:{self.input_batch.req_ids} "
                f"additional information update: {e}, with the multimodal_outputs "
                f"as {multimodal_outputs}"
            )
            import traceback

            traceback.print_exc()

    def _collect_additional_information_for_prefill(
        self,
        num_scheduled_tokens_np: np.ndarray,
    ) -> dict[str, dict]:
        """Overlay per-request prompt_embeds for the prefill portion and collect
        additional_information slices for this step. Returns a map req_id -> dict."""
        for req_index, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            pe_cpu = getattr(req_state, "prompt_embeds_cpu", None)
            num_computed_tokens = int(self.input_batch.num_computed_tokens_cpu[req_index])
            prompt_len = len(req_state.prompt_token_ids)
            prompt_remaining = max(0, prompt_len - num_computed_tokens)
            sched_tokens = int(num_scheduled_tokens_np[req_index])
            overlay_len = min(sched_tokens, prompt_remaining)
            if overlay_len <= 0:
                continue
            if overlay_len > 0 and pe_cpu is not None:
                src = pe_cpu[num_computed_tokens : num_computed_tokens + overlay_len].to(
                    dtype=self.dtype, device=self.device, non_blocking=True
                )
                start_offset = int(self.query_start_loc.cpu[req_index])
                self.inputs_embeds[start_offset : start_offset + overlay_len].copy_(src)

    def _update_additional_information(self, scheduler_output: "SchedulerOutput") -> None:
        for new_req in scheduler_output.scheduled_new_reqs:
            payload_info = getattr(new_req, "additional_information", None)
            if isinstance(payload_info, dict):
                logger.warning_once(
                    "additional_information on request data is deprecated, use model_intermediate_buffer"
                )
                self._update_intermediate_buffer(new_req.req_id, payload_info)

        if hasattr(scheduler_output.scheduled_cached_reqs, "additional_information"):
            logger.warning_once(
                "additional_information on scheduled_cached_reqs is deprecated, use model_intermediate_buffer"
            )
            cached_infos = getattr(scheduler_output.scheduled_cached_reqs, "additional_information", {})
            if isinstance(cached_infos, dict):
                for req_id, req_infos in cached_infos.items():
                    self._update_intermediate_buffer(req_id, req_infos)

    def _maybe_attach_mimo_audio_req_infos(
        self,
        req_state: CachedRequestState | None,
        req_infos: dict | None,
        req_id: str,
    ) -> dict | None:
        """Attach MiMoAudio-specific fields into req_infos if applicable.

        This helper is intentionally small and self-contained so that it can be
        unit-tested to prevent regressions when updating MiMoAudio handling.
        """
        if req_state is None or self.model.__class__.__name__ != "MiMoAudioForConditionalGeneration":
            return req_infos

        # Always operate on a dict copy to avoid mutating shared instances.
        req_infos = dict(req_infos) if isinstance(req_infos, dict) else {}
        mm_features = getattr(req_state, "mm_features", None)
        if mm_features and (not req_infos.get("mm_features")):
            req_infos["mm_features"] = mm_features
        req_infos["req_id"] = req_id

        return req_infos

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ):
        """Align with v0.14.0 preprocess and omni's additional information handling."""
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        is_first_rank = get_pp_group().is_first_rank
        is_encoder_decoder = self.model_config.is_encoder_decoder

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        ec_connector_output = None

        if self.supports_mm_inputs and is_first_rank and not is_encoder_decoder:
            # Run the multimodal encoder if any.
            with self.maybe_get_ec_connector_output(
                scheduler_output,
                encoder_cache=self.encoder_cache,
            ) as ec_connector_output:
                self._execute_mm_encoder(scheduler_output)
                mm_embeds, is_mm_embed = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.embed_input_ids(
                self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(inputs_embeds_scheduled)

            input_ids, inputs_embeds = self._prepare_mm_inputs(num_input_tokens)
            model_kwargs = {
                **self._init_model_kwargs(),
                **self._extract_mm_kwargs(scheduler_output),
            }
        elif self.enable_prompt_embeds and is_first_rank:
            # Get the input embeddings for the tokens that are not input embeds,
            # then put them into the appropriate positions.
            # TODO(qthequartermasterman): Since even when prompt embeds are
            # enabled, (a) not all requests will use prompt embeds, and (b)
            # after the initial prompt is processed, the rest of the generated
            # tokens will be token ids, it is not desirable to have the
            # embedding layer outside of the CUDA graph all the time. The v0
            # engine avoids this by "double compiling" the CUDA graph, once
            # with input_ids and again with inputs_embeds, for all num_tokens.
            # If a batch only has token ids, then including the embedding layer
            # in the CUDA graph will be more performant (like in the else case
            # below).
            token_ids_idx = self.is_token_ids.gpu[:num_scheduled_tokens].nonzero(as_tuple=False).squeeze(1)
            # Some tokens ids may need to become embeds
            if token_ids_idx.numel() > 0:
                token_ids = self.input_ids.gpu[token_ids_idx]
                tokens_to_embeds = self.model.embed_input_ids(input_ids=token_ids)
                self.inputs_embeds.gpu[token_ids_idx] = tokens_to_embeds

            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
            input_ids = self.input_ids.gpu[:num_input_tokens]
        elif getattr(self.model, "has_preprocess", False):
            # Use pre-allocated buffer for CUDA graph compatibility.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = self._init_model_kwargs()
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs()

        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        elif self.uses_xdrope_dim > 0:
            positions = self.xdrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None
            intermediate_tensors = self.sync_and_slice_intermediate_tensors(
                num_input_tokens, intermediate_tensors, True
            )

        if is_encoder_decoder and scheduler_output.scheduled_encoder_inputs:
            # Run the encoder, just like we do with other multimodal inputs.
            # For an encoder-decoder model, our processing here is a bit
            # simpler, because the outputs are just passed to the decoder.
            # We are not doing any prompt replacement. We also will only
            # ever have a single encoder input.
            encoder_outputs = self._execute_mm_encoder(scheduler_output)
            model_kwargs.update({"encoder_outputs": encoder_outputs})

        req_ids = self.input_batch.req_ids
        num_scheduled_tokens_np = np.array(
            [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids],
            dtype=np.int32,
        )
        self._omni_num_scheduled_tokens_np = num_scheduled_tokens_np

        # Note: only prefill need collect additional_information for now.
        # Decode don't need per_req_additional_information anymore.
        if inputs_embeds is not None:
            # Prefill: overlay prompt_embeds and collect additional_information
            self._collect_additional_information_for_prefill(num_scheduled_tokens_np)

        # Keep per-request additional_information in sync for both new and
        # cached requests. This is required for stages without preprocess
        # (e.g., code2wav) so runtime_additional_information can be refreshed
        # from scheduler cached infos on every step.
        if hasattr(self.model, "has_preprocess") or hasattr(self.model, "enable_update_additional_information"):
            if self.vllm_config.model_config.async_chunk:
                self._update_additional_information(scheduler_output)

        if hasattr(self.model, "has_preprocess") and self.model.has_preprocess:
            # Overlay custom prompt_embeds per request for the prompt portion;
            # collect additional_information (tensor/list) for prefill portion only
            decode_req_ids = []
            for req_index, req_id in enumerate(self.input_batch.req_ids):
                req_infos = self.model_intermediate_buffer.get(req_id, {})

                # mimo-audio check
                req_state = self.requests.get(req_id)
                req_infos = self._maybe_attach_mimo_audio_req_infos(req_state, req_infos, req_id)

                start_offset = int(self.query_start_loc.cpu[req_index])
                sched_tokens = int(num_scheduled_tokens_np[req_index])
                s, e = start_offset, start_offset + sched_tokens
                span_len = int(e) - int(s)

                # call the custom process function
                embed_slice = inputs_embeds[s:e] if inputs_embeds is not None else None
                req_input_ids, req_embeds, update_dict = self.model.preprocess(
                    input_ids=input_ids[s:e], input_embeds=embed_slice, **req_infos
                )
                if inputs_embeds is None:
                    inputs_embeds = torch.empty(
                        (input_ids.shape[0], req_embeds.shape[-1]),
                        device=req_embeds.device,
                        dtype=req_embeds.dtype,
                    )

                if self.has_talker_mtp and span_len == 1:
                    last_talker_hidden, text_step = update_dict.pop("mtp_inputs")
                    decode_slice = slice(len(decode_req_ids), len(decode_req_ids) + 1)
                    self.talker_mtp_input_ids.gpu[decode_slice].copy_(req_input_ids)
                    self.talker_mtp_inputs_embeds.gpu[decode_slice].copy_(req_embeds)
                    self.last_talker_hidden.gpu[decode_slice].copy_(last_talker_hidden)
                    self.text_step.gpu[decode_slice].copy_(text_step)
                    decode_req_ids.append(req_id)

                # TODO(Peiqi): the merge stage could move out from the critical path
                self._merge_additional_information_update(req_id, update_dict)

                # update the inputs_embeds and input_ids
                seg_len = min(span_len, req_embeds.shape[0])
                inputs_embeds[s : s + seg_len] = req_embeds[:seg_len]
                if isinstance(req_input_ids, torch.Tensor) and req_input_ids.numel() == seg_len:
                    input_ids[s : s + seg_len] = req_input_ids

            # run talker mtp decode
            if self.has_talker_mtp:
                self._talker_mtp_forward(decode_req_ids, inputs_embeds)

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
            model_kwargs,
            ec_connector_output,
        )

    def _talker_mtp_forward(self, decode_req_ids: list[str], inputs_embeds: torch.Tensor) -> None:
        decode_batch_size = len(decode_req_ids)
        if decode_batch_size == 0:
            return
        _cudagraph_mode, batch_desc, _, _, _ = self._determine_batch_execution_and_padding(
            num_tokens=decode_batch_size,
            num_reqs=decode_batch_size,
            num_scheduled_tokens_np=np.ones(decode_batch_size, dtype=np.int32),
            max_num_scheduled_tokens=1,
            use_cascade_attn=False,
        )
        # Force eager for unwrapped code predictors (AR loops / multinomial).
        # When talker_mtp is not a CUDAGraphWrapper, it manages its own CUDA
        # graphs internally (code_predictor has its own bucket sizes).
        if not isinstance(self.talker_mtp, CUDAGraphWrapper):
            _cudagraph_mode = CUDAGraphMode.NONE
            num_tokens_padded = decode_batch_size
        else:
            num_tokens_padded = batch_desc.num_tokens
        req_input_ids = self.talker_mtp_input_ids.gpu[:num_tokens_padded]
        req_embeds = self.talker_mtp_inputs_embeds.gpu[:num_tokens_padded]
        last_talker_hidden = self.last_talker_hidden.gpu[:num_tokens_padded]
        text_step = self.text_step.gpu[:num_tokens_padded]
        with set_forward_context(
            None, self.vllm_config, cudagraph_runtime_mode=_cudagraph_mode, batch_descriptor=batch_desc
        ):
            req_embeds, code_predictor_codes = self.talker_mtp(req_input_ids, req_embeds, last_talker_hidden, text_step)
        # code_predictor_codes stays on GPU here; _update_intermediate_buffer
        # keeps it device-resident when the key is in gpu_resident_buffer_keys.
        # D2H is deferred to sample_tokens where hidden_states.to("cpu") already
        # syncs the stream, avoiding a per-step cudaStreamSynchronize.
        out_key = getattr(self.model, "talker_mtp_output_key", "code_predictor_codes")
        for idx, req_id in enumerate(decode_req_ids):
            req_index = self.input_batch.req_ids.index(req_id)
            start_offset = int(self.query_start_loc.cpu[req_index])
            inputs_embeds[start_offset : start_offset + 1] = req_embeds[idx : idx + 1]
            update_dict = {out_key: code_predictor_codes[idx : idx + 1]}
            self._merge_additional_information_update(req_id, update_dict)

    def _model_forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs: dict[str, Any],
    ):
        """Inject omni-specific kwargs into forward and cache model output"""
        model_kwargs_extra = self._build_model_kwargs_extra()

        base_model_forward = getattr(super(), "_model_forward", None)
        if base_model_forward is not None:
            model_output = base_model_forward(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                **model_kwargs_extra,
            )
        else:
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
                **model_kwargs_extra,
            )
        if not isinstance(model_output, OmniOutput) and hasattr(self.model, "make_omni_output"):
            model_output = self.model.make_omni_output(model_output, **model_kwargs, **model_kwargs_extra)
        # Cache model output so later sample_tokens can consume multimodal results.
        self._omni_last_model_output = model_output
        return model_output

    def _update_intermediate_buffer(self, req_id: str, upd: dict) -> None:
        if not isinstance(upd, dict) or not upd:
            return
        req_state = self.requests.get(req_id)
        if req_state is None:
            return
        # Check if the model declares keys that should stay on GPU
        gpu_keys: set[str] = set()
        if hasattr(self, "model") and hasattr(self.model, "gpu_resident_buffer_keys"):
            gpu_keys = self.model.gpu_resident_buffer_keys
        existing = self.model_intermediate_buffer.setdefault(req_id, {})
        for k, v in upd.items():
            if isinstance(v, torch.Tensor):
                if k in gpu_keys:
                    existing[k] = v.detach().clone()
                else:
                    existing[k] = v.detach().to("cpu").contiguous()
            elif isinstance(v, list):
                existing[k] = [
                    (item.detach().to("cpu").contiguous() if isinstance(item, torch.Tensor) else item) for item in v
                ]
            else:
                existing[k] = v
        # Backward compatible: mirror to old setattr location
        setattr(req_state, "additional_information_cpu", existing)

    def _merge_additional_information_update(self, req_id, upd):
        logger.warning_once("_merge_additional_information_update is deprecated, use _update_intermediate_buffer")
        return self._update_intermediate_buffer(req_id, upd)
