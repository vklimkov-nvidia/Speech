import gc
import os

import torch
from vllm.logger import init_logger
from vllm.platforms import current_platform
try:
    from vllm.tracing import instrument
except ImportError:
    def instrument(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
from vllm.utils import GiB_bytes, MemorySnapshot
from vllm.model_executor.utils import set_random_seed
from vllm.v1.utils import report_usage_stats
from vllm.v1.worker.gpu_worker import init_worker_distributed_environment
try:
    from vllm.v1.worker.workspace import init_workspace_manager
except ImportError:
    def init_workspace_manager(*args, **kwargs):
        return None

from vllm_omni.worker.base import OmniGPUWorkerBase
from vllm_omni.worker.gpu_generation_model_runner import GPUGenerationModelRunner
from vllm_omni.worker.mixins import OmniWorkerMixin

logger = init_logger(__name__)


def format_gib(value: int | float) -> str:
    return f"{value / GiB_bytes:.2f}"


def _empty_accelerator_cache() -> None:
    accelerator_empty_cache = getattr(getattr(torch, "accelerator", None), "empty_cache", None)
    if callable(accelerator_empty_cache):
        accelerator_empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _enable_dbo(parallel_config) -> bool:
    return bool(getattr(parallel_config, "enable_dbo", False))


def _data_parallel_local_rank(parallel_config) -> int:
    value = getattr(parallel_config, "data_parallel_rank_local", None)
    if value is None:
        value = getattr(parallel_config, "data_parallel_index", 0)
    return int(value or 0)


def _nnodes_within_dp(parallel_config) -> int:
    value = getattr(parallel_config, "nnodes_within_dp", None)
    if value is not None:
        return int(value)
    dp_size = int(getattr(parallel_config, "data_parallel_size", 1) or 1)
    local_dp_size = int(getattr(parallel_config, "data_parallel_size_local", dp_size) or dp_size)
    return max(1, (dp_size + local_dp_size - 1) // local_dp_size)


def _local_world_size(parallel_config) -> int:
    value = getattr(parallel_config, "local_world_size", None)
    if value is not None:
        return int(value)
    local_dp_size = int(getattr(parallel_config, "data_parallel_size_local", 1) or 1)
    pp_size = int(getattr(parallel_config, "pipeline_parallel_size", 1) or 1)
    tp_size = int(getattr(parallel_config, "tensor_parallel_size", 1) or 1)
    return max(1, local_dp_size * pp_size * tp_size)



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


def _report_usage_stats_safely(vllm_config) -> None:
    try:
        _register_easymagpie_model_in_vllm(
            getattr(vllm_config, "model_config", None),
            "usage_stats",
        )
        report_usage_stats(vllm_config)
    except Exception as exc:
        logger.warning("Skipping vLLM usage stats for custom omni model: %s", exc)


class GPUGenerationWorker(OmniWorkerMixin, OmniGPUWorkerBase):
    """GPU Worker for Generation model (non-autoregressive waveform generation).

    Usage in stage config:
        worker_cls: "vllm_omni.worker.gpu_generation_model_runner.GPUGenerationModelRunner"
    """

    @instrument(span_name="Init device")
    def init_device(self):
        if self.device_config.device_type == "cuda":
            # This env var set by Ray causes exceptions with graph building.
            os.environ.pop("NCCL_ASYNC_ERROR_HANDLING", None)
            parallel_config = self.parallel_config
            if (
                parallel_config.distributed_executor_backend not in ("ray", "external_launcher")
                and parallel_config.data_parallel_backend != "ray"
                and _nnodes_within_dp(parallel_config) == 1
            ):
                # Use local DP rank if available, otherwise use global DP rank.
                dp_local_rank = _data_parallel_local_rank(self.parallel_config)

                tp_pp_world_size = (
                    self.parallel_config.pipeline_parallel_size * self.parallel_config.tensor_parallel_size
                )

                # DP_LOCAL_RANK * TP_PP_WORLD_SIZE + TP_LOCAL_RANK
                self.local_rank += dp_local_rank * tp_pp_world_size
                assert self.local_rank < torch.accelerator.device_count(), (
                    f"DP adjusted local rank {self.local_rank} is out of bounds. "
                )
                visible_device_count = torch.accelerator.device_count() if torch.cuda.is_available() else 0
                local_world_size = _local_world_size(self.parallel_config)
                assert local_world_size <= visible_device_count, (
                    f"local_world_size ({local_world_size}) must "
                    f"be less than or equal to the number of visible devices "
                    f"({visible_device_count})."
                )
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.accelerator.set_device_index(self.device)

            check_if_supports_dtype = getattr(current_platform, "check_if_supports_dtype", None)
            if callable(check_if_supports_dtype):
                check_if_supports_dtype(self.model_config.dtype)

            # Initialize the distributed environment BEFORE taking
            # memory snapshot
            # This ensures NCCL buffers are allocated before we measure
            # available memory
            init_worker_distributed_environment(
                self.vllm_config,
                self.rank,
                self.distributed_init_method,
                self.local_rank,
                current_platform.dist_backend,
            )

            # Set random seed.
            set_random_seed(self.model_config.seed)

            # Now take memory snapshot after NCCL is initialized
            gc.collect()
            _empty_accelerator_cache()

            # take current memory snapshot
            self.init_snapshot = init_snapshot = MemorySnapshot()
            self.requested_memory = int(init_snapshot.total_memory * self.cache_config.gpu_memory_utilization)
            if init_snapshot.free_memory < self.requested_memory:
                raise ValueError(
                    "Free memory on device "
                    f"({format_gib(init_snapshot.free_memory)}/"
                    f"{format_gib(init_snapshot.total_memory)} GiB) on startup "
                    "is less than desired GPU memory utilization "
                    f"({self.cache_config.gpu_memory_utilization}, "
                    f"{format_gib(self.requested_memory)} GiB)."
                )
            logger.debug("worker init memory snapshot: %r", self.init_snapshot)
            logger.debug("worker requested memory: %sGiB", format_gib(self.requested_memory))
        else:
            raise RuntimeError(f"Not support device type: {self.device_config.device}")

        # Initialize workspace manager
        num_ubatches = 2 if _enable_dbo(self.vllm_config.parallel_config) else 1
        init_workspace_manager(self.device, num_ubatches)

        if self.use_v2_model_runner:
            # OMNI: v2 model runner does not yet include omni hooks.
            logger.warning("OMNI GPUGenerationWorker forces v1 model runner for omni hooks.")
            self.use_v2_model_runner = False

        self.model_runner = GPUGenerationModelRunner(self.vllm_config, self.device)

        if self.rank == 0:
            # If usage stat is enabled, collect relevant info.
            _report_usage_stats_safely(self.vllm_config)
