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
"""Import-mode tests for EasyMagpie vLLM-Omni startup hooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path


def test_compat_mode_off_does_not_import_vllm() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["EASYMAGPIE_VLLM_COMPAT_MODE"] = "off"
    env["PYTHONPATH"] = str(root)

    code = """
import json
import sys
import easymagpie_vllm_omni
loaded = sorted(name for name in sys.modules if name == "vllm" or name.startswith("vllm."))
print(json.dumps(loaded))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    loaded = json.loads(proc.stdout.strip())
    assert loaded == []


def test_compat_mode_serial_imports_review_path() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["EASYMAGPIE_VLLM_COMPAT_MODE"] = "serial"
    env["PYTHONPATH"] = str(root)

    code = """
import easymagpie_vllm_omni
print("ok")
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert proc.stdout.strip().splitlines()[-1] == "ok"


def test_compat_mode_full_is_not_part_of_review_branch() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["EASYMAGPIE_VLLM_COMPAT_MODE"] = "full"
    env["PYTHONPATH"] = str(root)

    proc = subprocess.run(
        [sys.executable, "-c", "import easymagpie_vllm_omni"],
        check=False,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    assert proc.returncode != 0
    assert "not included in the review-friendly EasyMagpie RL branch" in proc.stderr


def test_refit_compat_installs_named_runtime_reset_rpc(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import install_easy_magpie_refit_rpc_compat

    class FakeWorker:
        pass

    worker_module = types.ModuleType("vllm_omni.worker.gpu_ar_worker")
    worker_module.GPUARWorker = FakeWorker
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker", types.ModuleType("vllm_omni.worker"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker.gpu_ar_worker", worker_module)
    monkeypatch.delitem(sys.modules, "vllm_omni.worker.gpu_generation_worker", raising=False)

    install_easy_magpie_refit_rpc_compat()

    worker = FakeWorker()
    worker.model_runner = types.SimpleNamespace(
        _easymagpie_active_mamba_request_ids={"req"},
        _easymagpie_mamba_request_cache_kwargs={"req": {}},
        _easymagpie_mamba_request_cache_state={"req": {}},
        _easymagpie_mamba_cache_batch_size=1,
        requests={"req": object()},
        encoder_cache={"req": object()},
        input_batch=None,
    )

    result = worker.easymagpie_reset_runtime_state()

    assert result["ok"] is True
    assert result["runtime_reset_rpc_compat_version"] >= 1
    assert set(result["runtime_state_reset"]["cleared_attrs"]) == {
        "_easymagpie_active_mamba_request_ids",
        "_easymagpie_mamba_request_cache_kwargs",
        "_easymagpie_mamba_request_cache_state",
        "_easymagpie_mamba_cache_batch_size",
    }
    assert result["runtime_state_reset"]["cleared_mappings"] == [
        {"name": "requests", "size_before": 1},
        {"name": "encoder_cache", "size_before": 1},
    ]


def test_runtime_compat_handles_new_vllm_init_model_kwargs_signature(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import install_easy_magpie_runtime_compat

    class NewParentRunner:
        def _init_model_kwargs(self):
            return {"source": "new-vllm"}

    class OmniGPUModelRunner(NewParentRunner):
        max_num_tokens = 17

        def _init_model_kwargs(self, num_tokens=None):
            if num_tokens is None:
                num_tokens = int(getattr(self, "max_num_tokens", 0) or 0)
            return super()._init_model_kwargs(num_tokens)

    class RuntimeRunner(OmniGPUModelRunner):
        pass

    runner_module = types.ModuleType("vllm_omni.worker.gpu_model_runner")
    runner_module.OmniGPUModelRunner = OmniGPUModelRunner
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker", types.ModuleType("vllm_omni.worker"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker.gpu_model_runner", runner_module)

    try:
        OmniGPUModelRunner()._init_model_kwargs()
    except TypeError as exc:
        assert "positional argument" in str(exc)
    else:
        raise AssertionError("test fixture should reproduce the vLLM 0.21 signature mismatch")

    install_easy_magpie_runtime_compat()
    install_easy_magpie_runtime_compat()

    assert OmniGPUModelRunner()._init_model_kwargs() == {"source": "new-vllm"}
    assert RuntimeRunner()._init_model_kwargs() == {"source": "new-vllm"}


def test_runtime_compat_preserves_old_vllm_init_model_kwargs_signature(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import install_easy_magpie_runtime_compat

    class OldParentRunner:
        def _init_model_kwargs(self, num_tokens=None):
            return {"source": "old-vllm", "num_tokens": num_tokens}

    class OmniGPUModelRunner(OldParentRunner):
        max_num_tokens = 19

        def _init_model_kwargs(self, num_tokens=None):
            if num_tokens is None:
                num_tokens = int(getattr(self, "max_num_tokens", 0) or 0)
            return super()._init_model_kwargs(num_tokens)

    class RuntimeRunner(OmniGPUModelRunner):
        pass

    runner_module = types.ModuleType("vllm_omni.worker.gpu_model_runner")
    runner_module.OmniGPUModelRunner = OmniGPUModelRunner
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker", types.ModuleType("vllm_omni.worker"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker.gpu_model_runner", runner_module)

    install_easy_magpie_runtime_compat()
    install_easy_magpie_runtime_compat()

    assert OmniGPUModelRunner()._init_model_kwargs() == {"source": "old-vllm", "num_tokens": 19}
    assert OmniGPUModelRunner()._init_model_kwargs(5) == {"source": "old-vllm", "num_tokens": 5}
    assert RuntimeRunner()._init_model_kwargs() == {"source": "old-vllm", "num_tokens": 19}
    assert RuntimeRunner()._init_model_kwargs(5) == {"source": "old-vllm", "num_tokens": 5}


def test_runtime_compat_reports_configured_cudagraph_mode_for_omni_fallback(monkeypatch) -> None:
    from vllm.config import CUDAGraphMode

    from easymagpie_vllm_omni.vllm_compat import install_easy_magpie_runtime_compat

    class BatchDescriptor:
        num_tokens = 8
        num_reqs = 4

    class OmniGPUModelRunner:
        def __init__(self, cudagraph_mode):
            self.vllm_config = types.SimpleNamespace(
                compilation_config=types.SimpleNamespace(cudagraph_mode=cudagraph_mode)
            )

        def _init_model_kwargs(self, num_tokens=None):
            return {}

        def _determine_batch_execution_and_padding(self, **kwargs):
            return CUDAGraphMode.NONE, BatchDescriptor(), False, None, None

    runner_module = types.ModuleType("vllm_omni.worker.gpu_model_runner")
    runner_module.OmniGPUModelRunner = OmniGPUModelRunner
    monkeypatch.setitem(sys.modules, "vllm_omni", types.ModuleType("vllm_omni"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker", types.ModuleType("vllm_omni.worker"))
    monkeypatch.setitem(sys.modules, "vllm_omni.worker.gpu_model_runner", runner_module)

    install_easy_magpie_runtime_compat()
    install_easy_magpie_runtime_compat()

    mixed_result = OmniGPUModelRunner(CUDAGraphMode.PIECEWISE)._determine_batch_execution_and_padding(
        num_tokens=8,
        num_reqs=4,
        force_eager=False,
        force_uniform_decode=False,
    )
    eager_result = OmniGPUModelRunner(CUDAGraphMode.PIECEWISE)._determine_batch_execution_and_padding(
        num_tokens=8,
        num_reqs=4,
        force_eager=True,
        force_uniform_decode=False,
    )
    full_and_piecewise_mixed = OmniGPUModelRunner(
        CUDAGraphMode.FULL_AND_PIECEWISE
    )._determine_batch_execution_and_padding(
        num_tokens=8,
        num_reqs=4,
        force_eager=False,
        force_uniform_decode=False,
    )
    full_and_piecewise_decode = OmniGPUModelRunner(
        CUDAGraphMode.FULL_AND_PIECEWISE
    )._determine_batch_execution_and_padding(
        num_tokens=8,
        num_reqs=4,
        force_eager=False,
        force_uniform_decode=True,
    )

    assert mixed_result[0] == CUDAGraphMode.PIECEWISE
    assert eager_result[0] == CUDAGraphMode.NONE
    assert full_and_piecewise_mixed[0] == CUDAGraphMode.PIECEWISE
    assert full_and_piecewise_decode[0] == CUDAGraphMode.FULL
