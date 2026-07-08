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
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeQueue(list):
    def add_request(self, request):
        self.append(request)


class _FakeBaseScheduler:
    pass


def _load_scheduler_with_stubs(monkeypatch):
    def fake_package(name: str):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def fake_module(name: str):
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    for name in [
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.core.sched",
        "vllm_omni",
        "vllm_omni.core",
        "vllm_omni.core.sched",
    ]:
        fake_package(name)

    request_queue = fake_module("vllm.v1.core.sched.request_queue")
    request_queue.create_request_queue = lambda policy: _FakeQueue()

    request = fake_module("vllm.v1.request")
    request.Request = object
    request.StreamingUpdate = object

    omni_scheduler = fake_module("vllm_omni.core.sched.omni_ar_scheduler")
    omni_scheduler.OmniARAsyncScheduler = _FakeBaseScheduler
    omni_scheduler.OmniARScheduler = _FakeBaseScheduler

    module_path = (
        Path(__file__).parents[1] / "easymagpie_vllm_omni" / "scheduler.py"
    )
    spec = importlib.util.spec_from_file_location(
        "easymagpie_scheduler_under_test", module_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_easy_scheduler_defers_waiting_prefills_while_running(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.policy = "fcfs"
    scheduler.running = [SimpleNamespace(request_id="decode-0")]
    scheduler.waiting = _FakeQueue(
        [
            SimpleNamespace(request_id="prefill-0"),
            SimpleNamespace(request_id="prefill-1"),
        ]
    )

    def fake_base_schedule(self):
        assert list(self.waiting) == []
        self.waiting.add_request(SimpleNamespace(request_id="preempted-decode"))
        return "scheduler-output"

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    assert scheduler.schedule() == "scheduler-output"
    assert [request.request_id for request in scheduler.waiting] == [
        "preempted-decode",
        "prefill-0",
        "prefill-1",
    ]
    assert scheduler._easymagpie_last_mixed_guard == {
        "deferred_waiting": 2,
        "guarded_scheduled_tokens": None,
        "fallback_to_waiting_prefill": False,
    }


def test_easy_scheduler_allows_prefill_when_guarded_decode_schedules_no_tokens(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.policy = "fcfs"
    scheduler.running = [
        SimpleNamespace(
            request_id="stale-decode",
            status=SimpleNamespace(name="RUNNING"),
            is_finished=lambda: False,
            num_tokens_with_spec=129,
            num_output_placeholders=0,
            num_computed_tokens=128,
        )
    ]
    scheduler.waiting = _FakeQueue([SimpleNamespace(request_id="prefill-0")])
    calls = []

    def fake_base_schedule(self):
        calls.append([request.request_id for request in self.waiting])
        if len(calls) == 1:
            assert calls[-1] == []
            return SimpleNamespace(num_scheduled_tokens={})
        assert calls[-1] == ["prefill-0"]
        return "prefill-output"

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    assert scheduler.schedule() == "prefill-output"
    assert calls == [[], ["prefill-0"]]
    assert scheduler._easymagpie_last_mixed_guard == {
        "deferred_waiting": 1,
        "guarded_scheduled_tokens": 0,
        "fallback_to_waiting_prefill": True,
    }


def test_easy_scheduler_does_not_defer_prefill_for_paused_running(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.running = [
        SimpleNamespace(
            request_id="old-stream",
            status=SimpleNamespace(name="WAITING_FOR_CHUNK"),
            is_finished=lambda: False,
        )
    ]
    scheduler.waiting = _FakeQueue([SimpleNamespace(request_id="prefill-0")])

    def fake_base_schedule(self):
        assert [request.request_id for request in self.waiting] == ["prefill-0"]
        return "scheduler-output"

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    assert scheduler.schedule() == "scheduler-output"


def test_easy_scheduler_does_not_defer_prefill_for_stale_running_without_pending_tokens(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.running = [
        SimpleNamespace(
            request_id="stale-running",
            status=SimpleNamespace(name="RUNNING"),
            is_finished=lambda: False,
            num_tokens_with_spec=128,
            num_output_placeholders=0,
            num_computed_tokens=128,
        )
    ]
    scheduler.waiting = _FakeQueue([SimpleNamespace(request_id="prefill-0")])

    def fake_base_schedule(self):
        assert [request.request_id for request in self.waiting] == ["prefill-0"]
        return "scheduler-output"

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    assert scheduler.schedule() == "scheduler-output"


def test_easy_scheduler_purges_stale_train_running_before_post_refit(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.purge_stale_train_before_post_refit = True
    scheduler.running = [
        SimpleNamespace(
            request_id="easymagpie-grpo-train-rank-00000-step-000000-0-0",
            status=SimpleNamespace(name="RUNNING"),
            is_finished=lambda: False,
            num_tokens_with_spec=129,
            num_output_placeholders=0,
            num_computed_tokens=128,
        )
    ]
    scheduler.waiting = _FakeQueue(
        [SimpleNamespace(request_id="easymagpie-grpo-train-post-refit-rank-00000-step-000000-0-0")]
    )

    def fake_base_schedule(self):
        assert self.running == []
        assert [request.request_id for request in self.waiting] == [
            "easymagpie-grpo-train-post-refit-rank-00000-step-000000-0-0"
        ]
        return "post-refit-output"

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    assert scheduler.schedule() == "post-refit-output"
    assert scheduler._easymagpie_last_post_refit_purge == {
        "num_purged_running": 1,
        "purged_running_head": ["easymagpie-grpo-train-rank-00000-step-000000-0-0"],
        "purged_running_tail": ["easymagpie-grpo-train-rank-00000-step-000000-0-0"],
    }


def test_easy_scheduler_detects_legacy_and_ranked_train_post_refit_requests(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler_cls = scheduler_mod.EasyMagpieARAsyncScheduler

    train_ids = [
        "easymagpie-grpo-train-step-000000-0-0",
        "easymagpie-grpo-train-rank-00007-step-000000-1-7",
    ]
    post_refit_ids = [
        "easymagpie-grpo-train-post-refit-step-000000-0-0",
        "easymagpie-grpo-train-post-refit-rank-00007-step-000000-0-7",
    ]
    for request_id in train_ids:
        assert scheduler_cls._is_train_rollout_request(SimpleNamespace(request_id=request_id))
        assert not scheduler_cls._is_post_refit_request(SimpleNamespace(request_id=request_id))
    for request_id in post_refit_ids:
        assert scheduler_cls._is_post_refit_request(SimpleNamespace(request_id=request_id))
        assert not scheduler_cls._is_train_rollout_request(SimpleNamespace(request_id=request_id))


def test_easy_scheduler_keeps_non_train_running_before_post_refit(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    scheduler.disable_mixed_prefill_decode = True
    scheduler.purge_stale_train_before_post_refit = True
    scheduler.policy = "fcfs"
    scheduler.running = [
        SimpleNamespace(
            request_id="interactive-request",
            status=SimpleNamespace(name="RUNNING"),
            is_finished=lambda: False,
            num_tokens_with_spec=129,
            num_output_placeholders=0,
            num_computed_tokens=128,
        )
    ]
    scheduler.waiting = _FakeQueue(
        [SimpleNamespace(request_id="easymagpie-grpo-train-post-refit-step-000000-0-0")]
    )

    def fake_base_schedule(self):
        assert [request.request_id for request in self.waiting] == []
        return SimpleNamespace(num_scheduled_tokens={"interactive-request": 1})

    monkeypatch.setattr(
        scheduler_mod._OmniARBaseScheduler,
        "schedule",
        fake_base_schedule,
        raising=False,
    )

    result = scheduler.schedule()

    assert result.num_scheduled_tokens == {"interactive-request": 1}
    assert [request.request_id for request in scheduler.running] == ["interactive-request"]
    assert [request.request_id for request in scheduler.waiting] == [
        "easymagpie-grpo-train-post-refit-step-000000-0-0"
    ]
    assert not hasattr(scheduler, "_easymagpie_last_post_refit_purge")


def _cfg_request(child_index: int, *, outputs: int = 8):
    return SimpleNamespace(
        request_id=f"{child_index}_parent-request",
        additional_information={
            "cfg_enabled": True,
            "cfg_num_outputs": outputs,
            "cfg_scale": 2.5,
        },
        num_computed_tokens=64,
        num_prompt_tokens=64,
    )


def test_easy_scheduler_annotates_parallel_cfg_children(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    conditional = _cfg_request(3)
    unconditional = _cfg_request(11)

    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(conditional)
    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(unconditional)

    assert conditional.additional_information["cfg_role"] == "conditional"
    assert unconditional.additional_information["cfg_role"] == "unconditional"
    assert conditional._easymagpie_cfg_parent_id == "parent-request"
    assert unconditional._easymagpie_cfg_parent_id == "parent-request"
    assert conditional._easymagpie_cfg_pair_index == 3
    assert unconditional._easymagpie_cfg_pair_index == 3


def test_easy_scheduler_accepts_complete_cfg_decode_pair(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    conditional = _cfg_request(3)
    unconditional = _cfg_request(11)
    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(conditional)
    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(unconditional)
    scheduler.running = [conditional, unconditional]
    scheduler.waiting = _FakeQueue()

    scheduler._assert_complete_cfg_decode_pairs(
        SimpleNamespace(
            num_scheduled_tokens={conditional.request_id: 1, unconditional.request_id: 1}
        )
    )


def test_easy_scheduler_rejects_split_cfg_decode_pair(monkeypatch):
    scheduler_mod = _load_scheduler_with_stubs(monkeypatch)
    scheduler = object.__new__(scheduler_mod.EasyMagpieARAsyncScheduler)
    conditional = _cfg_request(3)
    unconditional = _cfg_request(11)
    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(conditional)
    scheduler_mod.EasyMagpieARAsyncScheduler._annotate_cfg_request(unconditional)
    scheduler.running = [conditional, unconditional]
    scheduler.waiting = _FakeQueue()

    with pytest.raises(RuntimeError, match="split CFG decode pairs"):
        scheduler._assert_complete_cfg_decode_pairs(
            SimpleNamespace(num_scheduled_tokens={conditional.request_id: 1})
        )
