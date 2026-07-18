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
"""Tests for the vLLM-Omni 0.24 async scheduler compatibility layer."""
from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
from easymagpie_vllm_omni.scheduler import EasyMagpieARAsyncScheduler
from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


def test_no_stop_is_inert_for_non_resumable_requests(monkeypatch):
    """A plain HTTP request never hits a segment stop, so the override must pass
    ``super()`` through unchanged and leave request accounting untouched."""
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=0,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        return new_token_ids, False  # no stop this step

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        return "outputs"

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == 0
    assert request.num_computed_tokens == 20
    assert request.num_output_placeholders == 0


def test_terminal_stop_without_discard_is_inert(monkeypatch):
    """HTTP requests end on a normal audio-EOS stop (not a resumable segment
    stop), so omni arms no discard and the override must not roll anything back."""
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=0,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        req.num_output_placeholders = 0  # terminal stop, nothing else in flight
        return new_token_ids, True

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        return "outputs"  # omni does not arm a discard for a terminal stop

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == 0
    assert request.num_computed_tokens == 20


@pytest.mark.parametrize("remaining_placeholders", [0, 1, 2])
def test_segment_stop_discards_and_rolls_back_exact_inflight_count(monkeypatch, remaining_placeholders):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=remaining_placeholders + 1,
    )

    def fake_update_request_with_output(self, req, new_token_ids):
        req.num_output_placeholders -= 1  # stopping token returned
        return new_token_ids, True

    def fake_update_from_output(self, scheduler_output, model_runner_output):
        self._update_request_with_output(request, [7])
        request.async_tokens_to_discard = 1
        request.num_output_placeholders = 0
        return "outputs"

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_with_output", fake_update_request_with_output)
    monkeypatch.setattr(OmniARAsyncScheduler, "update_from_output", fake_update_from_output)

    outputs = scheduler.update_from_output(None, None)

    assert outputs == "outputs"
    assert request.async_tokens_to_discard == remaining_placeholders
    assert request.num_computed_tokens == 20 - remaining_placeholders


def test_final_streaming_sentinel_marks_session_non_resumable(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque([None]))

    def fake_handle_stopped_request(self, req):
        assert req.resumable is False
        return True

    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", fake_handle_stopped_request)

    assert scheduler._handle_stopped_request(request) is True
    assert request.resumable is False


def test_empty_streaming_queue_remains_resumable_while_waiting(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    request = SimpleNamespace(resumable=True, streaming_queue=deque())
    monkeypatch.setattr(OmniARAsyncScheduler, "_handle_stopped_request", lambda self, req: False)

    assert scheduler._handle_stopped_request(request) is False
    assert request.resumable is True


def test_resume_uses_exact_discard_count_and_forwards_chunk_metadata(monkeypatch):
    scheduler = object.__new__(EasyMagpieARAsyncScheduler)
    scheduler.vllm_config = SimpleNamespace(model_config=SimpleNamespace(stage_id=0))
    session = SimpleNamespace(
        async_tokens_to_discard=0,
        num_computed_tokens=20,
        num_output_placeholders=2,
        num_tokens=23,
        max_tokens=1,
        additional_information={"text_token": [1]},
    )
    update = SimpleNamespace(max_tokens=5, additional_information={"text_token": [2, 3]})

    def fake_update_request_as_session(self, req, streaming_update):
        req.async_tokens_to_discard = 1
        req.num_computed_tokens -= req.num_output_placeholders
        req.num_output_placeholders = 0

    monkeypatch.setattr(OmniARAsyncScheduler, "_update_request_as_session", fake_update_request_as_session)

    scheduler._update_request_as_session(session, update)

    assert session.async_tokens_to_discard == 2
    assert session.num_computed_tokens == 18
    assert session.max_tokens == 5
    assert session.additional_information == {"text_token": [2, 3]}
