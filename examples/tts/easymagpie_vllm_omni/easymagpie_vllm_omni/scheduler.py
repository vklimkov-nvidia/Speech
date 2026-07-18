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
"""Streaming scheduler that propagates EasyMagpie request metadata.

Configure it on a single-stage deployment with::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

from vllm.v1.request import Request, StreamingUpdate

from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler


class EasyMagpieARAsyncScheduler(OmniARAsyncScheduler):
    """Forward each chunk's token limit and additional information.

    This class also works around a bug in vLLM-Omni's async segment-stop
    handling that deadlocks paced streaming sessions. On a resumable segment
    stop, ``OmniARScheduler.update_from_output`` does::

        request.async_tokens_to_discard = 1        # hardcoded
        request.num_output_placeholders = 0

    i.e. it assumes exactly one async token is in flight and, unlike omni's own
    *resume* path, it never rolls ``num_computed_tokens`` back for the tokens it
    is about to discard. Combined with vLLM 0.24's async accounting (a discarded
    token returns early from ``AsyncScheduler._update_request_with_output``
    without decrementing ``num_output_placeholders``) this leaves the re-admitted
    session in an unschedulable state:

    * a leaked placeholder (``placeholders>0`` with ``num_computed==num_tokens``)
      permanently trips the scheduler's async skip-optimisation, or
    * ``num_computed_tokens == num_tokens`` with ``placeholders==0`` yields
      ``num_new_tokens==0``.

    Either way the request is never scheduled again and paced clients hang.

    The fix mirrors omni's resume path: snapshot the *true* number of in-flight
    async tokens at the moment of the stop, then after ``update_from_output`` set
    ``async_tokens_to_discard`` to that count (0 when nothing is in flight, so no
    spurious discard) and roll ``num_computed_tokens`` back by the same amount.

    TODO(upstream): fix ``OmniARScheduler.update_from_output`` directly so the
    segment-stop branch uses ``async_tokens_to_discard = num_output_placeholders``
    and ``num_computed_tokens -= num_output_placeholders`` (matching the resume
    branch), then drop this override.
    """

    def _update_request_with_output(self, request: Request, new_token_ids):
        new_token_ids, stopped = super()._update_request_with_output(request, new_token_ids)
        if stopped:
            # After super() has decremented the placeholder for the stopping
            # token, ``num_output_placeholders`` is the number of *other* async
            # tokens still in flight for this request — the value omni's stop
            # handler should have used but overwrites with a hardcoded 1. Record
            # it so update_from_output can restore the correct accounting. Only
            # tracked while inside update_from_output, so there is no per-step
            # cost beyond the (rare) segment stops themselves.
            pending = getattr(self, "_emp_stopped_this_step", None)
            if pending is not None:
                pending.append((request, request.num_output_placeholders))
        return new_token_ids, stopped

    def update_from_output(self, scheduler_output, model_runner_output):
        self._emp_stopped_this_step = []
        try:
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            for request, snap in self._emp_stopped_this_step:
                # Only correct resumable stops where omni actually armed a discard.
                if getattr(request, "async_tokens_to_discard", 0) > 0:
                    request.async_tokens_to_discard = snap
                    if snap > 0:
                        request.num_computed_tokens -= snap
        finally:
            self._emp_stopped_this_step = None
        return outputs

    def _handle_stopped_request(self, request: Request) -> bool:
        # The input engine queues ``None`` after the final StreamingInput but
        # leaves the existing session's ``resumable`` flag set. Clear it before
        # the base handler consumes the sentinel so the chunk-transfer adapter
        # emits a true terminal payload and releases request-persistent codec
        # state. An empty queue still means "waiting for more websocket input".
        streaming_queue = getattr(request, "streaming_queue", None)
        if getattr(request, "resumable", False) and streaming_queue and streaming_queue[0] is None:
            request.resumable = False
        return super()._handle_stopped_request(request)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        outstanding_async_tokens = getattr(session, "num_output_placeholders", 0)
        super()._update_request_as_session(session, update)

        # Upstream hardcodes one discard on resume even when multiple async
        # outputs are outstanding. Its rollback is otherwise correct, so retain
        # it and replace only the discard count with the captured real value.
        if outstanding_async_tokens > 0 and getattr(session, "async_tokens_to_discard", 0) > 0:
            session.async_tokens_to_discard = outstanding_async_tokens

        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info

        # Defensive guard: if a resumed session has every token already computed
        # (``num_computed_tokens >= num_tokens``), the upstream scheduler computes
        # ``num_new_tokens == 0`` and trips ``assert num_new_tokens > 0``. Roll
        # back one token so there is always something to recompute and sample from
        # — the same "recompute the last token" corrective vLLM applies on a full
        # prompt cache hit (see Scheduler._update_waiting_for_remote_kv).
        if session.num_computed_tokens >= session.num_tokens:
            session.num_computed_tokens = session.num_tokens - 1
