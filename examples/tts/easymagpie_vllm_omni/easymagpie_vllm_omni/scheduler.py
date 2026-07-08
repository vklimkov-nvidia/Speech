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
"""Streaming-aware scheduler for the single-stage EasyMagpieTTS engine.

vLLM-Omni's stage-0 streaming session update
(:meth:`OmniARScheduler._update_request_as_session`) extends the prompt token
ids from each ``StreamingInput`` chunk but **never updates**
``session.additional_information``. For EasyMagpie's streaming-text path that
silently drops the per-chunk ``text_token`` payload on the scheduler side: the
runner only ever sees the initial request's ``additional_information``, so every
decode step reads ``text_token=None``, the text channel is masked off, and the
model emits audio-EOS almost immediately (a handful of frames instead of the
full utterance).

:class:`EasyMagpieARAsyncScheduler` restores the missing propagation. It is a
drop-in replacement for ``OmniARAsyncScheduler``; wire it in via the stage's
``scheduler_cls``::

    "scheduler_cls": "easymagpie_vllm_omni.scheduler.EasyMagpieARAsyncScheduler"
"""
from __future__ import annotations

import os
import logging

from vllm.v1.core.sched.request_queue import create_request_queue
from vllm.v1.request import Request

try:
    from vllm.v1.request import StreamingUpdate
except ImportError:
    StreamingUpdate = object

try:
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARAsyncScheduler as _OmniARBaseScheduler
except ImportError:
    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler as _OmniARBaseScheduler


logger = logging.getLogger(__name__)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


class EasyMagpieARAsyncScheduler(_OmniARBaseScheduler):
    """``OmniARAsyncScheduler`` that forwards per-chunk ``additional_information``.

    Replace (not merge) is the correct session-level semantics: the session field
    is just a courier for the latest chunk's payload to ``OmniNewRequestData``.
    Per-key accumulation, where a model needs it, is handled by the runner's
    ``_update_streaming_input_additional_info`` against the model's
    ``streaming_accumulated_keys`` set, so the merge policy stays a per-model
    concern. ``None`` is treated as "this chunk omitted the field" (keep the prior
    value) rather than "clear the session", so a client may keep pumping
    placeholder chunks (e.g. the masking ``text_token=-1`` sentinel still sets a
    value; a truly empty chunk leaves the previous payload intact).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.disable_mixed_prefill_decode = _env_flag(
            "EASYMAGPIE_DISABLE_MIXED_PREFILL_DECODE", True
        )
        self.purge_stale_train_before_post_refit = _env_flag(
            "EASYMAGPIE_PURGE_STALE_TRAIN_BEFORE_POST_REFIT", True
        )

    @staticmethod
    def _request_id(request: Request) -> str:
        return str(getattr(request, "request_id", ""))

    @staticmethod
    def _request_ids_head(requests, limit: int = 8) -> list[str]:
        ids: list[str] = []
        for request in requests or []:
            ids.append(EasyMagpieARAsyncScheduler._request_id(request))
            if len(ids) >= limit:
                break
        return ids

    @staticmethod
    def _safe_len(value) -> int | None:
        try:
            return len(value)
        except Exception:
            return None

    @staticmethod
    def _is_train_rollout_request(request: Request) -> bool:
        request_id = EasyMagpieARAsyncScheduler._request_id(request)
        return request_id.startswith(
            (
                "easymagpie-grpo-train-step-",
                "easymagpie-grpo-train-rank-",
            )
        )

    @staticmethod
    def _is_post_refit_request(request: Request) -> bool:
        request_id = EasyMagpieARAsyncScheduler._request_id(request)
        return request_id.startswith(
            (
                "easymagpie-grpo-train-post-refit-step-",
                "easymagpie-grpo-train-post-refit-rank-",
            )
        )

    def _has_waiting_post_refit_request(self) -> bool:
        return any(self._is_post_refit_request(request) for request in self.waiting)

    def _purge_stale_train_running_before_post_refit(self) -> list[str]:
        if (
            not getattr(self, "purge_stale_train_before_post_refit", True)
            or not self.waiting
            or not self._has_waiting_post_refit_request()
            or not self.running
        ):
            return []

        kept = []
        purged = []
        for request in self.running:
            if self._is_train_rollout_request(request):
                purged.append(self._request_id(request))
            else:
                kept.append(request)

        if not purged:
            return []
        try:
            self.running[:] = kept
        except TypeError:
            self.running = kept
        self._easymagpie_last_post_refit_purge = {
            "num_purged_running": len(purged),
            "purged_running_head": purged[:16],
            "purged_running_tail": purged[-16:],
        }
        logger.info(
            "EasyMagpie post-refit scheduler purged stale train running requests: "
            "num_purged=%d waiting_head=%s purged_head=%s",
            len(purged),
            self._request_ids_head(self.waiting),
            purged[:8],
        )
        return purged

    @staticmethod
    def _is_active_decode_request(request: Request) -> bool:
        is_finished = getattr(request, "is_finished", None)
        if callable(is_finished) and is_finished():
            return False

        status = getattr(request, "status", None)
        status_name = getattr(status, "name", None)
        if status_name is None and status is not None:
            status_name = str(status)
        if status_name is not None and status_name != "RUNNING":
            return False

        pending = EasyMagpieARAsyncScheduler._num_pending_tokens(request)
        if pending is not None and pending <= 0:
            return False

        # Older tests/stubs may not carry vLLM RequestStatus. If no status is
        # present, keep the conservative old behavior and treat it as active.
        return True

    @staticmethod
    def _num_pending_tokens(request: Request) -> int | None:
        required = (
            "num_tokens_with_spec",
            "num_output_placeholders",
            "num_computed_tokens",
        )
        if not all(hasattr(request, name) for name in required):
            return None
        return (
            int(getattr(request, "num_tokens_with_spec") or 0)
            + int(getattr(request, "num_output_placeholders") or 0)
            - int(getattr(request, "num_computed_tokens") or 0)
        )

    def _has_active_decode_requests(self) -> bool:
        return any(self._is_active_decode_request(request) for request in self.running)

    @staticmethod
    def _payload_scalar(payload, key: str):
        if isinstance(payload, dict):
            return payload.get(key)
        entries = getattr(payload, "entries", None)
        entry = entries.get(key) if isinstance(entries, dict) else None
        if entry is None:
            return None
        scalar = getattr(entry, "scalar_data", None)
        if scalar is not None:
            return scalar
        values = getattr(entry, "list_data", None)
        return values[0] if isinstance(values, list) and values else None

    @classmethod
    def _annotate_cfg_request(cls, request: Request) -> None:
        if getattr(request, "_easymagpie_cfg_annotated", False):
            return
        payload = getattr(request, "additional_information", None)
        if not bool(cls._payload_scalar(payload, "cfg_enabled")):
            request._easymagpie_cfg_annotated = True
            return
        cfg_n = int(cls._payload_scalar(payload, "cfg_num_outputs") or 0)
        request_id = cls._request_id(request)
        child_text, separator, parent_id = request_id.partition("_")
        if cfg_n <= 0 or not separator or not child_text.isdigit():
            raise RuntimeError(
                f"Malformed EasyMagpie CFG child request {request_id!r}: cfg_num_outputs={cfg_n}"
            )
        child_index = int(child_text)
        if child_index >= 2 * cfg_n:
            raise RuntimeError(
                f"EasyMagpie CFG child index {child_index} is outside 2*n={2 * cfg_n}"
            )
        role = "conditional" if child_index < cfg_n else "unconditional"
        pair_index = child_index % cfg_n

        if isinstance(payload, dict):
            annotated = dict(payload)
            annotated.update(
                {
                    "cfg_role": role,
                    "cfg_pair_index": pair_index,
                    "cfg_parent_request_id": parent_id,
                }
            )
        else:
            from vllm_omni.engine import AdditionalInformationEntry, AdditionalInformationPayload

            entries = dict(getattr(payload, "entries", {}) or {})
            entries.update(
                {
                    "cfg_role": AdditionalInformationEntry(scalar_data=role),
                    "cfg_pair_index": AdditionalInformationEntry(scalar_data=pair_index),
                    "cfg_parent_request_id": AdditionalInformationEntry(scalar_data=parent_id),
                }
            )
            annotated = AdditionalInformationPayload(entries=entries)
        request.additional_information = annotated
        request._easymagpie_cfg_annotated = True
        request._easymagpie_cfg_parent_id = parent_id
        request._easymagpie_cfg_pair_index = pair_index
        request._easymagpie_cfg_role = role

    def _annotate_cfg_requests(self) -> None:
        for request in list(self.waiting or []) + list(self.running or []):
            self._annotate_cfg_request(request)

    def _assert_complete_cfg_decode_pairs(self, schedule_output) -> None:
        scheduled = getattr(schedule_output, "num_scheduled_tokens", None)
        if not isinstance(scheduled, dict):
            return
        requests = {
            self._request_id(request): request
            for request in list(self.running or []) + list(self.waiting or [])
        }
        pairs: dict[tuple[str, int], set[str]] = {}
        for request_id, token_count in scheduled.items():
            request = requests.get(str(request_id))
            if request is None or not getattr(request, "_easymagpie_cfg_parent_id", None):
                continue
            if int(getattr(request, "num_computed_tokens", 0) or 0) < int(
                getattr(request, "num_prompt_tokens", 0) or 0
            ):
                continue
            if int(token_count or 0) != 1:
                continue
            key = (
                str(request._easymagpie_cfg_parent_id),
                int(request._easymagpie_cfg_pair_index),
            )
            pairs.setdefault(key, set()).add(str(request._easymagpie_cfg_role))
        incomplete = {key: sorted(roles) for key, roles in pairs.items() if roles != {"conditional", "unconditional"}}
        if incomplete:
            raise RuntimeError(f"EasyMagpie scheduler split CFG decode pairs: {incomplete}")

    def _finalize_schedule(self, result):
        self._assert_complete_cfg_decode_pairs(result)
        return result

    @staticmethod
    def _scheduled_token_count(schedule_output) -> int | None:
        num_scheduled_tokens = getattr(schedule_output, "num_scheduled_tokens", None)
        if num_scheduled_tokens is None:
            return None
        if isinstance(num_scheduled_tokens, dict):
            try:
                return sum(int(count or 0) for count in num_scheduled_tokens.values())
            except Exception:
                return None
        try:
            return int(num_scheduled_tokens or 0)
        except Exception:
            return None

    @staticmethod
    def _restore_deferred_waiting(current_waiting, deferred_waiting):
        if current_waiting:
            for request in deferred_waiting:
                current_waiting.add_request(request)
            return current_waiting
        return deferred_waiting

    def schedule(self):  # type: ignore[override]
        self._annotate_cfg_requests()
        post_refit_waiting = self._has_waiting_post_refit_request()
        if post_refit_waiting:
            logger.info(
                "EasyMagpie post-refit scheduler before schedule: waiting=%s running=%s "
                "waiting_head=%s running_head=%s active_decode=%s",
                self._safe_len(self.waiting),
                self._safe_len(self.running),
                self._request_ids_head(self.waiting),
                self._request_ids_head(self.running),
                self._has_active_decode_requests(),
            )
        self._purge_stale_train_running_before_post_refit()

        if (
            not self.disable_mixed_prefill_decode
            or not self._has_active_decode_requests()
            or not self.waiting
        ):
            result = super().schedule()
            if post_refit_waiting:
                logger.info(
                    "EasyMagpie post-refit scheduler after direct schedule: scheduled_tokens=%s "
                    "waiting=%s running=%s waiting_head=%s running_head=%s",
                    self._scheduled_token_count(result),
                    self._safe_len(self.waiting),
                    self._safe_len(self.running),
                    self._request_ids_head(self.waiting),
                    self._request_ids_head(self.running),
                )
            return self._finalize_schedule(result)

        # The EasyMagpie Mamba path is unstable when vLLM batches cached
        # one-token decodes together with fresh prompt prefills.
        deferred_waiting = self.waiting
        self.waiting = create_request_queue(self.policy)
        restored_waiting = False
        try:
            result = super().schedule()
            scheduled_tokens = self._scheduled_token_count(result)
            self._easymagpie_last_mixed_guard = {
                "deferred_waiting": len(deferred_waiting),
                "guarded_scheduled_tokens": scheduled_tokens,
                "fallback_to_waiting_prefill": scheduled_tokens == 0,
            }
            if scheduled_tokens == 0:
                self.waiting = self._restore_deferred_waiting(self.waiting, deferred_waiting)
                restored_waiting = True
                result = super().schedule()
                if post_refit_waiting:
                    logger.info(
                        "EasyMagpie post-refit scheduler after guarded fallback: scheduled_tokens=%s "
                        "waiting=%s running=%s waiting_head=%s running_head=%s",
                        self._scheduled_token_count(result),
                        self._safe_len(self.waiting),
                        self._safe_len(self.running),
                        self._request_ids_head(self.waiting),
                        self._request_ids_head(self.running),
                    )
                return self._finalize_schedule(result)
            if post_refit_waiting:
                logger.info(
                    "EasyMagpie post-refit scheduler after guarded decode schedule: scheduled_tokens=%s "
                    "deferred_waiting=%s waiting=%s running=%s",
                    scheduled_tokens,
                    self._safe_len(deferred_waiting),
                    self._safe_len(self.waiting),
                    self._safe_len(self.running),
                )
            return self._finalize_schedule(result)
        finally:
            if not restored_waiting:
                self.waiting = self._restore_deferred_waiting(self.waiting, deferred_waiting)

    def _update_request_as_session(self, session: Request, update: StreamingUpdate) -> None:
        super()._update_request_as_session(session, update)

        # ``check_stop`` decides segment termination on ``session.max_tokens``,
        # a value cached once at request creation. The base session update swaps
        # ``session.sampling_params`` for each chunk but never refreshes the
        # cached ``session.max_tokens`` (even though ``StreamingUpdate`` carries
        # one). Without this, a chunk that raises ``max_tokens`` — e.g. handing
        # the request off to a free-running acoustic tail once the text stream is
        # exhausted — is silently capped at the request's *initial* ``max_tokens``
        # (1 in the one-frame-per-chunk streaming-text path), so every segment,
        # the tail included, stops after a single decoded frame.
        new_max_tokens = getattr(update, "max_tokens", None)
        if new_max_tokens is not None:
            session.max_tokens = new_max_tokens

        # At stage_id != 0 the base class already routed through
        # ``_replace_session_with_streaming_update`` (which sets
        # ``additional_information``); only stage 0 drops it.
        if self.vllm_config.model_config.stage_id == 0:
            new_info = getattr(update, "additional_information", None)
            if new_info is not None:
                session.additional_information = new_info
