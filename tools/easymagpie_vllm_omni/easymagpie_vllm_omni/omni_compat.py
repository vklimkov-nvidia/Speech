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
"""Compatibility fixes for the pinned vLLM-Omni streaming lifecycle."""
from __future__ import annotations

from functools import wraps


def preserve_resumable_segment(output, request_state) -> bool:
    """Keep a per-turn stop from completing the whole streaming request."""
    streaming = getattr(request_state, "streaming", None)
    is_segment = (
        bool(getattr(output, "finished", False))
        and bool(getattr(streaming, "enabled", False))
        and bool(getattr(streaming, "segment_finished", False))
    )
    if not is_segment:
        return False

    output.finished = False
    output.is_segment_finished = True
    return True


def patch_resumable_segment_routing() -> None:
    """Expose segment completion without letting vLLM-Omni clean up the request."""
    from vllm_omni.engine.orchestrator import Orchestrator

    original = Orchestrator._handle_processed_outputs
    if getattr(original, "_easymagpie_patched", False):
        return

    @wraps(original)
    async def patched(self, stage_id, replica_id, outputs):
        for output in outputs:
            request_state = self.request_states.get(output.request_id)
            if request_state is not None:
                preserve_resumable_segment(output, request_state)
        return await original(self, stage_id, replica_id, outputs)

    patched._easymagpie_patched = True
    Orchestrator._handle_processed_outputs = patched
