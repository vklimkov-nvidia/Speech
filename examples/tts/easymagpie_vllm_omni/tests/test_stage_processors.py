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

from collections import defaultdict
from types import SimpleNamespace

import torch
from easymagpie_vllm_omni.stage_processors import talker2code2wav_async_chunk


class _Request:
    external_req_id = "request-0"

    def __init__(self):
        self.finished = False
        self.resumable = True
        self.output_token_ids = []

    def is_finished(self):
        return self.finished


def _manager():
    return SimpleNamespace(
        config=SimpleNamespace(hf_config=SimpleNamespace(streaming_speech_delay=2)),
        connector=SimpleNamespace(
            config={
                "extra": {
                    "codec_chunk_frames": 2,
                    "codec_left_context_frames": 1,
                    "initial_codec_chunk_frames": 0,
                }
            }
        ),
        code_prompt_token_ids=defaultdict(list),
    )


def _output(value: int):
    return {"audio_codes": torch.tensor([[value, value + 100]], dtype=torch.long)}


def test_async_codec_state_stays_continuous_across_resumable_segments():
    manager = _manager()
    request = _Request()

    # Warm-up is counted over the whole request, including segment boundaries.
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(1), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    warmup_flush = talker2code2wav_async_chunk(manager, _output(2), request, is_finished=True)
    assert warmup_flush.codes.audio.numel() == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    # The framework buffer is reset per segment, but the processor's request
    # state retains the real acoustic frames and its emission high-water mark.
    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(3), request) is None
    request.output_token_ids = [0, 0]
    request.finished = True
    first = talker2code2wav_async_chunk(manager, _output(4), request, is_finished=True)
    torch.testing.assert_close(first.codes.audio, torch.tensor([3, 4, 103, 104]))
    assert first.meta.left_context_size == 0
    manager.code_prompt_token_ids.pop(request.external_req_id, None)

    request.finished = False
    request.output_token_ids = [0]
    assert talker2code2wav_async_chunk(manager, _output(5), request) is None
    request.output_token_ids = [0, 0]
    second = talker2code2wav_async_chunk(manager, _output(6), request)
    torch.testing.assert_close(second.codes.audio, torch.tensor([4, 5, 6, 104, 105, 106]))
    assert second.meta.left_context_size == 1

    # Repeated segment flushes at the same length must not duplicate audio.
    request.finished = True
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None

    # Terminal completion releases the request-persistent state.
    request.resumable = False
    assert talker2code2wav_async_chunk(manager, None, request, is_finished=True) is None
    assert request.external_req_id not in manager._emp_seen_frames
    assert request.external_req_id not in manager._emp_emitted_frames
    assert request.external_req_id not in manager._emp_frame_buffer
