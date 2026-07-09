#!/usr/bin/env python3
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
"""Thin HTTP server for the EasyMagpieTTS two-stage pipeline.

Wraps a single in-process :class:`AsyncOmni` engine (the same known-good path as
``synthesize_two_stage.py``) behind a minimal FastAPI app so the deployment can be
benchmarked over HTTP with a real client/server split. This does NOT use
``vllm-omni serve`` (the installed vLLM-Omni does not forward EasyMagpie's
``additional_information`` / ``prompt_token_ids`` over its OpenAI endpoints).

Endpoints:
    GET  /health         -> {"status": "ok", "ready": bool}
    POST /tts            -> synthesize; streaming NDJSON (default) or single JSON

Request body (POST /tts):
    {
      "text": "...",                # required
      "speaker_id": "eng",          # optional (default: --speaker-id)
      "context_text": "[EN]",       # optional
      "temperature": 0.7,           # optional (audio/local-transformer temp)
      "top_k": 80,                  # optional
      "max_new_tokens": 1024,       # optional
      "stream": true                # optional (default true)
    }

Streaming response (``application/x-ndjson``): one JSON object per line:
    {"type": "audio", "sr": 22050, "pcm_b64": "<base64 float32 mono PCM>"}
    ...
    {"type": "done",  "sr": 22050, "num_samples": 12345}
    {"type": "error", "message": "..."}          # on failure

Non-streaming response (``stream=false``): a single JSON object
    {"sr": 22050, "num_samples": 12345, "pcm_b64": "<base64 float32 mono PCM>"}

Usage:
    python tts_server.py --model ./easymp_vllm_model --port 8091
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import base64
import json
import logging
import sys
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Optional
from uuid import uuid4

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

# Import the known-good offline helpers from the sibling script so the serving
# path cannot drift from synthesize_two_stage.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synthesize_two_stage import (  # noqa: E402
    CONTEXT_TEXT,
    DEFAULT_DEPLOY,
    LT_TEMPERATURE,
    LT_TOPK,
    SPEAKER,
    _extract_audio,
    _load_meta,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("easymp_tts_server")


class _State:
    args: argparse.Namespace | None = None
    omni: Any = None
    meta: dict[str, Any] | None = None
    prompt_len_cache: dict[str, int] = {}


STATE = _State()


class TTSRequest(BaseModel):
    text: str
    speaker_id: Optional[str] = None
    context_text: Optional[str] = None
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    max_new_tokens: int = 1024
    stream: bool = True


def _prompt_len_for(speaker_id: str) -> int:
    """Assembled-context length for ``speaker_id`` (cached).

    EasyMagpie asserts ``len(prompt_token_ids) == estimate_prompt_len(...)``, and
    that length depends on the speaker embedding, so compute it per speaker.
    """
    cached = STATE.prompt_len_cache.get(speaker_id)
    if cached is not None:
        return cached
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    tokenizer = STATE.meta["tokenizer"]
    plen = int(
        EasyMagpieTTSForConditionalGeneration.get_prompt_len(
            speaker_id, STATE.args.model, tokenize=lambda t: tokenizer.encode(t)
        )
    )
    STATE.prompt_len_cache[speaker_id] = plen
    return plen


def _build_prompt(req: TTSRequest) -> tuple[dict[str, Any], str]:
    speaker_id = req.speaker_id or STATE.args.speaker_id
    prompt = {
        "prompt_token_ids": [0] * _prompt_len_for(speaker_id),
        "additional_information": {
            "context_text": req.context_text or CONTEXT_TEXT,
            "text": req.text,
            "temperature": req.temperature if req.temperature is not None else LT_TEMPERATURE,
            "top_k": req.top_k if req.top_k is not None else LT_TOPK,
            "speaker_id": speaker_id,
        },
    }
    return prompt, speaker_id


def _sampling_params(max_new_tokens: int) -> list[Any]:
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    talker_sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        detokenize=False,
        ignore_eos=False,
        stop_token_ids=[STATE.meta["stop_token_id"]],
        output_kind=RequestOutputKind.DELTA,
    )
    code2wav_sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, detokenize=True)
    return [talker_sp, code2wav_sp]


def _ndjson(obj: dict[str, Any]) -> str:
    return json.dumps(obj) + "\n"


async def _generate_stream(req: TTSRequest) -> AsyncIterator[str]:
    """Drive AsyncOmni and yield NDJSON lines (audio deltas + terminal event).

    ``_extract_audio`` returns the full cumulative waveform on each engine output;
    we track how much has been emitted and stream only the newly produced tail so
    the client sees audio as soon as the first codec window is decoded (TTFA).
    """
    prompt, _speaker_id = _build_prompt(req)
    sampling_params_list = _sampling_params(req.max_new_tokens)
    request_id = f"tts-{uuid4().hex[:12]}"

    sent = 0
    sr = 22050
    gen = STATE.omni.generate(prompt, sampling_params_list=sampling_params_list, request_id=request_id)
    async for stage_output in gen:
        extracted = _extract_audio(stage_output)
        if extracted is None:
            continue
        wav, sr = extracted
        if len(wav) > sent:
            tail = np.asarray(wav[sent:], dtype=np.float32)
            sent = len(wav)
            yield _ndjson(
                {"type": "audio", "sr": int(sr), "pcm_b64": base64.b64encode(tail.tobytes()).decode("ascii")}
            )
    yield _ndjson({"type": "done", "sr": int(sr), "num_samples": int(sent)})


async def _safe_stream(req: TTSRequest) -> AsyncIterator[str]:
    try:
        async for line in _generate_stream(req):
            yield line
    except Exception as exc:  # surface engine errors to the client as a terminal line
        logger.exception("TTS request failed")
        yield _ndjson({"type": "error", "message": repr(exc)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    import vllm_plugin_easymagpie_omni

    # Register the archs + two-stage pipeline in this (orchestrator) process.
    vllm_plugin_easymagpie_omni.register()

    from vllm_omni import AsyncOmni

    args = STATE.args
    logger.info("Loading meta for model=%s speaker=%s", args.model, args.speaker_id)
    STATE.meta = _load_meta(args.model, args.speaker_id)
    STATE.prompt_len_cache[args.speaker_id] = int(STATE.meta["prompt_len"])
    logger.info("prompt_len=%d stop_token_id=%d", STATE.meta["prompt_len"], STATE.meta["stop_token_id"])

    logger.info("Building AsyncOmni engine (deploy=%s) ...", args.deploy_config)
    STATE.omni = AsyncOmni(model=args.model, deploy_config=args.deploy_config, log_stats=False)
    logger.info("EasyMagpie TTS server ready.")
    try:
        yield
    finally:
        if STATE.omni is not None:
            STATE.omni.shutdown()


app = FastAPI(title="EasyMagpieTTS thin server", lifespan=lifespan)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "ready": STATE.omni is not None}


@app.post("/tts")
async def tts(req: TTSRequest):
    if not req.text or not req.text.strip():
        return JSONResponse(status_code=400, content={"error": "field 'text' is required"})

    if req.stream:
        return StreamingResponse(_safe_stream(req), media_type="application/x-ndjson")

    # Non-streaming: drain the same generator and return one consolidated blob.
    chunks: list[np.ndarray] = []
    sr = 22050
    async for line in _generate_stream(req):
        obj = json.loads(line)
        if obj["type"] == "audio":
            chunks.append(np.frombuffer(base64.b64decode(obj["pcm_b64"]), dtype=np.float32))
            sr = obj["sr"]
        elif obj["type"] == "done":
            sr = obj.get("sr", sr)
    wav = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    return {
        "sr": int(sr),
        "num_samples": int(wav.size),
        "pcm_b64": base64.b64encode(wav.astype(np.float32).tobytes()).decode("ascii"),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="EasyMagpieTTS thin HTTP server (AsyncOmni wrapper)")
    p.add_argument("--model", required=True, help="Converted EasyMagpie model dir (with bundled codec/)")
    p.add_argument("--deploy-config", default=DEFAULT_DEPLOY, help="Deploy YAML (default: deploy/easymagpie.yaml)")
    p.add_argument("--speaker-id", default=SPEAKER, help="Default speaker id (default: %(default)s)")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8091)
    return p.parse_args()


def main() -> None:
    import uvicorn

    STATE.args = parse_args()
    uvicorn.run(app, host=STATE.args.host, port=STATE.args.port, log_level="info")


if __name__ == "__main__":
    main()
