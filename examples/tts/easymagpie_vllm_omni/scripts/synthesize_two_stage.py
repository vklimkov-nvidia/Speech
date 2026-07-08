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
"""Offline text-to-waveform with the EasyMagpieTTS two-stage pipeline.

Runs the full talker -> code2wav pipeline in a single :class:`AsyncOmni` engine
(no external codec service) and writes a ``.wav`` per input line. The Stage-1
Code2Wav decodes the codec bundled into the converted model directory by
``easy_magpietts_convert_to_vllm.py`` (``--bundle_codec``, the default).

Usage:
    python synthesize_two_stage.py --model ./easymp_vllm_model \\
        --text "Hello, welcome to the voice synthesis demo." --out out.wav
    python synthesize_two_stage.py --model ./easymp_vllm_model \\
        --text-file lines.txt --outdir ./wavs
"""
from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

SPEAKER = "eng"
CONTEXT_TEXT = "[EN]"
LT_TEMPERATURE = 0.7
LT_TOPK = 80
DEFAULT_DEPLOY = str(Path(__file__).resolve().parent.parent / "deploy" / "easymagpie.yaml")


def _load_meta(model_dir: str, speaker_id: str):
    from transformers import AutoTokenizer

    from easymagpie_vllm_omni.config import EasyMagpieOmniArch
    from easymagpie_vllm_omni.easymagpie import EasyMagpieTTSForConditionalGeneration

    config = json.loads((Path(model_dir) / "config.json").read_text())
    arch = EasyMagpieOmniArch.from_hf_config(type("Cfg", (), config))
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    prompt_len = EasyMagpieTTSForConditionalGeneration.get_prompt_len(
        speaker_id, model_dir, tokenize=lambda t: tokenizer.encode(t)
    )
    stop_token_id = EasyMagpieTTSForConditionalGeneration.audio_eos_stop_token_id(type("Cfg", (), config))
    return {
        "arch": arch,
        "tokenizer": tokenizer,
        "prompt_len": int(prompt_len),
        "stop_token_id": int(stop_token_id),
        "speech_delay": int(getattr(arch, "streaming_speech_delay", 0) or 0),
    }


def _build_prompt(text: str, meta: dict, speaker_id: str) -> dict:
    return {
        "prompt_token_ids": [0] * meta["prompt_len"],
        "additional_information": {
            "context_text": CONTEXT_TEXT,
            "text": text,
            "temperature": LT_TEMPERATURE,
            "top_k": LT_TOPK,
            "speaker_id": speaker_id,
        },
    }


def _extract_audio(stage_output) -> Optional[tuple[Any, int]]:
    """Pull (waveform, sample_rate) from a final-stage OmniRequestOutput.

    In async_chunk mode Stage 1 emits one audio tensor per codec window; the
    engine's output_processor renames ``model_outputs`` -> ``audio`` and
    accumulates the per-chunk tensors into a list, consolidating (concatenating)
    them on the finished output. So the audio may arrive as a list of chunks
    (concatenate along time) or as a single tensor. Mirrors the reference
    ``qwen3_tts/end2end.py::_save_wav``.
    """
    import torch

    ro = getattr(stage_output, "request_output", stage_output)
    outputs = getattr(ro, "outputs", None)
    if not outputs:
        return None
    mm = getattr(outputs[0], "multimodal_output", None)
    if not isinstance(mm, dict):
        return None
    # Framework surfaces audio under "audio" (preferred) or the raw "model_outputs".
    audio_data = mm.get("audio")
    if audio_data is None:
        audio_data = mm.get("model_outputs")
    if audio_data is None:
        return None

    if isinstance(audio_data, list):
        chunks = [t for t in audio_data if isinstance(t, torch.Tensor) and t.numel() > 0]
        if not chunks:
            return None
        try:
            wav_t = torch.cat([t.reshape(-1) for t in chunks], dim=0)
        except RuntimeError:
            wav_t = chunks[-1].reshape(-1)
    elif isinstance(audio_data, torch.Tensor):
        wav_t = audio_data.reshape(-1)
    else:
        wav_t = torch.as_tensor(audio_data).reshape(-1)

    wav_t = wav_t.detach().float().cpu()
    if wav_t.numel() == 0:
        return None

    sr = mm.get("sr")
    sr_val = sr[-1] if isinstance(sr, (list, tuple)) and sr else sr
    if isinstance(sr_val, torch.Tensor):
        sr_val = int(sr_val.reshape(-1)[0].item())
    return wav_t.numpy(), int(sr_val or 22050)


async def _synthesize_one(omni, text: str, meta: dict, speaker_id: str, max_new_tokens: int):
    from vllm import SamplingParams
    from vllm.sampling_params import RequestOutputKind

    talker_sp = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        detokenize=False,
        ignore_eos=False,
        stop_token_ids=[meta["stop_token_id"]],
        output_kind=RequestOutputKind.DELTA,
    )
    code2wav_sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, detokenize=True)

    prompt = _build_prompt(text, meta, speaker_id)
    audio = None
    gen = omni.generate(
        prompt,
        sampling_params_list=[talker_sp, code2wav_sp],
        request_id=f"easymp-{abs(hash(text)) & 0xFFFF:x}",
    )
    # The engine consolidates the per-chunk audio into mm["audio"] on the finished
    # output; prefer it, but keep the last non-empty extraction as a fallback.
    async for stage_output in gen:
        extracted = _extract_audio(stage_output)
        if extracted is not None:
            audio = extracted
        if getattr(stage_output, "finished", False) and extracted is not None:
            audio = extracted
    return audio


async def main(args):
    import vllm_plugin_easymagpie_omni

    # Ensure the archs + pipeline are registered in this (orchestrator) process.
    vllm_plugin_easymagpie_omni.register()

    from vllm_omni import AsyncOmni

    if args.text_file:
        texts = [ln.strip() for ln in Path(args.text_file).read_text().splitlines() if ln.strip()]
    else:
        texts = [args.text]
    if not texts:
        logger.error("No input text provided.")
        return

    meta = _load_meta(args.model, args.speaker_id)
    logger.info("prompt_len=%d stop_token_id=%d", meta["prompt_len"], meta["stop_token_id"])

    omni = AsyncOmni(
        model=args.model,
        deploy_config=args.deploy_config,
        log_stats=False,
    )
    try:
        import soundfile as sf

        outdir = Path(args.outdir) if args.outdir else None
        if outdir:
            outdir.mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(texts):
            logger.info("Synthesizing [%d/%d]: %s", i + 1, len(texts), text[:60])
            audio = await _synthesize_one(omni, text, meta, args.speaker_id, args.max_new_tokens)
            if audio is None:
                logger.warning("  no audio produced for line %d", i + 1)
                continue
            wav, sr = audio
            out_path = Path(args.out) if (args.out and len(texts) == 1) else (outdir or Path(".")) / f"out_{i:04d}.wav"
            sf.write(str(out_path), wav, sr)
            logger.info("  wrote %s (%.2fs @ %dHz)", out_path, len(wav) / sr, sr)
    finally:
        omni.shutdown()


def parse_args():
    p = argparse.ArgumentParser(description="EasyMagpieTTS two-stage offline synthesis")
    p.add_argument("--model", required=True, help="Converted EasyMagpie model dir (with bundled codec/)")
    p.add_argument("--deploy-config", default=DEFAULT_DEPLOY, help="Deploy YAML (default: deploy/easymagpie.yaml)")
    p.add_argument("--text", default="Hello, welcome to the voice synthesis demo.")
    p.add_argument("--text-file", default=None, help="One utterance per line (overrides --text)")
    p.add_argument("--out", default="out.wav", help="Output wav (single --text mode)")
    p.add_argument("--outdir", default=None, help="Output dir (multi-line --text-file mode)")
    p.add_argument("--speaker-id", default=SPEAKER)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
