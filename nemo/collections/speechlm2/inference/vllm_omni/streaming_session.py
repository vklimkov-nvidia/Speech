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

"""Synchronous bridge between the streaming S2S wrapper and vLLM-Omni.

The S2S wrapper drives inference one chunk at a time from a synchronous
PyTorch loop (perception -> per-frame text + ASR -> audio codec decode).
vLLM-Omni's ``AsyncOmni.generate(...)`` exposes a fully asynchronous
``async for`` interface; bridging the two by spawning one event loop per
``infer_one_step`` would re-pay the request-init cost on every chunk and
break vllm-omni's session semantics.

:class:`OmniRuntime` owns a single long-lived asyncio loop in a daemon
thread, the ``AsyncOmni`` engine, and lazy wrapper-checkpoint construction.
Each stream gets an :class:`OmniStreamingSession` that runs ``omni.generate``
on that shared loop for the lifetime of the stream and exposes a
synchronous ``step(acoustic_embedding) -> (text_tok, asr_tok)`` plus
``drain_audio_codes() -> list[Tensor]``.

Per-step protocol (mirrors
``examples/offline_inference/nemotron_voicechat/run_nemotron_voicechat.py``):

1. **Prefill chunk** -- enqueued at session start; carries the system
   prompt string and the speaker latent. Produces a first stage-0 token
   ``t_0`` that we *don't* expose to the caller (the wrapper never sees
   it) but feed back as ``prompt_token_ids`` of the first decode chunk.
2. **Decode chunk k** (``k >= 1``) -- ``prompt_token_ids = [t_{k-1}]``,
   ``additional_information.acoustic_embedding = ac_emb[k-1]``. Produces
   stage-0 token ``t_k`` (returned to the caller as the per-frame text
   token) plus the corresponding ASR token; stage 1 emits acoustic codes
   asynchronously (lag of ~1 step in async-chunk mode), drained into a
   per-session buffer that the wrapper reads after each chunk.

The session is single-threaded on the synchronous side: only one
``step()`` call may be in flight at a time. Calling ``finish()`` (or
``abort()`` on error) terminates the streaming generator and frees
engine-side state.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import torch
import yaml

from nemo.collections.speechlm2.inference.vllm_omni import default_deploy_yaml
from nemo.utils import logging


# ---------------------------------------------------------------------------
#  Wrapper-checkpoint construction
# ---------------------------------------------------------------------------


_NEMOTRON_SUBDIR = "nemotron"
_EARTTS_SUBDIR = "eartts"
_WRAPPER_CONFIG = {"model_type": "nemotron_voicechat"}


def build_wrapper_checkpoint(
    model_path: str,
    wrapper_dir: str | None = None,
    *,
    nemotron_dtype: str = "float32",
    eartts_precompute_batch_size: int = 256,
) -> str:
    """Build a wrapper checkpoint directory consumed by ``AsyncOmni(model=...)``.

    Layout::

        <wrapper>/
            config.json         # {"model_type": "nemotron_voicechat"}
            nemotron/           # converted NemotronDuplexH checkpoint
            eartts/             # converted EarTTS checkpoint + speaker_latents/

    Args:
        model_path: Path to the source NemotronVoiceChat HF-format checkpoint
            directory (``config.json`` + ``model.safetensors``).
        wrapper_dir: Where to put the wrapper directory. Defaults to
            ``/tmp/<basename>_vllm_omni_wrapper``.
        nemotron_dtype: dtype for the converted Nemotron checkpoint.
        eartts_precompute_batch_size: batch size used when baking out the
            EarTTS subword-encoder lookup table.

    Returns:
        Absolute path to the wrapper directory. If the wrapper directory
        already exists and looks complete, the existing one is returned and
        nothing is re-converted.
    """
    src = os.path.normpath(model_path)
    if wrapper_dir is None:
        wrapper_dir = os.path.join("/tmp", os.path.basename(src) + "_vllm_omni_wrapper")
    wrapper_dir = os.path.abspath(wrapper_dir)

    nemotron_dir = os.path.join(wrapper_dir, _NEMOTRON_SUBDIR)
    eartts_dir = os.path.join(wrapper_dir, _EARTTS_SUBDIR)
    config_path = os.path.join(wrapper_dir, "config.json")

    nemotron_ready = (
        os.path.isdir(nemotron_dir)
        and os.path.isfile(os.path.join(nemotron_dir, "config.json"))
        and os.path.isfile(os.path.join(nemotron_dir, "model.safetensors"))
    )
    eartts_ready = (
        os.path.isdir(eartts_dir)
        and os.path.isfile(os.path.join(eartts_dir, "config.json"))
        and os.path.isfile(os.path.join(eartts_dir, "model.safetensors"))
    )
    config_ready = os.path.isfile(config_path)

    if nemotron_ready and eartts_ready and config_ready:
        logging.info(f"Reusing existing vllm-omni wrapper checkpoint at {wrapper_dir}")
        return wrapper_dir

    os.makedirs(wrapper_dir, exist_ok=True)

    if not nemotron_ready:
        # Convert the Nemotron LLM with the existing DuplexSTT converter.
        # That converter's output (HF NemotronH config + filtered weights)
        # is consumed directly by NemotronDuplexHForCausalLM's WeightsMapper.
        if os.path.isdir(nemotron_dir):
            shutil.rmtree(nemotron_dir)
        logging.info(f"Converting Nemotron LLM into {nemotron_dir} ...")
        from nemo.collections.speechlm2.inference.vllm_omni.scripts.convert_duplex_stt_checkpoint import (
            convert_to_vllm_format as convert_nemotron,
        )

        convert_nemotron(
            checkpoint_path=src,
            output_dir=nemotron_dir,
            dtype=nemotron_dtype,
        )

    if not eartts_ready:
        if os.path.isdir(eartts_dir):
            shutil.rmtree(eartts_dir)
        logging.info(f"Converting EarTTS into {eartts_dir} ...")
        from nemo.collections.speechlm2.inference.vllm_omni.scripts.convert_duplex_eartts_checkpoint import (
            convert_to_vllm_format as convert_eartts,
        )

        convert_eartts(
            outdir=eartts_dir,
            config=os.path.join(src, "config.json"),
            model_path=os.path.join(src, "model.safetensors"),
            precompute_batch_size=eartts_precompute_batch_size,
        )

    if not config_ready:
        with open(config_path, "w", encoding="utf-8") as fh:
            json.dump(_WRAPPER_CONFIG, fh, indent=2)

    return wrapper_dir


# ---------------------------------------------------------------------------
#  Async runtime (one per wrapper instance)
# ---------------------------------------------------------------------------


class OmniRuntime:
    """Long-lived AsyncOmni engine + background asyncio loop.

    Constructed once by the inference wrapper and shared across streams.
    Each stream creates its own :class:`OmniStreamingSession` that runs on
    the shared loop.
    """

    def __init__(
        self,
        wrapper_dir: str,
        *,
        stage_configs_path: str | None = None,
        stage_overrides: dict | None = None,
        log_stats: bool = False,
        stage_init_timeout: int = 600,
    ) -> None:
        # Resolve deploy YAML, applying any per-stage overrides.
        deploy_yaml = Path(stage_configs_path) if stage_configs_path else default_deploy_yaml()
        if not deploy_yaml.is_file():
            raise FileNotFoundError(f"Deploy YAML not found: {deploy_yaml}")
        self._stage_yaml_path = self._maybe_write_overridden_yaml(deploy_yaml, stage_overrides)

        # Start the background loop in a daemon thread first; ``AsyncOmni``
        # is constructed *on* that loop (its ``__init__`` allocates
        # ``asyncio.Condition`` / ``asyncio.Queue`` and the orchestrator
        # binds them to the current event loop, so the engine must be
        # built from inside that loop's thread).
        self._loop = asyncio.new_event_loop()
        self._ready_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._loop_runner,
            name="OmniRuntimeLoop",
            daemon=True,
        )
        self._thread.start()
        self._ready_evt.wait()

        from vllm_omni import AsyncOmni

        logging.info(f"Creating AsyncOmni engine from wrapper={wrapper_dir} ...")

        async def _build_engine() -> Any:
            return AsyncOmni(
                model=wrapper_dir,
                stage_configs_path=str(self._stage_yaml_path),
                log_stats=log_stats,
                stage_init_timeout=stage_init_timeout,
            )

        fut = asyncio.run_coroutine_threadsafe(_build_engine(), self._loop)
        self.engine = fut.result()
        logging.info(f"AsyncOmni ready ({self.engine.num_stages} stages)")

    # ------------------------------------------------------------------ #
    #  YAML override                                                     #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _maybe_write_overridden_yaml(deploy_yaml: Path, stage_overrides: dict | None) -> Path:
        """Apply per-stage overrides to the deploy YAML, write to a tmp file.

        ``stage_overrides`` shape::

            {
                "common": {<flat keys applied to every stage>},
                "stage_0": {<flat keys for stage 0>},
                "stage_1": {<flat keys for stage 1>},
            }

        Returns the path that ``AsyncOmni`` should load; the original YAML
        is returned untouched when no overrides are supplied.
        """
        if not stage_overrides:
            return deploy_yaml

        with open(deploy_yaml, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh)

        common = stage_overrides.get("common", {}) or {}
        per_stage = {
            int(k.split("_", 1)[1]): v
            for k, v in stage_overrides.items()
            if k.startswith("stage_") and v
        }

        for stage in cfg.get("stages", []):
            for key, value in common.items():
                stage[key] = value
            sid = int(stage.get("stage_id", -1))
            for key, value in per_stage.get(sid, {}).items():
                stage[key] = value

        tmp = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            prefix="nemotron_voicechat_",
            delete=False,
        )
        yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
        tmp.close()
        logging.info(f"Wrote overridden stage config to {tmp.name}")
        return Path(tmp.name)

    # ------------------------------------------------------------------ #
    #  Background loop                                                   #
    # ------------------------------------------------------------------ #

    def _loop_runner(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready_evt.set()
        try:
            self._loop.run_forever()
        finally:
            # ``run_forever`` returns when ``loop.stop()`` is called from
            # ``shutdown``. Tear down any pending tasks before closing.
            try:
                pending = asyncio.all_tasks(self._loop)
                for task in pending:
                    task.cancel()
            except RuntimeError:
                pass
            try:
                self._loop.close()
            except Exception:
                pass

    def submit(self, coro):
        """Schedule a coroutine on the background loop, return the concurrent ``Future``."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self) -> None:
        """Stop the engine and the background loop."""
        try:
            self.engine.shutdown()
        except Exception as exc:
            logging.warning(f"AsyncOmni.shutdown() raised: {exc!r}")
        try:
            self._loop.call_soon_threadsafe(self._loop.stop)
        except Exception:
            pass
        self._thread.join(timeout=10)

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # ``__del__`` may run during interpreter shutdown when globals
            # are torn down; suppress to avoid noisy tracebacks.
            pass


# ---------------------------------------------------------------------------
#  Per-stream session
# ---------------------------------------------------------------------------


class _Sentinel:
    """Marker placed on the sync output queues to signal end-of-stream or an error."""

    __slots__ = ("exc",)

    def __init__(self, exc: BaseException | None = None):
        self.exc = exc


_END_OF_STREAM = _Sentinel()


def _step_delta(value: Any, finished: bool):
    """Mirror of ``_step_delta`` in the vllm-omni example: pull the
    new-this-step multimodal chunk from an :class:`OmniStageOutput`'s
    ``multimodal_output`` value (which may be a tensor, a list of tensors,
    ``None``, or absent)."""
    if finished:
        return None
    if isinstance(value, torch.Tensor):
        return value if value.numel() > 0 else None
    if isinstance(value, list) and value:
        last = value[-1]
        return last if isinstance(last, torch.Tensor) and last.numel() > 0 else None
    return None


class OmniStreamingSession:
    """One streaming generation request driven step-by-step from sync code."""

    def __init__(
        self,
        runtime: OmniRuntime,
        request_id: str,
        system_prompt: str,
        speaker_latent: torch.Tensor,
        t_prefill: int,
        max_decode_steps: int,
        *,
        sampling_params: dict | None = None,
        step_timeout: float = 60.0,
    ) -> None:
        if speaker_latent is None or speaker_latent.numel() == 0:
            raise ValueError("speaker_latent is required for OmniStreamingSession")
        if t_prefill <= 0:
            raise ValueError(f"t_prefill must be > 0 (got {t_prefill})")

        self._runtime = runtime
        self.request_id = request_id
        self._step_timeout = step_timeout
        self._system_prompt = system_prompt
        self._speaker_latent = speaker_latent.detach().cpu().contiguous()
        self._t_prefill = int(t_prefill)
        self._max_decode_steps = int(max_decode_steps)
        self._sampling_overrides = dict(sampling_params or {})

        # Inter-thread queues. The sync side waits on these via .get().
        # Audio codes are accumulated into a per-session buffer that
        # ``drain_audio_codes`` returns and clears.
        from queue import Queue

        self._text_out_q: "Queue[tuple[int, int] | _Sentinel]" = Queue()
        self._audio_buf_lock = threading.Lock()
        self._audio_buf: list[torch.Tensor] = []

        self._closed = False
        self._error: BaseException | None = None
        self._loop = runtime._loop

        # Async-side queues (created on the loop)
        self._input_q: asyncio.Queue | None = None
        self._stage0_internal_q: asyncio.Queue | None = None

        self._consumer_future = runtime.submit(self._run_consumer())

    # ------------------------------------------------------------------ #
    #  Consumer (runs on the background event loop)                      #
    # ------------------------------------------------------------------ #

    async def _run_consumer(self) -> None:
        try:
            self._input_q = asyncio.Queue()
            self._stage0_internal_q = asyncio.Queue()

            from vllm import SamplingParams
            from vllm.engine.protocol import StreamingInput
            from vllm.sampling_params import RequestOutputKind

            stage0_params = SamplingParams(
                temperature=float(self._sampling_overrides.get("temperature", 0.0)),
                top_p=float(self._sampling_overrides.get("top_p", 1.0)),
                max_tokens=1,
                detokenize=False,
                ignore_eos=True,
                output_kind=RequestOutputKind.DELTA,
            )
            stage1_params = SamplingParams(
                temperature=0.0,
                top_p=1.0,
                max_tokens=self._max_decode_steps,
                detokenize=False,
                ignore_eos=True,
                output_kind=RequestOutputKind.DELTA,
            )

            prefill_chunk = StreamingInput(
                prompt={
                    "prompt_token_ids": [0] * self._t_prefill,
                    "additional_information": {
                        "system_prompt": self._system_prompt,
                        "speaker_latent": self._speaker_latent.clone(),
                    },
                },
                sampling_params=stage0_params,
            )

            async def input_generator():
                yield prefill_chunk
                while True:
                    item = await self._input_q.get()
                    if item is None:
                        return
                    ac_emb = item
                    # ``prev_tok`` comes from stage 0's prior output (or
                    # from prefill if this is the very first decode chunk).
                    prev_tok = await self._stage0_internal_q.get()
                    yield StreamingInput(
                        prompt={
                            "prompt_token_ids": [int(prev_tok)],
                            "additional_information": {
                                "system_prompt": None,
                                "acoustic_embedding": ac_emb,
                            },
                        },
                        sampling_params=stage0_params,
                    )

            stage0_count = 0
            async for stage_output in self._runtime.engine.generate(
                input_generator(),
                sampling_params_list=[stage0_params, stage1_params],
                request_id=self.request_id,
            ):
                sid = stage_output.stage_id
                req_out = stage_output.request_output
                mm = stage_output.multimodal_output or {}
                finished = bool(getattr(req_out, "finished", False))

                if sid == 0:
                    if req_out and req_out.outputs and req_out.outputs[0].token_ids:
                        text_tok = int(req_out.outputs[0].token_ids[-1])
                    else:
                        text_tok = 0
                    # Feed prev_tok into the input_generator side.
                    await self._stage0_internal_q.put(text_tok)

                    asr_delta = _step_delta(mm.get("asr_tokens"), finished)
                    asr_tok = int(asr_delta[-1].item()) if asr_delta is not None else 0

                    stage0_count += 1
                    # Skip the prefill's stage-0 output (t_0) -- it is fed
                    # back as ``prev_tok`` of the first decode chunk but is
                    # not a per-frame text token visible to the caller.
                    if stage0_count > 1:
                        self._text_out_q.put((text_tok, asr_tok))

                elif sid == 1:
                    audio = _step_delta(mm.get("audio_codes"), finished)
                    # using only acoustic tokens from decode
                    if audio is not None and audio.shape[0] == 1:
                        with self._audio_buf_lock:
                            self._audio_buf.append(audio.detach().cpu().to(torch.long))
        except asyncio.CancelledError:
            raise
        except BaseException as exc:
            self._error = exc
            self._text_out_q.put(_Sentinel(exc))
            raise
        finally:
            # Always close the sync side so a stuck step() unblocks.
            self._text_out_q.put(_END_OF_STREAM)

    # ------------------------------------------------------------------ #
    #  Sync API                                                          #
    # ------------------------------------------------------------------ #

    def step(self, acoustic_embedding: torch.Tensor) -> tuple[int, int]:
        """Submit one acoustic embedding and return its (text_tok, asr_tok)."""
        if self._closed:
            raise RuntimeError(f"OmniStreamingSession {self.request_id} is closed")

        ac_emb = acoustic_embedding.detach().cpu().contiguous()
        if ac_emb.dim() == 1:
            ac_emb = ac_emb.unsqueeze(0)
        elif ac_emb.dim() == 3:
            ac_emb = ac_emb.reshape(-1, ac_emb.shape[-1])
        if ac_emb.dim() != 2:
            raise ValueError(
                f"acoustic_embedding must be shapeable to 2D [n, hidden], got {tuple(acoustic_embedding.shape)}"
            )
        # vLLM-Omni expects float32 acoustic embeddings (matches the
        # Nemotron embed_tokens dtype at the input-merge point).
        ac_emb = ac_emb.to(torch.float32)

        asyncio.run_coroutine_threadsafe(self._input_q.put(ac_emb), self._loop).result()

        item = self._text_out_q.get(timeout=self._step_timeout)
        if isinstance(item, _Sentinel):
            if item.exc is not None:
                raise RuntimeError(
                    f"OmniStreamingSession {self.request_id} consumer raised"
                ) from item.exc
            raise RuntimeError(
                f"OmniStreamingSession {self.request_id} ended before producing a token"
            )
        return item

    def drain_audio_codes(self) -> list[torch.Tensor]:
        """Return all audio code chunks accumulated since the last drain."""
        with self._audio_buf_lock:
            out = self._audio_buf
            self._audio_buf = []
        return out

    def finish(self, *, drain_remaining_audio_s: float = 0.0) -> None:
        """Terminate the input stream gracefully and wait for the consumer
        to drain stage 1 (lagged audio frames).
        """
        if self._closed:
            return
        self._closed = True
        # Send the end-of-input sentinel so input_generator returns.
        try:
            asyncio.run_coroutine_threadsafe(self._input_q.put(None), self._loop).result(timeout=5)
        except Exception:
            pass
        # Wait for the consumer to finish; this also flushes any final
        # audio_codes into ``_audio_buf``.
        try:
            self._consumer_future.result(timeout=max(drain_remaining_audio_s, 5.0))
        except Exception as exc:
            logging.debug(f"OmniStreamingSession {self.request_id} consumer ended with: {exc!r}")
        # Drain the text queue so we don't leak the END_OF_STREAM sentinel.
        try:
            while True:
                self._text_out_q.get_nowait()
        except Exception:
            pass

    def abort(self) -> None:
        """Force-cancel the streaming generation, dropping any in-flight state."""
        if self._closed:
            return
        self._closed = True
        try:
            self._consumer_future.cancel()
        except Exception:
            pass
        # Best-effort engine-side abort.
        try:
            abort_coro = self._runtime.engine.abort(self.request_id)
            if asyncio.iscoroutine(abort_coro):
                asyncio.run_coroutine_threadsafe(abort_coro, self._loop).result(timeout=5)
        except Exception as exc:
            logging.debug(f"AsyncOmni.abort({self.request_id}) raised: {exc!r}")


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def load_speaker_latent(eartts_dir: str, speaker_name: str) -> torch.Tensor:
    """Load ``<eartts_dir>/speaker_latents/<speaker_name>.pt`` (saved by the
    EarTTS converter) and return a contiguous CPU tensor of shape
    ``(Tref, hidden_size)``.
    """
    latent_path = os.path.join(eartts_dir, "speaker_latents", f"{speaker_name}.pt")
    if not os.path.isfile(latent_path):
        raise FileNotFoundError(
            f"Speaker latent for '{speaker_name}' not found at {latent_path}. "
            "Either pick a speaker_name that is present in the EarTTS checkpoint, "
            "or re-run the EarTTS converter on a checkpoint that contains the "
            "requested audio_prompt_latents."
        )
    latent = torch.load(latent_path, weights_only=False)
    if isinstance(latent, torch.Tensor) and latent.dim() == 3:
        latent = latent[0]
    if not isinstance(latent, torch.Tensor) or latent.dim() != 2:
        raise ValueError(
            f"Expected speaker latent at {latent_path} to be a 2-D tensor [Tref, hidden], "
            f"got {type(latent).__name__} with shape "
            f"{tuple(latent.shape) if isinstance(latent, torch.Tensor) else 'n/a'}"
        )
    return latent.detach().to(torch.float32).cpu().contiguous()


def compute_prefill_len(model_dir: str, system_prompt: str) -> int:
    """Length of the prefill chunk fed to stage 0 (NemotronDuplexH) for a
    given system prompt. Mirrors the in-model tokenization:
    ``[BOS] + tokenizer.encode(prompt) + [EOS]``.
    """
    from transformers import AutoTokenizer

    from nemo.collections.speechlm2.inference.vllm_omni.nemotron_duplex_h.nemotron_duplex_h import (
        NemotronDuplexHForCausalLM,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    return NemotronDuplexHForCausalLM.compute_prefix_len(tokenizer, system_prompt)
