#!/bin/bash
# Launch the EasyMagpieTTS two-stage pipeline (talker -> code2wav) with the
# stock `vllm serve` CLI and serve it over the OpenAI-compatible speech API
# (POST /v1/audio/speech), fully in-engine (no external codec service).
#
# The vllm_plugin_easymagpie_omni entry point registers the model archs and the
# EASYMAGPIE_PIPELINE, and installs a TTS serving adapter so /v1/audio/speech
# accepts EasyMagpie requests and streams raw audio chunk-by-chunk.
#
# Requires vLLM / vLLM-Omni 0.24+ (see the `emp24` env).
#
# Usage:
#   ./run_thin_server.sh <converted_model_dir> [port]
#
# <converted_model_dir> is produced by scripts/easy_magpietts_convert_to_vllm.py
# (with --bundle_codec, the default, so the codec ships under <dir>/codec/).
#
# Query it with scripts/run_thin_service_request.ipynb.
set -e

MODEL="${1:?Usage: run_thin_server.sh <converted_model_dir> [port]}"
PORT="${2:-8091}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie.yaml"

echo "Starting EasyMagpieTTS via vllm serve: model=${MODEL} deploy=${DEPLOY_CONFIG} port=${PORT}"

# VLLM_PLUGINS ensures the EasyMagpie plugin (model + pipeline + TTS adapter) is
# loaded in the API-server process. --trust-remote-code is required for the
# Nemotron-H config; --omni enables the multi-stage pipeline runtime.
#
# Per-request StageRequestStats tables and Uvicorn access lines are disabled:
# formatting/writing them is noisy and can distort high-concurrency benchmarks.
# Warnings and errors remain visible.
VLLM_PLUGINS=easymagpie_omni vllm serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --disable-log-stats \
    --disable-uvicorn-access-log \
    --uvicorn-log-level warning \
    --omni
