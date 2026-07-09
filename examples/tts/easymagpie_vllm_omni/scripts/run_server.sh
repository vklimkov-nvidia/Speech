#!/bin/bash
# Launch a vLLM-Omni server for the EasyMagpieTTS two-stage pipeline
# (talker -> code2wav), fully in-engine (no external codec service).
#
# Usage:
#   ./run_server.sh <converted_model_dir> [port]
#
# <converted_model_dir> is produced by scripts/easy_magpietts_convert_to_vllm.py
# (with --bundle_codec, the default, so the codec ships under <dir>/codec/).
set -e

MODEL="${1:?Usage: run_server.sh <converted_model_dir> [port]}"
PORT="${2:-8091}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_CONFIG="${SCRIPT_DIR}/../deploy/easymagpie.yaml"

echo "Starting EasyMagpieTTS server: model=${MODEL} deploy=${DEPLOY_CONFIG} port=${PORT}"

# The vllm_plugin_easymagpie_omni entry point registers the model archs and the
# EASYMAGPIE_PIPELINE. --trust-remote-code is required for the Nemotron-H config.
vllm-omni serve "$MODEL" \
    --deploy-config "$DEPLOY_CONFIG" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --trust-remote-code \
    --omni
