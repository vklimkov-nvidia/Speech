#!/bin/bash
# Launch the EasyMagpieTTS *thin* HTTP server (in-process AsyncOmni wrapper).
#
# Unlike run_server.sh (which uses `vllm-omni serve`), this serves EasyMagpie via
# a minimal FastAPI app around AsyncOmni -- the same known-good pipeline as
# scripts/synthesize_two_stage.py -- so it can be benchmarked over HTTP without
# patching vLLM-Omni.
#
# Usage:
#   ./run_thin_server.sh <converted_model_dir> [port]
#
# <converted_model_dir> is produced by scripts/easy_magpietts_convert_to_vllm.py
# (with --bundle_codec, the default, so the codec ships under <dir>/codec/).
set -e

MODEL="${1:?Usage: run_thin_server.sh <converted_model_dir> [port]}"
PORT="${2:-8091}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting EasyMagpieTTS thin server: model=${MODEL} port=${PORT}"

python "${SCRIPT_DIR}/tts_server.py" \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT"
