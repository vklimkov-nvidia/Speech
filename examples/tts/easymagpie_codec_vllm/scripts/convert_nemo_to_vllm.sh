#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CODEC.nemo OUTPUT_DIR [convert_codec.py options]" >&2
    exit 2
fi

codec_nemo=$1
output_dir=$2
shift 2

script_dir=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
plugin_dir=$(realpath "${script_dir}/..")
codec_conda_env=${EASYMAGPIE_VLLM_ENV:-easymagpie-vllm}
plugin_pythonpath=${plugin_dir}
if [[ -n ${PYTHONPATH:-} ]]; then
    plugin_pythonpath=${plugin_pythonpath}:${PYTHONPATH}
fi

PYTHONPATH=${plugin_pythonpath} conda run -n "${codec_conda_env}" python \
    "${script_dir}/convert_codec.py" "${codec_nemo}" "${output_dir}" "$@"
