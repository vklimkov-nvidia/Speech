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
"""EasyMagpieTTS model definition for vLLM-Omni.

This package provides an inference-only re-implementation of EasyMagpieTTS
(decoder-only, Nemotron-H hybrid-Mamba backbone + autoregressive local
transformer over the stacked audio codebooks) that plugs into the vLLM-Omni
serving stack via the standard ``preprocess`` / ``postprocess`` /
``make_omni_output`` hooks.

The companion ``vllm_plugin_easymagpie_omni`` package registers the model with
vLLM's ``ModelRegistry`` through the ``vllm.general_plugins`` entry point.
"""

import os

_compat_mode = os.environ.get("EASYMAGPIE_VLLM_COMPAT_MODE", "refit").strip().lower()
if _compat_mode not in {"0", "false", "none", "off"}:
    from easymagpie_vllm_omni.vllm_compat import (
        install_easy_magpie_refit_rpc_compat,
        install_easy_magpie_runtime_compat,
        install_vllm_omni_compat,
    )

    if _compat_mode in {"full", "all"}:
        install_vllm_omni_compat()
    elif _compat_mode in {"serial", "runtime", "rl"}:
        install_easy_magpie_runtime_compat()
    else:
        install_easy_magpie_refit_rpc_compat()

from easymagpie_vllm_omni.config import EASYMAGPIE_SMALLMAMBA, EasyMagpieOmniArch

__all__ = ["EASYMAGPIE_SMALLMAMBA", "EasyMagpieOmniArch"]
