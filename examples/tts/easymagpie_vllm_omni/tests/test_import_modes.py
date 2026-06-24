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
"""Import-mode tests for EasyMagpie vLLM-Omni startup hooks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


def test_compat_mode_off_does_not_import_vllm() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["EASYMAGPIE_VLLM_COMPAT_MODE"] = "off"
    env["PYTHONPATH"] = str(root)

    code = """
import json
import sys
import easymagpie_vllm_omni
loaded = sorted(name for name in sys.modules if name == "vllm" or name.startswith("vllm."))
print(json.dumps(loaded))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=30,
    )
    loaded = json.loads(proc.stdout.strip())
    assert loaded == []
