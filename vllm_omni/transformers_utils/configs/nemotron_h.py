"""Nemotron-H config registration for EasyMagpie exports.

The EOS vLLM runtime can execute Nemotron-H models, but its paired
Transformers build does not always know the ``nemotron_h`` model_type. vLLM
asks Transformers to parse ``config.json`` before dispatching to the custom
EasyMagpie model class, so a lightweight config class is enough here.
"""

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class NemotronHConfig(PretrainedConfig):
    model_type = "nemotron_h"


try:
    AutoConfig.register(NemotronHConfig.model_type, NemotronHConfig)
except ValueError:
    # Already registered by the installed stack.
    pass
