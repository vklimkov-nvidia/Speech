from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm_omni"
    / "engine"
    / "parallel_sampling_compat.py"
)
SPEC = importlib.util.spec_from_file_location("parallel_sampling_compat", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_make_parent_request_uses_vllm_021_request_signature() -> None:
    request = object()

    class ParentRequest021:
        def __init__(self, actual_request) -> None:
            self.request = actual_request

    parent = MODULE.make_parent_request(
        ParentRequest021,
        request_id="external-id",
        params=object(),
        request=request,
    )

    assert parent.request is request


def test_make_parent_request_falls_back_to_legacy_signature() -> None:
    params = object()

    class ParentRequestLegacy:
        def __init__(self, request_id, sampling_params) -> None:
            self.request_id = request_id
            self.sampling_params = sampling_params

    parent = MODULE.make_parent_request(
        ParentRequestLegacy,
        request_id="external-id",
        params=params,
        request=object(),
    )

    assert parent.request_id == "external-id"
    assert parent.sampling_params is params
