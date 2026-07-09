# SPDX-License-Identifier: Apache-2.0
"""Compatibility helpers for vLLM parallel-sampling APIs."""

from typing import Any


def make_parent_request(
    parent_request_cls: type[Any],
    *,
    request_id: str,
    params: Any,
    request: Any,
) -> Any:
    """Construct ParentRequest across the vLLM 0.21 and legacy signatures."""

    try:
        return parent_request_cls(request)
    except TypeError as one_arg_error:
        try:
            return parent_request_cls(request_id, params)
        except TypeError:
            raise one_arg_error
