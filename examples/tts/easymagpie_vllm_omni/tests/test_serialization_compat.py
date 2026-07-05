# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused tests for EasyMagpie's vLLM V1 serialization compatibility."""

from __future__ import annotations

import sys
import types
from typing import Any

import msgspec


def _install_fake_serial_utils(monkeypatch):
    class MsgpackEncoder:
        size_threshold = 256

        def _encode_tensor(self, obj):
            return obj

    class MsgpackDecoder:
        def __init__(self, target_type: Any = None):
            args = () if target_type is None else (target_type,)
            self.decoder = msgspec.msgpack.Decoder(
                *args,
                ext_hook=self.ext_hook,
                dec_hook=self.dec_hook,
            )
            self.aux_buffers = ()

        def decode(self, bufs):
            if isinstance(bufs, (bytes, bytearray, memoryview)):
                return self.decoder.decode(bufs)
            self.aux_buffers = bufs
            try:
                return self.decoder.decode(bufs[0])
            finally:
                self.aux_buffers = ()

        @staticmethod
        def ext_hook(_code, data):
            return data

        @staticmethod
        def dec_hook(_target_type, obj):
            return obj

        def _decode_tensor(self, obj):
            return ("decoded_tensor", obj)

    serial_utils = types.ModuleType("vllm.v1.serial_utils")
    serial_utils.CUSTOM_TYPE_RAW_VIEW = 1
    serial_utils.MsgpackDecoder = MsgpackDecoder
    serial_utils.MsgpackEncoder = MsgpackEncoder
    vllm = types.ModuleType("vllm")
    vllm_v1 = types.ModuleType("vllm.v1")
    vllm.v1 = vllm_v1
    vllm_v1.serial_utils = serial_utils
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.serial_utils", serial_utils)
    return serial_utils


def test_decoder_drops_only_incompatible_array_scheduler_stats(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None
        timestamp: float = 0.0

    serial_utils = _install_fake_serial_utils(monkeypatch)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(EngineCoreOutputs)

    incompatible_wire = msgspec.msgpack.encode([0, [7], [123], 4.5])
    recovered = decoder.decode([incompatible_wire])

    assert recovered.outputs == [7]
    assert recovered.scheduler_stats is None
    assert recovered.timestamp == 4.5
    assert decoder._easymagpie_array_scheduler_stats_recoveries == 1

    compatible_wire = msgspec.msgpack.encode([0, [8], {"running": 3}, 5.5])
    compatible = decoder.decode([compatible_wire])

    assert compatible.outputs == [8]
    assert compatible.scheduler_stats == {"running": 3}
    assert compatible.timestamp == 5.5
    assert decoder._easymagpie_array_scheduler_stats_recoveries == 1


def test_decoder_does_not_hide_other_engine_output_schema_errors(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None

    serial_utils = _install_fake_serial_utils(monkeypatch)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(EngineCoreOutputs)

    wrong_output_type = msgspec.msgpack.encode([0, "not-a-list", None])

    try:
        decoder.decode([wrong_output_type])
    except msgspec.ValidationError as exc:
        assert "at `$[1]`" in str(exc)
    else:
        raise AssertionError("unrelated EngineCoreOutputs schema errors must propagate")
