# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Focused tests for EasyMagpie's vLLM V1 serialization compatibility."""

from __future__ import annotations

import sys
import types
from typing import Any

import msgspec


def _install_fake_serial_utils(monkeypatch, engine_core_outputs_type):
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
    vllm_engine = types.ModuleType("vllm.v1.engine")
    vllm_engine.EngineCoreOutputs = engine_core_outputs_type
    vllm.v1 = vllm_v1
    vllm_v1.serial_utils = serial_utils
    vllm_v1.engine = vllm_engine
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "vllm.v1", vllm_v1)
    monkeypatch.setitem(sys.modules, "vllm.v1.engine", vllm_engine)
    monkeypatch.setitem(sys.modules, "vllm.v1.serial_utils", serial_utils)
    return serial_utils


class _DecoderWithoutTypeMetadata:
    def __init__(self, delegate):
        self._delegate = delegate
        self.ext_hook = delegate.ext_hook
        self.dec_hook = delegate.dec_hook

    def decode(self, data):
        return self._delegate.decode(data)


def test_decoder_drops_only_incompatible_array_scheduler_stats(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None
        timestamp: float = 0.0

    serial_utils = _install_fake_serial_utils(monkeypatch, EngineCoreOutputs)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(EngineCoreOutputs)
    decoder.decoder = _DecoderWithoutTypeMetadata(decoder.decoder)

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


def test_decoder_recovers_vllm_omni_engine_output_subclass(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None
        timestamp: float = 0.0

    class OmniEngineCoreOutputs(EngineCoreOutputs):
        outputs: list[int] = []

    serial_utils = _install_fake_serial_utils(monkeypatch, EngineCoreOutputs)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(OmniEngineCoreOutputs)

    incompatible_wire = msgspec.msgpack.encode([0, [7], [123], 4.5])
    recovered = decoder.decode([incompatible_wire])

    assert isinstance(recovered, OmniEngineCoreOutputs)
    assert recovered.outputs == [7]
    assert recovered.scheduler_stats is None
    assert recovered.timestamp == 4.5
    assert decoder._easymagpie_array_scheduler_stats_recoveries == 1


def test_decoder_does_not_hide_other_engine_output_schema_errors(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None

    serial_utils = _install_fake_serial_utils(monkeypatch, EngineCoreOutputs)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(EngineCoreOutputs)

    wrong_output_type = msgspec.msgpack.encode([0, "not-a-list", None])

    try:
        decoder.decode([wrong_output_type])
    except msgspec.ValidationError as exc:
        assert "at `$[1]`" in str(exc)
    else:
        raise AssertionError("unrelated EngineCoreOutputs schema errors must propagate")


def test_decoder_does_not_retype_unrelated_array_like_outputs(monkeypatch) -> None:
    from easymagpie_vllm_omni.vllm_compat import (
        _install_v1_serial_utils_dense_tensor_compat,
    )

    class EngineCoreOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None

    class OtherOutputs(msgspec.Struct, array_like=True, omit_defaults=True):
        engine_index: int = 0
        outputs: list[int] = []
        scheduler_stats: dict[str, int] | None = None

    serial_utils = _install_fake_serial_utils(monkeypatch, EngineCoreOutputs)
    _install_v1_serial_utils_dense_tensor_compat()
    decoder = serial_utils.MsgpackDecoder(OtherOutputs)

    incompatible_wire = msgspec.msgpack.encode([0, [7], [123]])

    try:
        decoder.decode([incompatible_wire])
    except msgspec.ValidationError as exc:
        assert "at `$[2]`" in str(exc)
    else:
        raise AssertionError("unrelated typed decoder errors must propagate")
