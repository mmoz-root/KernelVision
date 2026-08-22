"""Portable validation for TensorRT backend setup errors."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from kernelvision.backends.tensorrt_backend import TensorRTBackend


def test_missing_engine_is_rejected_before_optional_import(tmp_path) -> None:
    missing_engine = tmp_path / "missing.engine"

    with pytest.raises(FileNotFoundError, match="engine not found"):
        TensorRTBackend(missing_engine)


def test_missing_tensorrt_dependency_has_actionable_error(
    tmp_path,
    monkeypatch,
) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"placeholder")
    monkeypatch.setitem(sys.modules, "tensorrt", None)

    with pytest.raises(RuntimeError, match="TensorRT and PyTorch"):
        TensorRTBackend(engine)


def test_cpu_device_is_rejected(tmp_path, monkeypatch) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"placeholder")
    fake_torch = SimpleNamespace(
        device=lambda value: SimpleNamespace(type=value),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(ValueError, match="requires a CUDA device"):
        TensorRTBackend(engine, device="cpu")


def test_unavailable_cuda_device_is_rejected(tmp_path, monkeypatch) -> None:
    engine = tmp_path / "model.engine"
    engine.write_bytes(b"placeholder")
    fake_torch = SimpleNamespace(
        device=lambda value: SimpleNamespace(type=value),
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "tensorrt", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    with pytest.raises(RuntimeError, match="available CUDA device"):
        TensorRTBackend(engine)
