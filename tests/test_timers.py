"""Tests for device-aware benchmark synchronization."""

import sys
from types import SimpleNamespace

import pytest

from kernelvision.benchmarking.timers import synchronize_device


class FakeAccelerator:
    """Expose availability and record synchronization calls."""

    def __init__(self, available: bool) -> None:
        self.available = available
        self.synchronize_calls = 0

    def is_available(self) -> bool:
        return self.available

    def synchronize(self) -> None:
        self.synchronize_calls += 1


def _fake_torch(*, cuda_available: bool, mps_available: bool) -> SimpleNamespace:
    cuda = FakeAccelerator(cuda_available)
    mps = FakeAccelerator(mps_available)
    return SimpleNamespace(
        cuda=cuda,
        mps=mps,
        backends=SimpleNamespace(mps=mps),
    )


def test_cpu_does_not_require_torch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)

    synchronize_device(" CPU ")


def test_mps_synchronizes_the_mps_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    torch = _fake_torch(cuda_available=False, mps_available=True)
    monkeypatch.setitem(sys.modules, "torch", torch)

    synchronize_device("mps")

    assert torch.mps.synchronize_calls == 1
    assert torch.cuda.synchronize_calls == 0


@pytest.mark.parametrize("device", ["cuda", "cuda:0", "0"])
def test_cuda_device_forms_synchronize_cuda(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _fake_torch(cuda_available=True, mps_available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    synchronize_device(device)

    assert torch.cuda.synchronize_calls == 1
    assert torch.mps.synchronize_calls == 0


@pytest.mark.parametrize("device", ["mps", "cuda", "0"])
def test_unavailable_accelerator_fails_loudly(
    device: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = _fake_torch(cuda_available=False, mps_available=False)
    monkeypatch.setitem(sys.modules, "torch", torch)

    with pytest.raises(RuntimeError, match="unavailable"):
        synchronize_device(device)


def test_unsupported_device_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported device"):
        synchronize_device("tpu")
