"""Tests for the Ultralytics model-execution boundary."""

from pathlib import Path
from typing import Any

import pytest

from kernelvision.backends.ultralytics_backend import UltralyticsBackend


class FakeModel:
    """Record prediction arguments and return configurable fake results."""

    def __init__(self, results: list[Any]) -> None:
        self.results = results
        self.predict_kwargs: dict[str, Any] = {}

    def predict(self, **kwargs: Any) -> list[Any]:
        self.predict_kwargs = kwargs
        return self.results


def _backend_with_model(model: FakeModel) -> UltralyticsBackend:
    backend = object.__new__(UltralyticsBackend)
    backend._model = model
    return backend


def test_predict_forwards_settings_and_returns_one_result() -> None:
    expected_result = object()
    model = FakeModel([expected_result])
    backend = _backend_with_model(model)

    result = backend.predict(
        Path("input.jpg"),
        confidence=0.35,
        image_size=640,
        device="cpu",
        precision="fp16",
    )

    assert result is expected_result
    assert model.predict_kwargs == {
        "source": "input.jpg",
        "conf": 0.35,
        "imgsz": 640,
        "device": "cpu",
        "quantize": 16,
        "verbose": False,
    }


def test_predict_rejects_an_unexpected_result_count() -> None:
    backend = _backend_with_model(FakeModel([]))

    with pytest.raises(RuntimeError, match="expected 1 result"):
        backend.predict(
            Path("input.jpg"),
            confidence=0.25,
            image_size=640,
            device="cpu",
            precision="fp32",
        )


def test_predict_maps_fp32_to_explicit_32_bit_quantization() -> None:
    model = FakeModel([object()])
    backend = _backend_with_model(model)

    backend.predict(
        Path("input.jpg"),
        confidence=0.25,
        image_size=640,
        device="cpu",
        precision="fp32",
    )

    assert model.predict_kwargs["quantize"] == 32
