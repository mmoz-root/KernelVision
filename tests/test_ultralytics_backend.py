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


def test_constructor_rejects_unknown_preprocessor() -> None:
    with pytest.raises(ValueError, match="preprocessor must be either"):
        UltralyticsBackend(
            "model.pt",
            preprocessor="unknown",  # type: ignore[arg-type]
        )


def _backend_with_model(
    model: FakeModel,
    *,
    preprocessor: str = "ultralytics",
) -> UltralyticsBackend:
    backend = object.__new__(UltralyticsBackend)
    backend._model = model
    backend._preprocessor = preprocessor
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


def test_predict_forwards_cuda_extension_predictor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakePredictor:
        pass

    monkeypatch.setattr(
        "kernelvision.backends.cuda_extension_predictor."
        "get_cuda_extension_predictor_class",
        lambda: FakePredictor,
    )
    model = FakeModel([object()])
    backend = _backend_with_model(model, preprocessor="cuda_extension")

    backend.predict(
        Path("input.jpg"),
        confidence=0.25,
        image_size=640,
        device="0",
        precision="fp16",
    )

    assert model.predict_kwargs["predictor"] is FakePredictor


def test_cuda_extension_preprocessor_rejects_cpu() -> None:
    backend = _backend_with_model(
        FakeModel([object()]),
        preprocessor="cuda_extension",
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        backend.predict(
            Path("input.jpg"),
            confidence=0.25,
            image_size=640,
            device="cpu",
            precision="fp32",
        )
