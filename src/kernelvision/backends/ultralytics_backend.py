"""Ultralytics model-execution backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kernelvision.config import Precision, Preprocessor, validate_preprocessor


class UltralyticsBackend:
    """Load and run an Ultralytics YOLO detector."""

    def __init__(
        self,
        model: str | Path,
        *,
        preprocessor: Preprocessor = "ultralytics",
    ) -> None:
        validate_preprocessor(preprocessor)
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Ultralytics is required for image inference. "
                "Install KernelVision with the 'inference' extra."
            ) from error

        self._model = YOLO(str(model), verbose=False)
        self._preprocessor = preprocessor

    def predict(
        self,
        image: Any,
        *,
        confidence: float,
        image_size: int,
        device: str,
        precision: Precision,
    ) -> Any:
        """Run detection for one image and return its Ultralytics result."""
        quantize = 16 if precision == "fp16" else 32
        if isinstance(image, Path):
            source = str(image)
        else:
            source = image
        prediction_arguments: dict[str, Any] = {
            "source": source,
            "conf": confidence,
            "imgsz": image_size,
            "device": device,
            "quantize": quantize,
            "verbose": False,
        }
        if self._preprocessor == "cuda_extension":
            normalized_device = device.strip().lower()
            if not (
                normalized_device == "cuda"
                or normalized_device.startswith("cuda:")
                or normalized_device.isdigit()
            ):
                raise ValueError(
                    "cuda_extension preprocessing requires a CUDA device"
                )
            from kernelvision.backends.cuda_extension_predictor import (
                get_cuda_extension_predictor_class,
            )

            prediction_arguments["predictor"] = (
                get_cuda_extension_predictor_class()
            )

        results = self._model.predict(
            **prediction_arguments,
        )

        if len(results) != 1:
            raise RuntimeError(f"expected 1 result, received {len(results)}")

        return results[0]
