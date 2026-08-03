"""Ultralytics model-execution backend."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class UltralyticsBackend:
    """Load and run an Ultralytics YOLO detector."""

    def __init__(self, model: str | Path) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Ultralytics is required for image inference. "
                "Install KernelVision with the 'inference' extra."
            ) from error

        self._model = YOLO(str(model), verbose=False)

    def predict(
        self,
        image: Any,
        *,
        confidence: float,
        image_size: int,
        device: str,
    ) -> Any:
        """Run detection for one image and return its Ultralytics result."""
        if isinstance(image, Path):
            source = str(image)
        else:
            source = image
        results = self._model.predict(
            source=source,
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )

        if len(results) != 1:
            raise RuntimeError(f"expected 1 result, received {len(results)}")

        return results[0]
