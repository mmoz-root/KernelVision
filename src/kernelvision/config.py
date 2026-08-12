"""Configuration objects for KernelVision workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal


Precision = Literal["fp32", "fp16"]
Preprocessor = Literal["ultralytics", "cuda_extension"]


def validate_precision(precision: str) -> None:
    """Reject precision modes that KernelVision does not benchmark."""
    if precision not in ("fp32", "fp16"):
        raise ValueError("precision must be either 'fp32' or 'fp16'")


def validate_preprocessor(preprocessor: str) -> None:
    """Reject preprocessing implementations KernelVision cannot select."""
    if preprocessor not in ("ultralytics", "cuda_extension"):
        raise ValueError(
            "preprocessor must be either 'ultralytics' or 'cuda_extension'"
        )


@dataclass(frozen=True, slots=True)
class ImageInferenceConfig:
    """Validated settings for one-image object-detection inference."""

    model: str
    image: Path
    output: Path
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 640
    precision: Precision = "fp32"
    preprocessor: Preprocessor = "ultralytics"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty path or model name")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        validate_precision(self.precision)
        validate_preprocessor(self.preprocessor)


@dataclass(frozen=True, slots=True)
class VideoInferenceConfig:
    """Validated settings for object detection over one video."""

    model: str
    video: Path
    output: Path
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 640
    precision: Precision = "fp32"
    preprocessor: Preprocessor = "ultralytics"

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty path or model name")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        validate_precision(self.precision)
        validate_preprocessor(self.preprocessor)
