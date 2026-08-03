"""Configuration objects for KernelVision workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ImageInferenceConfig:
    """Validated settings for one-image object-detection inference."""

    model: str
    image: Path
    output: Path
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 640

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty path or model name")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")


@dataclass(frozen=True, slots=True)
class VideoInferenceConfig:
    """Validated settings for object detection over one video."""

    model: str
    video: Path
    output: Path
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 640

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty path or model name")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
