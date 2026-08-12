"""Inference pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from kernelvision.backends import UltralyticsBackend
from kernelvision.config import ImageInferenceConfig, Precision, VideoInferenceConfig


class DetectionResult(Protocol):
    """Result behavior required by the image pipeline."""

    def save(self, filename: str | None = None, *args: Any, **kwargs: Any) -> str:
        """Save an annotated result and return its filename."""
        ...


class ImageBackend(Protocol):
    """Backend behavior required by the image pipeline."""

    def predict(
        self,
        image: Any,
        *,
        confidence: float,
        image_size: int,
        device: str,
        precision: Precision,
    ) -> DetectionResult:
        """Run object detection for one image."""
        ...


@dataclass(frozen=True, slots=True)
class VideoInferenceSummary:
    """Summary of an annotated-video inference run."""

    output: Path
    frames: int
    source_fps: float


def run_image_inference(
    config: ImageInferenceConfig,
    backend: ImageBackend | None = None,
) -> Path:
    """Run one-image inference and save its annotated result."""
    if not config.image.is_file():
        raise FileNotFoundError(f"input image does not exist: {config.image}")

    config.output.parent.mkdir(parents=True, exist_ok=True)
    selected_backend = backend or UltralyticsBackend(
        config.model,
        preprocessor=config.preprocessor,
    )
    result = selected_backend.predict(
        config.image,
        confidence=config.confidence,
        image_size=config.image_size,
        device=config.device,
        precision=config.precision,
    )
    result.save(filename=str(config.output))

    if not config.output.is_file():
        raise RuntimeError(f"annotated image was not created: {config.output}")
    return config.output


def run_video_inference(
    config: VideoInferenceConfig,
    backend: ImageBackend | None = None,
) -> VideoInferenceSummary:
    """Decode, annotate, and encode every frame in one video."""
    if not config.video.is_file():
        raise FileNotFoundError(f"input video does not exist: {config.video}")

    try:
        import cv2
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "OpenCV is required for video inference. "
            "Install KernelVision with the 'inference' extra."
        ) from error

    capture = cv2.VideoCapture(str(config.video))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"could not open input video: {config.video}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(
            "input video has invalid metadata: "
            f"fps={source_fps}, width={width}, height={height}"
        )

    config.output.parent.mkdir(parents=True, exist_ok=True)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(config.output), codec, source_fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        writer.release()
        raise RuntimeError(f"could not open output video: {config.output}")

    selected_backend = backend or UltralyticsBackend(
        config.model,
        preprocessor=config.preprocessor,
    )
    frame_count = 0
    try:
        while True:
            decoded, frame = capture.read()
            if not decoded:
                break
            result = selected_backend.predict(
                frame,
                confidence=config.confidence,
                image_size=config.image_size,
                device=config.device,
                precision=config.precision,
            )
            writer.write(result.plot())
            frame_count += 1
    finally:
        capture.release()
        writer.release()

    if frame_count == 0:
        raise RuntimeError(f"input video contained no decodable frames: {config.video}")
    if not config.output.is_file():
        raise RuntimeError(f"annotated video was not created: {config.output}")

    return VideoInferenceSummary(
        output=config.output,
        frames=frame_count,
        source_fps=source_fps,
    )
