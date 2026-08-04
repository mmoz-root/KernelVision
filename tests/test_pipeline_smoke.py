"""Smoke tests for inference pipeline orchestration."""

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from kernelvision.config import ImageInferenceConfig, VideoInferenceConfig
from kernelvision.pipeline import run_image_inference, run_video_inference


class FakeResult:
    """Create a small stand-in for an annotated image."""

    def save(self, filename: str | None = None, *args: Any, **kwargs: Any) -> str:
        assert filename is not None
        Path(filename).write_bytes(b"annotated image")
        return filename

    def plot(self) -> bytes:
        return b"annotated frame"


class FakeBackend:
    """Record the pipeline's inference request without loading a model."""

    def __init__(self) -> None:
        self.request: dict[str, Any] = {}
        self.requests: list[dict[str, Any]] = []

    def predict(
        self,
        image: Path,
        *,
        confidence: float,
        image_size: int,
        device: str,
        precision: str,
    ) -> FakeResult:
        self.request = {
            "image": image,
            "confidence": confidence,
            "image_size": image_size,
            "device": device,
            "precision": precision,
        }
        self.requests.append(self.request)
        return FakeResult()


class FakeCapture:
    """Provide deterministic decoded frames for the video pipeline."""

    def __init__(self, frames: list[Any]) -> None:
        self.frames = iter(frames)
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {1: 24.0, 2: 1280.0, 3: 720.0}[property_id]

    def read(self) -> tuple[bool, Any | None]:
        frame = next(self.frames, None)
        return (frame is not None, frame)

    def release(self) -> None:
        self.released = True


class FakeWriter:
    """Record encoded frames and create a stand-in output video."""

    def __init__(self, output: str) -> None:
        self.output = Path(output)
        self.frames: list[Any] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: Any) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        self.output.write_bytes(b"annotated video")
        self.released = True


def test_image_pipeline_saves_output_without_a_real_model(tmp_path: Path) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"input image")
    output = tmp_path / "nested" / "annotated.jpg"
    backend = FakeBackend()
    config = ImageInferenceConfig(
        model="fake.pt",
        image=image,
        output=output,
        device="cpu",
        confidence=0.4,
        image_size=320,
    )

    saved_path = run_image_inference(config, backend=backend)

    assert saved_path == output
    assert output.read_bytes() == b"annotated image"
    assert backend.request == {
        "image": image,
        "confidence": 0.4,
        "image_size": 320,
        "device": "cpu",
        "precision": "fp32",
    }


def test_image_pipeline_rejects_a_missing_input(tmp_path: Path) -> None:
    config = ImageInferenceConfig(
        model="fake.pt",
        image=tmp_path / "missing.jpg",
        output=tmp_path / "output.jpg",
    )

    with pytest.raises(FileNotFoundError, match="input image does not exist"):
        run_image_inference(config, backend=FakeBackend())


def test_video_pipeline_decodes_predicts_and_encodes_frames(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"input video")
    output = tmp_path / "output.mp4"
    decoded_frames = [object(), object()]
    capture = FakeCapture(decoded_frames)
    writer = FakeWriter(str(output))
    fake_cv2 = SimpleNamespace(
        CAP_PROP_FPS=1,
        CAP_PROP_FRAME_WIDTH=2,
        CAP_PROP_FRAME_HEIGHT=3,
        VideoCapture=lambda path: capture,
        VideoWriter_fourcc=lambda *codec: 0,
        VideoWriter=lambda path, codec, fps, size: writer,
    )
    monkeypatch.setitem(sys.modules, "cv2", fake_cv2)
    backend = FakeBackend()
    config = VideoInferenceConfig(
        model="fake.pt",
        video=video,
        output=output,
        device="cpu",
        confidence=0.3,
        image_size=320,
    )

    summary = run_video_inference(config, backend=backend)

    assert summary.output == output
    assert summary.frames == 2
    assert summary.source_fps == 24.0
    assert [request["image"] for request in backend.requests] == decoded_frames
    assert writer.frames == [b"annotated frame", b"annotated frame"]
    assert capture.released
    assert writer.released
