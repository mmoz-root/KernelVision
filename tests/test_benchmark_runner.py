"""Smoke tests for the baseline image benchmark runner."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from kernelvision.benchmarking.runner import (
    ImageBenchmarkConfig,
    run_image_benchmark,
)


class FakeFrame:
    shape = (480, 640, 3)


class FakeResult:
    speed = {"preprocess": 1.0, "inference": 2.0, "postprocess": 3.0}

    def plot(self) -> object:
        return object()


class FakeBackend:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def predict(self, frame: Any, **kwargs: Any) -> FakeResult:
        self.calls.append({"frame": frame, **kwargs})
        return FakeResult()


def test_runner_warms_up_and_records_each_measured_iteration(
    tmp_path: Path,
) -> None:
    image = tmp_path / "input.jpg"
    image.write_bytes(b"image")
    cv2 = SimpleNamespace(imread=lambda path: FakeFrame())
    backend = FakeBackend()
    config = ImageBenchmarkConfig(
        model="fake.pt",
        image=image,
        device="cpu",
        confidence=0.4,
        image_size=320,
        precision="fp16",
        warmup_iterations=2,
        measured_iterations=3,
    )

    report = run_image_benchmark(config, backend=backend, cv2_module=cv2)

    assert len(backend.calls) == 5
    assert len(report.samples) == 3
    assert report.summary_ms["inference_ms"]["median"] == 2.0
    assert report.summary_ms["end_to_end_ms"]["count"] == 3
    assert report.metadata["warmup_iterations"] == 2
    assert report.metadata["measured_iterations"] == 3
    assert report.metadata["dtype"] == "fp16"
    assert all(call["precision"] == "fp16" for call in backend.calls)
    assert report.metadata["host_to_device_timing"].startswith("included")
    assert report.throughput_fps > 0
