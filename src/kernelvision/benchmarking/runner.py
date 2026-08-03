"""Benchmark execution for the baseline image-inference pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter_ns
from typing import Any

from kernelvision.backends import UltralyticsBackend
from kernelvision.benchmarking.report import (
    BenchmarkReport,
    BenchmarkSample,
    build_report,
)
from kernelvision.benchmarking.timers import synchronize_device
from kernelvision.environment import collect_environment


@dataclass(frozen=True, slots=True)
class ImageBenchmarkConfig:
    """Validated settings for repeated baseline image inference."""

    model: str
    image: Path
    device: str = "cpu"
    confidence: float = 0.25
    image_size: int = 640
    warmup_iterations: int = 30
    measured_iterations: int = 200

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("model must be a non-empty path or model name")
        if not self.device.strip():
            raise ValueError("device must be non-empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations cannot be negative")
        if self.measured_iterations <= 0:
            raise ValueError("measured_iterations must be greater than zero")


def _elapsed_ms(start_ns: int) -> float:
    return (perf_counter_ns() - start_ns) / 1_000_000.0


def _is_cuda_device(device: str) -> bool:
    normalized = device.strip().lower()
    return (
        normalized == "cuda"
        or normalized.startswith("cuda:")
        or normalized.isdigit()
    )


def _cuda_device_index(device: str) -> int | None:
    normalized = device.strip().lower()
    if normalized.isdigit():
        return int(normalized)
    if normalized.startswith("cuda:"):
        return int(normalized.split(":", maxsplit=1)[1])
    return None


def _reset_peak_gpu_memory(device: str) -> None:
    if not _is_cuda_device(device):
        return
    import torch

    torch.cuda.reset_peak_memory_stats(_cuda_device_index(device))


def _peak_gpu_memory_mb(device: str) -> float | None:
    if not _is_cuda_device(device):
        return None
    import torch

    allocated_bytes = torch.cuda.max_memory_allocated(_cuda_device_index(device))
    return allocated_bytes / (1024.0**2)


def _stage_speeds(result: Any) -> tuple[float, float, float]:
    speed = getattr(result, "speed", None)
    required = ("preprocess", "inference", "postprocess")
    if not isinstance(speed, dict) or any(name not in speed for name in required):
        raise RuntimeError(
            "backend result did not provide preprocess, inference, and "
            "postprocess timings"
        )
    return tuple(float(speed[name]) for name in required)


def run_image_benchmark(
    config: ImageBenchmarkConfig,
    backend: Any | None = None,
    cv2_module: Any | None = None,
) -> BenchmarkReport:
    """Run warmed, repeated image inference and return raw measurements."""
    if not config.image.is_file():
        raise FileNotFoundError(f"input image does not exist: {config.image}")

    if cv2_module is None:
        try:
            import cv2 as cv2_module
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "OpenCV is required for benchmarking. "
                "Install KernelVision with the 'inference' extra."
            ) from error

    validation_frame = cv2_module.imread(str(config.image))
    if validation_frame is None:
        raise RuntimeError(f"could not decode input image: {config.image}")

    selected_backend = backend or UltralyticsBackend(config.model)
    for _ in range(config.warmup_iterations):
        warmup_result = selected_backend.predict(
            validation_frame,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
        )
        warmup_result.plot()
    synchronize_device(config.device)
    _reset_peak_gpu_memory(config.device)

    samples: list[BenchmarkSample] = []
    for iteration in range(1, config.measured_iterations + 1):
        synchronize_device(config.device)
        end_to_end_start = perf_counter_ns()

        decode_start = perf_counter_ns()
        frame = cv2_module.imread(str(config.image))
        decode_ms = _elapsed_ms(decode_start)
        if frame is None:
            raise RuntimeError(f"could not decode input image: {config.image}")

        synchronize_device(config.device)
        backend_start = perf_counter_ns()
        result = selected_backend.predict(
            frame,
            confidence=config.confidence,
            image_size=config.image_size,
            device=config.device,
        )
        synchronize_device(config.device)
        backend_ms = _elapsed_ms(backend_start)
        preprocess_ms, inference_ms, postprocess_ms = _stage_speeds(result)

        visualization_start = perf_counter_ns()
        result.plot()
        visualization_ms = _elapsed_ms(visualization_start)

        synchronize_device(config.device)
        end_to_end_ms = _elapsed_ms(end_to_end_start)
        samples.append(
            BenchmarkSample(
                iteration=iteration,
                decode_ms=decode_ms,
                preprocess_ms=preprocess_ms,
                host_to_device_ms=None,
                inference_ms=inference_ms,
                postprocess_ms=postprocess_ms,
                backend_ms=backend_ms,
                visualization_ms=visualization_ms,
                end_to_end_ms=end_to_end_ms,
            )
        )

    synchronize_device(config.device)
    peak_gpu_memory_mb = _peak_gpu_memory_mb(config.device)
    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "backend": "ultralytics",
        "model": config.model,
        "image": str(config.image),
        "source_shape_hwc": list(validation_frame.shape),
        "model_input_size": config.image_size,
        "batch_size": 1,
        "dtype": "float32",
        "device": config.device,
        "confidence": config.confidence,
        "warmup_iterations": config.warmup_iterations,
        "measured_iterations": config.measured_iterations,
        "peak_gpu_memory_allocated_mb": peak_gpu_memory_mb,
        "host_to_device_timing": (
            "included in Ultralytics preprocess_ms; not separately exposed"
        ),
        "allocation_included": True,
        "visualization_included_in_end_to_end": True,
        "file_output_included_in_end_to_end": False,
        "environment": collect_environment(),
    }
    return build_report(metadata, samples)
