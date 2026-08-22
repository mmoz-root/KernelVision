"""Benchmark PyTorch FP32/FP16 and TensorRT FP16 model execution on L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from typing import Any, Callable

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
PYTORCH_MODEL = PROJECT_ROOT / "yolov8n.pt"
FP16_ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp16.engine"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "ultralytics==8.4.115",
        "tensorrt>=11.2,<11.3",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_file(PYTORCH_MODEL, remote_path="/root/yolov8n.pt")
    .add_local_file(
        FP16_ENGINE_MODEL,
        remote_path="/root/yolov8n_fp16.engine",
    )
)

app = modal.App("kernelvision-tensorrt-model-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_l4(
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Run a correctness-gated, position-balanced model-only benchmark."""
    import torch
    from ultralytics import YOLO

    from kernelvision.backends import TensorRTBackend
    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment

    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"

    element_count = 3 * 640 * 640
    input_fp32 = (
        torch.arange(element_count, dtype=torch.int32, device="cuda") % 256
    ).to(torch.float32).div(255.0).reshape(1, 3, 640, 640)
    input_fp16 = input_fp32.to(torch.float16)

    pytorch_fp32 = YOLO("/root/yolov8n.pt", verbose=False).model
    pytorch_fp32 = pytorch_fp32.to("cuda").eval()

    pytorch_fp16 = YOLO("/root/yolov8n.pt", verbose=False).model
    pytorch_fp16 = pytorch_fp16.to("cuda").eval().half()

    tensorrt_fp16 = TensorRTBackend(
        "/root/yolov8n_fp16.engine",
        device="cuda",
    )

    default_stream = torch.cuda.current_stream()

    # Learning exercise: map each label to (operation, execution stream).
    # Each operation must perform only one raw model forward pass.
    operations: dict[str, tuple[Callable[[], Any], Any]] = {
        "pytorch_fp32": (
            lambda: pytorch_fp32(input_fp32)[0],
            default_stream
        ),
        "pytorch_fp16": (
            lambda: pytorch_fp16(input_fp16)[0],
            default_stream
        ),
        "tensorrt_fp16": (
            lambda: tensorrt_fp16.infer(input_fp16),
            tensorrt_fp16.stream
        )
    }

    def measure_one(
        operation: Callable[[], Any],
        stream: Any,
    ) -> tuple[Any, float]:
        """Measure one asynchronous GPU operation with CUDA events."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record(stream)
        output = operation()
        end.record(stream)

        end.synchronize()
        elapsed_ms = float(start.elapsed_time(end))
        return output, elapsed_ms

    labels = tuple(operations)
    if set(labels) != {"pytorch_fp32", "pytorch_fp16", "tensorrt_fp16"}:
        raise RuntimeError("benchmark operations are incomplete")

    orders = list(permutations(labels))
    samples_ms = {label: [] for label in labels}
    last_outputs: dict[str, Any] = {}

    with torch.inference_mode():
        for iteration in range(warmup_iterations):
            order = orders[iteration % len(orders)]
            for label in order:
                operation, _ = operations[label]
                last_outputs[label] = operation()
        torch.cuda.synchronize()

        for iteration in range(measured_iterations):
            order = orders[iteration % len(orders)]
            for label in order:
                torch.cuda.synchronize()
                operation, stream = operations[label]
                output, elapsed_ms = measure_one(operation, stream)
                last_outputs[label] = output
                samples_ms[label].append(elapsed_ms)

    torch.cuda.synchronize()
    summaries = {
        label: summarize(samples).to_dict()
        for label, samples in samples_ms.items()
    }
    trt_median = float(summaries["tensorrt_fp16"]["median"])

    return {
        "experiment": "PyTorch versus TensorRT model-only benchmark",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "input_shape": list(input_fp32.shape),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "timing_scope": "raw model execution only",
        "engine_loading_included": False,
        "preprocessing_included": False,
        "postprocessing_included": False,
        "position_balanced": True,
        "orders": [list(order) for order in orders],
        "summaries_ms": summaries,
        "samples_ms": samples_ms,
        "median_speedup_over_tensorrt": {
            label: float(summary["median"]) / trt_median
            for label, summary in summaries.items()
        },
        "environment": collect_environment(),
    }


@app.local_entrypoint()
def main(
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_tensorrt_model_benchmark.json",
) -> None:
    """Run the model-only benchmark and save its complete raw report."""
    report = benchmark_l4.remote(warmup, iterations)
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries_ms"], indent=2, sort_keys=True))
    print(f"Saved model-only benchmark report to {output}")


if __name__ == "__main__":
    main()
