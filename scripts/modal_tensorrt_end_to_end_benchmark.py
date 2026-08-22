"""Benchmark PyTorch and TensorRT complete image pipelines on Modal L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from itertools import permutations
from pathlib import Path
from time import perf_counter_ns
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

app = modal.App("kernelvision-tensorrt-end-to-end-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_l4(
    image_asset: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Benchmark file decode through rendered detections."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment

    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")

    image_path = ASSETS / image_asset
    if not image_path.is_file():
        raise FileNotFoundError(f"image asset does not exist: {image_path}")

    pytorch_model = YOLO(
        "/root/yolov8n.pt",
        task="detect",
        verbose=False,
    )
    tensorrt_model = YOLO(
        "/root/yolov8n_fp16.engine",
        task="detect",
        verbose=False,
    )

    def run_pipeline(model: Any, *, pytorch: bool) -> Any:
        """Decode, detect, apply NMS, and render one image."""
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"could not decode image: {image_path}")

        prediction_arguments: dict[str, Any] = {
            "source": frame,
            "imgsz": 640,
            "conf": 0.25,
            "rect": False,
            "device": "0",
            "verbose": False,
        }

        if pytorch:
            prediction_arguments["quantize"] = 32

        results = model.predict(**prediction_arguments)

        if len(results) != 1:
            raise RuntimeError(
                f"expected one result, received {len(results)}"
            )

        return results[0].plot()

    operations: dict[str, Callable[[], Any]] = {
        "pytorch_fp32": lambda: run_pipeline(
            pytorch_model,
            pytorch=True,
        ),
        "tensorrt_fp16": lambda: run_pipeline(
            tensorrt_model,
            pytorch=False,
        ),
    }

    def measure_wall_ms(operation: Callable[[], Any]) -> tuple[Any, float]:
        """Measure synchronized CPU and GPU pipeline latency."""
        torch.cuda.synchronize()
        start_ns = perf_counter_ns()

        output = operation()

        torch.cuda.synchronize()
        elapsed_ms = (perf_counter_ns() - start_ns) / 1_000_000.0
        return output, elapsed_ms

    labels = tuple(operations)
    orders = list(permutations(labels))
    samples_ms = {label: [] for label in labels}
    last_outputs: dict[str, Any] = {}

    for iteration in range(warmup_iterations):
        order = orders[iteration % len(orders)]
        for label in order:
            last_outputs[label] = operations[label]()
    torch.cuda.synchronize()

    for iteration in range(measured_iterations):
        order = orders[iteration % len(orders)]
        for label in order:
            output, elapsed_ms = measure_wall_ms(operations[label])
            last_outputs[label] = output
            samples_ms[label].append(elapsed_ms)

    summaries = {
        label: summarize(samples).to_dict()
        for label, samples in samples_ms.items()
    }
    pytorch_median = float(summaries["pytorch_fp32"]["median"])
    tensorrt_median = float(summaries["tensorrt_fp16"]["median"])

    return {
        "experiment": "PyTorch versus TensorRT end-to-end benchmark",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "image": image_asset,
        "image_size": 640,
        "confidence": 0.25,
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "timing_scope": (
            "image decode, preprocessing, host-to-device transfer, model, "
            "NMS, and visualization; excludes file output"
        ),
        "model_loading_included": False,
        "file_output_included": False,
        "position_balanced": True,
        "orders": [list(order) for order in orders],
        "summaries_ms": summaries,
        "samples_ms": samples_ms,
        "tensorrt_median_speedup": pytorch_median / tensorrt_median,
        "environment": collect_environment(),
    }


@app.local_entrypoint()
def main(
    image_asset: str = "bus.jpg",
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_tensorrt_end_to_end_benchmark.json",
) -> None:
    """Run the full-pipeline benchmark and save every raw sample."""
    report = benchmark_l4.remote(image_asset, warmup, iterations)
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summaries_ms"], indent=2, sort_keys=True))
    print(
        "TensorRT median speedup: "
        f"{report['tensorrt_median_speedup']:.3f}x"
    )
    print(f"Saved end-to-end benchmark report to {output}")


if __name__ == "__main__":
    main()
