"""Run order-reversed FP32/FP16 benchmarks in one Modal L4 container."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
)

app = modal.App("kernelvision-precision-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_pair_l4(
    model: str,
    image_asset: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Benchmark both precisions twice with their execution order reversed."""
    import torch
    from ultralytics.utils import ASSETS

    from kernelvision.benchmarking.runner import (
        ImageBenchmarkConfig,
        run_image_benchmark,
    )

    image = ASSETS / image_asset
    orders = (
        ("fp32_then_fp16", ("fp32", "fp16")),
        ("fp16_then_fp32", ("fp16", "fp32")),
    )
    runs = []
    for trial, precisions in orders:
        for order_position, precision in enumerate(precisions, start=1):
            gc.collect()
            torch.cuda.empty_cache()
            config = ImageBenchmarkConfig(
                model=model,
                image=image,
                device="0",
                confidence=0.25,
                image_size=640,
                precision=precision,
                warmup_iterations=warmup_iterations,
                measured_iterations=measured_iterations,
            )
            runs.append(
                {
                    "trial": trial,
                    "order_position": order_position,
                    "precision": precision,
                    "report": run_image_benchmark(config).to_dict(),
                }
            )

    return {
        "design": {
            "gpu": "NVIDIA L4",
            "model": model,
            "image": image_asset,
            "warmup_iterations_per_run": warmup_iterations,
            "measured_iterations_per_run": measured_iterations,
            "orders": [list(precisions) for _, precisions in orders],
            "same_modal_container": True,
        },
        "runs": runs,
    }


def _trial_comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate FP16 changes relative to FP32 for each execution order."""
    comparisons = []
    for trial in dict.fromkeys(run["trial"] for run in runs):
        trial_runs = {
            run["precision"]: run["report"]
            for run in runs
            if run["trial"] == trial
        }
        fp32 = trial_runs["fp32"]
        fp16 = trial_runs["fp16"]
        fp32_inference = fp32["summary_ms"]["inference_ms"]["median"]
        fp16_inference = fp16["summary_ms"]["inference_ms"]["median"]
        fp32_end_to_end = fp32["summary_ms"]["end_to_end_ms"]["median"]
        fp16_end_to_end = fp16["summary_ms"]["end_to_end_ms"]["median"]
        comparisons.append(
            {
                "trial": trial,
                "fp32_inference_median_ms": fp32_inference,
                "fp16_inference_median_ms": fp16_inference,
                "fp16_inference_latency_change_percent": (
                    (fp16_inference / fp32_inference) - 1.0
                )
                * 100.0,
                "fp32_end_to_end_median_ms": fp32_end_to_end,
                "fp16_end_to_end_median_ms": fp16_end_to_end,
                "fp16_end_to_end_latency_change_percent": (
                    (fp16_end_to_end / fp32_end_to_end) - 1.0
                )
                * 100.0,
                "fp32_throughput_fps": fp32["throughput_fps"],
                "fp16_throughput_fps": fp16["throughput_fps"],
                "fp16_throughput_change_percent": (
                    (fp16["throughput_fps"] / fp32["throughput_fps"]) - 1.0
                )
                * 100.0,
            }
        )
    return comparisons


@app.local_entrypoint()
def main(
    model: str = "yolov8n.pt",
    image_asset: str = "bus.jpg",
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_fp32_vs_fp16_performance.json",
) -> None:
    """Launch paired L4 benchmarks and save all summaries and raw samples."""
    report = benchmark_pair_l4.remote(model, image_asset, warmup, iterations)
    report["comparisons"] = _trial_comparisons(report["runs"])

    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for comparison in report["comparisons"]:
        print(
            f"{comparison['trial']}: inference FP32="
            f"{comparison['fp32_inference_median_ms']:.3f} ms, FP16="
            f"{comparison['fp16_inference_median_ms']:.3f} ms, change="
            f"{comparison['fp16_inference_latency_change_percent']:+.2f}%; "
            "end-to-end change="
            f"{comparison['fp16_end_to_end_latency_change_percent']:+.2f}%; "
            "throughput change="
            f"{comparison['fp16_throughput_change_percent']:+.2f}%"
        )
    print(f"Saved paired performance report to {output}")
