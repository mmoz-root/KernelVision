"""Run the KernelVision baseline benchmark on a Modal NVIDIA L4."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics>=8.3,<9")
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

app = modal.App("kernelvision-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_l4(
    model: str,
    image_asset: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Execute the baseline benchmark inside a Modal L4 container."""
    from ultralytics.utils import ASSETS

    from kernelvision.benchmarking.runner import (
        ImageBenchmarkConfig,
        run_image_benchmark,
    )

    image = ASSETS / image_asset
    config = ImageBenchmarkConfig(
        model=model,
        image=image,
        device="0",
        confidence=0.25,
        image_size=640,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    return run_image_benchmark(config).to_dict()


def _save_remote_report(
    report: dict[str, Any],
    json_output: Path,
    csv_output: Path,
) -> None:
    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    samples = report["samples"]
    if not samples:
        raise RuntimeError("Modal benchmark returned no raw samples")
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    with csv_output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)


@app.local_entrypoint()
def main(
    model: str = "yolov8n.pt",
    image_asset: str = "bus.jpg",
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_baseline.json",
    csv_out: str = "benchmarks/raw/modal_l4_baseline.csv",
) -> None:
    """Launch the L4 benchmark and persist its report on the local machine."""
    report = benchmark_l4.remote(model, image_asset, warmup, iterations)
    json_output = Path(json_out)
    csv_output = Path(csv_out)
    _save_remote_report(report, json_output, csv_output)

    end_to_end = report["summary_ms"]["end_to_end_ms"]
    inference = report["summary_ms"]["inference_ms"]
    print(
        f"L4 end-to-end median={end_to_end['median']:.3f} ms, "
        f"P95={end_to_end['p95']:.3f} ms; "
        f"inference median={inference['median']:.3f} ms; "
        f"throughput={report['throughput_fps']:.3f} FPS"
    )
    print(f"Saved JSON report to {json_output}")
    print(f"Saved raw CSV samples to {csv_output}")
