"""Benchmark PyTorch and Triton preprocessing on a Modal NVIDIA L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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

app = modal.App("kernelvision-triton-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_l4(
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Run GPU-event microbenchmarks and Triton launch experiments."""
    import torch

    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        torch_reference_preprocess,
        triton_preprocess,
    )

    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")

    def measure_gpu_ms(operation: Callable[[], Any]) -> list[float]:
        result = None
        for _ in range(warmup_iterations):
            result = operation()
        torch.cuda.synchronize()

        event_pairs = [
            (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            for _ in range(measured_iterations)
        ]
        for start, end in event_pairs:
            start.record()
            result = operation()
            end.record()
        torch.cuda.synchronize()
        if result is None:
            raise RuntimeError("benchmark operation did not produce an output")
        return [float(start.elapsed_time(end)) for start, end in event_pairs]

    torch.manual_seed(0)
    shapes = ((320, 320), (384, 640), (640, 640), (720, 1280))
    dtypes = (("fp32", torch.float32, 4), ("fp16", torch.float16, 2))
    block_sizes = (128, 256, 512, 1024)
    warp_counts = (2, 4, 8)
    cases = []
    for height, width in shapes:
        image = torch.randint(
            0,
            256,
            (height, width, 3),
            dtype=torch.uint8,
            device="cuda",
        )
        for dtype_name, output_dtype, output_element_bytes in dtypes:
            reference_operation = lambda: torch_reference_preprocess(
                image,
                output_dtype=output_dtype,
            )
            reference_before_samples = measure_gpu_ms(reference_operation)
            triton_runs = []
            for block_size in block_sizes:
                for num_warps in warp_counts:
                    operation = lambda block_size=block_size, num_warps=num_warps: triton_preprocess(
                        image,
                        output_dtype=output_dtype,
                        block_size=block_size,
                        num_warps=num_warps,
                    )
                    samples = measure_gpu_ms(operation)
                    summary = summarize(samples).to_dict()
                    transferred_bytes = height * width * 3 * (
                        1 + output_element_bytes
                    )
                    median_seconds = float(summary["median"]) / 1000.0
                    triton_runs.append(
                        {
                            "block_size": block_size,
                            "num_warps": num_warps,
                            "summary_ms": summary,
                            "effective_bandwidth_gbps": (
                                transferred_bytes / median_seconds / 1e9
                            ),
                            "samples_ms": samples,
                        }
                    )

            reference_after_samples = measure_gpu_ms(reference_operation)
            reference_samples = (
                reference_before_samples + reference_after_samples
            )
            reference_summary = summarize(reference_samples).to_dict()
            best_triton = min(
                triton_runs,
                key=lambda run: float(run["summary_ms"]["median"]),
            )
            reference_median = float(reference_summary["median"])
            triton_median = float(best_triton["summary_ms"]["median"])
            cases.append(
                {
                    "height": height,
                    "width": width,
                    "dtype": dtype_name,
                    "input_bytes": height * width * 3,
                    "output_bytes": (
                        height * width * 3 * output_element_bytes
                    ),
                    "pytorch": {
                        "summary_ms": reference_summary,
                        "before_samples_ms": reference_before_samples,
                        "after_samples_ms": reference_after_samples,
                    },
                    "triton_runs": triton_runs,
                    "best_triton": {
                        "block_size": best_triton["block_size"],
                        "num_warps": best_triton["num_warps"],
                        "summary_ms": best_triton["summary_ms"],
                        "effective_bandwidth_gbps": best_triton[
                            "effective_bandwidth_gbps"
                        ],
                    },
                    "speedup_over_pytorch": reference_median / triton_median,
                    "latency_reduction_percent": (
                        1.0 - triton_median / reference_median
                    )
                    * 100.0,
                }
            )

    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": "NVIDIA L4",
            "warmup_iterations_per_operation": warmup_iterations,
            "measured_iterations_per_operation": measured_iterations,
            "timing": "CUDA events; first-use compilation excluded",
            "scope": (
                "GPU preprocessing only; input already on CUDA; resize, "
                "letterbox, host-to-device transfer, and Python wall time excluded"
            ),
            "allocation": "steady-state output/intermediate allocations included",
            "environment": collect_environment(),
        },
        "cases": cases,
    }


@app.local_entrypoint()
def main(
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_triton_preprocess_benchmark.json",
) -> None:
    """Launch the parameter experiment and save raw measurements."""
    report = benchmark_l4.remote(warmup, iterations)
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    for case in report["cases"]:
        pytorch_median = case["pytorch"]["summary_ms"]["median"]
        triton_median = case["best_triton"]["summary_ms"]["median"]
        print(
            f"{case['height']}x{case['width']} {case['dtype']}: "
            f"PyTorch={pytorch_median:.6f} ms, "
            f"Triton={triton_median:.6f} ms, "
            f"speedup={case['speedup_over_pytorch']:.2f}x, "
            f"block={case['best_triton']['block_size']}, "
            f"warps={case['best_triton']['num_warps']}"
        )
    print(f"Saved Triton benchmark report to {output}")
