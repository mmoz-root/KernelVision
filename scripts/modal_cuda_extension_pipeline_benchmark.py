"""Compare standard and CUDA-extension YOLO preprocessing on a Modal L4."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
CUDA_DIR = PROJECT_ROOT / "csrc"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115", "ninja")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_dir(CUDA_DIR, remote_path="/root/csrc")
)

app = modal.App("kernelvision-cuda-extension-pipeline-benchmark")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_pipeline_l4(
    model: str,
    image_asset: str,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict[str, Any]:
    """Benchmark both preprocessors with execution order reversed."""
    import torch
    from ultralytics.utils import ASSETS

    from kernelvision.benchmarking.runner import (
        ImageBenchmarkConfig,
        run_image_benchmark,
    )
    from kernelvision.preprocessing import cuda_extension_preprocess

    # Compile before the controlled runs. Compilation is deployment/setup cost,
    # not steady-state inference latency.
    cuda_extension_preprocess(
        torch.zeros((2, 2, 3), dtype=torch.uint8, device="cuda"),
        output_dtype=torch.float32,
    )

    image = ASSETS / image_asset
    orders = (
        ("baseline_then_extension", ("ultralytics", "cuda_extension")),
        ("extension_then_baseline", ("cuda_extension", "ultralytics")),
    )
    runs = []
    for precision in ("fp32", "fp16"):
        for trial, preprocessors in orders:
            for order_position, preprocessor in enumerate(
                preprocessors,
                start=1,
            ):
                gc.collect()
                torch.cuda.empty_cache()
                config = ImageBenchmarkConfig(
                    model=model,
                    image=image,
                    device="0",
                    confidence=0.25,
                    image_size=640,
                    precision=precision,
                    preprocessor=preprocessor,
                    warmup_iterations=warmup_iterations,
                    measured_iterations=measured_iterations,
                )
                runs.append(
                    {
                        "precision": precision,
                        "trial": trial,
                        "order_position": order_position,
                        "preprocessor": preprocessor,
                        "report": run_image_benchmark(config).to_dict(),
                    }
                )

    return {
        "design": {
            "gpu": "NVIDIA L4",
            "model": model,
            "image": image_asset,
            "precision_cases": ["fp32", "fp16"],
            "warmup_iterations_per_run": warmup_iterations,
            "measured_iterations_per_run": measured_iterations,
            "orders": [list(preprocessors) for _, preprocessors in orders],
            "same_modal_container": True,
            "extension_compilation_excluded": True,
        },
        "runs": runs,
    }


def _comparisons(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Calculate extension changes relative to baseline for each trial."""
    comparisons = []
    keys = ("preprocess_ms", "backend_ms", "end_to_end_ms")
    precisions = dict.fromkeys(run["precision"] for run in runs)
    for precision in precisions:
        trials = dict.fromkeys(
            run["trial"] for run in runs if run["precision"] == precision
        )
        for trial in trials:
            pair = {
                run["preprocessor"]: run["report"]
                for run in runs
                if run["precision"] == precision and run["trial"] == trial
            }
            comparison: dict[str, Any] = {
                "precision": precision,
                "trial": trial,
            }
            for key in keys:
                baseline = pair["ultralytics"]["summary_ms"][key]["median"]
                extension = pair["cuda_extension"]["summary_ms"][key][
                    "median"
                ]
                comparison[key] = {
                    "baseline_median": baseline,
                    "extension_median": extension,
                    "extension_change_ms": extension - baseline,
                    "extension_change_percent": (
                        ((extension / baseline) - 1.0) * 100.0
                    ),
                }
            comparisons.append(comparison)
    return comparisons


@app.local_entrypoint()
def main(
    model: str = "yolov8n.pt",
    image_asset: str = "bus.jpg",
    warmup: int = 30,
    iterations: int = 200,
    json_out: str = "results/modal_l4_cuda_extension_pipeline_benchmark.json",
) -> None:
    """Run the paired benchmark and save all raw measurements."""
    report = benchmark_pipeline_l4.remote(
        model,
        image_asset,
        warmup,
        iterations,
    )
    report["comparisons"] = _comparisons(report["runs"])
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for comparison in report["comparisons"]:
        preprocess = comparison["preprocess_ms"]
        end_to_end = comparison["end_to_end_ms"]
        print(
            f"{comparison['precision']} {comparison['trial']}: "
            f"preprocess {preprocess['baseline_median']:.3f} -> "
            f"{preprocess['extension_median']:.3f} ms "
            f"({preprocess['extension_change_percent']:+.2f}%); "
            f"end-to-end {end_to_end['baseline_median']:.3f} -> "
            f"{end_to_end['extension_median']:.3f} ms "
            f"({end_to_end['extension_change_percent']:+.2f}%)"
        )
    print(f"Saved pipeline benchmark report to {output}")


if __name__ == "__main__":
    main()
