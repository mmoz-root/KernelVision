"""Explore naive CUDA block sizes on a Modal NVIDIA L4."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

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
    .add_local_dir(CUDA_DIR, remote_path="/root/csrc")
)

app = modal.App("kernelvision-cuda-block-size-experiment")


def _run_checked(
    command: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command while retaining useful compiler/runtime diagnostics."""
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_native_samples(path: Path) -> list[float]:
    """Read per-launch milliseconds emitted by the CUDA executable."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [float(row["latency_ms"]) for row in rows]


def _balanced_orders(block_sizes: Sequence[int]) -> list[list[int]]:
    """Rotate unique block sizes so each occupies every position once."""
    sizes = list(block_sizes)
    if not sizes:
        raise ValueError("at least one block size is required")
    if len(set(sizes)) != len(sizes):
        raise ValueError("block sizes must be unique")
    return [sizes[offset:] + sizes[:offset] for offset in range(len(sizes))]


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def sweep_l4(
    warmup_iterations: int,
    measured_iterations: int,
    launches_per_sample: int,
    block_sizes: list[int],
    git_commit: str,
    git_worktree_dirty: bool,
) -> dict[str, Any]:
    """Correctness-gate and time one kernel across CUDA block sizes."""
    import torch

    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        deterministic_bgr_image,
        read_standalone_output,
        torch_reference_preprocess,
    )

    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")
    if launches_per_sample < 1:
        raise ValueError("launches_per_sample must be positive")
    if 256 not in block_sizes:
        raise ValueError("block sizes must include the 256-thread baseline")
    if any(size <= 0 or size > 1024 for size in block_sizes):
        raise ValueError("every block size must be in [1, 1024]")
    if any(size % 32 for size in block_sizes):
        raise ValueError("every block size must be a multiple of one warp")

    round_orders = _balanced_orders(block_sizes)
    cuda_source = Path("/root/csrc/preprocessing/standalone_preprocess.cu")
    binary = Path("/tmp/kernelvision_cuda_block_size_experiment")
    compile_command = [
        "nvcc",
        "-O3",
        "--std=c++17",
        "-lineinfo",
        str(cuda_source),
        "-o",
        str(binary),
    ]
    _run_checked(compile_command)
    nvcc_version = _run_checked(["nvcc", "--version"]).stdout.strip()

    def run_native(
        *,
        block_size: int,
        input_path: Path,
        output_path: Path,
        samples_path: Path,
        warmup: int,
        iterations: int,
        launches: int,
    ) -> list[float]:
        command = [
            str(binary),
            "--height",
            "640",
            "--width",
            "640",
            "--dtype",
            "fp32",
            "--block-size",
            str(block_size),
            "--warmup",
            str(warmup),
            "--iterations",
            str(iterations),
            "--launches-per-sample",
            str(launches),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--samples",
            str(samples_path),
        ]
        _run_checked(command)
        samples = _read_native_samples(samples_path)
        if len(samples) != iterations:
            raise RuntimeError("native CUDA sample count is incorrect")
        return samples

    height = 640
    width = 640
    dtype_name = "fp32"
    tolerance = 1e-7
    cpu_image = deterministic_bgr_image(
        height,
        width,
        torch_module=torch,
        device="cpu",
    )
    input_bytes = cpu_image.numpy().tobytes(order="C")
    image = cpu_image.to(device="cuda")
    expected = torch_reference_preprocess(
        image,
        output_dtype=torch.float32,
    )

    correctness_by_block: dict[int, dict[str, Any]] = {}
    rounds_by_block: dict[int, list[dict[str, Any]]] = {
        block_size: [] for block_size in block_sizes
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        input_path = temporary / "input_640x640.bin"
        input_path.write_bytes(input_bytes)

        for block_size in block_sizes:
            output_path = temporary / f"correctness_block_{block_size}.bin"
            samples_path = temporary / f"correctness_block_{block_size}.csv"
            run_native(
                block_size=block_size,
                input_path=input_path,
                output_path=output_path,
                samples_path=samples_path,
                warmup=1,
                iterations=1,
                launches=1,
            )
            actual = read_standalone_output(
                output_path,
                height=height,
                width=width,
                dtype_name=dtype_name,
                torch_module=torch,
            ).to(device="cuda")
            difference = (actual.float() - expected.float()).abs()
            maximum_difference = float(difference.max().item())
            mean_difference = float(difference.mean().item())
            mismatched_values = int(
                (difference > tolerance).sum().item()
            )
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=tolerance,
            )
            correctness_by_block[block_size] = {
                "passed": True,
                "tolerance": tolerance,
                "maximum_absolute_difference": maximum_difference,
                "mean_absolute_difference": mean_difference,
                "mismatched_values": mismatched_values,
            }

        for round_id, order in enumerate(round_orders, start=1):
            for position, block_size in enumerate(order, start=1):
                output_path = temporary / (
                    f"round_{round_id}_block_{block_size}.bin"
                )
                samples_path = temporary / (
                    f"round_{round_id}_block_{block_size}.csv"
                )
                samples = run_native(
                    block_size=block_size,
                    input_path=input_path,
                    output_path=output_path,
                    samples_path=samples_path,
                    warmup=warmup_iterations,
                    iterations=measured_iterations,
                    launches=launches_per_sample,
                )
                rounds_by_block[block_size].append(
                    {
                        "round_id": round_id,
                        "order": order,
                        "position": position,
                        "summary_ms": summarize(samples).to_dict(),
                        "samples_ms": samples,
                    }
                )

    configurations = []
    for block_size in block_sizes:
        rounds = rounds_by_block[block_size]
        combined_samples = [
            sample
            for round_result in rounds
            for sample in round_result["samples_ms"]
        ]
        round_medians = [
            float(round_result["summary_ms"]["median"])
            for round_result in rounds
        ]
        configurations.append(
            {
                "block_size": block_size,
                "threads_per_block": block_size,
                "warps_per_block": block_size // 32,
                "grid_blocks": (height * width + block_size - 1)
                // block_size,
                "correctness": correctness_by_block[block_size],
                "summary_ms": summarize(combined_samples).to_dict(),
                "round_median_span_ms": (
                    max(round_medians) - min(round_medians)
                ),
                "rounds": rounds,
            }
        )

    baseline = next(
        configuration
        for configuration in configurations
        if configuration["block_size"] == 256
    )
    baseline_median = float(baseline["summary_ms"]["median"])
    for configuration in configurations:
        candidate_median = float(configuration["summary_ms"]["median"])
        configuration["comparison_with_256"] = {
            "speedup_over_256": baseline_median / candidate_median,
            "saving_vs_256_ms": baseline_median - candidate_median,
        }

    nominal_best = min(
        configurations,
        key=lambda configuration: float(
            configuration["summary_ms"]["median"]
        ),
    )
    nominal_best_median = float(nominal_best["summary_ms"]["median"])
    observed_round_span = max(
        float(baseline["round_median_span_ms"]),
        float(nominal_best["round_median_span_ms"]),
    )

    return {
        "schema_version": 1,
        "experiment": "naive CUDA exploratory block-size sweep",
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "gpu": torch.cuda.get_device_name(),
            "height": height,
            "width": width,
            "dtype": dtype_name,
            "warmup_iterations_per_configuration": warmup_iterations,
            "measured_samples_per_round": measured_iterations,
            "launches_per_sample": launches_per_sample,
            "round_orders": round_orders,
            "compile_command": compile_command,
            "nvcc_version": nvcc_version,
            "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
            "cuda_source_sha256": hashlib.sha256(
                cuda_source.read_bytes()
            ).hexdigest(),
            "timing": (
                "native CUDA events; repeated launches per interval; elapsed "
                "interval divided by launch count"
            ),
            "scope": (
                "warm-cache standalone CUDA only; input/output resident; H2D, "
                "D2H, compilation, context creation, subprocess startup, and "
                "outer wall time excluded"
            ),
            "selection_policy": (
                "exploratory only; nominal winner requires fresh confirmation "
                "against the declared 256-thread baseline"
            ),
            "environment": collect_environment(),
        },
        "configurations": configurations,
        "nominal_best": {
            "block_size": nominal_best["block_size"],
            "median_ms": nominal_best_median,
            "saving_vs_256_ms": baseline_median - nominal_best_median,
            "observed_round_median_span_ms": observed_round_span,
            "saving_exceeds_observed_round_span": (
                baseline_median - nominal_best_median
            )
            > observed_round_span,
            "confirmed": False,
        },
    }


def _write_raw_csv(report: dict[str, Any], path: Path) -> None:
    """Flatten every exploratory timing sample into one CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "height",
        "width",
        "dtype",
        "block_size",
        "warps_per_block",
        "round_id",
        "order",
        "position",
        "sample_index",
        "latency_ms",
        "launches_per_sample",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for configuration in report["configurations"]:
            for round_result in configuration["rounds"]:
                for sample_index, latency_ms in enumerate(
                    round_result["samples_ms"]
                ):
                    writer.writerow(
                        {
                            "height": report["metadata"]["height"],
                            "width": report["metadata"]["width"],
                            "dtype": report["metadata"]["dtype"],
                            "block_size": configuration["block_size"],
                            "warps_per_block": configuration[
                                "warps_per_block"
                            ],
                            "round_id": round_result["round_id"],
                            "order": ">".join(
                                str(value) for value in round_result["order"]
                            ),
                            "position": round_result["position"],
                            "sample_index": sample_index,
                            "latency_ms": latency_ms,
                            "launches_per_sample": report["metadata"][
                                "launches_per_sample"
                            ],
                        }
                    )


@app.local_entrypoint()
def main(
    warmup: int = 30,
    iterations: int = 200,
    launches_per_sample: int = 100,
    block_sizes: str = "128,256,512,1024",
    json_out: str = "results/modal_l4_cuda_block_size_experiment.json",
    csv_out: str = "benchmarks/raw/modal_l4_cuda_block_size_experiment.csv",
) -> None:
    """Run the exploratory sweep and save summary plus raw measurements."""
    parsed_block_sizes = [
        int(value.strip())
        for value in block_sizes.split(",")
        if value.strip()
    ]
    git_commit = _run_checked(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
    ).stdout.strip()
    git_worktree_dirty = bool(
        _run_checked(
            ["git", "status", "--short"],
            cwd=PROJECT_ROOT,
        ).stdout.strip()
    )
    report = sweep_l4.remote(
        warmup,
        iterations,
        launches_per_sample,
        parsed_block_sizes,
        git_commit,
        git_worktree_dirty,
    )
    report["metadata"]["orchestrator_sha256"] = hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest()

    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = Path(csv_out)
    _write_raw_csv(report, csv_path)

    for configuration in report["configurations"]:
        print(
            f"block={configuration['block_size']}: "
            f"median={configuration['summary_ms']['median']:.9f} ms, "
            f"round-span={configuration['round_median_span_ms']:.9f} ms"
        )
    print(
        "Nominal exploratory winner: "
        f"block={report['nominal_best']['block_size']} "
        f"(confirmed={report['nominal_best']['confirmed']})"
    )
    print(f"Saved experiment report to {json_path}")
    print(f"Saved raw samples to {csv_path}")
