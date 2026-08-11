"""Benchmark reference, Triton, and CUDA preprocessing variants on an L4."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

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

app = modal.App("kernelvision-cuda-optimization-benchmark")


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run a native command and preserve compiler/runtime diagnostics."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_native_samples(path: Path) -> list[float]:
    """Read per-launch milliseconds emitted by the CUDA executable."""
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return [float(row["latency_ms"]) for row in rows]


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def benchmark_l4(
    warmup_iterations: int,
    measured_iterations: int,
    launches_per_sample: int,
    block_size: int,
    num_warps: int,
    git_commit: str,
    git_worktree_dirty: bool,
) -> dict[str, Any]:
    """Correctness-gate and benchmark all five implementations."""
    import torch

    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        deterministic_bgr_image,
        read_standalone_output,
        torch_reference_preprocess,
        triton_preprocess_into,
    )

    if warmup_iterations < 1:
        raise ValueError("warmup_iterations must be positive")
    if measured_iterations < 1:
        raise ValueError("measured_iterations must be positive")
    if launches_per_sample < 1:
        raise ValueError("launches_per_sample must be positive")
    if block_size <= 0 or block_size > 1024:
        raise ValueError("block_size must be in [1, 1024]")
    if num_warps not in (1, 2, 4, 8):
        raise ValueError("num_warps must be one of 1, 2, 4, or 8")

    cuda_source = Path("/root/csrc/preprocessing/standalone_preprocess.cu")
    triton_source = Path(
        "/root/src/kernelvision/preprocessing/triton_kernel.py"
    )
    binary = Path("/tmp/kernelvision_cuda_optimization_benchmark")
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

    def measure_torch_gpu_ms(
        operation: Callable[[], Any],
    ) -> list[float]:
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
        for start, stop in event_pairs:
            start.record()
            for _ in range(launches_per_sample):
                result = operation()
            stop.record()
        torch.cuda.synchronize()
        if result is None:
            raise RuntimeError("benchmark operation did not produce output")
        return [
            float(start.elapsed_time(stop)) / launches_per_sample
            for start, stop in event_pairs
        ]

    def run_native(
        *,
        height: int,
        width: int,
        dtype_name: str,
        input_path: Path,
        output_path: Path,
        samples_path: Path,
        implementation: str,
        warmup: int,
        iterations: int,
        launches: int,
    ) -> list[float]:
        command = [
            str(binary),
            "--height",
            str(height),
            "--width",
            str(width),
            "--dtype",
            dtype_name,
            "--block-size",
            str(block_size),
            "--implementation",
            implementation,
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

    shapes = ((320, 320), (384, 640), (640, 640), (720, 1280))
    dtype_cases = (
        ("fp32", torch.float32, 4, 1e-7),
        ("fp16", torch.float16, 2, 5e-4),
    )
    round_orders = (
        (
            "pytorch",
            "triton",
            "naive_cuda",
            "coalesced_cuda",
            "warp_packed_cuda",
        ),
        (
            "triton",
            "naive_cuda",
            "coalesced_cuda",
            "warp_packed_cuda",
            "pytorch",
        ),
        (
            "naive_cuda",
            "coalesced_cuda",
            "warp_packed_cuda",
            "pytorch",
            "triton",
        ),
        (
            "coalesced_cuda",
            "warp_packed_cuda",
            "pytorch",
            "triton",
            "naive_cuda",
        ),
        (
            "warp_packed_cuda",
            "pytorch",
            "triton",
            "naive_cuda",
            "coalesced_cuda",
        ),
    )
    cases = []

    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary = Path(temporary_directory)
        for height, width in shapes:
            cpu_image = deterministic_bgr_image(
                height,
                width,
                torch_module=torch,
                device="cpu",
            )
            input_bytes = cpu_image.numpy().tobytes(order="C")
            input_path = temporary / f"input_{height}x{width}.bin"
            input_path.write_bytes(input_bytes)
            image = cpu_image.to(device="cuda")

            for dtype_name, output_dtype, output_bytes, tolerance in dtype_cases:
                expected = torch_reference_preprocess(
                    image,
                    output_dtype=output_dtype,
                )
                triton_output = torch.empty_like(expected)
                triton_preprocess_into(
                    image,
                    triton_output,
                    block_size=block_size,
                    num_warps=num_warps,
                )
                torch.cuda.synchronize()
                triton_difference = (
                    triton_output.float() - expected.float()
                ).abs()
                triton_maximum_difference = float(
                    triton_difference.max().item()
                )
                triton_mean_difference = float(
                    triton_difference.mean().item()
                )
                triton_mismatches = int(
                    (triton_difference > tolerance).sum().item()
                )
                torch.testing.assert_close(
                    triton_output,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                )

                cuda_correctness = {}
                for native_implementation in (
                    "naive",
                    "coalesced",
                    "warp_packed",
                ):
                    correctness_output = temporary / (
                        f"correctness_{native_implementation}_{height}x"
                        f"{width}_{dtype_name}.bin"
                    )
                    correctness_samples = temporary / (
                        f"correctness_{native_implementation}_{height}x"
                        f"{width}_{dtype_name}.csv"
                    )
                    run_native(
                        height=height,
                        width=width,
                        dtype_name=dtype_name,
                        input_path=input_path,
                        output_path=correctness_output,
                        samples_path=correctness_samples,
                        implementation=native_implementation,
                        warmup=1,
                        iterations=1,
                        launches=1,
                    )
                    cuda_output = read_standalone_output(
                        correctness_output,
                        height=height,
                        width=width,
                        dtype_name=dtype_name,
                        torch_module=torch,
                    ).to(device="cuda")
                    cuda_difference = (
                        cuda_output.float() - expected.float()
                    ).abs()
                    cuda_maximum_difference = float(
                        cuda_difference.max().item()
                    )
                    cuda_mean_difference = float(
                        cuda_difference.mean().item()
                    )
                    cuda_mismatches = int(
                        (cuda_difference > tolerance).sum().item()
                    )
                    torch.testing.assert_close(
                        cuda_output,
                        expected,
                        rtol=0.0,
                        atol=tolerance,
                    )
                    cuda_correctness[f"{native_implementation}_cuda"] = {
                        "passed": True,
                        "maximum_absolute_difference": (
                            cuda_maximum_difference
                        ),
                        "mean_absolute_difference": cuda_mean_difference,
                        "mismatched_values": cuda_mismatches,
                    }

                pytorch_operation = lambda: torch_reference_preprocess(
                    image,
                    output_dtype=output_dtype,
                )
                triton_operation = lambda: triton_preprocess_into(
                    image,
                    triton_output,
                    block_size=block_size,
                    num_warps=num_warps,
                )
                round_results: dict[str, list[dict[str, Any]]] = {
                    "pytorch": [],
                    "triton": [],
                    "naive_cuda": [],
                    "coalesced_cuda": [],
                    "warp_packed_cuda": [],
                }

                for round_id, order in enumerate(round_orders, start=1):
                    for position, implementation in enumerate(order, start=1):
                        if implementation == "pytorch":
                            samples = measure_torch_gpu_ms(
                                pytorch_operation
                            )
                        elif implementation == "triton":
                            samples = measure_torch_gpu_ms(
                                triton_operation
                            )
                        else:
                            output_path = temporary / (
                                f"round{round_id}_{height}x{width}_"
                                f"{dtype_name}_{implementation}.bin"
                            )
                            samples_path = temporary / (
                                f"round{round_id}_{height}x{width}_"
                                f"{dtype_name}_{implementation}.csv"
                            )
                            samples = run_native(
                                height=height,
                                width=width,
                                dtype_name=dtype_name,
                                input_path=input_path,
                                output_path=output_path,
                                samples_path=samples_path,
                                implementation={
                                    "naive_cuda": "naive",
                                    "coalesced_cuda": "coalesced",
                                    "warp_packed_cuda": "warp_packed",
                                }[implementation],
                                warmup=warmup_iterations,
                                iterations=measured_iterations,
                                launches=launches_per_sample,
                            )

                        round_results[implementation].append(
                            {
                                "round_id": round_id,
                                "order": list(order),
                                "position": position,
                                "summary_ms": summarize(samples).to_dict(),
                                "samples_ms": samples,
                            }
                        )

                implementations = {}
                for implementation, rounds in round_results.items():
                    combined_samples = [
                        sample
                        for round_result in rounds
                        for sample in round_result["samples_ms"]
                    ]
                    implementations[implementation] = {
                        "summary_ms": summarize(
                            combined_samples
                        ).to_dict(),
                        "rounds": rounds,
                    }

                pytorch_median = float(
                    implementations["pytorch"]["summary_ms"]["median"]
                )
                triton_median = float(
                    implementations["triton"]["summary_ms"]["median"]
                )
                cuda_median = float(
                    implementations["naive_cuda"]["summary_ms"]["median"]
                )
                coalesced_median = float(
                    implementations["coalesced_cuda"]["summary_ms"][
                        "median"
                    ]
                )
                warp_packed_median = float(
                    implementations["warp_packed_cuda"]["summary_ms"][
                        "median"
                    ]
                )
                cases.append(
                    {
                        "height": height,
                        "width": width,
                        "dtype": dtype_name,
                        "input_sha256": hashlib.sha256(
                            input_bytes
                        ).hexdigest(),
                        "input_bytes": len(input_bytes),
                        "output_bytes": height * width * 3 * output_bytes,
                        "correctness": {
                            "tolerance": tolerance,
                            "triton": {
                                "passed": True,
                                "maximum_absolute_difference": (
                                    triton_maximum_difference
                                ),
                                "mean_absolute_difference": (
                                    triton_mean_difference
                                ),
                                "mismatched_values": triton_mismatches,
                            },
                            **cuda_correctness,
                        },
                        "implementations": implementations,
                        "comparisons": {
                            "triton_speedup_over_pytorch": (
                                pytorch_median / triton_median
                            ),
                            "naive_cuda_speedup_over_pytorch": (
                                pytorch_median / cuda_median
                            ),
                            "naive_cuda_speedup_over_triton": (
                                triton_median / cuda_median
                            ),
                            "coalesced_cuda_speedup_over_pytorch": (
                                pytorch_median / coalesced_median
                            ),
                            "coalesced_cuda_speedup_over_triton": (
                                triton_median / coalesced_median
                            ),
                            "coalesced_cuda_speedup_over_naive_cuda": (
                                cuda_median / coalesced_median
                            ),
                            "warp_packed_cuda_speedup_over_pytorch": (
                                pytorch_median / warp_packed_median
                            ),
                            "warp_packed_cuda_speedup_over_triton": (
                                triton_median / warp_packed_median
                            ),
                            "warp_packed_cuda_speedup_over_naive_cuda": (
                                cuda_median / warp_packed_median
                            ),
                            "warp_packed_cuda_speedup_over_coalesced_cuda": (
                                coalesced_median / warp_packed_median
                            ),
                            "triton_saving_vs_pytorch_ms": (
                                pytorch_median - triton_median
                            ),
                            "naive_cuda_saving_vs_pytorch_ms": (
                                pytorch_median - cuda_median
                            ),
                            "naive_cuda_saving_vs_triton_ms": (
                                triton_median - cuda_median
                            ),
                            "coalesced_cuda_saving_vs_pytorch_ms": (
                                pytorch_median - coalesced_median
                            ),
                            "coalesced_cuda_saving_vs_triton_ms": (
                                triton_median - coalesced_median
                            ),
                            "coalesced_cuda_saving_vs_naive_cuda_ms": (
                                cuda_median - coalesced_median
                            ),
                            "warp_packed_cuda_saving_vs_pytorch_ms": (
                                pytorch_median - warp_packed_median
                            ),
                            "warp_packed_cuda_saving_vs_triton_ms": (
                                triton_median - warp_packed_median
                            ),
                            "warp_packed_cuda_saving_vs_naive_cuda_ms": (
                                cuda_median - warp_packed_median
                            ),
                            "warp_packed_cuda_saving_vs_coalesced_cuda_ms": (
                                coalesced_median - warp_packed_median
                            ),
                        },
                    }
                )

    return {
        "schema_version": 3,
        "experiment": "final profile-guided CUDA preprocessing comparison",
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "gpu": torch.cuda.get_device_name(),
            "warmup_iterations_per_operation": warmup_iterations,
            "measured_samples_per_round": measured_iterations,
            "launches_per_sample": launches_per_sample,
            "round_orders": [list(order) for order in round_orders],
            "block_size": block_size,
            "num_warps": num_warps,
            "compile_command": compile_command,
            "nvcc_version": nvcc_version,
            "source_sha256": {
                "standalone_cuda": hashlib.sha256(
                    cuda_source.read_bytes()
                ).hexdigest(),
                "triton_wrapper": hashlib.sha256(
                    triton_source.read_bytes()
                ).hexdigest(),
            },
            "timing": (
                "CUDA events; 100 repeated launches per interval by default; "
                "elapsed interval divided by launch count"
            ),
            "scope": (
                "warm-cache repeated-launch preprocessing; input resident; "
                "H2D, D2H, compilation, context creation, subprocess startup, "
                "and outer wall time excluded; stream gaps caused by each "
                "implementation's submission path may remain"
            ),
            "allocation": (
                "Triton and native CUDA outputs preallocated; PyTorch keeps "
                "its normal multi-operation reference behavior"
            ),
            "environment": collect_environment(),
        },
        "cases": cases,
    }


def _write_raw_csv(report: dict[str, Any], path: Path) -> None:
    """Flatten all per-round timing samples into one portable CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "height",
        "width",
        "dtype",
        "implementation",
        "round_id",
        "order",
        "position",
        "sample_index",
        "latency_ms",
        "launches_per_sample",
        "block_size",
        "num_warps",
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for case in report["cases"]:
            for implementation, result in case["implementations"].items():
                for round_result in result["rounds"]:
                    for sample_index, latency_ms in enumerate(
                        round_result["samples_ms"]
                    ):
                        writer.writerow(
                            {
                                "height": case["height"],
                                "width": case["width"],
                                "dtype": case["dtype"],
                                "implementation": implementation,
                                "round_id": round_result["round_id"],
                                "order": ">".join(round_result["order"]),
                                "position": round_result["position"],
                                "sample_index": sample_index,
                                "latency_ms": latency_ms,
                                "launches_per_sample": report["metadata"][
                                    "launches_per_sample"
                                ],
                                "block_size": report["metadata"][
                                    "block_size"
                                ],
                                "num_warps": (
                                    report["metadata"]["num_warps"]
                                    if implementation == "triton"
                                    else ""
                                ),
                            }
                        )


@app.local_entrypoint()
def main(
    warmup: int = 30,
    iterations: int = 200,
    launches_per_sample: int = 100,
    block_size: int = 256,
    num_warps: int = 4,
    json_out: str = "results/modal_l4_cuda_final_benchmark.json",
    csv_out: str = "benchmarks/raw/modal_l4_cuda_final_benchmark.csv",
) -> None:
    """Run the comparison and save summary plus raw measurements locally."""
    git_commit = _run_checked(
        ["git", "rev-parse", "HEAD"]
    ).stdout.strip()
    git_worktree_dirty = bool(
        _run_checked(["git", "status", "--short"]).stdout.strip()
    )
    report = benchmark_l4.remote(
        warmup,
        iterations,
        launches_per_sample,
        block_size,
        num_warps,
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

    for case in report["cases"]:
        implementations = case["implementations"]
        print(
            f"{case['height']}x{case['width']} {case['dtype']}: "
            f"PyTorch={implementations['pytorch']['summary_ms']['median']:.6f} ms, "
            f"Triton={implementations['triton']['summary_ms']['median']:.6f} ms, "
            f"naive CUDA={implementations['naive_cuda']['summary_ms']['median']:.6f} ms, "
            f"coalesced CUDA={implementations['coalesced_cuda']['summary_ms']['median']:.6f} ms, "
            f"warp-packed CUDA={implementations['warp_packed_cuda']['summary_ms']['median']:.6f} ms"
        )
    print(f"Saved benchmark report to {json_path}")
    print(f"Saved raw samples to {csv_path}")
