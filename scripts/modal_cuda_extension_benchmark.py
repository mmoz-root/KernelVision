"""Benchmark the integrated PyTorch CUDA extension on a Modal L4."""

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

app = modal.App("kernelvision-pytorch-cuda-extension-benchmark")


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


def _read_native_samples(path: Path) -> list[float]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [
            float(row["latency_ms"])
            for row in csv.DictReader(stream)
        ]


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
    """Correctness-gate and time four preprocessing boundaries."""
    import torch

    from kernelvision.benchmarking.statistics import summarize
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        cuda_extension_preprocess,
        deterministic_bgr_image,
        read_standalone_output,
        torch_reference_preprocess,
        triton_preprocess,
    )

    if warmup_iterations < 1 or measured_iterations < 1:
        raise ValueError("warmup and measured iterations must be positive")
    if launches_per_sample < 1:
        raise ValueError("launches_per_sample must be positive")

    cuda_source = Path("/root/csrc/preprocessing/standalone_preprocess.cu")
    extension_cpp = Path(
        "/root/csrc/preprocessing/torch_extension.cpp"
    )
    extension_cuda = Path(
        "/root/csrc/preprocessing/torch_extension_kernel.cu"
    )
    binary = Path("/tmp/kernelvision_extension_benchmark_standalone")
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
            raise RuntimeError("operation did not produce output")
        return [
            float(start.elapsed_time(stop)) / launches_per_sample
            for start, stop in event_pairs
        ]

    def run_standalone(
        *,
        height: int,
        width: int,
        dtype_name: str,
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
            str(height),
            "--width",
            str(width),
            "--dtype",
            dtype_name,
            "--block-size",
            str(block_size),
            "--implementation",
            "naive",
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
        return _read_native_samples(samples_path)

    shapes = ((320, 320), (384, 640), (640, 640), (720, 1280))
    dtype_cases = (
        ("fp32", torch.float32, False, 1e-7),
        ("fp16", torch.float16, True, 5e-4),
    )
    round_orders = (
        ("pytorch", "triton", "standalone_cuda", "cuda_extension"),
        ("triton", "standalone_cuda", "cuda_extension", "pytorch"),
        ("standalone_cuda", "cuda_extension", "pytorch", "triton"),
        ("cuda_extension", "pytorch", "triton", "standalone_cuda"),
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

            for (
                dtype_name,
                output_dtype,
                output_fp16,
                tolerance,
            ) in dtype_cases:
                expected = torch_reference_preprocess(
                    image,
                    output_dtype=output_dtype,
                )
                triton_actual = triton_preprocess(
                    image,
                    output_dtype=output_dtype,
                    block_size=block_size,
                    num_warps=num_warps,
                )
                extension_actual = cuda_extension_preprocess(
                    image,
                    output_dtype=output_dtype,
                )
                torch.testing.assert_close(
                    triton_actual,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                )
                torch.testing.assert_close(
                    extension_actual,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                )

                native_output = temporary / (
                    f"correctness_{height}x{width}_{dtype_name}.bin"
                )
                native_samples = temporary / (
                    f"correctness_{height}x{width}_{dtype_name}.csv"
                )
                run_standalone(
                    height=height,
                    width=width,
                    dtype_name=dtype_name,
                    input_path=input_path,
                    output_path=native_output,
                    samples_path=native_samples,
                    warmup=1,
                    iterations=1,
                    launches=1,
                )
                native_actual = read_standalone_output(
                    native_output,
                    height=height,
                    width=width,
                    dtype_name=dtype_name,
                    torch_module=torch,
                ).to(device="cuda")
                torch.testing.assert_close(
                    native_actual,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                )

                operations = {
                    "pytorch": lambda: torch_reference_preprocess(
                        image,
                        output_dtype=output_dtype,
                    ),
                    "triton": lambda: triton_preprocess(
                        image,
                        output_dtype=output_dtype,
                        block_size=block_size,
                        num_warps=num_warps,
                    ),
                    "cuda_extension": lambda: cuda_extension_preprocess(
                        image,
                        output_dtype=output_dtype,
                    ),
                }
                round_results: dict[str, list[dict[str, Any]]] = {
                    implementation: []
                    for implementation in round_orders[0]
                }
                for round_id, order in enumerate(round_orders, start=1):
                    for position, implementation in enumerate(order, start=1):
                        if implementation == "standalone_cuda":
                            output_path = temporary / (
                                f"round{round_id}_{height}x{width}_"
                                f"{dtype_name}.bin"
                            )
                            samples_path = temporary / (
                                f"round{round_id}_{height}x{width}_"
                                f"{dtype_name}.csv"
                            )
                            samples = run_standalone(
                                height=height,
                                width=width,
                                dtype_name=dtype_name,
                                input_path=input_path,
                                output_path=output_path,
                                samples_path=samples_path,
                                warmup=warmup_iterations,
                                iterations=measured_iterations,
                                launches=launches_per_sample,
                            )
                        else:
                            samples = measure_torch_gpu_ms(
                                operations[implementation]
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
                    combined = [
                        sample
                        for round_result in rounds
                        for sample in round_result["samples_ms"]
                    ]
                    implementations[implementation] = {
                        "summary_ms": summarize(combined).to_dict(),
                        "rounds": rounds,
                    }
                extension_median = implementations["cuda_extension"][
                    "summary_ms"
                ]["median"]
                standalone_median = implementations["standalone_cuda"][
                    "summary_ms"
                ]["median"]
                triton_median = implementations["triton"]["summary_ms"][
                    "median"
                ]
                cases.append(
                    {
                        "height": height,
                        "width": width,
                        "dtype": dtype_name,
                        "input_sha256": hashlib.sha256(
                            input_bytes
                        ).hexdigest(),
                        "correctness_passed": True,
                        "implementations": implementations,
                        "comparisons": {
                            "extension_overhead_vs_standalone_ms": (
                                extension_median - standalone_median
                            ),
                            "extension_slowdown_vs_standalone": (
                                extension_median / standalone_median
                            ),
                            "extension_speedup_over_triton": (
                                triton_median / extension_median
                            ),
                        },
                    }
                )

    return {
        "schema_version": 1,
        "experiment": "PyTorch CUDA extension integration boundary",
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "git_commit": git_commit,
            "git_worktree_dirty": git_worktree_dirty,
            "gpu": torch.cuda.get_device_name(),
            "warmup_iterations": warmup_iterations,
            "measured_samples_per_round": measured_iterations,
            "launches_per_sample": launches_per_sample,
            "round_orders": [list(order) for order in round_orders],
            "block_size": block_size,
            "num_warps": num_warps,
            "allocation_boundary": (
                "PyTorch, Triton, and extension use allocating public APIs; "
                "standalone CUDA reuses preallocated native buffers"
            ),
            "timing_scope": (
                "CUDA events around repeated calls; input resident; H2D, "
                "D2H, compilation, and context creation excluded"
            ),
            "source_sha256": {
                "standalone_cuda": hashlib.sha256(
                    cuda_source.read_bytes()
                ).hexdigest(),
                "extension_cpp": hashlib.sha256(
                    extension_cpp.read_bytes()
                ).hexdigest(),
                "extension_cuda": hashlib.sha256(
                    extension_cuda.read_bytes()
                ).hexdigest(),
            },
            "environment": collect_environment(),
        },
        "cases": cases,
    }


def _write_raw_csv(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "height",
        "width",
        "dtype",
        "implementation",
        "round_id",
        "position",
        "order",
        "sample_index",
        "latency_ms",
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
                                "position": round_result["position"],
                                "order": ">".join(round_result["order"]),
                                "sample_index": sample_index,
                                "latency_ms": latency_ms,
                            }
                        )


@app.local_entrypoint()
def main(
    warmup: int = 30,
    iterations: int = 200,
    launches_per_sample: int = 100,
    block_size: int = 256,
    num_warps: int = 4,
    json_out: str = "results/modal_l4_cuda_extension_benchmark.json",
    csv_out: str = "benchmarks/raw/modal_l4_cuda_extension_benchmark.csv",
) -> None:
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
    json_path = Path(json_out)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    csv_path = Path(csv_out)
    _write_raw_csv(report, csv_path)
    for case in report["cases"]:
        results = case["implementations"]
        print(
            f"{case['height']}x{case['width']} {case['dtype']}: "
            f"PyTorch={results['pytorch']['summary_ms']['median']:.6f} ms, "
            f"Triton={results['triton']['summary_ms']['median']:.6f} ms, "
            f"standalone={results['standalone_cuda']['summary_ms']['median']:.6f} ms, "
            f"extension={results['cuda_extension']['summary_ms']['median']:.6f} ms"
        )
    print(f"Saved benchmark report to {json_path}")
    print(f"Saved raw samples to {csv_path}")
