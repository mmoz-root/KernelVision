"""Compile and validate standalone CUDA preprocessing on a Modal L4."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import UTC, datetime
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

app = modal.App("kernelvision-cuda-preprocessing")


def _run_checked(command: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one native command and preserve useful compiler/runtime errors."""
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def validate_l4(block_size: int) -> dict[str, Any]:
    """Compile raw CUDA and compare it with the trusted PyTorch reference."""
    import torch

    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        deterministic_bgr_image,
        read_standalone_output,
        torch_reference_preprocess,
    )

    if block_size <= 0 or block_size > 1024:
        raise ValueError("block_size must be in [1, 1024]")

    source = Path("/root/csrc/preprocessing/standalone_preprocess.cu")
    binary = Path("/tmp/kernelvision_cuda_preprocess")
    compile_command = [
        "nvcc",
        "-O3",
        "--std=c++17",
        "-lineinfo",
        str(source),
        "-o",
        str(binary),
    ]
    _run_checked(compile_command)
    nvcc_version = _run_checked(["nvcc", "--version"]).stdout.strip()

    shapes = ((2, 3), (5, 7), (384, 640), (640, 640), (641, 639))
    dtype_cases = (
        ("fp32", torch.float32, 1e-7),
        ("fp16", torch.float16, 5e-4),
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
            input_path = temporary / f"input_{height}x{width}.bin"
            input_path.write_bytes(cpu_image.numpy().tobytes(order="C"))
            cuda_image = cpu_image.to(device="cuda")

            for dtype_name, output_dtype, tolerance in dtype_cases:
                output_path = temporary / (
                    f"output_{height}x{width}_{dtype_name}.bin"
                )
                samples_path = temporary / (
                    f"samples_{height}x{width}_{dtype_name}.csv"
                )
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
                    "--warmup",
                    "1",
                    "--iterations",
                    "1",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--samples",
                    str(samples_path),
                ]
                _run_checked(command)

                expected = torch_reference_preprocess(
                    cuda_image,
                    output_dtype=output_dtype,
                )
                actual = read_standalone_output(
                    output_path,
                    height=height,
                    width=width,
                    dtype_name=dtype_name,
                    torch_module=torch,
                ).to(device="cuda")
                absolute_difference = (actual.float() - expected.float()).abs()
                maximum_difference = float(absolute_difference.max().item())
                mean_difference = float(absolute_difference.mean().item())
                mismatched_values = int(
                    (absolute_difference > tolerance).sum().item()
                )
                torch.testing.assert_close(
                    actual,
                    expected,
                    rtol=0.0,
                    atol=tolerance,
                )
                cases.append(
                    {
                        "height": height,
                        "width": width,
                        "dtype": dtype_name,
                        "element_count": actual.numel(),
                        "maximum_absolute_difference": maximum_difference,
                        "mean_absolute_difference": mean_difference,
                        "tolerance": tolerance,
                        "mismatched_values": mismatched_values,
                        "output_contiguous": actual.is_contiguous(),
                    }
                )

    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": "NVIDIA L4",
            "implementation": "naive standalone CUDA",
            "block_size": block_size,
            "compile_command": compile_command,
            "nvcc_version": nvcc_version,
            "timing_scope": "correctness only; samples are not benchmark results",
            "environment": collect_environment(),
        },
        "all_cases_passed": True,
        "cases": cases,
    }


@app.local_entrypoint()
def main(
    block_size: int = 256,
    json_out: str = "results/modal_l4_cuda_preprocess_correctness.json",
) -> None:
    """Run the naive CUDA correctness matrix and save its report locally."""
    report = validate_l4.remote(block_size)
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    maximum_difference = max(
        case["maximum_absolute_difference"] for case in report["cases"]
    )
    print(
        f"Passed {len(report['cases'])} CUDA correctness cases; "
        f"maximum absolute difference={maximum_difference:.9f}"
    )
    print(f"Saved correctness report to {output}")
