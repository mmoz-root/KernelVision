"""Validate the PyTorch CUDA preprocessing extension on a Modal L4."""

from __future__ import annotations

import hashlib
import json
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

app = modal.App("kernelvision-pytorch-cuda-extension-correctness")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def validate_l4() -> dict[str, Any]:
    """Build the extension and compare it with the PyTorch reference."""
    import torch
    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        cuda_extension_preprocess,
        deterministic_bgr_image,
        torch_reference_preprocess,
    )

    cpp_source = Path("/root/csrc/preprocessing/torch_extension.cpp")
    cuda_source = Path(
        "/root/csrc/preprocessing/torch_extension_kernel.cu"
    )
    shapes = ((2, 3), (5, 7), (384, 640), (640, 640), (641, 639))
    dtype_cases = (
        ("fp32", False, torch.float32, 1e-7),
        ("fp16", True, torch.float16, 5e-4),
    )
    cases = []
    for height, width in shapes:
        image = deterministic_bgr_image(
            height,
            width,
            torch_module=torch,
            device="cuda",
        )
        for dtype_name, output_fp16, output_dtype, tolerance in dtype_cases:
            expected = torch_reference_preprocess(
                image,
                output_dtype=output_dtype,
            )
            actual = cuda_extension_preprocess(
                image,
                output_dtype=output_dtype,
            )
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
                    "output_device": str(actual.device),
                }
            )

    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": torch.cuda.get_device_name(),
            "implementation": "PyTorch C++/CUDA extension naive kernel",
            "block_size": 256,
            "sources": {
                str(cpp_source): hashlib.sha256(
                    cpp_source.read_bytes()
                ).hexdigest(),
                str(cuda_source): hashlib.sha256(
                    cuda_source.read_bytes()
                ).hexdigest(),
            },
            "environment": collect_environment(),
        },
        "all_cases_passed": True,
        "cases": cases,
    }


@app.local_entrypoint()
def main(
    json_out: str = "results/modal_l4_cuda_extension_correctness.json",
) -> None:
    """Run the extension correctness matrix and save its report."""
    report = validate_l4.remote()
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
        f"Passed {len(report['cases'])} extension correctness cases; "
        f"maximum absolute difference={maximum_difference:.9f}"
    )
    print(f"Saved correctness report to {output}")
