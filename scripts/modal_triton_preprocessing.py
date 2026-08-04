"""Validate fused Triton preprocessing on a Modal NVIDIA L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
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

app = modal.App("kernelvision-triton-preprocessing")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def validate_l4(block_size: int, num_warps: int) -> dict[str, Any]:
    """Compare Triton output with PyTorch over varied shapes and dtypes."""
    import torch

    from kernelvision.environment import collect_environment
    from kernelvision.preprocessing import (
        torch_reference_preprocess,
        triton_preprocess,
    )

    torch.manual_seed(0)
    shapes = ((2, 3), (5, 7), (384, 640), (640, 640), (641, 639))
    dtype_cases = (
        ("fp32", torch.float32, 1e-7),
        ("fp16", torch.float16, 5e-4),
    )
    cases = []
    for height, width in shapes:
        image = torch.randint(
            0,
            256,
            (height, width, 3),
            dtype=torch.uint8,
            device="cuda",
        )
        for dtype_name, output_dtype, tolerance in dtype_cases:
            expected = torch_reference_preprocess(
                image,
                output_dtype=output_dtype,
            )
            actual = triton_preprocess(
                image,
                output_dtype=output_dtype,
                block_size=block_size,
                num_warps=num_warps,
            )
            torch.cuda.synchronize()
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
            "block_size": block_size,
            "num_warps": num_warps,
            "environment": collect_environment(),
        },
        "all_cases_passed": True,
        "cases": cases,
    }


@app.local_entrypoint()
def main(
    block_size: int = 256,
    num_warps: int = 4,
    json_out: str = "results/modal_l4_triton_preprocess_correctness.json",
) -> None:
    """Run the correctness matrix and save its report locally."""
    report = validate_l4.remote(block_size, num_warps)
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
        f"Passed {len(report['cases'])} Triton correctness cases; "
        f"maximum absolute difference={maximum_difference:.9f}"
    )
    print(f"Saved correctness report to {output}")
