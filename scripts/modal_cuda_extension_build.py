"""Compile and import the Milestone 6 PyTorch CUDA extension on Modal L4."""

from __future__ import annotations

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

app = modal.App("kernelvision-pytorch-cuda-extension-build")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def build_l4() -> dict[str, Any]:
    """Build both translation units and verify that the module imports."""
    import torch
    from torch.utils.cpp_extension import load

    from kernelvision.environment import collect_environment

    cpp_source = Path("/root/csrc/preprocessing/torch_extension.cpp")
    cuda_source = Path(
        "/root/csrc/preprocessing/torch_extension_kernel.cu"
    )
    module = load(
        name="kernelvision_cuda_extension",
        sources=[str(cpp_source), str(cuda_source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-lineinfo"],
        with_cuda=True,
        verbose=True,
    )
    cuda_translation_unit_loaded = bool(
        module._cuda_translation_unit_loaded()
    )
    if not cuda_translation_unit_loaded:
        raise RuntimeError("compiled CUDA translation unit found no device")

    image = torch.zeros((2, 3, 3), dtype=torch.uint8, device="cuda")
    output_metadata = {}
    for output_fp16, dtype_name, expected_dtype in (
        (False, "fp32", torch.float32),
        (True, "fp16", torch.float16),
    ):
        output = module.preprocess(image, output_fp16)
        passed = (
            output.shape == (3, 2, 3)
            and output.dtype == expected_dtype
            and output.device == image.device
            and output.is_contiguous()
        )
        output_metadata[dtype_name] = {
            "passed": passed,
            "shape": list(output.shape),
            "dtype": str(output.dtype),
            "device": str(output.device),
            "contiguous": output.is_contiguous(),
        }
    if not all(case["passed"] for case in output_metadata.values()):
        raise RuntimeError(f"output metadata failures: {output_metadata}")

    def expect_error(tensor: Any, expected_message: str) -> bool:
        try:
            module.preprocess(tensor, False)
        except RuntimeError as error:
            return expected_message in str(error)
        return False

    validation_cases = {
        "cpu_device": expect_error(
            torch.zeros((2, 3, 3), dtype=torch.uint8),
            "cuda device",
        ),
        "rank": expect_error(
            torch.zeros((2, 3), dtype=torch.uint8, device="cuda"),
            "3 dimensions",
        ),
        "channels": expect_error(
            torch.zeros((2, 3, 4), dtype=torch.uint8, device="cuda"),
            "3 channels",
        ),
        "dtype": expect_error(
            torch.zeros((2, 3, 3), dtype=torch.float32, device="cuda"),
            "torch.uint8 dtype",
        ),
        "contiguous": expect_error(
            torch.zeros(
                (2, 4, 3), dtype=torch.uint8, device="cuda"
            ).transpose(0, 1),
            "must be contiguous",
        ),
        "positive_shape": expect_error(
            torch.zeros((0, 3, 3), dtype=torch.uint8, device="cuda"),
            "height and width must be positive",
        ),
    }
    if not all(validation_cases.values()):
        raise RuntimeError(
            f"native validation failures: {validation_cases}"
        )

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "extension_name": module.__name__,
        "cuda_translation_unit_loaded": cuda_translation_unit_loaded,
        "output_metadata_cases": output_metadata,
        "native_validation_cases": validation_cases,
        "sources": [str(cpp_source), str(cuda_source)],
        "environment": collect_environment(),
    }


@app.local_entrypoint()
def main(
    json_out: str = "results/modal_l4_cuda_extension_build.json",
) -> None:
    """Run the L4 build smoke test and save its environment record."""
    report = build_l4.remote()
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"Built {report['extension_name']} on {report['gpu']}; "
        "CUDA translation unit loaded"
    )
    print(f"Saved build report to {output}")
