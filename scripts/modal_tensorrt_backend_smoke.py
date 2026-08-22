"""Smoke-test the reusable TensorRT backend on a Modal NVIDIA L4."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp16.engine"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .uv_pip_install(
        "ultralytics==8.4.115",
        "tensorrt>=11.2,<11.3",
    )
    .env({"PYTHONPATH": "/root/src"})
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_file(
        ENGINE_MODEL,
        remote_path="/root/yolov8n_fp16.engine",
    )
)

app = modal.App("kernelvision-tensorrt-backend-smoke")


@app.function(image=runtime_image, gpu="L4", timeout=20 * 60)
def smoke_l4() -> dict[str, Any]:
    """Run the same fixed input twice through one backend instance."""
    import torch

    from kernelvision.backends import TensorRTBackend

    element_count = 3 * 640 * 640
    input_tensor = (
        torch.arange(element_count, dtype=torch.int32, device="cuda") % 256
    ).to(torch.float16).div(255.0).reshape(1, 3, 640, 640)

    backend = TensorRTBackend(
        "/root/yolov8n_fp16.engine",
        device="cuda",
    )

    first_output = backend.infer(input_tensor)
    backend.synchronize()
    first_pointer = first_output.data_ptr()
    first_snapshot = first_output.clone()

    second_output = backend.infer(input_tensor)
    backend.synchronize()
    second_pointer = second_output.data_ptr()

    outputs_repeat = bool(torch.equal(first_snapshot, second_output))
    output_buffer_reused = first_pointer == second_pointer
    non_default_stream = (
        backend.stream.cuda_stream
        != torch.cuda.default_stream().cuda_stream
    )

    if not outputs_repeat:
        raise RuntimeError("repeated TensorRT outputs did not match")
    if not output_buffer_reused:
        raise RuntimeError("TensorRT output buffer was not reused")
    if not non_default_stream:
        raise RuntimeError("TensorRT backend used the default CUDA stream")

    return {
        "gpu": torch.cuda.get_device_name(),
        "output_shape": list(second_output.shape),
        "output_dtype": str(second_output.dtype),
        "outputs_repeat": outputs_repeat,
        "output_buffer_reused": output_buffer_reused,
        "non_default_stream": non_default_stream,
    }


@app.local_entrypoint()
def main() -> None:
    """Launch the reusable-backend smoke test."""
    report = smoke_l4.remote()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
