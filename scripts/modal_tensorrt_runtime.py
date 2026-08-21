"""Validate the FP32 TensorRT engine against PyTorch on a Modal L4."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_MODEL = PROJECT_ROOT / "yolov8n.pt"
ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp32.engine"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115", "tensorrt>=11.2,<11.3")
    .env(
        {
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_file(PYTORCH_MODEL, remote_path="/root/yolov8n.pt")
    .add_local_file(ENGINE_MODEL, remote_path="/root/yolov8n_fp32.engine")
)

app = modal.App("kernelvision-tensorrt-runtime")


def run_tensorrt(
    engine_path: Path,
    input_tensor: Any,
) -> tuple[Any, dict[str, Any]]:
    """Execute one static-shape TensorRT inference using PyTorch buffers."""
    import tensorrt as trt
    import torch

    # Milestone 7 learning exercise:
    # 1. Create a Logger and Runtime, then deserialize engine_path bytes.
    # 2. Create one execution context.
    # 3. Find input/output names through the engine's named I/O metadata.
    # 4. Validate the supplied input shape and FP32 CUDA properties.
    # 5. Allocate a FP32 CUDA output tensor using the engine output shape.
    # 6. Bind both tensor data_ptr() addresses to the context.
    # 7. Launch execute_async_v3 on torch.cuda.current_stream().cuda_stream.
    # 8. Synchronize for this correctness script, then return output + metadata.
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    engine_bytes = engine_path.read_bytes()
    engine = runtime.deserialize_cuda_engine(engine_bytes)

    if engine is None:
        raise RuntimeError("failed to deserialize TensorRT engine")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("failed to create TensorRT execution context")

    io_names = [
        engine.get_tensor_name(index)
        for index in range(engine.num_io_tensors)
    ]

    input_names = [
        name
        for name in io_names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT
    ]

    output_names = [
        name
        for name in io_names
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT
    ]

    if len(input_names) != 1 or len(output_names) != 1:
        raise RuntimeError(
            f"expected one input and one output. Got "
            f"{len(input_names)} inputs and {len(output_names)} outputs"
        )
    
    input_name = input_names[0]
    output_name = output_names[0]

    expected_input_shape = tuple(engine.get_tensor_shape(input_name))
    engine_input_dtype = engine.get_tensor_dtype(input_name)

    if tuple(input_tensor.shape) != expected_input_shape:
        raise ValueError(
            f"expected input shape {expected_input_shape}, "
            f"received {tuple(input_tensor.shape)}"
        )
    if input_tensor.dtype != torch.float32:
        raise ValueError(f"expected torch.float32, received {input_tensor.dtype}")

    if not input_tensor.is_cuda:
        raise ValueError("input tensor must be on a CUDA device")

    if not input_tensor.is_contiguous():
        raise ValueError("input tensor must be contiguous")

    if engine_input_dtype != trt.DataType.FLOAT:
        raise ValueError(
            f"expected a TensorRT FP32 input, received {engine_input_dtype}"
        )

    output_shape = tuple(engine.get_tensor_shape(output_name))
    engine_output_dtype = engine.get_tensor_dtype(output_name)

    if engine_output_dtype != trt.DataType.FLOAT:
        raise ValueError(
            f"expected a TensorRT fp32 output, received {engine_output_dtype}"
        )
    
    output_tensor = torch.empty(
        output_shape,
        dtype=torch.float32,
        device=input_tensor.device,
    )

    if not context.set_tensor_address(
        input_name,
        input_tensor.data_ptr(),
    ):
        raise RuntimeError(f"failed to bind TensorRT input {input_name!r}")

    if not context.set_tensor_address(
        output_name,
        output_tensor.data_ptr(),
    ):
        raise RuntimeError(f"failed to bind TensorRT output {output_name!r}")

    stream = torch.cuda.current_stream(input_tensor.device)

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError("TensorRT execution failed")

    stream.synchronize()

    metadata = {
        "tensorrt_version": trt.__version__,
        "input_name": input_name,
        "input_shape": list(expected_input_shape),
        "input_dtype": str(engine_input_dtype),
        "output_name": output_name,
        "output_shape": list(output_shape),
        "output_dtype": str(engine_output_dtype),
    }

    return output_tensor, metadata


@app.function(image=runtime_image, gpu="L4", timeout=20 * 60)
def validate_fp32_l4() -> dict[str, Any]:
    """Compare TensorRT and PyTorch raw output for identical CUDA input."""
    import torch
    from ultralytics import YOLO
    
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"

    element_count = 3 * 640 * 640
    input_tensor = (
        torch.arange(element_count, dtype=torch.float32, device="cuda") % 256
    ).div(255.0).reshape(1, 3, 640, 640)

    pytorch_model = YOLO("/root/yolov8n.pt", verbose=False).model
    pytorch_model = pytorch_model.to("cuda").eval()
    with torch.inference_mode():
        pytorch_output = pytorch_model(input_tensor)[0]

    tensorrt_output, runtime_metadata = run_tensorrt(
        Path("/root/yolov8n_fp32.engine"),
        input_tensor,
    )
    absolute_difference = (pytorch_output - tensorrt_output).abs()
    report = {
        "experiment": "TensorRT FP32 raw-output correctness",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "engine_sha256": hashlib.sha256(
            Path("/root/yolov8n_fp32.engine").read_bytes()
        ).hexdigest(),
        "shape": list(tensorrt_output.shape),
        "element_count": tensorrt_output.numel(),
        "maximum_absolute_difference": float(
            absolute_difference.max().item()
        ),
        "mean_absolute_difference": float(
            absolute_difference.mean().item()
        ),
        "allclose_atol_1e-3_rtol_1e-3": bool(
            torch.allclose(
                pytorch_output,
                tensorrt_output,
                atol=1e-3,
                rtol=1e-3,
            )
        ),
        "runtime": runtime_metadata,
    }
    if not report["allclose_atol_1e-3_rtol_1e-3"]:
        raise RuntimeError(f"TensorRT FP32 correctness gate failed: {report}")
    return report


@app.local_entrypoint()
def main(
    json_out: str = "results/modal_l4_tensorrt_fp32_correctness.json",
) -> None:
    """Run the L4 correctness gate and save its report locally."""
    if not ENGINE_MODEL.is_file():
        raise FileNotFoundError(f"TensorRT engine does not exist: {ENGINE_MODEL}")
    report = validate_fp32_l4.remote()
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved TensorRT correctness report to {output}")


if __name__ == "__main__":
    main()
