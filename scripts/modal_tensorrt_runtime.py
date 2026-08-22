"""Validate TensorRT engines against reference runtimes on a Modal L4."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTORCH_MODEL = PROJECT_ROOT / "yolov8n.pt"
FP16_ONNX_MODEL = PROJECT_ROOT / "artifacts/yolov8n_mixed_fp16.onnx"
FP32_ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp32.engine"
FP16_ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp16.engine"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install(
        "ultralytics==8.4.115",
        "tensorrt>=11.2,<11.3",
        "onnxruntime-gpu>=1.27,<1.30",
    )
    .env(
        {
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_file(PYTORCH_MODEL, remote_path="/root/yolov8n.pt")
    .add_local_file(
        FP16_ONNX_MODEL,
        remote_path="/root/yolov8n_mixed_fp16.onnx",
    )
    .add_local_file(
        FP32_ENGINE_MODEL,
        remote_path="/root/yolov8n_fp32.engine",
    )
    .add_local_file(
        FP16_ENGINE_MODEL,
        remote_path="/root/yolov8n_fp16.engine",
    )
)

app = modal.App("kernelvision-tensorrt-runtime")


def run_onnx_reference(
    model_path: Path,
    input_tensor: Any,
) -> Any:
    """Execute the mixed ONNX graph and return its output as a CUDA tensor."""
    import onnxruntime as ort
    import torch

    # Milestone 7 learning exercise:
    # 1. Create an InferenceSession with CUDA and CPU providers.
    # 2. Discover the single input and output names from session metadata.
    # 3. Convert input_tensor to a detached CPU NumPy array.
    # 4. Call session.run with a list containing the output name.
    # 5. Convert the first NumPy output back to a CUDA tensor on the same
    #    device as input_tensor.
    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ],
    )

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    input_array = input_tensor.detach().cpu().numpy()

    output_array = session.run(
        [output_name],
        {input_name: input_array},
    )[0]

    return torch.from_numpy(output_array).to(
        device=input_tensor.device,
    )


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
    # 4. Validate the supplied input shape, dtype, and CUDA properties.
    # 5. Allocate a CUDA output tensor using the engine output contract.
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

    trt_to_torch_dtype = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
    }
    expected_input_shape = tuple(engine.get_tensor_shape(input_name))
    engine_input_dtype = engine.get_tensor_dtype(input_name)

    if tuple(input_tensor.shape) != expected_input_shape:
        raise ValueError(
            f"expected input shape {expected_input_shape}, "
            f"received {tuple(input_tensor.shape)}"
        )

    if not input_tensor.is_cuda:
        raise ValueError("input tensor must be on a CUDA device")

    if not input_tensor.is_contiguous():
        raise ValueError("input tensor must be contiguous")

    if engine_input_dtype not in trt_to_torch_dtype:
        raise ValueError(f"unsupported TensorRT input dtype: {engine_input_dtype}")

    expected_torch_input_dtype = trt_to_torch_dtype[engine_input_dtype]

    if input_tensor.dtype != expected_torch_input_dtype:
        raise ValueError(
            f"expected {expected_torch_input_dtype}, "
            f"received {input_tensor.dtype}"
        )

    output_shape = tuple(engine.get_tensor_shape(output_name))
    engine_output_dtype = engine.get_tensor_dtype(output_name)

    if engine_output_dtype not in trt_to_torch_dtype:
        raise ValueError(
            f"unsupported TensorRT output dtype: {engine_output_dtype}"
        )

    output_torch_dtype = trt_to_torch_dtype[engine_output_dtype]

    output_tensor = torch.empty(
        output_shape,
        dtype=output_torch_dtype,
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
def validate_l4(precision: str) -> dict[str, Any]:
    """Compare TensorRT and the selected raw-output reference on an L4."""
    import torch
    from ultralytics import YOLO

    engine_paths = {
        "fp32": Path("/root/yolov8n_fp32.engine"),
        "fp16": Path("/root/yolov8n_fp16.engine"),
    }
    tolerances = {
        "fp32": (1e-3, 1e-3),
        "fp16": (1e-2, 1e-2),
    }
    engine_path = engine_paths[precision]
    atol, rtol = tolerances[precision]

    torch.backends.cuda.matmul.fp32_precision = "ieee"
    torch.backends.cudnn.fp32_precision = "ieee"

    element_count = 3 * 640 * 640
    input_tensor = (
        torch.arange(element_count, dtype=torch.float32, device="cuda") % 256
    ).div(255.0).reshape(1, 3, 640, 640)

    if precision == "fp16":
        input_tensor = input_tensor.to(torch.float16)
        reference_output = run_onnx_reference(
            Path("/root/yolov8n_mixed_fp16.onnx"),
            input_tensor,
        )
        reference_name = "ModelOpt mixed FP16 ONNX"
    else:
        pytorch_model = YOLO("/root/yolov8n.pt", verbose=False).model
        pytorch_model = pytorch_model.to("cuda").eval()

        with torch.inference_mode():
            reference_output = pytorch_model(input_tensor)[0]

        reference_name = "PyTorch FP32"

    tensorrt_output, runtime_metadata = run_tensorrt(
        engine_path,
        input_tensor,
    )
    absolute_difference = (reference_output - tensorrt_output).abs()
    box_difference = absolute_difference[:, :4, :].float()
    class_difference = absolute_difference[:, 4:, :].float()

    close_fraction = torch.isclose(
        reference_output,
        tensorrt_output,
        atol=atol,
        rtol=rtol,
    ).float().mean()
    outputs_finite = bool(
        torch.isfinite(reference_output).all()
        and torch.isfinite(tensorrt_output).all()
    )

    all_outputs_allclose = bool(
        torch.allclose(
            reference_output,
            tensorrt_output,
            atol=atol,
            rtol=rtol,
        )
    )

    class_scores_allclose = bool(
        torch.allclose(
            reference_output[:, 4:, :],
            tensorrt_output[:, 4:, :],
            atol=atol,
            rtol=rtol,
        )
    )

    raw_gate_passed = outputs_finite and (
        all_outputs_allclose
        if precision == "fp32"
        else class_scores_allclose
    )
    report = {
        "experiment": f"TensorRT {precision.upper()} raw-output correctness",
        "reference": reference_name,
        "precision": precision,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "engine_sha256": hashlib.sha256(engine_path.read_bytes()).hexdigest(),
        "shape": list(tensorrt_output.shape),
        "element_count": tensorrt_output.numel(),
        "maximum_absolute_difference": float(
            absolute_difference.max().item()
        ),
        "mean_absolute_difference": float(
            absolute_difference.mean().item()
        ),
        "p99_absolute_difference": float(
            torch.quantile(absolute_difference.float(), 0.99).item()
        ),
        "box_maximum_difference": float(box_difference.max().item()),
        "box_mean_difference": float(box_difference.mean().item()),
        "class_maximum_difference": float(class_difference.max().item()),
        "class_mean_difference": float(class_difference.mean().item()),
        "atol": atol,
        "rtol": rtol,
        "allclose": all_outputs_allclose,
        "outputs_finite": outputs_finite,
        "class_scores_allclose": class_scores_allclose,
        "raw_gate_passed": raw_gate_passed,
        "runtime": runtime_metadata,
        "close_fraction": float(close_fraction.item()),
    }
    if not raw_gate_passed:
        raise RuntimeError(
            f"TensorRT {precision.upper()} correctness gate failed: {report}"
        )
    return report


@app.local_entrypoint()
def main(
    precision: str = "fp32",
    json_out: str = "",
) -> None:
    """Run the L4 correctness gate and save its report locally."""
    local_engines = {
        "fp32": FP32_ENGINE_MODEL,
        "fp16": FP16_ENGINE_MODEL,
    }
    if precision not in local_engines:
        raise ValueError("precision must be either 'fp32' or 'fp16'")
    engine_model = local_engines[precision]
    if not engine_model.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: {engine_model}"
        )

    report = validate_l4.remote(precision)
    output = Path(
        json_out
        or f"results/modal_l4_tensorrt_{precision}_correctness.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved TensorRT correctness report to {output}")


if __name__ == "__main__":
    main()
