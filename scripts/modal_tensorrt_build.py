"""Build fixed-shape TensorRT engines on a Modal NVIDIA L4."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FP32_ONNX_MODEL = PROJECT_ROOT / "yolov8n.onnx"
FP16_ONNX_MODEL = PROJECT_ROOT / "artifacts/yolov8n_mixed_fp16.onnx"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .uv_pip_install("tensorrt>=11.2,<11.3")
    .add_local_file(FP32_ONNX_MODEL, remote_path="/root/yolov8n.onnx")
    .add_local_file(
        FP16_ONNX_MODEL,
        remote_path="/root/yolov8n_mixed_fp16.onnx",
    )
)

app = modal.App("kernelvision-tensorrt-build")


def build_serialized_engine(
    onnx_path: Path,
    *,
    precision: str,
    workspace_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    """Parse a strongly typed ONNX graph and build engine bytes."""
    import tensorrt as trt

    # Milestone 7 learning exercise:
    # 1. Create a WARNING logger, Builder, empty Network, and OnnxParser.
    # 2. Parse onnx_path and raise an error containing every parser error
    #    when parsing fails.
    # 3. Create a BuilderConfig and set its WORKSPACE memory-pool limit.
    # 4. Call builder.build_serialized_network(network, config).
    # 5. Reject a None result and return bytes(serialized_engine) plus the
    #    metadata dictionary described in the lesson.
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    parsed = parser.parse_from_file(str(onnx_path))
    parser_errors = [
        str(parser.get_error(index)) for index in range(parser.num_errors)
    ]

    if not parsed:
        raise RuntimeError(
            f"TensorRT could not parse the ONNX model: {parser_errors}"
        )

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)

    # Milestone 7 learning exercise:
    # Verify that the strongly typed graph exposes the expected TensorRT
    # input and output dtype before spending time building the engine.
    expected_dtype = (
        trt.DataType.HALF
        if precision == "fp16"
        else trt.DataType.FLOAT
    )

    if (
        input_tensor.dtype!=expected_dtype
        or output_tensor.dtype!=expected_dtype
    ):
        raise RuntimeError(
        f"expected {expected_dtype}, "
        f"received input={input_tensor.dtype}, "
        f"output={output_tensor.dtype}"
    )

    config = builder.create_builder_config()
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        workspace_bytes,
    )

    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        raise RuntimeError("TensorRT engine build failed")

    engine_bytes = bytes(serialized_engine)

    return engine_bytes, {
        "tensorrt_version": trt.__version__,
        "precision": precision,
        "tf32_allowed": False,
        "network_layer_count": network.num_layers,
        "input": {
            "name": input_tensor.name,
            "shape": list(input_tensor.shape),
            "dtype": str(input_tensor.dtype),
        },
        "output": {
            "name": output_tensor.name,
            "shape": list(output_tensor.shape),
            "dtype": str(output_tensor.dtype),
        },
    }


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def build_l4(precision: str, workspace_bytes: int) -> dict[str, Any]:
    """Build on the L4 and return the portable report plus engine artifact."""
    remote_models = {
        "fp32": Path("/root/yolov8n.onnx"),
        "fp16": Path("/root/yolov8n_mixed_fp16.onnx"),
    }
    onnx_path = remote_models[precision]
    engine_bytes, build_metadata = build_serialized_engine(
        onnx_path,
        precision=precision,
        workspace_bytes=workspace_bytes,
    )
    return {
        "engine_bytes": engine_bytes,
        "report": {
            "experiment": f"TensorRT {precision.upper()} engine build",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": "NVIDIA L4",
            "onnx_model": onnx_path.name,
            "onnx_sha256": hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
            "engine_size_bytes": len(engine_bytes),
            "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(),
            "workspace_bytes": workspace_bytes,
            **build_metadata,
        },
    }


@app.local_entrypoint()
def main(
    precision: str = "fp32",
    workspace_gib: float = 1.0,
    engine_out: str = "",
    json_out: str = "",
) -> None:
    """Launch the L4 build and save the engine and report locally."""
    local_models = {
        "fp32": FP32_ONNX_MODEL,
        "fp16": FP16_ONNX_MODEL,
    }
    if precision not in local_models:
        raise ValueError("precision must be either 'fp32' or 'fp16'")
    onnx_model = local_models[precision]
    if not onnx_model.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {onnx_model}")
    if workspace_gib <= 0:
        raise ValueError("workspace_gib must be greater than zero")

    workspace_bytes = int(workspace_gib * 1024**3)
    payload = build_l4.remote(precision, workspace_bytes)
    engine_bytes = payload["engine_bytes"]
    report = payload["report"]

    engine_output = Path(
        engine_out or f"artifacts/yolov8n_{precision}.engine"
    )
    engine_output.parent.mkdir(parents=True, exist_ok=True)
    engine_output.write_bytes(engine_bytes)
    report_output = Path(
        json_out or f"results/modal_l4_tensorrt_{precision}_build.json"
    )
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved TensorRT engine to {engine_output}")
    print(f"Saved build report to {report_output}")


if __name__ == "__main__":
    main()
