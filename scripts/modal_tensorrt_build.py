"""Build the fixed-shape FP32 TensorRT engine on a Modal NVIDIA L4."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ONNX_MODEL = PROJECT_ROOT / "yolov8n.onnx"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .uv_pip_install("tensorrt>=11.2,<11.3")
    .add_local_file(ONNX_MODEL, remote_path="/root/yolov8n.onnx")
)

app = modal.App("kernelvision-tensorrt-build")


def build_serialized_engine(
    onnx_path: Path,
    *,
    workspace_bytes: int,
) -> tuple[bytes, dict[str, Any]]:
    """Parse a strongly typed FP32 ONNX graph and build engine bytes."""
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

    input_tensor = network.get_input(0)
    output_tensor = network.get_output(0)

    return engine_bytes, {
        "tensorrt_version": trt.__version__,
        "precision": "fp32",
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
def build_fp32_l4(workspace_bytes: int) -> dict[str, Any]:
    """Build on the L4 and return the portable report plus engine artifact."""
    engine_bytes, build_metadata = build_serialized_engine(
        Path("/root/yolov8n.onnx"),
        workspace_bytes=workspace_bytes,
    )
    return {
        "engine_bytes": engine_bytes,
        "report": {
            "experiment": "TensorRT FP32 engine build",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": "NVIDIA L4",
            "onnx_model": "yolov8n.onnx",
            "onnx_sha256": hashlib.sha256(
                Path("/root/yolov8n.onnx").read_bytes()
            ).hexdigest(),
            "engine_size_bytes": len(engine_bytes),
            "engine_sha256": hashlib.sha256(engine_bytes).hexdigest(),
            "workspace_bytes": workspace_bytes,
            **build_metadata,
        },
    }


@app.local_entrypoint()
def main(
    workspace_gib: float = 1.0,
    engine_out: str = "artifacts/yolov8n_fp32.engine",
    json_out: str = "results/modal_l4_tensorrt_fp32_build.json",
) -> None:
    """Launch the L4 build and save the engine and report locally."""
    if not ONNX_MODEL.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {ONNX_MODEL}")
    if workspace_gib <= 0:
        raise ValueError("workspace_gib must be greater than zero")

    workspace_bytes = int(workspace_gib * 1024**3)
    payload = build_fp32_l4.remote(workspace_bytes)
    engine_bytes = payload["engine_bytes"]
    report = payload["report"]

    engine_output = Path(engine_out)
    engine_output.parent.mkdir(parents=True, exist_ok=True)
    engine_output.write_bytes(engine_bytes)
    report_output = Path(json_out)
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
