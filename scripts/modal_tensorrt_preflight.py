"""Verify TensorRT 11 and parse the Milestone 7 ONNX graph on Modal L4."""

from __future__ import annotations

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

app = modal.App("kernelvision-tensorrt-preflight")


@app.function(image=runtime_image, gpu="L4", timeout=20 * 60)
def preflight_l4() -> dict[str, Any]:
    """Create TensorRT build objects and parse the fixed ONNX model."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)
    parsed = parser.parse_from_file("/root/yolov8n.onnx")
    parser_errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
    if not parsed:
        raise RuntimeError(f"TensorRT ONNX parsing failed: {parser_errors}")

    def tensor_metadata(tensor: Any) -> dict[str, Any]:
        return {
            "name": tensor.name,
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
        }

    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": "NVIDIA L4",
        "cuda_base_image": "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        "tensorrt_version": trt.__version__,
        "parsed": parsed,
        "parser_errors": parser_errors,
        "network_layer_count": network.num_layers,
        "inputs": [
            tensor_metadata(network.get_input(index))
            for index in range(network.num_inputs)
        ],
        "outputs": [
            tensor_metadata(network.get_output(index))
            for index in range(network.num_outputs)
        ],
        "builder_has_build_serialized_network": hasattr(
            builder,
            "build_serialized_network",
        ),
        "builder_flag_has_fp16": hasattr(trt.BuilderFlag, "FP16"),
        "network_creation_flag_has_strongly_typed": hasattr(
            trt.NetworkDefinitionCreationFlag,
            "STRONGLY_TYPED",
        ),
    }


@app.local_entrypoint()
def main(
    json_out: str = "results/modal_l4_tensorrt_preflight.json",
) -> None:
    """Run the L4 preflight and save its report locally."""
    if not ONNX_MODEL.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {ONNX_MODEL}")
    report = preflight_l4.remote()
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved TensorRT preflight report to {output}")


if __name__ == "__main__":
    main()
