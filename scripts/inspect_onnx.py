"""Validate and summarize a KernelVision ONNX model."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _value_info_metadata(value_info: Any, onnx_module: Any) -> dict[str, Any]:
    """Convert ONNX tensor type/shape protobuf fields into plain values."""
    tensor_type = value_info.type.tensor_type
    dimensions: list[int | str | None] = []
    for dimension in tensor_type.shape.dim:
        if dimension.HasField("dim_value"):
            dimensions.append(int(dimension.dim_value))
        elif dimension.HasField("dim_param"):
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return {
        "name": value_info.name,
        "dtype": onnx_module.TensorProto.DataType.Name(
            tensor_type.elem_type
        ),
        "shape": dimensions,
    }


def inspect_onnx(model_path: Path) -> dict[str, Any]:
    """Load, validate, and summarize one ONNX computation graph."""
    try:
        import onnx
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "ONNX is required for model inspection. Install the deployment "
            "dependencies first."
        ) from error

    # Milestone 7 learning exercise:
    # 1. Load model_path with onnx.load.
    # 2. Validate the ModelProto with onnx.checker.check_model.
    # 3. Read model.graph.
    # 4. Count graph nodes by node.op_type with Counter.
    # 5. Return the model/graph summary consumed by main below.
    model = onnx.load(model_path)
    onnx.checker.check_model(model)
    graph = model.graph
    operator_counts = Counter(node.op_type for node in graph.node)
    return {
        "path": str(model_path),
        "file_size_bytes": model_path.stat().st_size,
        "ir_version": int(model.ir_version),
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": [
            {
                "domain": imported.domain or "ai.onnx",
                "version": int(imported.version),
            }
            for imported in model.opset_import
        ],
        "node_count": len(graph.node),
        "initializer_count": len(graph.initializer),
        "inputs": [
            _value_info_metadata(value, onnx) for value in graph.input
        ],
        "outputs": [
            _value_info_metadata(value, onnx) for value in graph.output
        ],
        "operator_counts": dict(operator_counts.most_common()),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Print a JSON summary of an ONNX artifact."""
    parser = argparse.ArgumentParser(
        description="Validate and inspect a KernelVision ONNX graph."
    )
    parser.add_argument(
        "model",
        nargs="?",
        type=Path,
        default=Path("yolov8n.onnx"),
        help="ONNX model path (default: yolov8n.onnx)",
    )
    args = parser.parse_args(argv)
    if not args.model.is_file():
        parser.error(f"ONNX model does not exist: {args.model}")

    try:
        summary = inspect_onnx(args.model)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
