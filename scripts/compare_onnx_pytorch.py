"""Compare PyTorch and ONNX Runtime raw YOLO predictions."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def deterministic_input(image_size: int) -> np.ndarray:
    """Create repeatable normalized BCHW input without image preprocessing."""
    element_count = 3 * image_size * image_size
    values = np.arange(element_count, dtype=np.float32) % 256
    return (values / 255.0).reshape(1, 3, image_size, image_size)


def run_pytorch(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    """Return raw PyTorch YOLO predictions as a CPU NumPy array."""
    # Milestone 7 exercise:
    # 1. Load YOLO(model_path).
    # 2. Put its underlying .model in evaluation mode.
    # 3. Convert input_array with torch.from_numpy.
    # 4. Run under torch.inference_mode().
    # 5. The evaluation output is a tuple; select its first tensor.
    # 6. Detach it, move it to CPU, and return its NumPy array.
    import torch
    from ultralytics import YOLO

    model = YOLO(str(model_path), verbose=False).model
    model.eval()
    input_tensor = torch.from_numpy(input_array)
    with torch.inference_mode():
        output = model(input_tensor)
    pred_tensor = output[0]
    return pred_tensor.detach().cpu().numpy()


def run_onnx(model_path: Path, input_array: np.ndarray) -> np.ndarray:
    """Return raw ONNX Runtime predictions as a NumPy array."""
    # Milestone 7 exercise:
    # 1. Create ort.InferenceSession with CPUExecutionProvider.
    # 2. Read the first input and output names from session metadata.
    # 3. Call session.run with a name-to-array input feed.
    # 4. Return the first output array.
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(model_path),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    requested_output_names = session.get_outputs()[0].name

    outputs = session.run(
        [requested_output_names],
        {
            input_name: input_array,
        },
    )
    return outputs[0]


def comparison_summary(
    pytorch_output: np.ndarray,
    onnx_output: np.ndarray,
) -> dict[str, Any]:
    """Summarize elementwise FP32 differences between two raw outputs."""
    if pytorch_output.shape != onnx_output.shape:
        raise ValueError(
            "raw output shape mismatch: "
            f"PyTorch={pytorch_output.shape}, ONNX={onnx_output.shape}"
        )
    absolute_difference = np.abs(pytorch_output - onnx_output)
    return {
        "shape": list(pytorch_output.shape),
        "element_count": int(pytorch_output.size),
        "maximum_absolute_difference": float(absolute_difference.max()),
        "mean_absolute_difference": float(absolute_difference.mean()),
        "p99_absolute_difference": float(
            np.percentile(absolute_difference, 99)
        ),
        "allclose_atol_1e-4_rtol_1e-4": bool(
            np.allclose(
                pytorch_output,
                onnx_output,
                atol=1e-4,
                rtol=1e-4,
            )
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run both model formats and print their numerical comparison."""
    parser = argparse.ArgumentParser(
        description="Compare PyTorch and ONNX Runtime raw YOLO output."
    )
    parser.add_argument(
        "--pytorch-model",
        type=Path,
        default=Path("yolov8n.pt"),
    )
    parser.add_argument(
        "--onnx-model",
        type=Path,
        default=Path("yolov8n.onnx"),
    )
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/onnx_pytorch_raw_correctness.json"),
    )
    args = parser.parse_args(argv)
    for model_path in (args.pytorch_model, args.onnx_model):
        if not model_path.is_file():
            parser.error(f"model does not exist: {model_path}")
    if args.img_size <= 0:
        parser.error("--img-size must be greater than zero")

    input_array = deterministic_input(args.img_size)
    try:
        pytorch_output = run_pytorch(args.pytorch_model, input_array)
        onnx_output = run_onnx(args.onnx_model, input_array)
        summary = comparison_summary(pytorch_output, onnx_output)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"Saved raw-output report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
