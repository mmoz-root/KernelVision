"""Export the fixed-shape YOLOv8n ONNX graph used in Milestone 7."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def export_fixed_onnx(
    model_path: Path,
    *,
    image_size: int,
    opset: int,
) -> Path:
    """Export an FP32, batch-one, static-shape graph without embedded NMS."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Ultralytics is required for ONNX export. Install the inference "
            "dependencies first."
        ) from error

    model = YOLO(str(model_path), verbose=False)

    exported_path = model.export(
        format="onnx",
        imgsz=image_size,
        batch=1,
        dynamic=False,
        simplify=False,
        opset=opset,
        nms=False,
        quantize=32,
        device="cpu",  # Export creates a device-independent graph description.
    )
    return Path(exported_path)


def main(argv: Sequence[str] | None = None) -> int:
    """Parse export settings and print the generated ONNX artifact path."""
    parser = argparse.ArgumentParser(
        description="Export KernelVision's fixed-shape YOLO ONNX graph."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("yolov8n.pt"),
        help="PyTorch model path (default: yolov8n.pt)",
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="fixed square model input size (default: 640)",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX operator-set version (default: 17)",
    )
    args = parser.parse_args(argv)

    if not args.model.is_file():
        parser.error(f"model does not exist: {args.model}")
    if args.img_size <= 0:
        parser.error("--img-size must be greater than zero")
    if args.opset <= 0:
        parser.error("--opset must be greater than zero")

    try:
        output = export_fixed_onnx(
            args.model,
            image_size=args.img_size,
            opset=args.opset,
        )
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))
    if not output.is_file():
        parser.error(f"export did not create the expected file: {output}")

    print(f"Exported ONNX graph to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
