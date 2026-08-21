"""Compare final PyTorch and ONNX Runtime detections on one image."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from kernelvision.precision_comparison import compare_precision_results


def run_detection(
    model_path: Path,
    image_path: Path,
    *,
    image_size: int,
    confidence: float,
) -> Any:
    """Run the common Ultralytics preprocessing and postprocessing path."""
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "Ultralytics is required for detection comparison"
        ) from error

    results = YOLO(str(model_path), verbose=False).predict(
        source=str(image_path),
        imgsz=image_size,
        conf=confidence,
        rect=False,
        device="cpu",
        quantize=32,
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one result, received {len(results)}")
    return results[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run both formats, match detections, and save a JSON report."""
    from ultralytics.utils import ASSETS

    parser = argparse.ArgumentParser(
        description="Compare final PyTorch and ONNX YOLO detections."
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
    parser.add_argument("--image", type=Path, default=ASSETS / "bus.jpg")
    parser.add_argument("--img-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("results/onnx_pytorch_detection_correctness.json"),
    )
    args = parser.parse_args(argv)
    for path in (args.pytorch_model, args.onnx_model, args.image):
        if not path.is_file():
            parser.error(f"input does not exist: {path}")

    pytorch_result = run_detection(
        args.pytorch_model,
        args.image,
        image_size=args.img_size,
        confidence=args.conf,
    )
    onnx_result = run_detection(
        args.onnx_model,
        args.image,
        image_size=args.img_size,
        confidence=args.conf,
    )
    comparison = compare_precision_results(
        pytorch_result,
        onnx_result,
        minimum_iou=0.5,
    )
    report = {
        "pytorch_detection_count": comparison.fp32_detection_count,
        "onnx_detection_count": comparison.fp16_detection_count,
        "matched_detection_count": comparison.matched_detection_count,
        "unmatched_pytorch_count": comparison.unmatched_fp32_count,
        "unmatched_onnx_count": comparison.unmatched_fp16_count,
        "mean_matched_iou": comparison.mean_matched_iou,
        "minimum_matched_iou": comparison.minimum_matched_iou,
        "maximum_coordinate_difference_px": (
            comparison.maximum_coordinate_difference_px
        ),
        "maximum_confidence_difference": (
            comparison.maximum_confidence_difference
        ),
        "image": str(args.image),
        "image_size": args.img_size,
        "confidence": args.conf,
    }
    if (
        report["unmatched_pytorch_count"] != 0
        or report["unmatched_onnx_count"] != 0
    ):
        raise RuntimeError(f"detection correctness gate failed: {report}")

    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(serialized, encoding="utf-8")
    print(serialized, end="")
    print(f"Saved detection report to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
