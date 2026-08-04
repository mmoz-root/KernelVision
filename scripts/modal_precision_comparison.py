"""Compare FP32 and FP16 YOLO detections on a Modal NVIDIA L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"

runtime_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
)

app = modal.App("kernelvision-precision-comparison")


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def compare_l4(
    model: str,
    image_asset: str,
    confidence: float,
    image_size: int,
    minimum_iou: float,
) -> dict[str, Any]:
    """Run both precisions on one decoded image and compare detections."""
    import cv2
    from ultralytics.utils import ASSETS

    from kernelvision.backends import UltralyticsBackend
    from kernelvision.environment import collect_environment
    from kernelvision.precision_comparison import compare_precision_results

    image = ASSETS / image_asset
    frame = cv2.imread(str(image))
    if frame is None:
        raise RuntimeError(f"could not decode comparison image: {image}")

    fp32_backend = UltralyticsBackend(model)
    fp16_backend = UltralyticsBackend(model)
    fp32_result = fp32_backend.predict(
        frame,
        confidence=confidence,
        image_size=image_size,
        device="0",
        precision="fp32",
    )
    fp16_result = fp16_backend.predict(
        frame,
        confidence=confidence,
        image_size=image_size,
        device="0",
        precision="fp16",
    )
    comparison = compare_precision_results(
        fp32_result,
        fp16_result,
        minimum_iou=minimum_iou,
    )
    return {
        "metadata": {
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "model": model,
            "image": image_asset,
            "source_shape_hwc": list(frame.shape),
            "model_input_size": image_size,
            "confidence": confidence,
            "device": "0",
            "gpu": "NVIDIA L4",
            "reference_precision": "fp32",
            "candidate_precision": "fp16",
            "environment": collect_environment(),
        },
        "comparison": comparison.to_dict(),
    }


@app.local_entrypoint()
def main(
    model: str = "yolov8n.pt",
    image_asset: str = "bus.jpg",
    confidence: float = 0.25,
    image_size: int = 640,
    minimum_iou: float = 0.5,
    json_out: str = "results/modal_l4_fp32_vs_fp16_correctness.json",
) -> None:
    """Launch the L4 comparison and save its report locally."""
    report = compare_l4.remote(
        model,
        image_asset,
        confidence,
        image_size,
        minimum_iou,
    )
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    comparison = report["comparison"]
    print(
        "Detection counts: "
        f"FP32={comparison['fp32_detection_count']}, "
        f"FP16={comparison['fp16_detection_count']}, "
        f"matched={comparison['matched_detection_count']}"
    )
    print(
        f"Matched IoU mean={comparison['mean_matched_iou']:.6f}, "
        f"minimum={comparison['minimum_matched_iou']:.6f}; "
        "maximum coordinate difference="
        f"{comparison['maximum_coordinate_difference_px']:.6f} px; "
        "maximum confidence difference="
        f"{comparison['maximum_confidence_difference']:.6f}"
    )
    print(f"Saved correctness report to {output}")
