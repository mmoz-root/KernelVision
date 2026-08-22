"""Compare final PyTorch FP32 and TensorRT FP16 detections on an L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
PYTORCH_MODEL = PROJECT_ROOT / "yolov8n.pt"
FP16_ENGINE_MODEL = PROJECT_ROOT / "artifacts/yolov8n_fp16.engine"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115", "tensorrt>=11.2,<11.3")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_file(PYTORCH_MODEL, remote_path="/root/yolov8n.pt")
    .add_local_file(
        FP16_ENGINE_MODEL,
        remote_path="/root/yolov8n_fp16.engine",
    )
)

app = modal.App("kernelvision-tensorrt-detection-comparison")


@app.function(image=runtime_image, gpu="L4", timeout=20 * 60)
def compare_detections_l4(
    image_asset: str,
    confidence: float,
    image_size: int,
    minimum_iou: float,
) -> dict[str, Any]:
    """Compare post-NMS detections from FP32 PyTorch and FP16 TensorRT."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import ASSETS

    from kernelvision.precision_comparison import compare_precision_results

    image_path = ASSETS / image_asset
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"could not decode comparison image: {image_path}")

    # Milestone 7 learning exercise:
    # 1. Load the PyTorch checkpoint and FP16 TensorRT engine once each.
    # 2. Run predict on the same decoded frame with identical confidence,
    #    image size, rect=False, device="0", and verbose=False settings.
    # 3. Request FP32 for the PyTorch path. TensorRT obtains FP16 from its
    #    strongly typed engine contract.
    # 4. Require exactly one result from each call and select result index 0.
    pytorch_model = YOLO(
        "/root/yolov8n.pt",
        task="detect",
        verbose=False
    )
    tensorrt_model = YOLO(
        "/root/yolov8n_fp16.engine",
        task="detect",
        verbose=False
    )

    pytorch_results = pytorch_model.predict(
        source=frame,
        imgsz=image_size,
        conf=confidence,
        rect=False,
        device="0",
        quantize=32,
        verbose=False,
    )
    tensorrt_results = tensorrt_model.predict(
        source=frame,
        imgsz=image_size,
        conf=confidence,
        rect=False,
        device="0",
        verbose=False,
    )

    if len(pytorch_results) != 1 or len(tensorrt_results) != 1:
        raise RuntimeError(
            f"expected one result per backend, received "
            f"PyTorch={len(pytorch_results)}, "
            f"TensorRT={len(tensorrt_results)}"
        )

    pytorch_result = pytorch_results[0]
    tensorrt_result = tensorrt_results[0]


    comparison = compare_precision_results(
        pytorch_result,
        tensorrt_result,
        minimum_iou=minimum_iou,
    )
    report = {
        "experiment": "PyTorch FP32 vs TensorRT FP16 detection correctness",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "gpu": torch.cuda.get_device_name(),
        "image": image_asset,
        "source_shape_hwc": list(frame.shape),
        "image_size": image_size,
        "confidence": confidence,
        "reference": "PyTorch FP32",
        "candidate": "TensorRT FP16",
        "comparison": comparison.to_dict(),
    }
    if (
        comparison.unmatched_fp32_count != 0
        or comparison.unmatched_fp16_count != 0
    ):
        raise RuntimeError(f"detection correctness gate failed: {report}")
    return report


@app.local_entrypoint()
def main(
    image_asset: str = "bus.jpg",
    confidence: float = 0.25,
    image_size: int = 640,
    minimum_iou: float = 0.5,
    json_out: str = "results/modal_l4_tensorrt_fp16_detections.json",
) -> None:
    """Run the final-detection gate and save its report locally."""
    if not PYTORCH_MODEL.is_file():
        raise FileNotFoundError(f"PyTorch model does not exist: {PYTORCH_MODEL}")
    if not FP16_ENGINE_MODEL.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: {FP16_ENGINE_MODEL}"
        )

    report = compare_detections_l4.remote(
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
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved TensorRT detection report to {output}")


if __name__ == "__main__":
    main()
