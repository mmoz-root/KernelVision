"""Generate an annotated TensorRT FP16 image demo on a Modal L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
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
    )
    .env(
        {
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_file(
        FP16_ENGINE_MODEL,
        remote_path="/root/yolov8n_fp16.engine",
    )
)

app = modal.App("kernelvision-tensorrt-demo")


@app.function(image=runtime_image, gpu="L4", timeout=20 * 60)
def generate_demo_l4(
    image_asset: str,
    confidence: float,
    image_size: int,
) -> dict[str, Any]:
    """Run the TensorRT engine and return an annotated JPEG plus metadata."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.autobackend import default_class_names
    from ultralytics.utils import ASSETS

    image_path = ASSETS / image_asset
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"could not decode demo image: {image_path}")

    model = YOLO(
        "/root/yolov8n_fp16.engine",
        task="detect",
        verbose=False,
    )
    results = model.predict(
        source=frame,
        imgsz=image_size,
        conf=confidence,
        rect=False,
        device="0",
        verbose=False,
    )
    if len(results) != 1:
        raise RuntimeError(f"expected one result, received {len(results)}")

    result = results[0]
    result.names = default_class_names("coco8.yaml")
    annotated = result.plot()
    encoded, jpeg = cv2.imencode(".jpg", annotated)
    if not encoded:
        raise RuntimeError("failed to encode annotated TensorRT image")

    detections = []
    for box, confidence_value, class_id in zip(
        result.boxes.xyxy.detach().cpu().tolist(),
        result.boxes.conf.detach().cpu().tolist(),
        result.boxes.cls.detach().cpu().tolist(),
        strict=True,
    ):
        integer_class_id = int(class_id)
        detections.append(
            {
                "box_xyxy": [float(coordinate) for coordinate in box],
                "class_id": integer_class_id,
                "class_name": result.names[integer_class_id],
                "confidence": float(confidence_value),
            }
        )

    return {
        "image_bytes": jpeg.tobytes(),
        "report": {
            "experiment": "TensorRT FP16 annotated-image demo",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "gpu": torch.cuda.get_device_name(),
            "backend": "TensorRT FP16 engine",
            "image": image_asset,
            "source_shape_hwc": list(frame.shape),
            "image_size": image_size,
            "confidence": confidence,
            "detection_count": len(detections),
            "detections": detections,
            "stage_speed_ms": {
                name: float(value)
                for name, value in result.speed.items()
            },
        },
    }


@app.local_entrypoint()
def main(
    image_asset: str = "bus.jpg",
    confidence: float = 0.25,
    image_size: int = 640,
    image_out: str = "results/tensorrt_fp16_bus.jpg",
    json_out: str = "results/tensorrt_fp16_bus_demo.json",
) -> None:
    """Generate and save the final annotated-image demo."""
    if not FP16_ENGINE_MODEL.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: {FP16_ENGINE_MODEL}"
        )

    payload = generate_demo_l4.remote(
        image_asset,
        confidence,
        image_size,
    )
    image_output = Path(image_out)
    image_output.parent.mkdir(parents=True, exist_ok=True)
    image_output.write_bytes(payload["image_bytes"])

    report_output = Path(json_out)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(payload["report"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["report"], indent=2, sort_keys=True))
    print(f"Saved annotated TensorRT image to {image_output}")
    print(f"Saved TensorRT demo report to {report_output}")


if __name__ == "__main__":
    main()
