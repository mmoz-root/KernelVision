"""Convert the FP32 YOLO ONNX graph to mixed FP16/FP32 with ModelOpt."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FP32_ONNX_MODEL = PROJECT_ROOT / "yolov8n.onnx"

autocast_image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("nvidia-modelopt[onnx]==0.45.0")
    .add_local_file(FP32_ONNX_MODEL, remote_path="/root/yolov8n.onnx")
)

app = modal.App("kernelvision-onnx-autocast")


@app.function(image=autocast_image, timeout=20 * 60)
def convert_fp16_onnx() -> dict[str, Any]:
    """Create a strongly typed mixed FP16/FP32 ONNX graph."""
    import importlib.metadata
    import subprocess
    import sys

    source = Path("/root/yolov8n.onnx")
    output = Path("/tmp/yolov8n_mixed_fp16.onnx")

    # Milestone 7 learning exercise:
    # 1. Build the ModelOpt AutoCast command as a list of strings.
    # 2. Request FP16 low precision and preserve ONNX opset 17.
    # 3. Run the command with subprocess.run(..., capture_output=True,
    #    text=True), then raise a useful error if it fails.
    command = [
        sys.executable,
        "-m",
        "modelopt.onnx.autocast",
        "--onnx_path",
        str(source),
        "--output_path",
        str(output),
        "--low_precision_type",
        "fp16",
        "--opset",
        "17",
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if completed.returncode != 0:
        raise RuntimeError(f"{completed.stdout}, {completed.stderr}")

    if not output.is_file():
        raise RuntimeError(f"ModelOpt did not create the expected file: {output}")

    model_bytes = output.read_bytes()
    return {
        "model_bytes": model_bytes,
        "report": {
            "experiment": "ModelOpt ONNX FP16 AutoCast",
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
            "source_model": source.name,
            "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "output_model": output.name,
            "output_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "output_size_bytes": len(model_bytes),
            "low_precision_type": "fp16",
            "opset": 17,
        },
    }


@app.local_entrypoint()
def main(
    onnx_out: str = "artifacts/yolov8n_mixed_fp16.onnx",
    json_out: str = "results/modal_onnx_fp16_autocast.json",
) -> None:
    """Run AutoCast remotely and save the converted graph and report."""
    if not FP32_ONNX_MODEL.is_file():
        raise FileNotFoundError(f"ONNX model does not exist: {FP32_ONNX_MODEL}")

    payload = convert_fp16_onnx.remote()
    model_bytes = payload["model_bytes"]
    report = payload["report"]

    model_output = Path(onnx_out)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_bytes(model_bytes)

    report_output = Path(json_out)
    report_output.parent.mkdir(parents=True, exist_ok=True)
    report_output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"Saved mixed-precision ONNX graph to {model_output}")
    print(f"Saved AutoCast report to {report_output}")


if __name__ == "__main__":
    main()
