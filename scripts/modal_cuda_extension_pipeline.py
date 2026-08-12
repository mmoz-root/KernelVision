"""Validate CUDA-extension preprocessing inside YOLO on a Modal L4."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = PROJECT_ROOT / "src"
CUDA_DIR = PROJECT_ROOT / "csrc"

runtime_image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("libgl1", "libglib2.0-0")
    .uv_pip_install("ultralytics==8.4.115", "ninja")
    .env(
        {
            "PYTHONPATH": "/root/src",
            "TORCH_EXTENSIONS_DIR": "/tmp/torch_extensions",
            "YOLO_CONFIG_DIR": "/tmp/ultralytics",
            "MPLCONFIGDIR": "/tmp/matplotlib",
            "XDG_CACHE_HOME": "/tmp/cache",
        }
    )
    .add_local_dir(SOURCE_DIR, remote_path="/root/src")
    .add_local_dir(CUDA_DIR, remote_path="/root/csrc")
)

app = modal.App("kernelvision-cuda-extension-pipeline")


def _intersection_over_union(first: list[float], second: list[float]) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(
        0.0, first[3] - first[1]
    )
    second_area = max(0.0, second[2] - second[0]) * max(
        0.0, second[3] - second[1]
    )
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _match_detections(
    baseline_rows: list[list[float]],
    extension_rows: list[list[float]],
) -> list[tuple[list[float], list[float], float]]:
    """Greedily pair same-class boxes by highest IoU, independent of order."""
    if len(baseline_rows) != len(extension_rows):
        raise RuntimeError(
            "detection count changed: "
            f"baseline={len(baseline_rows)}, extension={len(extension_rows)}"
        )

    unmatched = list(extension_rows)
    matches = []
    for baseline in baseline_rows:
        same_class = [
            (index, candidate)
            for index, candidate in enumerate(unmatched)
            if int(candidate[5]) == int(baseline[5])
        ]
        if not same_class:
            raise RuntimeError(
                f"no extension detection matched class {int(baseline[5])}"
            )
        best_index, best_candidate = max(
            same_class,
            key=lambda indexed: _intersection_over_union(
                baseline,
                indexed[1],
            ),
        )
        iou = _intersection_over_union(baseline, best_candidate)
        matches.append((baseline, best_candidate, iou))
        unmatched.pop(best_index)
    return matches


@app.function(image=runtime_image, gpu="L4", timeout=30 * 60)
def validate_pipeline_l4(model: str, image_asset: str) -> dict[str, Any]:
    """Compare complete detections using standard and extension preprocessing."""
    import torch
    from ultralytics.utils import ASSETS

    from kernelvision.backends import UltralyticsBackend
    from kernelvision.environment import collect_environment

    image = ASSETS / image_asset
    precision_cases = []
    for precision in ("fp32", "fp16"):
        baseline_backend = UltralyticsBackend(
            model,
            preprocessor="ultralytics",
        )
        extension_backend = UltralyticsBackend(
            model,
            preprocessor="cuda_extension",
        )
        common = {
            "confidence": 0.25,
            "image_size": 640,
            "device": "0",
            "precision": precision,
        }
        baseline = baseline_backend.predict(image, **common)
        extension = extension_backend.predict(image, **common)
        torch.cuda.synchronize()

        baseline_rows = baseline.boxes.data.detach().float().cpu().tolist()
        extension_rows = extension.boxes.data.detach().float().cpu().tolist()
        matches = _match_detections(baseline_rows, extension_rows)
        maximum_box_difference = max(
            (
                max(abs(first[index] - second[index]) for index in range(4))
                for first, second, _ in matches
            ),
            default=0.0,
        )
        maximum_confidence_difference = max(
            (abs(first[4] - second[4]) for first, second, _ in matches),
            default=0.0,
        )
        minimum_iou = min((iou for _, _, iou in matches), default=1.0)
        passed = (
            maximum_box_difference <= 1e-3
            and maximum_confidence_difference <= 1e-5
            and minimum_iou >= 0.99999
        )
        precision_cases.append(
            {
                "precision": precision,
                "passed": passed,
                "detection_count": len(matches),
                "class_ids": [int(row[5]) for row in baseline_rows],
                "maximum_box_difference": maximum_box_difference,
                "maximum_confidence_difference": (
                    maximum_confidence_difference
                ),
                "minimum_iou": minimum_iou,
                "baseline_speed_ms": baseline.speed,
                "extension_speed_ms": extension.speed,
            }
        )

    if not all(case["passed"] for case in precision_cases):
        raise RuntimeError(f"pipeline correctness failure: {precision_cases}")

    return {
        "experiment": "CUDA extension YOLO pipeline correctness",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model": model,
        "image": image_asset,
        "device": "NVIDIA L4",
        "confidence": 0.25,
        "image_size": 640,
        "matching": "same class, then maximum intersection-over-union",
        "tolerances": {
            "box_max_abs": 1e-3,
            "confidence_max_abs": 1e-5,
            "minimum_iou": 0.99999,
        },
        "cases": precision_cases,
        "environment": collect_environment(),
    }


@app.local_entrypoint()
def main(
    model: str = "yolov8n.pt",
    image_asset: str = "bus.jpg",
    json_out: str = "results/modal_l4_cuda_extension_pipeline.json",
) -> None:
    """Run pipeline validation and save its report locally."""
    report = validate_pipeline_l4.remote(model, image_asset)
    output = Path(json_out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for case in report["cases"]:
        print(
            f"{case['precision']}: {case['detection_count']} detections, "
            f"box max diff={case['maximum_box_difference']:.8f}, "
            f"confidence max diff="
            f"{case['maximum_confidence_difference']:.8f}, "
            f"minimum IoU={case['minimum_iou']:.8f}"
        )
    print(f"Saved pipeline correctness report to {output}")


if __name__ == "__main__":
    main()
