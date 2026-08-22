"""Create reproducible final-result plots from KernelVision JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"benchmark report does not exist: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _annotate_bars(axis: Any, bars: Any) -> None:
    for bar in bars:
        value = float(bar.get_height())
        axis.annotate(
            f"{value:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2.0, value),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_tensorrt_latency(
    model_report: dict[str, Any],
    pipeline_report: dict[str, Any],
    output: Path,
) -> None:
    """Plot model-only and complete-pipeline latency as separate panels."""
    import matplotlib.pyplot as plt

    model_labels = ("PyTorch FP32", "PyTorch FP16", "TensorRT FP16")
    model_keys = ("pytorch_fp32", "pytorch_fp16", "tensorrt_fp16")
    pipeline_labels = ("PyTorch FP32", "TensorRT FP16")
    pipeline_keys = ("pytorch_fp32", "tensorrt_fp16")
    colors = ("#5276A7", "#77AADD", "#49A078")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    figure.suptitle("YOLOv8n latency on NVIDIA L4 (batch 1, 640×640)")

    model_summaries = model_report["summaries_ms"]
    model_medians = [
        float(model_summaries[key]["median"]) for key in model_keys
    ]
    model_p95 = [float(model_summaries[key]["p95"]) for key in model_keys]
    model_errors = [
        maximum - median
        for maximum, median in zip(model_p95, model_medians, strict=True)
    ]
    model_bars = axes[0].bar(
        model_labels,
        model_medians,
        yerr=([0.0] * len(model_errors), model_errors),
        capsize=4,
        color=colors,
    )
    axes[0].set_title("Raw model execution")
    axes[0].set_ylabel("Median latency (ms); upper whisker = P95")
    axes[0].tick_params(axis="x", rotation=18)
    axes[0].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[0], model_bars)

    pipeline_summaries = pipeline_report["summaries_ms"]
    pipeline_medians = [
        float(pipeline_summaries[key]["median"])
        for key in pipeline_keys
    ]
    pipeline_p95 = [
        float(pipeline_summaries[key]["p95"])
        for key in pipeline_keys
    ]
    pipeline_errors = [
        maximum - median
        for maximum, median in zip(
            pipeline_p95,
            pipeline_medians,
            strict=True,
        )
    ]
    pipeline_bars = axes[1].bar(
        pipeline_labels,
        pipeline_medians,
        yerr=([0.0] * len(pipeline_errors), pipeline_errors),
        capsize=4,
        color=(colors[0], colors[2]),
    )
    axes[1].set_title("Decode-to-visualization pipeline")
    axes[1].set_ylabel("Median latency (ms); upper whisker = P95")
    axes[1].tick_params(axis="x", rotation=18)
    axes[1].grid(axis="y", alpha=0.25)
    _annotate_bars(axes[1], pipeline_bars)

    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def plot_preprocessing_latency(
    cuda_report: dict[str, Any],
    output: Path,
) -> None:
    """Plot final FP16 preprocessing implementations at 640×640."""
    import matplotlib.pyplot as plt

    selected_case = next(
        case
        for case in cuda_report["cases"]
        if case["height"] == 640
        and case["width"] == 640
        and case["dtype"] == "fp16"
    )
    labels_and_keys = (
        ("PyTorch", "pytorch"),
        ("Triton", "triton"),
        ("Naive CUDA", "naive_cuda"),
        ("Warp-packed", "warp_packed_cuda"),
        ("Coalesced + shared", "coalesced_cuda"),
    )
    labels = [label for label, _ in labels_and_keys]
    medians = [
        float(selected_case["implementations"][key]["summary_ms"]["median"])
        for _, key in labels_and_keys
    ]
    p95 = [
        float(selected_case["implementations"][key]["summary_ms"]["p95"])
        for _, key in labels_and_keys
    ]
    errors = [
        maximum - median
        for maximum, median in zip(p95, medians, strict=True)
    ]

    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    bars = axis.bar(
        labels,
        medians,
        yerr=([0.0] * len(errors), errors),
        capsize=4,
        color=("#5276A7", "#77AADD", "#49A078", "#D9A441", "#C56A5A"),
    )
    axis.set_title("FP16 preprocessing latency on NVIDIA L4 (640×640)")
    axis.set_ylabel("Median kernel latency (ms); upper whisker = P95")
    axis.tick_params(axis="x", rotation=16)
    axis.grid(axis="y", alpha=0.25)
    _annotate_bars(axis, bars)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot final KernelVision benchmark reports."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results/figures",
    )
    args = parser.parse_args()

    model_report = _load_json(
        args.results_dir / "modal_l4_tensorrt_model_benchmark.json"
    )
    pipeline_report = _load_json(
        args.results_dir / "modal_l4_tensorrt_end_to_end_benchmark.json"
    )
    cuda_report = _load_json(
        args.results_dir / "modal_l4_cuda_final_benchmark.json"
    )

    tensor_output = args.output_dir / "tensorrt_latency.png"
    preprocessing_output = args.output_dir / "preprocessing_latency.png"
    plot_tensorrt_latency(model_report, pipeline_report, tensor_output)
    plot_preprocessing_latency(cuda_report, preprocessing_output)
    print(f"Saved TensorRT latency plot to {tensor_output}")
    print(f"Saved preprocessing latency plot to {preprocessing_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
