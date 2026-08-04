"""Command-line interface for KernelVision."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from kernelvision import __version__
from kernelvision.benchmarking.report import save_csv_samples, save_json_report
from kernelvision.benchmarking.runner import ImageBenchmarkConfig, run_image_benchmark
from kernelvision.config import ImageInferenceConfig, VideoInferenceConfig
from kernelvision.environment import collect_environment, format_environment
from kernelvision.pipeline import run_image_inference, run_video_inference


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level KernelVision argument parser."""
    parser = argparse.ArgumentParser(
        prog="kernelvision",
        description="Run and inspect the KernelVision inference pipeline.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser(
        "environment",
        help="report Python, package, and accelerator information",
    )
    image_parser = subparsers.add_parser(
        "image",
        help="run object detection on one image",
    )
    image_parser.add_argument(
        "--model",
        required=True,
        help="model path or Ultralytics model name",
    )
    image_parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="input image path",
    )
    image_parser.add_argument(
        "--device",
        default="cpu",
        help="inference device, such as cpu, mps, or 0",
    )
    image_parser.add_argument(
        "--conf",
        dest="confidence",
        default=0.25,
        type=float,
        help="confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    image_parser.add_argument(
        "--img-size",
        dest="image_size",
        default=640,
        type=int,
        help="square model input size in pixels (default: 640)",
    )
    image_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="model compute precision (default: fp32)",
    )
    image_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="annotated output image path",
    )
    video_parser = subparsers.add_parser(
        "video",
        help="run object detection over a video",
    )
    video_parser.add_argument(
        "--model",
        required=True,
        help="model path or Ultralytics model name",
    )
    video_parser.add_argument(
        "--input",
        dest="video",
        required=True,
        type=Path,
        help="input video path",
    )
    video_parser.add_argument(
        "--device",
        default="cpu",
        help="inference device, such as cpu, mps, or 0",
    )
    video_parser.add_argument(
        "--conf",
        dest="confidence",
        default=0.25,
        type=float,
        help="confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    video_parser.add_argument(
        "--img-size",
        dest="image_size",
        default=640,
        type=int,
        help="square model input size in pixels (default: 640)",
    )
    video_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="model compute precision (default: fp32)",
    )
    video_parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="annotated output video path",
    )
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="benchmark the baseline image-inference pipeline",
    )
    benchmark_parser.add_argument(
        "--model",
        required=True,
        help="model path or Ultralytics model name",
    )
    benchmark_parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="input image path",
    )
    benchmark_parser.add_argument(
        "--device",
        default="cpu",
        help="inference device, such as cpu, mps, or 0",
    )
    benchmark_parser.add_argument(
        "--conf",
        dest="confidence",
        default=0.25,
        type=float,
        help="confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    benchmark_parser.add_argument(
        "--img-size",
        dest="image_size",
        default=640,
        type=int,
        help="square model input size in pixels (default: 640)",
    )
    benchmark_parser.add_argument(
        "--precision",
        choices=("fp32", "fp16"),
        default="fp32",
        help="model compute precision (default: fp32)",
    )
    benchmark_parser.add_argument(
        "--warmup",
        dest="warmup_iterations",
        default=30,
        type=int,
        help="unmeasured warm-up iterations (default: 30)",
    )
    benchmark_parser.add_argument(
        "--iterations",
        dest="measured_iterations",
        default=200,
        type=int,
        help="measured iterations (default: 200)",
    )
    benchmark_parser.add_argument(
        "--json-out",
        default=Path("results/benchmark_report.json"),
        type=Path,
        help="JSON report path",
    )
    benchmark_parser.add_argument(
        "--csv-out",
        default=Path("benchmarks/raw/benchmark_samples.csv"),
        type=Path,
        help="raw sample CSV path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the KernelVision command-line interface."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "environment":
        print(format_environment(collect_environment()))
    elif args.command == "image":
        try:
            config = ImageInferenceConfig(
                model=args.model,
                image=args.image,
                output=args.output,
                device=args.device,
                confidence=args.confidence,
                image_size=args.image_size,
                precision=args.precision,
            )
            output = run_image_inference(config)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(f"Saved annotated image to {output}")
    elif args.command == "video":
        try:
            config = VideoInferenceConfig(
                model=args.model,
                video=args.video,
                output=args.output,
                device=args.device,
                confidence=args.confidence,
                image_size=args.image_size,
                precision=args.precision,
            )
            summary = run_video_inference(config)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))
        print(
            f"Saved {summary.frames} annotated frames to {summary.output} "
            f"at {summary.source_fps:.3f} FPS"
        )
    elif args.command == "benchmark":
        try:
            config = ImageBenchmarkConfig(
                model=args.model,
                image=args.image,
                device=args.device,
                confidence=args.confidence,
                image_size=args.image_size,
                precision=args.precision,
                warmup_iterations=args.warmup_iterations,
                measured_iterations=args.measured_iterations,
            )
            report = run_image_benchmark(config)
            json_output = save_json_report(report, args.json_out)
            csv_output = save_csv_samples(report.samples, args.csv_out)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            parser.error(str(error))

        end_to_end = report.summary_ms["end_to_end_ms"]
        inference = report.summary_ms["inference_ms"]
        assert end_to_end is not None and inference is not None
        print(
            f"End-to-end median={end_to_end['median']:.3f} ms, "
            f"P95={end_to_end['p95']:.3f} ms; "
            f"inference median={inference['median']:.3f} ms; "
            f"throughput={report.throughput_fps:.3f} FPS"
        )
        print(f"Saved JSON report to {json_output}")
        print(f"Saved raw CSV samples to {csv_output}")
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
