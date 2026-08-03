"""Benchmark sample models and report serialization."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from kernelvision.benchmarking.statistics import summarize


@dataclass(frozen=True, slots=True)
class BenchmarkSample:
    """Raw component timings for one measured iteration, in milliseconds."""

    iteration: int
    decode_ms: float
    preprocess_ms: float
    host_to_device_ms: float | None
    inference_ms: float
    postprocess_ms: float
    backend_ms: float
    visualization_ms: float
    end_to_end_ms: float

    def to_dict(self) -> dict[str, int | float | None]:
        """Return a serialization-friendly sample mapping."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    """Reproducibility metadata, summaries, and raw benchmark measurements."""

    metadata: dict[str, Any]
    summary_ms: dict[str, dict[str, int | float] | None]
    throughput_fps: float
    samples: list[BenchmarkSample]

    def to_dict(self) -> dict[str, Any]:
        """Return the full report as JSON-compatible data."""
        return {
            "metadata": self.metadata,
            "summary_ms": self.summary_ms,
            "throughput_fps": self.throughput_fps,
            "samples": [sample.to_dict() for sample in self.samples],
        }


def build_report(
    metadata: dict[str, Any],
    samples: list[BenchmarkSample],
) -> BenchmarkReport:
    """Build summaries and throughput from non-empty raw measurements."""
    if not samples:
        raise ValueError("cannot build a benchmark report without samples")

    metric_names = (
        "decode_ms",
        "preprocess_ms",
        "host_to_device_ms",
        "inference_ms",
        "postprocess_ms",
        "backend_ms",
        "visualization_ms",
        "end_to_end_ms",
    )
    summary_ms: dict[str, dict[str, int | float] | None] = {}
    for metric_name in metric_names:
        values = [
            float(value)
            for sample in samples
            if (value := getattr(sample, metric_name)) is not None
        ]
        summary_ms[metric_name] = summarize(values).to_dict() if values else None

    end_to_end_summary = summary_ms["end_to_end_ms"]
    assert end_to_end_summary is not None
    mean_end_to_end_ms = float(end_to_end_summary["mean"])
    throughput_fps = 1000.0 / mean_end_to_end_ms
    return BenchmarkReport(
        metadata=metadata,
        summary_ms=summary_ms,
        throughput_fps=throughput_fps,
        samples=samples,
    )


def save_json_report(report: BenchmarkReport, output: Path) -> Path:
    """Write a benchmark report as indented JSON."""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output


def save_csv_samples(samples: list[BenchmarkSample], output: Path) -> Path:
    """Write raw per-iteration measurements as CSV."""
    if not samples:
        raise ValueError("cannot save an empty benchmark sample collection")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [sample.to_dict() for sample in samples]
    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return output
