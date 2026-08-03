"""Tests for benchmark report construction and serialization."""

import csv
import json
from pathlib import Path

import pytest

from kernelvision.benchmarking.report import (
    BenchmarkSample,
    build_report,
    save_csv_samples,
    save_json_report,
)


def _samples() -> list[BenchmarkSample]:
    return [
        BenchmarkSample(1, 1.0, 2.0, None, 3.0, 4.0, 10.0, 1.0, 12.0),
        BenchmarkSample(2, 2.0, 3.0, None, 4.0, 5.0, 12.0, 2.0, 16.0),
    ]


def test_build_report_summarizes_raw_samples() -> None:
    report = build_report({"device": "cpu"}, _samples())

    assert report.summary_ms["decode_ms"] == {
        "count": 2,
        "mean": 1.5,
        "median": 1.5,
        "p95": pytest.approx(1.95),
        "minimum": 1.0,
        "maximum": 2.0,
    }
    assert report.summary_ms["host_to_device_ms"] is None
    assert report.throughput_fps == pytest.approx(1000.0 / 14.0)


def test_reports_are_saved_as_json_and_csv(tmp_path: Path) -> None:
    samples = _samples()
    report = build_report({"device": "cpu"}, samples)
    json_path = save_json_report(report, tmp_path / "reports" / "report.json")
    csv_path = save_csv_samples(samples, tmp_path / "raw" / "samples.csv")

    json_data = json.loads(json_path.read_text(encoding="utf-8"))
    assert json_data["metadata"]["device"] == "cpu"
    assert len(json_data["samples"]) == 2

    with csv_path.open(encoding="utf-8", newline="") as csv_file:
        rows = list(csv.DictReader(csv_file))
    assert [row["iteration"] for row in rows] == ["1", "2"]
    assert rows[0]["host_to_device_ms"] == ""
