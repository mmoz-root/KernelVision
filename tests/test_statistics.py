"""Tests for benchmark statistics."""

import pytest

from kernelvision.benchmarking.statistics import percentile, summarize


def test_summarize_reports_required_statistics() -> None:
    summary = summarize([1.0, 2.0, 3.0, 4.0])

    assert summary.count == 4
    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.p95 == pytest.approx(3.85)
    assert summary.minimum == 1.0
    assert summary.maximum == 4.0


def test_single_value_summary_uses_that_value_everywhere() -> None:
    summary = summarize([2.5])

    assert summary.mean == 2.5
    assert summary.median == 2.5
    assert summary.p95 == 2.5
    assert summary.minimum == 2.5
    assert summary.maximum == 2.5


@pytest.mark.parametrize("probability", [-0.01, 1.01])
def test_percentile_rejects_invalid_probability(probability: float) -> None:
    with pytest.raises(ValueError, match="probability"):
        percentile([1.0], probability)


def test_empty_measurements_are_rejected() -> None:
    with pytest.raises(ValueError, match="empty"):
        summarize([])
