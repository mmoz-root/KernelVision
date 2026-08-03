"""Statistical summaries for raw benchmark measurements."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import floor
from statistics import fmean, median


@dataclass(frozen=True, slots=True)
class MetricSummary:
    """Descriptive statistics for one latency metric."""

    count: int
    mean: float
    median: float
    p95: float
    minimum: float
    maximum: float

    def to_dict(self) -> dict[str, int | float]:
        """Return a JSON-serializable representation."""
        return asdict(self)


def percentile(values: list[float], probability: float) -> float:
    """Calculate a linearly interpolated percentile for non-empty values."""
    if not values:
        raise ValueError("cannot calculate a percentile for an empty sample")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between 0.0 and 1.0")

    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower_index = floor(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (
        ordered[upper_index] - ordered[lower_index]
    )


def summarize(values: list[float]) -> MetricSummary:
    """Summarize a non-empty collection of latency measurements."""
    if not values:
        raise ValueError("cannot summarize an empty sample")
    return MetricSummary(
        count=len(values),
        mean=fmean(values),
        median=median(values),
        p95=percentile(values, 0.95),
        minimum=min(values),
        maximum=max(values),
    )
