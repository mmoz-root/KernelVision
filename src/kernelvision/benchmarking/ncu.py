"""Helpers for parsing Nsight Compute command-line CSV output."""

from __future__ import annotations

import csv
from typing import Any


def _ncu_rows(text: str) -> list[dict[str, Any]]:
    """Return normalized rows from Nsight Compute's mixed CSV output."""
    lines = text.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Section Name" in line
            and "Metric Name" in line
            and "Metric Value" in line
        ),
        None,
    )
    if header_index is None:
        raise ValueError("Nsight Compute output did not contain a metric table")

    reader = csv.DictReader(lines[header_index:])
    return [
        {
            str(key).strip(): value.strip() if value is not None else ""
            for key, value in row.items()
            if key is not None
        }
        for row in reader
    ]


def parse_ncu_csv(text: str) -> list[dict[str, Any]]:
    """Extract metric rows after Nsight Compute's CSV header."""
    rows = [row for row in _ncu_rows(text) if row.get("Metric Name")]
    if not rows:
        raise ValueError("Nsight Compute metric table was empty")
    return rows


def parse_ncu_rules(text: str) -> list[dict[str, Any]]:
    """Extract profiler recommendations and warnings from the CSV table."""
    return [row for row in _ncu_rows(text) if row.get("Rule Name")]
