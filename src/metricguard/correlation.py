"""Deterministic cross-metric correlation analysis for evaluation reports."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import SuiteReport


@dataclass(frozen=True, slots=True)
class MetricCorrelation:
    """Correlation between two reports over pairwise-complete case scores."""

    left: str
    right: str
    count: int
    pearson: float | None
    spearman: float | None
    reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-compatible representation."""

        return {
            "left": self.left,
            "right": self.right,
            "count": self.count,
            "pearson": self.pearson,
            "spearman": self.spearman,
            "reason": self.reason,
        }


@dataclass(frozen=True, slots=True)
class CorrelationReport:
    """A symmetric correlation matrix with an auditable pair list."""

    metrics: tuple[str, ...]
    pairs: tuple[MetricCorrelation, ...]

    def pair(self, left: str, right: str) -> MetricCorrelation:
        """Return one pair regardless of argument order."""

        if left == right:
            raise ValueError("correlation pairs must contain two distinct metrics")
        first, second = sorted((left, right))
        for item in self.pairs:
            if item.left == first and item.right == second:
                return item
        raise KeyError(f"unknown metric pair: {left!r}, {right!r}")

    def to_dict(self) -> dict[str, Any]:
        """Return a stable matrix and pair-list representation."""

        pearson: dict[str, dict[str, float | None]] = {
            name: {other: None for other in self.metrics} for name in self.metrics
        }
        spearman: dict[str, dict[str, float | None]] = {
            name: {other: None for other in self.metrics} for name in self.metrics
        }
        counts = {name: {other: 0 for other in self.metrics} for name in self.metrics}
        for name in self.metrics:
            pearson[name][name] = 1.0
            spearman[name][name] = 1.0
            counts[name][name] = 0
        for item in self.pairs:
            pearson[item.left][item.right] = pearson[item.right][item.left] = item.pearson
            spearman[item.left][item.right] = spearman[item.right][item.left] = item.spearman
            counts[item.left][item.right] = counts[item.right][item.left] = item.count
        return {
            "schema_version": 1,
            "metrics": list(self.metrics),
            "pearson": pearson,
            "spearman": spearman,
            "pair_counts": counts,
            "pairs": [item.to_dict() for item in self.pairs],
        }


def correlate_reports(
    reports: Mapping[str, SuiteReport], *, minimum_count: int = 2
) -> CorrelationReport:
    """Correlate metric reports using pairwise-complete case scores.

    Case IDs are used for alignment, so report order does not affect the result.
    Undefined or skipped scores are omitted only from the pair that contains them;
    this preserves useful information when metrics have different undefined domains.
    Pearson uses the population denominator because these values describe the
    observed evaluation suite. Spearman uses average ranks for ties. A constant
    vector, or fewer than ``minimum_count`` complete pairs, yields ``None`` with
    an explicit reason rather than a misleading zero.
    """

    if not isinstance(reports, Mapping) or not reports:
        raise ValueError("reports must be a non-empty mapping")
    if isinstance(minimum_count, bool) or not isinstance(minimum_count, int) or minimum_count < 2:
        raise ValueError("minimum_count must be an integer of at least 2")
    names = tuple(sorted(reports))
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("report names must be non-empty strings")
    values: dict[str, dict[str, float]] = {}
    for name in names:
        report = reports[name]
        if not isinstance(report, SuiteReport):
            raise TypeError("reports must contain SuiteReport values")
        rows: dict[str, float] = {}
        for result in report.results:
            if result.case_id in rows:
                raise ValueError(f"report {name!r} contains duplicate case ID {result.case_id!r}")
            if result.resolved_score is not None:
                rows[result.case_id] = result.resolved_score
        values[name] = rows

    pairs: list[MetricCorrelation] = []
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared = sorted(set(values[left]) & set(values[right]))
            x = [values[left][case_id] for case_id in shared]
            y = [values[right][case_id] for case_id in shared]
            if len(x) < minimum_count:
                pairs.append(
                    MetricCorrelation(
                        left,
                        right,
                        len(x),
                        None,
                        None,
                        f"fewer than {minimum_count} pairwise-complete cases",
                    )
                )
                continue
            pearson, pearson_reason = _correlation(x, y)
            spearman, spearman_reason = _correlation(_ranks(x), _ranks(y))
            reason = pearson_reason or spearman_reason
            pairs.append(MetricCorrelation(left, right, len(x), pearson, spearman, reason))
    return CorrelationReport(names, tuple(pairs))


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: (item[1], item[0]))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(ordered):
        end = position + 1
        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1
        average = (position + 1 + end) / 2
        for item_index in range(position, end):
            ranks[ordered[item_index][0]] = average
        position = end
    return ranks


def _correlation(left: list[float], right: list[float]) -> tuple[float | None, str | None]:
    left_mean = math.fsum(left) / len(left)
    right_mean = math.fsum(right) / len(right)
    centered_left = [value - left_mean for value in left]
    centered_right = [value - right_mean for value in right]
    left_norm = math.sqrt(math.fsum(value * value for value in centered_left))
    right_norm = math.sqrt(math.fsum(value * value for value in centered_right))
    if left_norm == 0 or right_norm == 0:
        return None, "correlation undefined for a constant score vector"
    value = math.fsum(a * b for a, b in zip(centered_left, centered_right, strict=True)) / (
        left_norm * right_norm
    )
    return max(-1.0, min(1.0, value)), None


__all__ = ["CorrelationReport", "MetricCorrelation", "correlate_reports"]
