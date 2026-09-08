"""Confidence calibration and reliability summaries for evaluation cases."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from .models import EvaluationCase


@dataclass(frozen=True, slots=True)
class CalibrationBin:
    """One equal-width confidence bin."""

    index: int
    lower: float
    upper: float
    count: int
    mean_confidence: float | None
    accuracy: float | None
    gap: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "mean_confidence": self.mean_confidence,
            "accuracy": self.accuracy,
            "gap": self.gap,
        }


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Reliability diagnostics for binary outcomes and confidence values."""

    confidence_field: str
    outcome_field: str
    bin_count: int
    case_count: int
    brier_score: float
    expected_calibration_error: float
    maximum_calibration_error: float
    bins: tuple[CalibrationBin, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "operation": "calibration",
            "confidence_field": self.confidence_field,
            "outcome_field": self.outcome_field,
            "bin_count": self.bin_count,
            "case_count": self.case_count,
            "brier_score": self.brier_score,
            "expected_calibration_error": self.expected_calibration_error,
            "maximum_calibration_error": self.maximum_calibration_error,
            "bins": [item.to_dict() for item in self.bins],
        }


def calibration_report(
    cases: Iterable[EvaluationCase],
    *,
    confidence_field: str = "confidence",
    outcome_field: str = "correct",
    bins: int = 10,
) -> CalibrationReport:
    """Compute deterministic Brier, ECE, and MCE diagnostics.

    Confidence and boolean outcome values are read from case metadata. Dotted
    field paths support nested metadata (for example ``model.confidence``).
    Missing, non-finite, or out-of-range confidence values fail explicitly so a
    reliability report cannot silently change its evaluation population.
    """

    _validate_field(confidence_field, "confidence_field")
    _validate_field(outcome_field, "outcome_field")
    if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 100:
        raise ValueError("bins must be an integer between 2 and 100")
    buckets: list[list[tuple[float, bool]]] = [[] for _ in range(bins)]
    seen: set[str] = set()
    brier_total = 0.0
    count = 0
    for case in cases:
        if not isinstance(case, EvaluationCase):
            raise TypeError("cases must contain EvaluationCase values")
        if case.case_id in seen:
            raise ValueError(f"cases contain duplicate ID {case.case_id!r}")
        seen.add(case.case_id)
        confidence = _confidence(_metadata_value(case.metadata, confidence_field), case.case_id)
        outcome = _outcome(_metadata_value(case.metadata, outcome_field), case.case_id)
        index = min(bins - 1, int(confidence * bins))
        buckets[index].append((confidence, outcome))
        brier_total += (confidence - float(outcome)) ** 2
        count += 1
    if not count:
        raise ValueError("calibration requires at least one case")
    summaries: list[CalibrationBin] = []
    weighted_gap = 0.0
    maximum_gap = 0.0
    for index, values in enumerate(buckets):
        mean_confidence = sum(item[0] for item in values) / len(values) if values else None
        accuracy = sum(item[1] for item in values) / len(values) if values else None
        gap = None if mean_confidence is None or accuracy is None else accuracy - mean_confidence
        if gap is not None:
            weighted_gap += len(values) / count * abs(gap)
            maximum_gap = max(maximum_gap, abs(gap))
        summaries.append(
            CalibrationBin(
                index=index,
                lower=index / bins,
                upper=(index + 1) / bins,
                count=len(values),
                mean_confidence=mean_confidence,
                accuracy=accuracy,
                gap=gap,
            )
        )
    return CalibrationReport(
        confidence_field=confidence_field,
        outcome_field=outcome_field,
        bin_count=bins,
        case_count=count,
        brier_score=brier_total / count,
        expected_calibration_error=weighted_gap,
        maximum_calibration_error=maximum_gap,
        bins=tuple(summaries),
    )


def _validate_field(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or not value.strip()
        or any(part == "" for part in value.split("."))
    ):
        raise ValueError(f"{name} must be a non-empty dotted field path")


def _metadata_value(metadata: Mapping[str, Any], path: str) -> Any:
    value: Any = metadata
    for part in path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            raise ValueError(f"metadata field {path!r} is missing")
        value = value[part]
    return value


def _confidence(value: Any, case_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"case {case_id!r} confidence must be a number")
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"case {case_id!r} confidence must be finite and between 0 and 1")
    return float(value)


def _outcome(value: Any, case_id: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"case {case_id!r} outcome must be boolean")
    return value


__all__ = ["CalibrationBin", "CalibrationReport", "calibration_report"]
