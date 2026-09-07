"""Deterministic metadata-slice evaluation summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any

from .models import EvaluationCase, SuiteReport


@dataclass(frozen=True, slots=True)
class SliceSummary:
    """Macro score summary for one metadata slice."""

    field: str
    value: str
    case_count: int
    scored_count: int
    skipped_count: int
    mean_score: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field,
            "value": self.value,
            "case_count": self.case_count,
            "scored_count": self.scored_count,
            "skipped_count": self.skipped_count,
            "mean_score": self.mean_score,
        }


def summarize_by_metadata(
    report: SuiteReport,
    cases: Iterable[EvaluationCase],
    field: str,
    *,
    missing: str = "<missing>",
    min_count: int = 1,
) -> tuple[SliceSummary, ...]:
    """Aggregate resolved scores by a dotted metadata field.

    The function validates that report and cases describe the same population,
    serializes nested JSON values canonically, and never mutates case metadata.
    Missing values are assigned to the explicit ``missing`` bucket.
    """

    if not isinstance(field, str) or not field.strip():
        raise ValueError("field must be a non-empty metadata path")
    if not isinstance(missing, str) or not missing:
        raise ValueError("missing must be a non-empty string")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    rows = tuple(cases)
    case_map = {case.case_id: case for case in rows}
    if len(case_map) != len(rows):
        raise ValueError("cases contain duplicate IDs")
    results = {result.case_id: result for result in report.results}
    if set(case_map) != set(results):
        raise ValueError("report and cases must contain the same case IDs")
    values: dict[str, list[float | None]] = {}
    for case in rows:
        key = _metadata_key(case.metadata, field.split("."), missing)
        values.setdefault(key, []).append(results[case.case_id].resolved_score)
    summaries = []
    for key, scores in values.items():
        if len(scores) < min_count:
            continue
        resolved = tuple(score for score in scores if score is not None)
        summaries.append(
            SliceSummary(
                field=field,
                value=key,
                case_count=len(scores),
                scored_count=len(resolved),
                skipped_count=len(scores) - len(resolved),
                mean_score=fmean(resolved) if resolved else None,
            )
        )
    return tuple(sorted(summaries, key=lambda item: item.value))


def _metadata_key(value: Mapping[str, Any], path: list[str], missing: str) -> str:
    current: Any = value
    for part in path:
        if not isinstance(current, Mapping) or part not in current:
            return missing
        current = current[part]
    if current is None:
        return missing
    if isinstance(current, str):
        return current
    try:
        return json.dumps(current, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError) as error:
        raise ValueError("metadata slice values must be JSON-compatible") from error
