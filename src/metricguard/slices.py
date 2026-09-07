"""Deterministic metadata-slice evaluation summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from statistics import fmean
from typing import Any, Literal

from .metrics import Metric
from .models import EvaluationCase, SuiteReport, UndefinedPolicy
from .statistics import BootstrapConfig, PairedComparison, paired_comparison
from .suite import EvaluationSuite


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


@dataclass(frozen=True, slots=True)
class SliceComparison:
    """One metadata slice's paired baseline/candidate result."""

    field: str
    value: str
    case_count: int
    comparison: PairedComparison | None = None
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.error is None and self.comparison is not None and self.comparison.passed_gate

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "field": self.field,
            "value": self.value,
            "case_count": self.case_count,
            "passed": self.passed,
            "error": self.error,
        }
        if self.comparison is not None:
            payload["comparison"] = {
                "baseline_mean": self.comparison.baseline_mean,
                "candidate_mean": self.comparison.candidate_mean,
                "improvement": self.comparison.improvement.point,
                "lower_bound": self.comparison.improvement.lower,
                "upper_bound": self.comparison.improvement.upper,
                "p_value": self.comparison.two_sided_p_value,
                "passed_gate": self.comparison.passed_gate,
            }
        return payload


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


def compare_by_metadata(
    baseline_cases: tuple[EvaluationCase, ...],
    candidate_cases: tuple[EvaluationCase, ...],
    *,
    metric: Metric,
    field: str,
    undefined_policy: UndefinedPolicy,
    bootstrap: BootstrapConfig | None = None,
    missing: str = "<missing>",
    min_count: int = 1,
    minimum_delta: float = 0.0,
    minimum_lower_bound: float | None = None,
    direction: Literal["higher", "lower"] = "higher",
) -> tuple[SliceComparison, ...]:
    """Run paired statistical comparisons independently for metadata slices."""

    if len({case.case_id for case in baseline_cases}) != len(baseline_cases):
        raise ValueError("baseline cases contain duplicate IDs")
    if len({case.case_id for case in candidate_cases}) != len(candidate_cases):
        raise ValueError("candidate cases contain duplicate IDs")
    baseline_by_id = {case.case_id: case for case in baseline_cases}
    candidate_by_id = {case.case_id: case for case in candidate_cases}
    if baseline_by_id.keys() != candidate_by_id.keys():
        raise ValueError("baseline and candidate cases must contain the same case IDs")
    groups: dict[str, list[str]] = {}
    for case_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id[case_id]
        baseline_value = _metadata_key(baseline.metadata, field.split("."), missing)
        candidate_value = _metadata_key(candidate.metadata, field.split("."), missing)
        if baseline_value != candidate_value:
            raise ValueError(f"case {case_id!r} changed metadata slice value")
        groups.setdefault(baseline_value, []).append(case_id)
    results: list[SliceComparison] = []
    for value, case_ids in sorted(groups.items()):
        if len(case_ids) < min_count:
            continue
        left = tuple(baseline_by_id[case_id] for case_id in case_ids)
        right = tuple(candidate_by_id[case_id] for case_id in case_ids)
        try:
            baseline_report = EvaluationSuite(left, undefined_policy=undefined_policy).run(metric)
            candidate_report = EvaluationSuite(right, undefined_policy=undefined_policy).run(metric)
            comparison = paired_comparison(
                baseline_report,
                candidate_report,
                config=bootstrap,
                minimum_delta=minimum_delta,
                minimum_lower_bound=minimum_lower_bound,
                direction=direction,
            )
            results.append(SliceComparison(field, value, len(case_ids), comparison))
        except (TypeError, ValueError) as error:
            results.append(SliceComparison(field, value, len(case_ids), error=str(error)))
    return tuple(results)


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
