"""Deterministic metadata-slice evaluation summaries."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any, Literal

from .metrics import Metric
from .models import EvaluationCase, SuiteReport, UndefinedPolicy
from .statistics import (
    BootstrapConfig,
    CorrectionMethod,
    PairedComparison,
    adjust_p_values,
    paired_comparison,
)
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
    adjusted_p_value: float | None = None
    significance_passed: bool | None = None

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
        if self.adjusted_p_value is not None:
            payload["adjusted_p_value"] = self.adjusted_p_value
            payload["significance_passed"] = self.significance_passed
        return payload


@dataclass(frozen=True, slots=True)
class MetadataFamilyComparison:
    """A corrected family of comparisons across multiple metadata fields."""

    fields: tuple[str, ...]
    comparisons: tuple[SliceComparison, ...]
    correction: CorrectionMethod
    alpha: float

    @property
    def passed(self) -> bool:
        """Return whether every slice passes its effect and family significance gate."""

        if not self.comparisons:
            return False
        return all(
            row.passed
            and (
                self.correction == "none"
                or row.comparison is None
                or row.significance_passed is True
            )
            for row in self.comparisons
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fields": list(self.fields),
            "correction": self.correction,
            "alpha": self.alpha,
            "passed": self.passed,
            "comparisons": [item.to_dict() for item in self.comparisons],
        }


def correct_slice_p_values(
    comparisons: Iterable[SliceComparison],
    *,
    method: CorrectionMethod,
    alpha: float = 0.05,
) -> tuple[SliceComparison, ...]:
    """Apply one family-level p-value correction to successful slices.

    Ordering and error entries are preserved. Significance is reported
    separately from the existing confidence/delta gate, so callers can choose
    whether a family-level claim should block a release.
    """

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be a real number")
    if not 0 < alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    rows = tuple(comparisons)
    active = tuple(row for row in rows if row.comparison is not None and row.error is None)
    adjusted = adjust_p_values(
        (row.comparison.two_sided_p_value for row in active if row.comparison is not None),
        method,
    )
    by_identity = {id(row): value for row, value in zip(active, adjusted, strict=True)}
    return tuple(
        replace(
            row,
            adjusted_p_value=by_identity[id(row)],
            significance_passed=by_identity[id(row)] <= alpha,
        )
        if id(row) in by_identity
        else row
        for row in rows
    )


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

    _validate_slice_arguments(field, missing, min_count)
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
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


def compare_metadata_family(
    baseline_cases: tuple[EvaluationCase, ...],
    candidate_cases: tuple[EvaluationCase, ...],
    *,
    metric: Metric,
    fields: Iterable[str],
    undefined_policy: UndefinedPolicy,
    bootstrap: BootstrapConfig | None = None,
    missing: str = "<missing>",
    min_count: int = 1,
    minimum_delta: float = 0.0,
    minimum_lower_bound: float | None = None,
    direction: Literal["higher", "lower"] = "higher",
    correction: CorrectionMethod = "none",
    alpha: float = 0.05,
) -> MetadataFamilyComparison:
    """Compare several metadata dimensions and correct one shared p-value family.

    Unlike repeated calls to :func:`compare_by_metadata`, all successful slice
    p-values across every requested field enter one correction family. This
    prevents a caller from accidentally treating a collection of exploratory
    slice reports as independent confirmatory claims.
    """

    names = tuple(fields)
    if not names or len(names) != len(set(names)):
        raise ValueError("fields must contain at least one unique metadata path")
    if correction not in {"none", "bonferroni", "holm", "benjamini-hochberg"}:
        raise ValueError(f"unsupported p-value correction {correction!r}")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not 0 < alpha <= 1:
        raise ValueError("alpha must be between zero and one")
    rows: list[SliceComparison] = []
    for field in names:
        rows.extend(
            compare_by_metadata(
                baseline_cases,
                candidate_cases,
                metric=metric,
                field=field,
                undefined_policy=undefined_policy,
                bootstrap=bootstrap,
                missing=missing,
                min_count=min_count,
                minimum_delta=minimum_delta,
                minimum_lower_bound=minimum_lower_bound,
                direction=direction,
            )
        )
    corrected = (
        correct_slice_p_values(rows, method=correction, alpha=float(alpha))
        if correction != "none"
        else tuple(rows)
    )
    return MetadataFamilyComparison(names, corrected, correction, float(alpha))


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


def _validate_slice_arguments(field: str, missing: str, min_count: int) -> None:
    if (
        not isinstance(field, str)
        or not field.strip()
        or any(not part.strip() for part in field.split("."))
    ):
        raise ValueError("field must be a non-empty dotted metadata path")
    if not isinstance(missing, str) or not missing:
        raise ValueError("missing must be a non-empty string")
    if isinstance(min_count, bool) or not isinstance(min_count, int) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
