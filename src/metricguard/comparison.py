"""High-level paired evaluation with dataset alignment checks."""

from __future__ import annotations

from typing import Any, Literal

from .metrics import Metric
from .models import EvaluationCase, UndefinedPolicy
from .statistics import BootstrapConfig, PairedComparison, paired_comparison
from .suite import EvaluationSuite


def compare_case_sets(
    baseline_cases: tuple[EvaluationCase, ...],
    candidate_cases: tuple[EvaluationCase, ...],
    *,
    metric: Metric,
    undefined_policy: UndefinedPolicy = UndefinedPolicy.ERROR,
    bootstrap: BootstrapConfig | None = None,
    minimum_delta: float = 0.0,
    minimum_lower_bound: float | None = None,
    direction: Literal["higher", "lower"] = "higher",
) -> PairedComparison:
    """Evaluate aligned datasets with one metric configuration on both sides."""

    _validate_case_alignment(baseline_cases, candidate_cases)
    baseline_report = EvaluationSuite(baseline_cases, undefined_policy=undefined_policy).run(metric)
    candidate_report = EvaluationSuite(candidate_cases, undefined_policy=undefined_policy).run(
        metric
    )
    return paired_comparison(
        baseline_report,
        candidate_report,
        config=bootstrap,
        minimum_delta=minimum_delta,
        minimum_lower_bound=minimum_lower_bound,
        direction=direction,
    )


def _validate_case_alignment(
    baseline_cases: tuple[EvaluationCase, ...], candidate_cases: tuple[EvaluationCase, ...]
) -> None:
    baseline_by_id = _case_map("baseline", baseline_cases)
    candidate_by_id = _case_map("candidate", candidate_cases)
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing_candidate = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        missing_baseline = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        details: list[str] = []
        if missing_candidate:
            details.append(f"missing from candidate: {', '.join(missing_candidate)}")
        if missing_baseline:
            details.append(f"missing from baseline: {', '.join(missing_baseline)}")
        raise ValueError("case files have different IDs (" + "; ".join(details) + ")")
    for case_id, baseline in baseline_by_id.items():
        candidate = candidate_by_id[case_id]
        if not _same_value(baseline.reference, candidate.reference):
            raise ValueError(f"case {case_id!r} has different references")
        if frozenset(baseline.tags) != frozenset(candidate.tags):
            raise ValueError(f"case {case_id!r} has different tags")


def _case_map(label: str, cases: tuple[EvaluationCase, ...]) -> dict[str, EvaluationCase]:
    output: dict[str, EvaluationCase] = {}
    for case in cases:
        if case.case_id in output:
            raise ValueError(f"{label} cases contain duplicate ID {case.case_id!r}")
        output[case.case_id] = case
    return output


def _same_value(left: Any, right: Any) -> bool:
    """Compare JSON-like values without treating booleans as integers."""

    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _same_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(
            _same_value(left_item, right_item)
            for left_item, right_item in zip(left, right, strict=True)
        )
    return bool(left == right)
