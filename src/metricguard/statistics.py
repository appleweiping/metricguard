"""Deterministic uncertainty estimates and paired score summaries."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from statistics import fmean
from typing import Literal

from .models import CaseResult, EvaluationCase, SuiteReport


@dataclass(frozen=True, slots=True)
class BootstrapConfig:
    """Configuration for reproducible paired resampling estimates.

    ``samples`` controls both percentile-bootstrap replicates and the separate
    Monte Carlo sign-flip randomization test used by :func:`paired_comparison`.
    """

    samples: int = 2_000
    confidence: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.samples, bool) or not isinstance(self.samples, int):
            raise TypeError("bootstrap samples must be an integer")
        if self.samples < 1:
            raise ValueError("bootstrap samples must be positive")
        if isinstance(self.confidence, bool) or not isinstance(self.confidence, (int, float)):
            raise TypeError("bootstrap confidence must be a real number")
        if not math.isfinite(self.confidence) or not 0 < self.confidence < 1:
            raise ValueError("bootstrap confidence must be between zero and one")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("bootstrap seed must be an integer")


@dataclass(frozen=True, slots=True)
class ConfidenceInterval:
    """Point estimate and percentile confidence interval."""

    point: float
    lower: float
    upper: float
    confidence: float
    samples: int


@dataclass(frozen=True, slots=True)
class PairedComparison:
    """Paired candidate-minus-baseline comparison for aligned cases."""

    metric_name: str
    baseline_mean: float
    candidate_mean: float
    delta: ConfidenceInterval
    probability_improvement: float
    two_sided_p_value: float
    compared_count: int
    omitted_count: int
    minimum_delta: float
    minimum_lower_bound: float | None
    direction: Literal["higher", "lower"] = "higher"

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'")

    @property
    def improvement(self) -> ConfidenceInterval:
        """Return the delta oriented so positive always means improvement."""

        if self.direction == "higher":
            return self.delta
        return ConfidenceInterval(
            point=-self.delta.point,
            lower=-self.delta.upper,
            upper=-self.delta.lower,
            confidence=self.delta.confidence,
            samples=self.delta.samples,
        )

    @property
    def passed_gate(self) -> bool:
        """Return whether the observed and optional confidence gates pass."""

        if self.improvement.point < self.minimum_delta:
            return False
        return (
            self.minimum_lower_bound is None or self.improvement.lower >= self.minimum_lower_bound
        )


@dataclass(frozen=True, slots=True)
class TagSummary:
    """Macro score summary for one case tag."""

    tag: str
    case_count: int
    scored_count: int
    mean_score: float | None


def confidence_interval(
    values: tuple[float, ...], config: BootstrapConfig | None = None
) -> ConfidenceInterval:
    """Estimate a deterministic percentile interval for a macro mean."""

    active = config or BootstrapConfig()
    clean = _validated_values(values)
    replicates = _bootstrap_means(clean, active)
    tail = (1.0 - active.confidence) / 2.0
    return ConfidenceInterval(
        point=fmean(clean),
        lower=_quantile(replicates, tail),
        upper=_quantile(replicates, 1.0 - tail),
        confidence=active.confidence,
        samples=active.samples,
    )


def report_confidence_interval(
    report: SuiteReport, config: BootstrapConfig | None = None
) -> ConfidenceInterval:
    """Estimate a confidence interval over a report's resolved scores."""

    return confidence_interval(_resolved_scores(report), config)


def paired_comparison(
    baseline: SuiteReport,
    candidate: SuiteReport,
    *,
    config: BootstrapConfig | None = None,
    minimum_delta: float = 0.0,
    minimum_lower_bound: float | None = None,
    direction: Literal["higher", "lower"] = "higher",
) -> PairedComparison:
    """Compare reports using paired bootstrap resampling by case ID.

    Cases skipped by both reports are omitted. A case resolved on only one side
    is rejected because silently changing the evaluation population would make
    a regression gate misleading.
    """

    if baseline.metric_name != candidate.metric_name:
        raise ValueError("paired reports must use the same metric name")
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'")
    _validate_gate("minimum_delta", minimum_delta)
    if minimum_lower_bound is not None:
        _validate_gate("minimum_lower_bound", minimum_lower_bound)
    baseline_by_id = _result_map("baseline", baseline)
    candidate_by_id = _result_map("candidate", candidate)
    if baseline_by_id.keys() != candidate_by_id.keys():
        missing_candidate = sorted(baseline_by_id.keys() - candidate_by_id.keys())
        missing_baseline = sorted(candidate_by_id.keys() - baseline_by_id.keys())
        details: list[str] = []
        if missing_candidate:
            details.append(f"missing from candidate: {', '.join(missing_candidate)}")
        if missing_baseline:
            details.append(f"missing from baseline: {', '.join(missing_baseline)}")
        raise ValueError("paired reports have different case IDs (" + "; ".join(details) + ")")

    baseline_scores: list[float] = []
    candidate_scores: list[float] = []
    omitted = 0
    for case_id in baseline_by_id:
        left = baseline_by_id[case_id].resolved_score
        right = candidate_by_id[case_id].resolved_score
        if left is None and right is None:
            omitted += 1
            continue
        if left is None or right is None:
            raise ValueError(f"case {case_id!r} is resolved on only one side")
        baseline_scores.append(left)
        candidate_scores.append(right)
    if not baseline_scores:
        raise ValueError("paired comparison requires at least one resolved case")

    clean_baseline = _validated_values(tuple(baseline_scores))
    clean_candidate = _validated_values(tuple(candidate_scores))
    active = config or BootstrapConfig()
    deltas = _validated_values(
        tuple(right - left for left, right in zip(clean_baseline, clean_candidate, strict=True))
    )
    delta_replicates = _bootstrap_means(deltas, active)
    tail = (1.0 - active.confidence) / 2.0
    delta_interval = ConfidenceInterval(
        point=fmean(deltas),
        lower=_quantile(delta_replicates, tail),
        upper=_quantile(delta_replicates, 1.0 - tail),
        confidence=active.confidence,
        samples=active.samples,
    )
    orientation = 1.0 if direction == "higher" else -1.0
    oriented_replicates = tuple(orientation * value for value in delta_replicates)
    greater = sum(value > 0 for value in oriented_replicates)
    equal = sum(value == 0 for value in oriented_replicates)
    probability = (greater + 0.5 * equal) / len(delta_replicates)
    p_value = _sign_flip_p_value(deltas, active)
    return PairedComparison(
        metric_name=baseline.metric_name,
        baseline_mean=fmean(clean_baseline),
        candidate_mean=fmean(clean_candidate),
        delta=delta_interval,
        probability_improvement=probability,
        two_sided_p_value=p_value,
        compared_count=len(deltas),
        omitted_count=omitted,
        minimum_delta=float(minimum_delta),
        minimum_lower_bound=(None if minimum_lower_bound is None else float(minimum_lower_bound)),
        direction=direction,
    )


def summarize_by_tag(
    report: SuiteReport, cases: tuple[EvaluationCase, ...]
) -> tuple[TagSummary, ...]:
    """Summarize resolved scores for each tag without double-counting cases."""

    case_by_id = {case.case_id: case for case in cases}
    if len(case_by_id) != len(cases):
        raise ValueError("cases contain duplicate IDs")
    if set(case_by_id) != {result.case_id for result in report.results}:
        raise ValueError("report and cases must contain the same case IDs")
    buckets: dict[str, list[float | None]] = {}
    result_by_id = _result_map("report", report)
    for case in cases:
        for tag in case.tags:
            buckets.setdefault(tag, []).append(result_by_id[case.case_id].resolved_score)
    return tuple(
        TagSummary(
            tag=tag,
            case_count=len(scores),
            scored_count=sum(score is not None for score in scores),
            mean_score=fmean(score for score in scores if score is not None)
            if any(score is not None for score in scores)
            else None,
        )
        for tag, scores in sorted(buckets.items())
    )


def _resolved_scores(report: SuiteReport) -> tuple[float, ...]:
    scores = tuple(
        result.resolved_score for result in report.results if result.resolved_score is not None
    )
    if not scores:
        raise ValueError("confidence interval requires at least one resolved score")
    return _validated_values(scores)


def _result_map(label: str, report: SuiteReport) -> dict[str, CaseResult]:
    output: dict[str, CaseResult] = {}
    for result in report.results:
        if result.case_id in output:
            raise ValueError(f"{label} report contains duplicate case ID {result.case_id!r}")
        output[result.case_id] = result
    return output


def _validated_values(values: tuple[float, ...]) -> tuple[float, ...]:
    if not isinstance(values, tuple):
        raise TypeError("bootstrap values must be a tuple")
    if not values:
        raise ValueError("bootstrap requires at least one value")
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("bootstrap values must be real numbers")
        if not math.isfinite(value):
            raise ValueError("bootstrap values must be finite")
    return tuple(float(value) for value in values)


def _bootstrap_means(values: tuple[float, ...], config: BootstrapConfig) -> tuple[float, ...]:
    sample_size = len(values)
    replicates: list[float] = []
    for replicate in range(config.samples):
        total = 0.0
        for position in range(sample_size):
            index = _deterministic_index(config.seed, replicate, position, sample_size)
            total += values[index]
        replicates.append(total / sample_size)
    return tuple(sorted(replicates))


def _deterministic_index(seed: int, replicate: int, position: int, size: int) -> int:
    payload = f"metricguard-bootstrap-v1\0{seed}\0{replicate}\0{position}".encode()
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[:8], "big") % size


def _sign_flip_p_value(values: tuple[float, ...], config: BootstrapConfig) -> float:
    """Return a deterministic Monte Carlo paired sign-flip p-value.

    Under the sharp null, exchanging baseline and candidate within each pair
    changes only the sign of its delta. The add-one correction prevents a zero
    Monte Carlo p-value and includes the observed assignment conceptually.
    """

    observed = abs(fmean(values))
    extreme = 0
    for replicate in range(config.samples):
        total = 0.0
        for position, value in enumerate(values):
            payload = f"metricguard-sign-flip-v1\0{config.seed}\0{replicate}\0{position}".encode()
            positive = hashlib.sha256(payload).digest()[0] & 1
            total += value if positive else -value
        if abs(total / len(values)) >= observed:
            extreme += 1
    return (extreme + 1) / (config.samples + 1)


def _quantile(sorted_values: tuple[float, ...], probability: float) -> float:
    """Return the R-7/NumPy-linear quantile of an already sorted sample."""

    if len(sorted_values) == 1:
        return sorted_values[0]
    location = (len(sorted_values) - 1) * probability
    lower_index = math.floor(location)
    upper_index = math.ceil(location)
    fraction = location - lower_index
    return sorted_values[lower_index] * (1.0 - fraction) + sorted_values[upper_index] * fraction


def _validate_gate(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
