"""Bounded-memory metric aggregation for newline-delimited case suites."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

from .metrics import Metric
from .models import EvaluationCase, MetricValue, UndefinedPolicy


@dataclass(frozen=True, slots=True)
class StreamingTagSummary:
    """Aggregate for cases carrying one tag."""

    tag: str
    case_count: int
    scored_count: int
    skipped_count: int
    mean_score: float | None


@dataclass(frozen=True, slots=True)
class StreamingReport:
    """Aggregate-only result that never retains references or predictions."""

    metric_name: str
    case_count: int
    scored_count: int
    skipped_count: int
    mean_score: float | None
    errors: tuple[tuple[str, str], ...]
    by_tag: tuple[StreamingTagSummary, ...]

    @property
    def passed(self) -> bool:
        """Return whether all cases produced a resolved or explicitly skipped outcome."""

        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "metric": self.metric_name,
            "summary": {
                "case_count": self.case_count,
                "scored_count": self.scored_count,
                "skipped_count": self.skipped_count,
                "mean_score": self.mean_score,
                "errors": len(self.errors),
            },
            "errors": [list(error) for error in self.errors],
            "by_tag": [
                {
                    "tag": item.tag,
                    "case_count": item.case_count,
                    "scored_count": item.scored_count,
                    "skipped_count": item.skipped_count,
                    "mean_score": item.mean_score,
                }
                for item in self.by_tag
            ],
        }


@dataclass
class _TagAccumulator:
    count: int = 0
    scored: int = 0
    skipped: int = 0
    scores: list[float] | None = None

    def add(self, score: float | None, skipped: bool) -> None:
        self.count += 1
        if skipped:
            self.skipped += 1
        if score is not None:
            self.scored += 1
            if self.scores is None:
                self.scores = []
            self.scores.append(score)


def evaluate_stream(
    cases: Iterable[EvaluationCase],
    metric: Metric,
    *,
    undefined_policy: UndefinedPolicy = UndefinedPolicy.ERROR,
    fail_fast: bool = True,
) -> StreamingReport:
    """Evaluate an iterable without retaining case payloads or per-case results.

    ``fail_fast=False`` records metric exceptions and continues, which is useful
    for data-quality sweeps. Undefined scores follow the same four policies as
    :class:`EvaluationSuite`; ``ERROR`` raises unless non-fail-fast mode is used.
    """

    if not isinstance(metric, Metric):
        raise TypeError("metric must implement Metric")
    if not isinstance(undefined_policy, UndefinedPolicy):
        raise TypeError("undefined_policy must be an UndefinedPolicy")
    if not isinstance(fail_fast, bool):
        raise TypeError("fail_fast must be a boolean")
    seen: set[str] = set()
    errors: list[tuple[str, str]] = []
    tag_data: dict[str, _TagAccumulator] = defaultdict(_TagAccumulator)
    total = scored = skipped = 0
    scores: list[float] = []
    for case in cases:
        if not isinstance(case, EvaluationCase):
            raise TypeError("cases must contain EvaluationCase values")
        if case.case_id in seen:
            raise ValueError(f"duplicate case IDs: {case.case_id}")
        seen.add(case.case_id)
        total += 1
        try:
            raw = metric.evaluate(case.reference, case.prediction)
            if not isinstance(raw, MetricValue):
                raise TypeError("metric did not return MetricValue")
            resolved, was_skipped = _resolve(case.case_id, raw, undefined_policy)
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            errors.append((case.case_id, detail))
            if fail_fast:
                raise ValueError(f"metric failed for {case.case_id!r}: {error}") from error
            continue
        if resolved is not None:
            scored += 1
            scores.append(resolved)
        if was_skipped:
            skipped += 1
        for tag in case.tags:
            tag_data[tag].add(resolved, was_skipped)
    summaries = tuple(
        StreamingTagSummary(
            tag,
            accumulator.count,
            accumulator.scored,
            accumulator.skipped,
            fmean(accumulator.scores) if accumulator.scores else None,
        )
        for tag, accumulator in sorted(tag_data.items())
    )
    return StreamingReport(
        metric.name,
        total,
        scored,
        skipped,
        fmean(scores) if scores else None,
        tuple(errors),
        summaries,
    )


def _resolve(case_id: str, raw: MetricValue, policy: UndefinedPolicy) -> tuple[float | None, bool]:
    if raw.score is not None:
        return raw.score, False
    if policy is UndefinedPolicy.ERROR:
        raise ValueError(f"metric is undefined for {case_id!r}: {raw.reason}")
    if policy is UndefinedPolicy.SKIP:
        return None, True
    return (0.0 if policy is UndefinedPolicy.ZERO else 1.0), False
