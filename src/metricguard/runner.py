"""Resumable, ordered metric execution for larger case suites."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from decimal import getcontext, localcontext
from pathlib import Path
from typing import Any

from ._runner_cache import fingerprint, load_cache, make_row, metric_identity, raw_value, save_cache
from .metrics import Metric
from .models import CaseResult, EvaluationCase, MetricValue, UndefinedPolicy


@dataclass(frozen=True, slots=True)
class RunnerReport:
    """Case results and cache/error accounting."""

    metric_name: str
    results: tuple[CaseResult, ...]
    errors: tuple[tuple[str, str], ...]
    cached: int

    @property
    def mean_score(self) -> float | None:
        scores = [item.resolved_score for item in self.results if item.resolved_score is not None]
        return sum(scores) / len(scores) if scores else None


class MetricRunner:
    """Run a metric with optional worker parallelism and JSONL checkpoint cache."""

    def __init__(
        self,
        cases: list[EvaluationCase] | tuple[EvaluationCase, ...],
        metric: Metric,
        *,
        undefined_policy: UndefinedPolicy = UndefinedPolicy.ERROR,
        workers: int = 1,
    ) -> None:
        if not all(isinstance(case, EvaluationCase) for case in cases):
            raise TypeError("cases must contain EvaluationCase values")
        if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
            raise ValueError("workers must be a positive integer")
        if not isinstance(undefined_policy, UndefinedPolicy):
            raise TypeError("undefined_policy must be an UndefinedPolicy")
        ids = [case.case_id for case in cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        self.cases = tuple(cases)
        self.metric = metric
        self.undefined_policy = undefined_policy
        self.workers = workers

    def run(
        self,
        *,
        cache: str | Path | None = None,
        fail_fast: bool = True,
        checkpoint_interval: int = 100,
    ) -> RunnerReport:
        """Evaluate in input order and resume validated raw scores under the current policy.

        Successful results are checkpointed periodically and on exit, including
        ordinary evaluation failure. Exceptions are always retried on resume.
        """
        if not isinstance(fail_fast, bool):
            raise TypeError("fail_fast must be a boolean")
        if type(checkpoint_interval) is not int or checkpoint_interval < 1:
            raise ValueError("checkpoint_interval must be a positive integer")
        path = Path(cache) if cache is not None else None
        decimal_context = getcontext().copy()
        identity = metric_identity(self.metric) if path is not None else None
        cached_rows = load_cache(path) if path is not None and path.exists() else {}
        results: list[CaseResult | None] = [None] * len(self.cases)
        pending: list[tuple[int, EvaluationCase, str]] = []
        cached_count = 0
        for index, case in enumerate(self.cases):
            key = fingerprint(identity, case) if identity is not None else ""
            saved = cached_rows.get(case.case_id)
            if saved is not None and saved["fingerprint"] == key:
                results[index] = _resolve(case.case_id, raw_value(saved), self.undefined_policy)
                cached_count += 1
            else:
                pending.append((index, case, key))

        def evaluate(
            item: tuple[int, EvaluationCase, str],
        ) -> tuple[int, EvaluationCase, str, MetricValue, str | None]:
            index, case, key = item
            try:
                # Worker threads must use the caller's numeric context, including
                # precision and traps, rather than their interpreter defaults.
                with localcontext(decimal_context):
                    value = self.metric.evaluate(case.reference, case.prediction)
                if not isinstance(value, MetricValue):
                    raise TypeError("metric did not return MetricValue")
                return index, case, key, value, None
            except Exception as error:
                return (
                    index,
                    case,
                    key,
                    MetricValue(None, reason=str(error) or type(error).__name__),
                    type(error).__name__,
                )

        errors: list[tuple[str, str]] = []
        rows: dict[str, dict[str, Any]] = dict(cached_rows)
        unsaved = 0
        checkpoint_failed = False

        def checkpoint() -> None:
            nonlocal unsaved, checkpoint_failed
            if path is not None:
                checkpoint_failed = True
                save_cache(path, rows)
                unsaved = 0
                checkpoint_failed = False

        def consume(item: tuple[int, EvaluationCase, str, MetricValue, str | None]) -> None:
            nonlocal unsaved
            index, case, key, raw, error_type = item
            if error_type is not None:
                errors.append((case.case_id, f"{error_type}: {raw.reason}"))
                if fail_fast:
                    raise ValueError(f"metric failed for {case.case_id!r}: {raw.reason}")
            elif path is not None:
                rows[case.case_id] = make_row(case.case_id, key, raw)
                raw = raw_value(rows[case.case_id])
                unsaved += 1
            results[index] = _resolve(case.case_id, raw, self.undefined_policy)
            if path is not None and unsaved >= checkpoint_interval:
                checkpoint()

        try:
            if pending and self.workers > 1:
                with ThreadPoolExecutor(max_workers=self.workers) as executor:
                    for item in executor.map(evaluate, pending):
                        consume(item)
            else:
                for pending_item in pending:
                    consume(evaluate(pending_item))
        finally:
            if path is not None and not checkpoint_failed and (unsaved or not path.exists()):
                checkpoint()
        complete = tuple(result for result in results if result is not None)
        return RunnerReport(self.metric.name, complete, tuple(errors), cached_count)


def _resolve(case_id: str, raw: MetricValue, policy: UndefinedPolicy) -> CaseResult:
    if raw.score is not None:
        return CaseResult(case_id, raw, raw.score)
    if policy is UndefinedPolicy.ERROR:
        raise ValueError(f"metric is undefined for {case_id!r}: {raw.reason}")
    if policy is UndefinedPolicy.SKIP:
        return CaseResult(case_id, raw, None, skipped=True)
    return CaseResult(case_id, raw, 0.0 if policy is UndefinedPolicy.ZERO else 1.0)
