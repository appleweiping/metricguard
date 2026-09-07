"""Resumable, ordered metric execution for larger case suites."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


def _fingerprint(metric: Metric, case: EvaluationCase) -> str:
    try:
        body = json.dumps(
            {"metric": metric.name, "reference": case.reference, "prediction": case.prediction},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"case {case.case_id!r} is not JSON-serializable for caching") from error
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


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

    def run(self, *, cache: str | Path | None = None, fail_fast: bool = True) -> RunnerReport:
        """Evaluate in input order; valid matching cache rows are reused."""
        cached_rows = (
            _load_cache(Path(cache)) if cache is not None and Path(cache).is_file() else {}
        )
        results: list[CaseResult | None] = [None] * len(self.cases)
        pending: list[tuple[int, EvaluationCase, str]] = []
        cached_count = 0
        for index, case in enumerate(self.cases):
            fingerprint = _fingerprint(self.metric, case)
            saved = cached_rows.get(case.case_id)
            if saved is not None and saved.get("fingerprint") == fingerprint:
                results[index] = _case_from_json(saved)
                cached_count += 1
            else:
                pending.append((index, case, fingerprint))

        def evaluate(
            item: tuple[int, EvaluationCase, str],
        ) -> tuple[int, EvaluationCase, str, MetricValue, str | None]:
            index, case, fingerprint = item
            try:
                value = self.metric.evaluate(case.reference, case.prediction)
                if not isinstance(value, MetricValue):
                    raise TypeError("metric did not return MetricValue")
                return index, case, fingerprint, value, None
            except Exception as error:
                return (
                    index,
                    case,
                    fingerprint,
                    MetricValue(None, reason=str(error)),
                    type(error).__name__,
                )

        if pending and self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                evaluated = tuple(executor.map(evaluate, pending))
        else:
            evaluated = tuple(evaluate(item) for item in pending)
        errors: list[tuple[str, str]] = []
        rows: dict[str, dict[str, Any]] = dict(cached_rows)
        for index, case, fingerprint, raw, error_type in evaluated:
            if error_type is not None:
                errors.append((case.case_id, f"{error_type}: {raw.reason}"))
                if fail_fast:
                    raise ValueError(f"metric failed for {case.case_id!r}: {raw.reason}")
            result = _resolve(case.case_id, raw, self.undefined_policy)
            results[index] = result
            rows[case.case_id] = _case_to_json(result, fingerprint)
        complete = tuple(result for result in results if result is not None)
        if cache is not None:
            _save_cache(Path(cache), rows)
        return RunnerReport(self.metric.name, complete, tuple(errors), cached_count)


def _resolve(case_id: str, raw: MetricValue, policy: UndefinedPolicy) -> CaseResult:
    if raw.score is not None:
        return CaseResult(case_id, raw, raw.score)
    if policy is UndefinedPolicy.ERROR:
        raise ValueError(f"metric is undefined for {case_id!r}: {raw.reason}")
    if policy is UndefinedPolicy.SKIP:
        return CaseResult(case_id, raw, None, skipped=True)
    return CaseResult(case_id, raw, 0.0 if policy is UndefinedPolicy.ZERO else 1.0)


def _case_to_json(result: CaseResult, fingerprint: str) -> dict[str, Any]:
    return {
        "case_id": result.case_id,
        "fingerprint": fingerprint,
        "score": result.raw.score,
        "reason": result.raw.reason,
        "details": result.raw.details,
        "resolved_score": result.resolved_score,
        "skipped": result.skipped,
    }


def _case_from_json(raw: dict[str, Any]) -> CaseResult:
    return CaseResult(
        raw["case_id"],
        MetricValue(raw["score"], raw.get("reason"), raw.get("details", {})),
        raw.get("resolved_score"),
        raw.get("skipped", False),
    )


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            raw = json.loads(line)
            if isinstance(raw, dict) and isinstance(raw.get("case_id"), str):
                rows[raw["case_id"]] = raw
    return rows


def _save_cache(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(rows[key], sort_keys=True, allow_nan=False) + "\n" for key in sorted(rows)
        ),
        encoding="utf-8",
    )
