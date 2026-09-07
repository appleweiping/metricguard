"""Deterministic metric experiment matrices for repeatable evaluations."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from .metrics import Metric
from .models import EvaluationCase, UndefinedPolicy
from .runner import MetricRunner, RunnerReport


@dataclass(frozen=True, slots=True)
class ExperimentSpec:
    """One named metric/case combination in a matrix."""

    name: str
    metric: Metric
    cases: tuple[EvaluationCase, ...]
    undefined_policy: UndefinedPolicy = UndefinedPolicy.ERROR
    workers: int = 1

    def __post_init__(self) -> None:
        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or any(char.isspace() for char in self.name)
        ):
            raise ValueError("experiment name must be a non-empty token")
        if not isinstance(self.cases, tuple) or not all(
            isinstance(case, EvaluationCase) for case in self.cases
        ):
            raise TypeError("cases must be a tuple of EvaluationCase values")
        if not isinstance(self.undefined_policy, UndefinedPolicy):
            raise TypeError("undefined_policy must be an UndefinedPolicy")
        if isinstance(self.workers, bool) or not isinstance(self.workers, int) or self.workers < 1:
            raise ValueError("workers must be a positive integer")


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    """One completed matrix cell."""

    name: str
    metric_name: str
    report: RunnerReport
    cases: int

    @property
    def mean_score(self) -> float | None:
        """Macro mean over resolved case scores."""
        return self.report.mean_score


class ExperimentMatrix:
    """Run named metric experiments in parallel while preserving input order."""

    def __init__(self, specs: Iterable[ExperimentSpec]) -> None:
        self.specs = tuple(specs)
        names = [spec.name for spec in self.specs]
        if len(names) != len(set(names)):
            raise ValueError("experiment names must be unique")

    def run(
        self,
        *,
        cache_dir: str | Path | None = None,
        max_workers: int = 1,
        fail_fast: bool = True,
    ) -> tuple[ExperimentResult, ...]:
        """Evaluate all cells, optionally with one JSONL cache per cell."""
        if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError("max_workers must be a positive integer")
        directory = Path(cache_dir) if cache_dir is not None else None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)

        def evaluate(spec: ExperimentSpec) -> ExperimentResult:
            cache = None
            if directory is not None:
                suffix = hashlib.sha256(spec.name.encode("utf-8")).hexdigest()[:16]
                cache = directory / f"{suffix}.jsonl"
            report = MetricRunner(
                spec.cases,
                spec.metric,
                undefined_policy=spec.undefined_policy,
                workers=spec.workers,
            ).run(cache=cache, fail_fast=fail_fast)
            return ExperimentResult(spec.name, spec.metric.name, report, len(spec.cases))

        if max_workers == 1:
            return tuple(evaluate(spec) for spec in self.specs)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            return tuple(executor.map(evaluate, self.specs))

    @staticmethod
    def means(results: Iterable[ExperimentResult]) -> Mapping[str, float | None]:
        """Return stable name-to-mean scores for reporting."""
        return MappingProxyType({result.name: result.mean_score for result in results})
