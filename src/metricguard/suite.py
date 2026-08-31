"""Evaluation orchestration and undefined-result policy handling."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable

from .contracts import ContractAuditor
from .metrics import Metric
from .models import CaseResult, EvaluationCase, MetricValue, SuiteReport, UndefinedPolicy


class EvaluationSuite:
    """Run a metric over frozen case records and optionally audit its contracts."""

    def __init__(
        self,
        cases: Iterable[EvaluationCase],
        *,
        undefined_policy: UndefinedPolicy = UndefinedPolicy.ERROR,
    ) -> None:
        self.cases = tuple(cases)
        if not all(isinstance(case, EvaluationCase) for case in self.cases):
            raise TypeError("cases must contain only EvaluationCase values")
        if not isinstance(undefined_policy, UndefinedPolicy):
            raise TypeError("undefined_policy must be an UndefinedPolicy")
        self.undefined_policy = undefined_policy
        identifiers = [case.case_id for case in self.cases]
        duplicates = sorted(key for key, count in Counter(identifiers).items() if count > 1)
        if duplicates:
            raise ValueError(f"duplicate case IDs: {', '.join(duplicates)}")

    def run(
        self,
        metric: Metric,
        *,
        auditor: ContractAuditor | None = None,
    ) -> SuiteReport:
        """Evaluate every case and return a report.

        ``UndefinedPolicy.ERROR`` raises immediately with the case ID, making
        silent changes in metric semantics visible in CI.
        """

        results: list[CaseResult] = []
        for case in self.cases:
            raw = metric.evaluate(case.reference, case.prediction)
            if not isinstance(raw, MetricValue):
                raise TypeError(
                    f"metric {metric.name!r} returned {type(raw).__name__}, expected MetricValue"
                )
            if raw.score is not None:
                results.append(CaseResult(case.case_id, raw, raw.score))
                continue
            if self.undefined_policy is UndefinedPolicy.ERROR:
                raise ValueError(
                    f"metric {metric.name!r} is undefined for {case.case_id!r}: {raw.reason}"
                )
            if self.undefined_policy is UndefinedPolicy.SKIP:
                results.append(CaseResult(case.case_id, raw, None, skipped=True))
            elif self.undefined_policy is UndefinedPolicy.ZERO:
                results.append(CaseResult(case.case_id, raw, 0.0))
            elif self.undefined_policy is UndefinedPolicy.ONE:
                results.append(CaseResult(case.case_id, raw, 1.0))
            else:  # pragma: no cover - Enum construction prevents this path
                raise AssertionError(f"unsupported undefined policy: {self.undefined_policy}")

        findings = (
            auditor._audit_observed(
                metric,
                self.cases,
                tuple(result.raw for result in results),
            )
            if auditor
            else ()
        )
        return SuiteReport(metric_name=metric.name, results=tuple(results), findings=findings)
