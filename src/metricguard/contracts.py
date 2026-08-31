"""Metric contracts and metamorphic audits."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Any

from .metrics import Metric
from .models import AuditFinding, EvaluationCase, MetricValue, Severity


@dataclass(frozen=True, slots=True)
class Contract:
    """Properties expected from a metric implementation.

    ``identity_score`` checks ``metric(x, x)`` for every distinct case value.
    ``symmetric`` checks whether swapping reference and prediction is safe.
    Neither property is assumed automatically because some evaluation metrics
    are intentionally directional.
    """

    minimum: float = 0.0
    maximum: float = 1.0
    identity_score: float | None = 1.0
    symmetric: bool = False
    deterministic: bool = True
    tolerance: float = 1e-12

    def __post_init__(self) -> None:
        for name in ("minimum", "maximum", "tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"contract {name} must be a real number")
        if self.identity_score is not None and (
            isinstance(self.identity_score, bool)
            or not isinstance(self.identity_score, (int, float))
        ):
            raise TypeError("contract identity_score must be a real number or None")
        for name in ("symmetric", "deterministic"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"contract {name} must be a boolean")
        numeric_values = (self.minimum, self.maximum, self.tolerance)
        if not all(isfinite(value) for value in numeric_values):
            raise ValueError("contract bounds and tolerance must be finite")
        if self.identity_score is not None and not isfinite(self.identity_score):
            raise ValueError("identity_score must be finite or None")
        if self.minimum > self.maximum:
            raise ValueError("contract minimum cannot exceed maximum")
        if self.tolerance < 0:
            raise ValueError("contract tolerance must be non-negative")
        if self.identity_score is not None and not (
            self.minimum <= self.identity_score <= self.maximum
        ):
            raise ValueError("identity_score must fall inside the score range")


class ContractAuditor:
    """Audit observed and generated metric behavior against a contract."""

    def __init__(self, contract: Contract | None = None) -> None:
        if contract is not None and not isinstance(contract, Contract):
            raise TypeError("contract must be a Contract or None")
        self.contract = contract or Contract()

    def audit(self, metric: Metric, cases: Iterable[EvaluationCase]) -> tuple[AuditFinding, ...]:
        """Run boundedness, determinism, identity, and optional symmetry checks."""

        materialized = tuple(cases)
        return self._audit(metric, materialized, None)

    def _audit_observed(
        self,
        metric: Metric,
        cases: tuple[EvaluationCase, ...],
        observed: tuple[MetricValue, ...],
    ) -> tuple[AuditFinding, ...]:
        """Audit against values already observed by an evaluation suite."""

        if len(cases) != len(observed):
            raise ValueError("observed metric values must align with the audited cases")
        return self._audit(metric, cases, observed)

    def _audit(
        self,
        metric: Metric,
        materialized: tuple[EvaluationCase, ...],
        observed: tuple[MetricValue, ...] | None,
    ) -> tuple[AuditFinding, ...]:
        findings: list[AuditFinding] = []
        for index, case in enumerate(materialized):
            first = (
                metric.evaluate(case.reference, case.prediction)
                if observed is None
                else observed[index]
            )
            if first.score is not None and (
                first.score < self.contract.minimum - self.contract.tolerance
                or first.score > self.contract.maximum + self.contract.tolerance
            ):
                findings.append(
                    AuditFinding(
                        rule="bounded",
                        severity=Severity.ERROR,
                        message=(
                            f"score {first.score} is outside "
                            f"[{self.contract.minimum}, {self.contract.maximum}]"
                        ),
                        case_id=case.case_id,
                        observed=first.score,
                    )
                )
            if self.contract.deterministic:
                second = metric.evaluate(case.reference, case.prediction)
                if not _equivalent_values(first.score, second.score, self.contract.tolerance):
                    findings.append(
                        AuditFinding(
                            rule="deterministic",
                            severity=Severity.ERROR,
                            message="repeated evaluation returned a different score",
                            case_id=case.case_id,
                            observed=[first.score, second.score],
                        )
                    )
            if self.contract.symmetric:
                reverse = metric.evaluate(case.prediction, case.reference)
                if not _equivalent_values(first.score, reverse.score, self.contract.tolerance):
                    findings.append(
                        AuditFinding(
                            rule="symmetric",
                            severity=Severity.ERROR,
                            message="swapping reference and prediction changed the score",
                            case_id=case.case_id,
                            observed=[first.score, reverse.score],
                        )
                    )

        if self.contract.identity_score is not None:
            for label, value in _distinct_values(materialized):
                identity = metric.evaluate(value, value)
                if identity.score is None or not isclose(
                    identity.score,
                    self.contract.identity_score,
                    rel_tol=0.0,
                    abs_tol=self.contract.tolerance,
                ):
                    findings.append(
                        AuditFinding(
                            rule="identity",
                            severity=Severity.ERROR,
                            message=f"self-comparison did not score {self.contract.identity_score}",
                            case_id=label,
                            observed=identity.score,
                        )
                    )
        if not materialized:
            findings.append(
                AuditFinding(
                    rule="coverage",
                    severity=Severity.WARNING,
                    message="no cases were available for contract auditing",
                )
            )
        return tuple(findings)


def _equivalent_values(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left is right
    return isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def _distinct_values(cases: tuple[EvaluationCase, ...]) -> tuple[tuple[str, Any], ...]:
    """Deduplicate JSON-like values without requiring them to be hashable."""

    output: list[tuple[str, Any]] = []
    representations: set[str] = set()
    for case in cases:
        for side, value in (("reference", case.reference), ("prediction", case.prediction)):
            marker = repr(value)
            if marker in representations:
                continue
            representations.add(marker)
            output.append((f"{case.case_id}:{side}", value))
    return tuple(output)
