"""Typed value objects used across MetricGuard."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from statistics import fmean
from typing import Any


class UndefinedPolicy(str, Enum):
    """How a suite resolves a metric result that has no mathematical value."""

    ERROR = "error"
    SKIP = "skip"
    ZERO = "zero"
    ONE = "one"


class Severity(str, Enum):
    """Severity attached to a contract finding."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """One frozen case record; nested JSON payloads are not deeply frozen."""

    case_id: str
    reference: Any
    prediction: Any
    tags: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str):
            raise TypeError("case_id must be a string")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not isinstance(self.tags, tuple) or not all(isinstance(tag, str) for tag in self.tags):
            raise TypeError("tags must be a tuple of strings")
        if len(set(self.tags)) != len(self.tags):
            raise ValueError(f"case {self.case_id!r} contains duplicate tags")
        if not isinstance(self.metadata, dict) or not all(
            isinstance(key, str) for key in self.metadata
        ):
            raise TypeError("metadata must be a dictionary with string keys")


@dataclass(frozen=True, slots=True)
class MetricValue:
    """A frozen metric outcome; ``details`` is not deeply frozen."""

    score: float | None
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.score is not None and not isfinite(self.score):
            raise ValueError("metric scores must be finite or None")
        if self.score is None and not self.reason:
            raise ValueError("an undefined score requires a reason")

    @property
    def defined(self) -> bool:
        """Return whether this outcome contains a numeric score."""

        return self.score is not None


@dataclass(frozen=True, slots=True)
class CaseResult:
    """A resolved result for one case."""

    case_id: str
    raw: MetricValue
    resolved_score: float | None
    skipped: bool = False


@dataclass(frozen=True, slots=True)
class AuditFinding:
    """A contract violation or noteworthy observation."""

    rule: str
    severity: Severity
    message: str
    case_id: str | None = None
    observed: Any = None


@dataclass(frozen=True, slots=True)
class SuiteReport:
    """Complete result of one metric over a suite."""

    metric_name: str
    results: tuple[CaseResult, ...]
    findings: tuple[AuditFinding, ...] = ()

    @property
    def scored_count(self) -> int:
        """Number of cases included in the aggregate."""

        return sum(result.resolved_score is not None for result in self.results)

    @property
    def skipped_count(self) -> int:
        """Number of cases omitted by the undefined policy."""

        return sum(result.skipped for result in self.results)

    @property
    def mean_score(self) -> float | None:
        """Macro mean over resolved scores, or ``None`` for an empty report."""

        scores = [
            result.resolved_score for result in self.results if result.resolved_score is not None
        ]
        return fmean(scores) if scores else None

    @property
    def passed_contracts(self) -> bool:
        """Return ``True`` when the audit produced no error findings."""

        return not any(finding.severity is Severity.ERROR for finding in self.findings)
