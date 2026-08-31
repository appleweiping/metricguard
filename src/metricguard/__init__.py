"""MetricGuard's public API."""

from .contracts import Contract, ContractAuditor
from .metrics import Metric, build_metric
from .models import (
    AuditFinding,
    CaseResult,
    EvaluationCase,
    MetricValue,
    Severity,
    SuiteReport,
    UndefinedPolicy,
)
from .normalizers import TextNormalizer
from .suite import EvaluationSuite

__all__ = [
    "AuditFinding",
    "CaseResult",
    "Contract",
    "ContractAuditor",
    "EvaluationCase",
    "EvaluationSuite",
    "Metric",
    "MetricValue",
    "Severity",
    "SuiteReport",
    "TextNormalizer",
    "UndefinedPolicy",
    "build_metric",
]

__version__ = "0.1.0"
