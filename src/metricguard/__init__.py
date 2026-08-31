"""MetricGuard's public API."""

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .comparison import compare_case_sets
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
from .registry import MetricPluginError, MetricRegistry
from .statistics import (
    BootstrapConfig,
    ConfidenceInterval,
    PairedComparison,
    TagSummary,
    confidence_interval,
    paired_comparison,
    report_confidence_interval,
    summarize_by_tag,
)
from .suite import EvaluationSuite

__all__ = [
    "AuditFinding",
    "BootstrapConfig",
    "CaseResult",
    "ConfidenceInterval",
    "Contract",
    "ContractAuditor",
    "EvaluationCase",
    "EvaluationSuite",
    "LevenshteinSimilarity",
    "Metric",
    "MetricPluginError",
    "MetricRegistry",
    "MetricValue",
    "PairedComparison",
    "RougeL",
    "SentenceBleu",
    "Severity",
    "SuiteReport",
    "TagSummary",
    "TextNormalizer",
    "UndefinedPolicy",
    "build_metric",
    "compare_case_sets",
    "confidence_interval",
    "paired_comparison",
    "report_confidence_interval",
    "summarize_by_tag",
]

__version__ = "0.2.0"
