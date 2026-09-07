"""MetricGuard's public API."""

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .experiment import ExperimentMatrix, ExperimentResult, ExperimentSpec
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
from .ranking import RankingMetric
from .registry import MetricPluginError, MetricRegistry
from .runner import MetricRunner, RunnerReport
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
    "ExperimentMatrix",
    "ExperimentResult",
    "ExperimentSpec",
    "LevenshteinSimilarity",
    "Metric",
    "MetricPluginError",
    "MetricRegistry",
    "MetricRunner",
    "MetricValue",
    "PairedComparison",
    "RankingMetric",
    "RougeL",
    "RunnerReport",
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
