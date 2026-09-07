"""MetricGuard's public API."""

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .edit_metrics import CharacterErrorRate, WordErrorRate
from .experiment import ExperimentMatrix, ExperimentResult, ExperimentSpec, LeaderboardEntry
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
from .ocr import OcrBenchmarkReport, load_ocr_cases, run_ocr_benchmark
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
    "CharacterErrorRate",
    "ConfidenceInterval",
    "Contract",
    "ContractAuditor",
    "EvaluationCase",
    "EvaluationSuite",
    "ExperimentMatrix",
    "ExperimentResult",
    "ExperimentSpec",
    "LeaderboardEntry",
    "LevenshteinSimilarity",
    "Metric",
    "MetricPluginError",
    "MetricRegistry",
    "MetricRunner",
    "MetricValue",
    "OcrBenchmarkReport",
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
    "WordErrorRate",
    "build_metric",
    "compare_case_sets",
    "confidence_interval",
    "load_ocr_cases",
    "paired_comparison",
    "report_confidence_interval",
    "run_ocr_benchmark",
    "summarize_by_tag",
]

__version__ = "0.2.0"
