"""MetricGuard's public API."""

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .document import (
    OcrDocumentPage,
    OcrDocumentReport,
    OcrDocumentSummary,
    load_ocr_document_cases,
    run_document_ocr_benchmark,
)
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
from .ocr import (
    CommandOcrBackend,
    OcrBenchmarkReport,
    OcrImageCase,
    load_ocr_cases,
    load_ocr_image_cases,
    run_ocr_backend_benchmark,
    run_ocr_benchmark,
)
from .ranking import RankingMetric
from .registry import MetricPluginError, MetricRegistry
from .runner import MetricRunner, RunnerReport
from .slices import (
    SliceComparison,
    SliceSummary,
    compare_by_metadata,
    correct_slice_p_values,
    summarize_by_metadata,
)
from .statistics import (
    BootstrapConfig,
    ConfidenceInterval,
    PairedComparison,
    TagSummary,
    adjust_p_values,
    confidence_interval,
    paired_comparison,
    report_confidence_interval,
    summarize_by_tag,
)
from .streaming import StreamingReport, StreamingTagSummary, evaluate_stream
from .suite import EvaluationSuite

__all__ = [
    "AuditFinding",
    "BootstrapConfig",
    "CaseResult",
    "CharacterErrorRate",
    "CommandOcrBackend",
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
    "OcrDocumentPage",
    "OcrDocumentReport",
    "OcrDocumentSummary",
    "OcrImageCase",
    "PairedComparison",
    "RankingMetric",
    "RougeL",
    "RunnerReport",
    "SentenceBleu",
    "Severity",
    "SliceComparison",
    "SliceSummary",
    "StreamingReport",
    "StreamingTagSummary",
    "SuiteReport",
    "TagSummary",
    "TextNormalizer",
    "UndefinedPolicy",
    "WordErrorRate",
    "adjust_p_values",
    "build_metric",
    "compare_by_metadata",
    "compare_case_sets",
    "confidence_interval",
    "correct_slice_p_values",
    "evaluate_stream",
    "load_ocr_cases",
    "load_ocr_document_cases",
    "load_ocr_image_cases",
    "paired_comparison",
    "report_confidence_interval",
    "run_document_ocr_benchmark",
    "run_ocr_backend_benchmark",
    "run_ocr_benchmark",
    "summarize_by_metadata",
    "summarize_by_tag",
]

__version__ = "0.2.0"
