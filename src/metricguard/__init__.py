"""MetricGuard's public API."""

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .comparison import compare_case_sets
from .contracts import Contract, ContractAuditor
from .correlation import CorrelationReport, MetricCorrelation, correlate_reports
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
from .migration import migrate_report
from .models import (
    AuditFinding,
    CaseResult,
    EvaluationCase,
    MetricValue,
    Severity,
    SuiteReport,
    UndefinedPolicy,
)
from .native_ocr import WindowsOcrBackend
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
from .ocr_inputs import OcrInputLimits, load_ocr_pages
from .ocr_pipeline import PageOcrBackend, transcribe_document
from .ocr_types import (
    OcrDocumentResult,
    OcrLine,
    OcrPageImage,
    OcrPageInfo,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
)
from .ranking import RankingMetric
from .registry import MetricPluginError, MetricRegistry
from .reliability import CalibrationBin, CalibrationReport, calibration_report
from .runner import MetricRunner, RunnerReport
from .service import MetricService, create_server
from .slices import (
    MetadataFamilyComparison,
    SliceComparison,
    SliceSummary,
    compare_by_metadata,
    compare_metadata_family,
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
    "CalibrationBin",
    "CalibrationReport",
    "CaseResult",
    "CharacterErrorRate",
    "CommandOcrBackend",
    "ConfidenceInterval",
    "Contract",
    "ContractAuditor",
    "CorrelationReport",
    "EvaluationCase",
    "EvaluationSuite",
    "ExperimentMatrix",
    "ExperimentResult",
    "ExperimentSpec",
    "LeaderboardEntry",
    "LevenshteinSimilarity",
    "MetadataFamilyComparison",
    "Metric",
    "MetricCorrelation",
    "MetricPluginError",
    "MetricRegistry",
    "MetricRunner",
    "MetricService",
    "MetricValue",
    "OcrBenchmarkReport",
    "OcrDocumentPage",
    "OcrDocumentReport",
    "OcrDocumentResult",
    "OcrDocumentSummary",
    "OcrImageCase",
    "OcrInputLimits",
    "OcrLine",
    "OcrPageImage",
    "OcrPageInfo",
    "OcrPageResult",
    "OcrProviderIdentity",
    "OcrWord",
    "PageOcrBackend",
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
    "WindowsOcrBackend",
    "WordErrorRate",
    "adjust_p_values",
    "build_metric",
    "calibration_report",
    "compare_by_metadata",
    "compare_case_sets",
    "compare_metadata_family",
    "confidence_interval",
    "correct_slice_p_values",
    "correlate_reports",
    "create_server",
    "evaluate_stream",
    "load_ocr_cases",
    "load_ocr_document_cases",
    "load_ocr_image_cases",
    "load_ocr_pages",
    "migrate_report",
    "paired_comparison",
    "report_confidence_interval",
    "run_document_ocr_benchmark",
    "run_ocr_backend_benchmark",
    "run_ocr_benchmark",
    "summarize_by_metadata",
    "summarize_by_tag",
    "transcribe_document",
]

__version__ = "0.2.0"
