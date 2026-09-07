"""Document-level OCR orchestration with deterministic page ordering."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .io import CaseFormatError, _read_jsonl, _strict_json_loads
from .models import EvaluationCase, UndefinedPolicy
from .ocr import OcrBenchmarkReport, run_ocr_benchmark


@dataclass(frozen=True, slots=True)
class OcrDocumentPage:
    """One ordered page in a document OCR evaluation manifest."""

    page_id: str
    document_id: str
    page_index: int
    reference: str
    image: Path | None = None

    def __post_init__(self) -> None:
        for name, value in (("page_id", self.page_id), ("document_id", self.document_id)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        if isinstance(self.page_index, bool) or not isinstance(self.page_index, int):
            raise TypeError("page_index must be an integer")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if not isinstance(self.reference, str):
            raise TypeError("reference must be a string")
        if self.image is not None and not isinstance(self.image, Path):
            raise TypeError("image must be a pathlib.Path or None")


@dataclass(frozen=True, slots=True)
class OcrDocumentSummary:
    """Metrics for one document, retaining its page-level report."""

    document_id: str
    pages: int
    report: OcrBenchmarkReport

    def to_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "pages": self.pages, **self.report.to_dict()}


@dataclass(frozen=True, slots=True)
class OcrDocumentReport:
    """Page-level and per-document OCR metrics in one auditable result."""

    page_report: OcrBenchmarkReport
    documents: tuple[OcrDocumentSummary, ...]

    @property
    def pages(self) -> int:
        return self.page_report.cases

    @property
    def document_count(self) -> int:
        return len(self.documents)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.document_count,
            "pages": self.pages,
            "page_report": self.page_report.to_dict(),
            "document_reports": [item.to_dict() for item in self.documents],
        }


def run_document_ocr_benchmark(
    cases: Iterable[OcrDocumentPage],
    backend: Callable[[OcrDocumentPage], str],
    *,
    undefined_policy: UndefinedPolicy = UndefinedPolicy.SKIP,
) -> OcrDocumentReport:
    """Run a page backend and report both global and document-level metrics.

    Pages are sorted by ``(document_id, page_index, page_id)`` before invoking
    the backend. Duplicate page IDs or document/page positions are rejected so
    a manifest cannot silently overwrite a page in a downstream report.
    """

    if not callable(backend):
        raise TypeError("backend must be callable")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one OCR document page is required")
    ordered = tuple(
        sorted(materialized, key=lambda item: (item.document_id, item.page_index, item.page_id))
    )
    page_ids = [item.page_id for item in ordered]
    positions = [(item.document_id, item.page_index) for item in ordered]
    if len(set(page_ids)) != len(page_ids):
        raise ValueError("OCR document page IDs must be unique")
    if len(set(positions)) != len(positions):
        raise ValueError("OCR document page positions must be unique")

    predictions: list[EvaluationCase] = []
    grouped: dict[str, list[EvaluationCase]] = {}
    for page in ordered:
        prediction = backend(page)
        if not isinstance(prediction, str):
            raise TypeError("OCR document backend must return text")
        case = EvaluationCase(
            page.page_id,
            page.reference,
            prediction,
            tags=("ocr", "document", f"document:{page.document_id}"),
            metadata={"document_id": page.document_id, "page_index": page.page_index},
        )
        predictions.append(case)
        grouped.setdefault(page.document_id, []).append(case)

    page_report = run_ocr_benchmark(predictions, undefined_policy=undefined_policy)
    documents = tuple(
        OcrDocumentSummary(
            document_id,
            len(document_cases),
            run_ocr_benchmark(document_cases, undefined_policy=undefined_policy),
        )
        for document_id, document_cases in sorted(grouped.items())
    )
    return OcrDocumentReport(page_report, documents)


def load_ocr_document_cases(path: str | Path) -> tuple[OcrDocumentPage, ...]:
    """Load strict JSON/JSONL pages with optional paths relative to the manifest."""

    source = Path(path)
    if not source.is_file():
        raise CaseFormatError(f"OCR document case file does not exist: {source}")
    try:
        if source.suffix.lower() == ".jsonl":
            raw_cases = list(_read_jsonl(source))
        else:
            loaded = _strict_json_loads(source.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise CaseFormatError("OCR document JSON files must contain an array")
            raw_cases = loaded
    except json.JSONDecodeError as error:
        raise CaseFormatError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise CaseFormatError(f"cannot read OCR document case file {source}: {error}") from error

    pages: list[OcrDocumentPage] = []
    seen_ids: set[str] = set()
    seen_positions: set[tuple[str, int]] = set()
    required = {"id", "document_id", "page", "reference"}
    allowed = required | {"image"}
    for position, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise CaseFormatError(f"OCR document page {position} must be an object")
        missing = required - raw.keys()
        unexpected = set(raw) - allowed
        if missing:
            raise CaseFormatError(
                f"OCR document page {position} is missing: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise CaseFormatError(
                f"OCR document page {position} has unknown fields: {', '.join(sorted(unexpected))}"
            )
        page_id = raw["id"]
        document_id = raw["document_id"]
        page_index = raw["page"]
        reference = raw["reference"]
        if not isinstance(page_id, str) or not page_id.strip():
            raise CaseFormatError(f"OCR document page {position} id must be a non-empty string")
        if not isinstance(document_id, str) or not document_id.strip():
            raise CaseFormatError(
                f"OCR document page {position} document_id must be a non-empty string"
            )
        if isinstance(page_index, bool) or not isinstance(page_index, int) or page_index < 0:
            raise CaseFormatError(
                f"OCR document page {position} page must be a non-negative integer"
            )
        if not isinstance(reference, str):
            raise CaseFormatError(f"OCR document page {position} reference must be a string")
        image_value = raw.get("image")
        image: Path | None = None
        if image_value is not None:
            if not isinstance(image_value, str) or not image_value.strip():
                raise CaseFormatError(f"OCR document page {position} image must be a string")
            image = Path(image_value)
            if not image.is_absolute():
                image = source.parent / image
        if page_id in seen_ids:
            raise CaseFormatError(f"OCR document page {position} duplicates id {page_id!r}")
        page_position = (document_id, page_index)
        if page_position in seen_positions:
            raise CaseFormatError(
                f"OCR document page {position} duplicates position {document_id!r}/{page_index}"
            )
        pages.append(OcrDocumentPage(page_id, document_id, page_index, reference, image))
        seen_ids.add(page_id)
        seen_positions.add(page_position)
    if not pages:
        raise CaseFormatError(f"OCR document case file {source} contains no pages")
    return tuple(pages)
