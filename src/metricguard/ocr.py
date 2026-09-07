"""OCR-oriented benchmark orchestration over recognized text pairs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edit_metrics import CharacterErrorRate, WordErrorRate
from .io import load_cases
from .metrics import ExactMatch
from .models import EvaluationCase, SuiteReport, UndefinedPolicy
from .suite import EvaluationSuite


@dataclass(frozen=True, slots=True)
class OcrBenchmarkReport:
    """Aggregate OCR recognition reports with explicit undefined-case counts."""

    cases: int
    character_error_rate: SuiteReport
    word_error_rate: SuiteReport
    exact_match: SuiteReport

    @property
    def exact_match_rate(self) -> float | None:
        """Return the macro exact-match rate."""
        return self.exact_match.mean_score

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-compatible summary."""
        return {
            "cases": self.cases,
            "character_error_rate": {
                "mean": self.character_error_rate.mean_score,
                "scored": self.character_error_rate.scored_count,
                "skipped": self.character_error_rate.skipped_count,
            },
            "word_error_rate": {
                "mean": self.word_error_rate.mean_score,
                "scored": self.word_error_rate.scored_count,
                "skipped": self.word_error_rate.skipped_count,
            },
            "exact_match": {
                "mean": self.exact_match.mean_score,
                "scored": self.exact_match.scored_count,
                "skipped": self.exact_match.skipped_count,
            },
        }


def run_ocr_benchmark(
    cases: Iterable[EvaluationCase],
    *,
    undefined_policy: UndefinedPolicy = UndefinedPolicy.SKIP,
) -> OcrBenchmarkReport:
    """Run CER, WER, and normalized exact-match over OCR text pairs."""
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one OCR case is required")
    suite = EvaluationSuite(materialized, undefined_policy=undefined_policy)
    return OcrBenchmarkReport(
        len(materialized),
        suite.run(CharacterErrorRate()),
        suite.run(WordErrorRate()),
        suite.run(ExactMatch()),
    )


def load_ocr_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load the standard JSON/JSONL case format for OCR evaluation."""
    cases = load_cases(path)
    if not cases:
        raise ValueError(f"OCR case file {path} contains no cases")
    return cases
