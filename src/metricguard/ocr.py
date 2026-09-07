"""OCR-oriented benchmark orchestration over recognized text pairs."""

from __future__ import annotations

import json
import subprocess  # nosec B404 - the backend is an explicit, shell-free argv adapter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .edit_metrics import CharacterErrorRate, WordErrorRate
from .io import CaseFormatError, _read_jsonl, _strict_json_loads, load_cases
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


@dataclass(frozen=True, slots=True)
class OcrImageCase:
    """One image path and its human transcription for backend evaluation."""

    case_id: str
    image: Path
    reference: str

    def __post_init__(self) -> None:
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.image, Path):
            raise TypeError("image must be a pathlib.Path")
        if not isinstance(self.reference, str):
            raise TypeError("reference must be a string")


class CommandOcrBackend:
    """Run an external OCR executable without invoking a shell.

    The command is an argv template containing the literal ``{image}``
    placeholder. Stdout is the recognized text; stderr is never returned to
    callers. This supports local OCR tools and model servers wrapped by a small
    executable while keeping credentials and process arguments out of metric
    reports.
    """

    def __init__(
        self,
        command: Sequence[str],
        *,
        timeout: float = 120.0,
        max_output_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if not command or not all(isinstance(item, str) and item for item in command):
            raise ValueError("command must be a non-empty sequence of strings")
        if "{image}" not in command:
            raise ValueError("command must contain an {image} placeholder")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise ValueError("timeout must be a positive number")
        if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
            raise TypeError("max_output_bytes must be an integer")
        if max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        self.command = tuple(command)
        self.timeout = float(timeout)
        self.max_output_bytes = max_output_bytes

    def __call__(self, image: Path) -> str:
        """Transcribe one image, rejecting process and output failures."""

        if not isinstance(image, Path):
            raise TypeError("image must be a pathlib.Path")
        if not image.is_file():
            raise ValueError(f"OCR image does not exist: {image}")
        argv = tuple(str(image) if item == "{image}" else item for item in self.command)
        try:
            completed = subprocess.run(  # nosec B603 - shell=False and argv are explicit
                argv,
                check=False,
                capture_output=True,
                timeout=self.timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as error:
            raise ValueError("OCR backend timed out") from error
        except OSError as error:
            raise ValueError(f"OCR backend could not start: {type(error).__name__}") from error
        if completed.returncode != 0:
            raise ValueError(f"OCR backend exited with status {completed.returncode}")
        if len(completed.stdout) > self.max_output_bytes:
            raise ValueError(f"OCR output exceeds {self.max_output_bytes} bytes")
        try:
            return completed.stdout.decode("utf-8").rstrip("\r\n")
        except UnicodeDecodeError as error:
            raise ValueError("OCR backend output is not valid UTF-8") from error


def run_ocr_backend_benchmark(
    cases: Iterable[OcrImageCase],
    backend: Callable[[Path], str],
    *,
    undefined_policy: UndefinedPolicy = UndefinedPolicy.SKIP,
) -> OcrBenchmarkReport:
    """Run a pluggable image backend and evaluate its text outputs."""

    if not callable(backend):
        raise TypeError("backend must be callable")
    materialized = tuple(cases)
    if not materialized:
        raise ValueError("at least one OCR image case is required")
    predictions: list[EvaluationCase] = []
    for case in materialized:
        prediction = backend(case.image)
        if not isinstance(prediction, str):
            raise TypeError("OCR backend must return text")
        predictions.append(
            EvaluationCase(
                case.case_id,
                case.reference,
                prediction,
                tags=("ocr",),
                metadata={"image": str(case.image)},
            )
        )
    return run_ocr_benchmark(predictions, undefined_policy=undefined_policy)


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


def load_ocr_image_cases(path: str | Path) -> tuple[OcrImageCase, ...]:
    """Load image/reference records for a command or model OCR backend.

    The file may be a JSON array or JSONL. Relative image paths are resolved
    against the case file directory and are intentionally not required to
    exist until a backend is run, allowing manifests to be prepared before a
    dataset mount is available.
    """

    source = Path(path)
    if not source.is_file():
        raise CaseFormatError(f"OCR image case file does not exist: {source}")
    try:
        if source.suffix.lower() == ".jsonl":
            raw_cases = list(_read_jsonl(source))
        else:
            loaded = _strict_json_loads(source.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise CaseFormatError("OCR image JSON files must contain an array")
            raw_cases = loaded
    except json.JSONDecodeError as error:
        raise CaseFormatError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise CaseFormatError(f"cannot read OCR image case file {source}: {error}") from error

    cases: list[OcrImageCase] = []
    seen: set[str] = set()
    for position, raw in enumerate(raw_cases, start=1):
        if not isinstance(raw, dict):
            raise CaseFormatError(f"OCR image case {position} must be an object")
        required = {"id", "image", "reference"}
        missing = required - raw.keys()
        unexpected = set(raw) - required
        if missing:
            raise CaseFormatError(
                f"OCR image case {position} is missing: {', '.join(sorted(missing))}"
            )
        if unexpected:
            raise CaseFormatError(
                f"OCR image case {position} has unknown fields: {', '.join(sorted(unexpected))}"
            )
        case_id = raw["id"]
        image_name = raw["image"]
        reference = raw["reference"]
        if not isinstance(case_id, str) or not case_id.strip():
            raise CaseFormatError(f"OCR image case {position} id must be a non-empty string")
        if case_id in seen:
            raise CaseFormatError(f"OCR image case {position} duplicates id {case_id!r}")
        if not isinstance(image_name, str) or not image_name.strip():
            raise CaseFormatError(f"OCR image case {position} image must be a non-empty string")
        if not isinstance(reference, str):
            raise CaseFormatError(f"OCR image case {position} reference must be a string")
        image = Path(image_name)
        if not image.is_absolute():
            image = source.parent / image
        cases.append(OcrImageCase(case_id, image, reference))
        seen.add(case_id)
    if not cases:
        raise CaseFormatError(f"OCR image case file {source} contains no cases")
    return tuple(cases)
