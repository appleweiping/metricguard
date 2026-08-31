"""Strict JSON and JSONL input handling."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from .models import EvaluationCase


class CaseFormatError(ValueError):
    """Raised when an evaluation case file is not valid MetricGuard input."""


def load_cases(path: str | Path) -> tuple[EvaluationCase, ...]:
    """Load cases from a JSON array or newline-delimited JSON objects."""

    source = Path(path)
    if not source.is_file():
        raise CaseFormatError(f"case file does not exist: {source}")
    try:
        if source.suffix.lower() == ".jsonl":
            raw_cases = list(_read_jsonl(source))
        else:
            loaded = _strict_json_loads(source.read_text(encoding="utf-8"))
            if not isinstance(loaded, list):
                raise CaseFormatError("JSON case files must contain an array")
            raw_cases = loaded
    except json.JSONDecodeError as error:
        raise CaseFormatError(
            f"invalid JSON in {source} at line {error.lineno}, column {error.colno}"
        ) from error
    except (OSError, UnicodeError) as error:
        raise CaseFormatError(f"cannot read case file {source}: {error}") from error
    return tuple(_parse_case(item, position) for position, item in enumerate(raw_cases, start=1))


def _read_jsonl(path: Path) -> Iterable[Any]:
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield _strict_json_loads(line)
        except (json.JSONDecodeError, CaseFormatError) as error:
            column = error.colno if isinstance(error, json.JSONDecodeError) else 1
            raise CaseFormatError(
                f"invalid JSON in {path} at line {line_number}, column {column}: {error}"
            ) from error


def _parse_case(raw: Any, position: int) -> EvaluationCase:
    if not isinstance(raw, dict):
        raise CaseFormatError(f"case {position} must be an object")
    required = {"id", "reference", "prediction"}
    missing = required - raw.keys()
    if missing:
        raise CaseFormatError(f"case {position} is missing: {', '.join(sorted(missing))}")
    tags = raw.get("tags", [])
    metadata = raw.get("metadata", {})
    if not isinstance(raw["id"], str):
        raise CaseFormatError(f"case {position} id must be a string")
    if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
        raise CaseFormatError(f"case {position} tags must be an array of strings")
    if not isinstance(metadata, dict):
        raise CaseFormatError(f"case {position} metadata must be an object")
    unexpected = set(raw) - {"id", "reference", "prediction", "tags", "metadata"}
    if unexpected:
        raise CaseFormatError(
            f"case {position} has unknown fields: {', '.join(sorted(unexpected))}"
        )
    try:
        return EvaluationCase(
            case_id=raw["id"],
            reference=raw["reference"],
            prediction=raw["prediction"],
            tags=tuple(tags),
            metadata=metadata,
        )
    except ValueError as error:
        raise CaseFormatError(f"case {position}: {error}") from error


def load_metric_config(path: str | Path) -> dict[str, Any]:
    """Load a metric configuration object."""

    source = Path(path)
    try:
        value = _strict_json_loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, CaseFormatError) as error:
        raise CaseFormatError(f"cannot load metric configuration {source}: {error}") from error
    if not isinstance(value, dict):
        raise CaseFormatError("metric configuration must be a JSON object")
    return value


def _strict_json_loads(text: str) -> Any:
    return json.loads(
        text,
        parse_constant=_reject_non_finite,
        object_pairs_hook=_unique_object,
    )


def _reject_non_finite(value: str) -> None:
    raise CaseFormatError(f"non-finite JSON number {value!r} is not allowed")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise CaseFormatError(f"duplicate JSON object key {key!r}")
        output[key] = value
    return output
