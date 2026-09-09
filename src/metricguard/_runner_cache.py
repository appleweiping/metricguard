"""Versioned identities and validated, atomic metric checkpoints."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import unicodedata
from dataclasses import fields
from decimal import Decimal, getcontext
from pathlib import Path
from typing import Any, cast

from .advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from .edit_metrics import CharacterErrorRate, WordErrorRate
from .metrics import CharacterF1, ExactMatch, Metric, NumericEquivalence, TokenF1
from .models import EvaluationCase, MetricValue
from .normalizers import TextNormalizer
from .ranking import RankingMetric

_FORMAT_VERSION = 2
# Bump when built-in score semantics change, including normalization helpers.
_BUILTIN_REVISION = 1
_BUILTIN_TYPES = (
    ExactMatch,
    TokenF1,
    CharacterF1,
    NumericEquivalence,
    CharacterErrorRate,
    WordErrorRate,
    RougeL,
    SentenceBleu,
    LevenshteinSimilarity,
    RankingMetric,
)


def _json(value: Any) -> str:
    """Serialize only lossless JSON values, rejecting coerced dictionary keys."""
    if isinstance(value, dict):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        for item in value.values():
            _json(item)
    elif isinstance(value, list):
        for item in value:
            _json(item)
    elif type(value) not in (str, int, float, bool, type(None)):
        raise ValueError(f"unsupported JSON value: {type(value).__name__}")
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _builtin_config(value: Any) -> Any:
    if type(value) in (*_BUILTIN_TYPES, TextNormalizer):
        return {
            "type": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {
                field.name: _builtin_config(getattr(value, field.name)) for field in fields(value)
            },
        }
    if isinstance(value, Decimal):
        return {"decimal": str(value)}
    # Custom normalizers must not be mistaken for a built-in configuration.
    _json(value)
    return value


def metric_identity(metric: Metric) -> dict[str, Any]:
    """Return configuration identity without discovering or importing plugins."""
    if type(metric) in _BUILTIN_TYPES:
        identity = {
            "builtin_revision": _BUILTIN_REVISION,
            "unicode_version": unicodedata.unidata_version,
            "config": _builtin_config(metric),
        }
        if type(metric) is NumericEquivalence:
            context = getcontext()
            identity["decimal_context"] = {
                "prec": context.prec,
                "rounding": context.rounding,
                "Emin": context.Emin,
                "Emax": context.Emax,
                "capitals": context.capitals,
                "clamp": context.clamp,
                "traps": {signal.__name__: enabled for signal, enabled in context.traps.items()},
            }
        return identity
    hook = getattr(metric, "cache_identity", None)
    if not callable(hook):
        raise ValueError("custom metrics require cache_identity() when a cache is requested")
    identity = hook()
    if not isinstance(identity, dict) or not identity:
        raise ValueError("cache_identity() must return a non-empty JSON object")
    _json(identity)
    return {
        "type": f"{type(metric).__module__}.{type(metric).__qualname__}",
        "name": metric.name,
        "identity": identity,
    }


def fingerprint(identity: dict[str, Any], case: EvaluationCase) -> str:
    try:
        return _digest(
            {"metric": identity, "reference": case.reference, "prediction": case.prediction}
        )
    except (TypeError, ValueError, RecursionError) as error:
        raise ValueError(
            f"case {case.case_id!r} is not losslessly JSON-serializable for caching"
        ) from error


def make_row(case_id: str, key: str, value: MetricValue) -> dict[str, Any]:
    row = {
        "version": _FORMAT_VERSION,
        "case_id": case_id,
        "fingerprint": key,
        "score": value.score,
        "reason": value.reason,
        "details": value.details,
    }
    _validate_value(row)
    # MetricValue is shallowly frozen; a plugin may reuse its details dictionary.
    row = cast(dict[str, Any], json.loads(_json(row)))
    row["checksum"] = _digest(row)
    return row


def raw_value(row: dict[str, Any]) -> MetricValue:
    return MetricValue(row["score"], row["reason"], row["details"])


def _validate_value(row: dict[str, Any]) -> None:
    score, reason, details = row["score"], row["reason"], row["details"]
    if score is not None and (type(score) not in (float, int) or not math.isfinite(score)):
        raise ValueError("cached score must be a finite number or null")
    if reason is not None and not isinstance(reason, str):
        raise ValueError("cached reason must be a string or null")
    if score is None and (reason is None or not reason.strip()):
        raise ValueError("cached undefined score requires a reason")
    if not isinstance(details, dict):
        raise ValueError("cached details must be a JSON object")
    _json(details)


def _is_digest(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, object_pairs_hook=_unique_object)
                if not isinstance(row, dict):
                    raise ValueError("cache row must be a JSON object")
                if "version" not in row:
                    # Legacy name-only identities cannot prove configuration equivalence.
                    continue
                if type(row["version"]) is not int or row["version"] != _FORMAT_VERSION:
                    raise ValueError("unsupported cache version")
                if set(row) != {
                    "version",
                    "case_id",
                    "fingerprint",
                    "score",
                    "reason",
                    "details",
                    "checksum",
                }:
                    raise ValueError("cache row has missing or unexpected fields")
                case_id = row["case_id"]
                if not isinstance(case_id, str) or not case_id.strip():
                    raise ValueError("cache case_id must be a non-empty string")
                if case_id in rows:
                    raise ValueError(f"duplicate cached case_id: {case_id!r}")
                if not _is_digest(row["fingerprint"]) or not _is_digest(row["checksum"]):
                    raise ValueError("cache fingerprint and checksum must be SHA-256 hex digests")
                _validate_value(row)
                body = {key: value for key, value in row.items() if key != "checksum"}
                if row["checksum"] != _digest(body):
                    raise ValueError("cache checksum mismatch")
                rows[case_id] = row
            except (TypeError, ValueError, OverflowError, RecursionError) as error:
                raise ValueError(
                    f"invalid metric cache {path} at line {number}: {error}"
                ) from error
    return rows


def save_cache(path: Path, rows: dict[str, dict[str, Any]]) -> None:
    """Replace the checkpoint only after a complete, flushed sibling file exists."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            for key in sorted(rows):
                stream.write(_json(rows[key]) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
