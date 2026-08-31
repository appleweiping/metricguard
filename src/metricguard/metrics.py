"""Built-in metrics and their configuration factory."""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol, runtime_checkable

from .models import MetricValue
from .normalizers import TextNormalizer


@runtime_checkable
class Metric(Protocol):
    """Structural protocol implemented by MetricGuard metrics."""

    @property
    def name(self) -> str:
        """Stable display name."""

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        """Evaluate one pair without applying an undefined policy."""


@dataclass(frozen=True, slots=True)
class ExactMatch:
    """Normalized exact string equality."""

    normalizer: TextNormalizer = field(default_factory=TextNormalizer)
    name: str = "exact_match"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        left = self.normalizer(reference)
        right = self.normalizer(prediction)
        return MetricValue(float(left == right), details={"reference": left, "prediction": right})


@dataclass(frozen=True, slots=True)
class TokenF1:
    """Bag-of-token F1 that preserves duplicate token counts."""

    normalizer: TextNormalizer = field(
        default_factory=lambda: TextNormalizer(lowercase=True, strip_punctuation=True)
    )
    name: str = "token_f1"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = Counter(self.normalizer.tokenize(reference))
        actual = Counter(self.normalizer.tokenize(prediction))
        if not expected and not actual:
            return MetricValue(
                1.0, details={"overlap": 0, "reference_tokens": 0, "prediction_tokens": 0}
            )
        if not expected or not actual:
            return MetricValue(
                0.0,
                details={
                    "overlap": 0,
                    "reference_tokens": sum(expected.values()),
                    "prediction_tokens": sum(actual.values()),
                },
            )
        overlap = sum((expected & actual).values())
        precision = overlap / sum(actual.values())
        recall = overlap / sum(expected.values())
        score = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return MetricValue(
            score,
            details={
                "overlap": overlap,
                "precision": precision,
                "recall": recall,
                "reference_tokens": sum(expected.values()),
                "prediction_tokens": sum(actual.values()),
            },
        )


@dataclass(frozen=True, slots=True)
class CharacterF1:
    """Character multiset F1 after normalization."""

    normalizer: TextNormalizer = field(default_factory=lambda: TextNormalizer(lowercase=True))
    name: str = "character_f1"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = Counter(self.normalizer(reference))
        actual = Counter(self.normalizer(prediction))
        if not expected and not actual:
            return MetricValue(1.0)
        if not expected or not actual:
            return MetricValue(0.0)
        overlap = sum((expected & actual).values())
        precision = overlap / sum(actual.values())
        recall = overlap / sum(expected.values())
        score = 2 * precision * recall / (precision + recall) if overlap else 0.0
        return MetricValue(score, details={"overlap": overlap})


_NUMBER = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")
_GROUPED_NUMBER = re.compile(r"^[+-]?\d{1,3}(?:,\d{3})+(?:\.\d*)?(?:[eE][+-]?\d+)?$")


def _parse_decimal(value: Any, *, allow_percent: bool, allow_commas: bool) -> Decimal | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, int):
        return Decimal(value)
    if isinstance(value, float):
        return Decimal(str(value)) if math.isfinite(value) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    percent = text.endswith("%")
    if percent:
        if not allow_percent:
            return None
        text = text[:-1].strip()
    if "," in text:
        if not allow_commas or not _GROUPED_NUMBER.fullmatch(text):
            return None
        text = text.replace(",", "")
    if not _NUMBER.fullmatch(text):
        return None
    try:
        parsed = Decimal(text)
    except InvalidOperation:
        return None
    return parsed / 100 if percent else parsed


@dataclass(frozen=True, slots=True)
class NumericEquivalence:
    """Binary numeric equivalence with explicit parsing and tolerances."""

    absolute_tolerance: Decimal = Decimal("0")
    relative_tolerance: Decimal = Decimal("0")
    allow_percent: bool = False
    allow_commas: bool = False
    name: str = "numeric_equivalence"

    def __post_init__(self) -> None:
        tolerances = (self.absolute_tolerance, self.relative_tolerance)
        if not all(isinstance(value, Decimal) for value in tolerances):
            raise TypeError("numeric tolerances must be Decimal values")
        if not all(value.is_finite() for value in tolerances):
            raise ValueError("numeric tolerances must be finite")
        if self.absolute_tolerance < 0 or self.relative_tolerance < 0:
            raise ValueError("numeric tolerances must be non-negative")
        if type(self.allow_percent) is not bool or type(self.allow_commas) is not bool:
            raise TypeError("numeric parsing flags must be booleans")

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = _parse_decimal(
            reference, allow_percent=self.allow_percent, allow_commas=self.allow_commas
        )
        actual = _parse_decimal(
            prediction, allow_percent=self.allow_percent, allow_commas=self.allow_commas
        )
        if expected is None or actual is None:
            invalid = []
            if expected is None:
                invalid.append("reference")
            if actual is None:
                invalid.append("prediction")
            return MetricValue(None, reason=f"unparseable {' and '.join(invalid)}")
        difference = abs(expected - actual)
        scale = max(abs(expected), abs(actual))
        allowed = max(self.absolute_tolerance, self.relative_tolerance * scale)
        return MetricValue(
            float(difference <= allowed),
            details={
                "reference": str(expected),
                "prediction": str(actual),
                "difference": str(difference),
                "allowed_difference": str(allowed),
            },
        )


def _normalizer_from_config(config: dict[str, Any]) -> TextNormalizer:
    raw = config.get("normalizer", {})
    if not isinstance(raw, dict):
        raise ValueError("normalizer must be an object")
    allowed = {"unicode_form", "lowercase", "collapse_whitespace", "strip_punctuation"}
    unexpected = set(raw) - allowed
    if unexpected:
        raise ValueError(f"unknown normalizer options: {', '.join(sorted(unexpected))}")
    for name in ("lowercase", "collapse_whitespace", "strip_punctuation"):
        if name in raw and type(raw[name]) is not bool:
            raise ValueError(f"normalizer option {name!r} must be a boolean")
    unicode_form = raw.get("unicode_form")
    if unicode_form is not None and not isinstance(unicode_form, str):
        raise ValueError("normalizer option 'unicode_form' must be a string")
    return TextNormalizer(**raw)


def _decimal_option(options: dict[str, Any], name: str) -> Decimal:
    value = options.get(name, 0)
    if isinstance(value, bool) or not isinstance(value, (int, float, str, Decimal)):
        raise ValueError(f"metric option {name!r} must be a finite decimal")
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as error:
        raise ValueError(f"metric option {name!r} must be a finite decimal") from error
    if not parsed.is_finite():
        raise ValueError(f"metric option {name!r} must be a finite decimal")
    return parsed


def _boolean_option(options: dict[str, Any], name: str) -> bool:
    value = options.get(name, False)
    if type(value) is not bool:
        raise ValueError(f"metric option {name!r} must be a boolean")
    return value


def build_metric(config: str | dict[str, Any]) -> Metric:
    """Build a built-in metric from a name or JSON-compatible object."""

    if isinstance(config, str):
        kind = config
        options: dict[str, Any] = {}
    elif isinstance(config, dict):
        options = dict(config)
        kind_value = options.pop("kind", None)
        if not isinstance(kind_value, str):
            raise ValueError("metric configuration requires a string 'kind'")
        kind = kind_value
    else:
        raise TypeError("metric configuration must be a string or object")

    if kind == "exact_match":
        normalizer = (
            _normalizer_from_config(options) if "normalizer" in options else TextNormalizer()
        )
        unexpected = set(options) - {"normalizer"}
        if unexpected:
            raise ValueError(f"unknown exact_match options: {', '.join(sorted(unexpected))}")
        return ExactMatch(normalizer=normalizer)
    if kind == "token_f1":
        normalizer = (
            _normalizer_from_config(options)
            if "normalizer" in options
            else TextNormalizer(lowercase=True, strip_punctuation=True)
        )
        unexpected = set(options) - {"normalizer"}
        if unexpected:
            raise ValueError(f"unknown token_f1 options: {', '.join(sorted(unexpected))}")
        return TokenF1(normalizer=normalizer)
    if kind == "character_f1":
        normalizer = (
            _normalizer_from_config(options)
            if "normalizer" in options
            else TextNormalizer(lowercase=True)
        )
        unexpected = set(options) - {"normalizer"}
        if unexpected:
            raise ValueError(f"unknown character_f1 options: {', '.join(sorted(unexpected))}")
        return CharacterF1(normalizer=normalizer)
    if kind == "numeric_equivalence":
        allowed = {"absolute_tolerance", "relative_tolerance", "allow_percent", "allow_commas"}
        unexpected = set(options) - allowed
        if unexpected:
            raise ValueError(
                f"unknown numeric_equivalence options: {', '.join(sorted(unexpected))}"
            )
        return NumericEquivalence(
            absolute_tolerance=_decimal_option(options, "absolute_tolerance"),
            relative_tolerance=_decimal_option(options, "relative_tolerance"),
            allow_percent=_boolean_option(options, "allow_percent"),
            allow_commas=_boolean_option(options, "allow_commas"),
        )
    raise ValueError(f"unknown metric kind: {kind}")
