"""Sequence-edit metrics for OCR and long-context text evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .models import MetricValue
from .normalizers import TextNormalizer


def _distance(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Return Levenshtein distance using memory proportional to the shorter side."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_item in left:
        current = [previous[0] + 1]
        for index, right_item in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[index] + 1,
                    previous[index - 1] + (left_item != right_item),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class CharacterErrorRate:
    """Normalized character edit rate (lower is better)."""

    normalizer: Callable[[Any], str] = field(default_factory=TextNormalizer)
    name: str = "character_error_rate"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = tuple(self.normalizer(reference))
        actual = tuple(self.normalizer(prediction))
        if not expected:
            if not actual:
                return MetricValue(0.0, details={"distance": 0, "reference_characters": 0})
            return MetricValue(
                None,
                reason="character error rate is undefined for an empty reference",
                details={"distance": len(actual), "reference_characters": 0},
            )
        distance = _distance(expected, actual)
        return MetricValue(
            distance / len(expected),
            details={
                "distance": distance,
                "reference_characters": len(expected),
                "prediction_characters": len(actual),
            },
        )


@dataclass(frozen=True, slots=True)
class WordErrorRate:
    """Normalized word-level edit rate (lower is better)."""

    normalizer: TextNormalizer = field(
        default_factory=lambda: TextNormalizer(lowercase=True, strip_punctuation=True)
    )
    name: str = "word_error_rate"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = tuple(self.normalizer.tokenize(reference))
        actual = tuple(self.normalizer.tokenize(prediction))
        if not expected:
            if not actual:
                return MetricValue(0.0, details={"distance": 0, "reference_words": 0})
            return MetricValue(
                None,
                reason="word error rate is undefined for an empty reference",
                details={"distance": len(actual), "reference_words": 0},
            )
        distance = _distance(expected, actual)
        return MetricValue(
            distance / len(expected),
            details={
                "distance": distance,
                "reference_words": len(expected),
                "prediction_words": len(actual),
            },
        )
