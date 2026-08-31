"""Dependency-free text metrics with explicit, reviewable semantics."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .models import MetricValue
from .normalizers import TextNormalizer


def _strict_tokens(value: Any, normalizer: TextNormalizer) -> tuple[str, ...]:
    """Normalize and tokenize text without coercing non-string payloads."""

    return tuple(normalizer.tokenize(value))


def _lcs_length(left: tuple[str, ...], right: tuple[str, ...]) -> int:
    """Return LCS length using memory proportional to the shorter sequence."""

    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_item in left:
        current = [0]
        for index, right_item in enumerate(right, start=1):
            if left_item == right_item:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class RougeL:
    """Token-level ROUGE-L F-score based on longest common subsequence.

    This implementation deliberately reports the pairwise F-score. It does not
    perform stemming, sentence splitting, bootstrap aggregation, or the
    multi-reference conventions of a particular external ROUGE package.
    """

    normalizer: TextNormalizer = field(
        default_factory=lambda: TextNormalizer(lowercase=True, strip_punctuation=True)
    )
    beta: float = 1.0
    name: str = "rouge_l"

    def __post_init__(self) -> None:
        if isinstance(self.beta, bool) or not isinstance(self.beta, (int, float)):
            raise TypeError("ROUGE-L beta must be a real number")
        if not math.isfinite(self.beta) or self.beta <= 0:
            raise ValueError("ROUGE-L beta must be finite and positive")

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = _strict_tokens(reference, self.normalizer)
        actual = _strict_tokens(prediction, self.normalizer)
        if not expected and not actual:
            return MetricValue(
                1.0, details={"lcs": 0, "reference_tokens": 0, "prediction_tokens": 0}
            )
        if not expected or not actual:
            return MetricValue(
                0.0,
                details={
                    "lcs": 0,
                    "reference_tokens": len(expected),
                    "prediction_tokens": len(actual),
                },
            )
        lcs = _lcs_length(expected, actual)
        recall = lcs / len(expected)
        precision = lcs / len(actual)
        beta_squared = self.beta * self.beta
        denominator = recall + beta_squared * precision
        score = 0.0 if denominator == 0 else (1 + beta_squared) * precision * recall / denominator
        return MetricValue(
            score,
            details={
                "lcs": lcs,
                "precision": precision,
                "recall": recall,
                "reference_tokens": len(expected),
                "prediction_tokens": len(actual),
            },
        )


def _ngrams(tokens: tuple[str, ...], order: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + order]) for index in range(len(tokens) - order + 1))


@dataclass(frozen=True, slots=True)
class SentenceBleu:
    """Deterministic single-reference sentence BLEU.

    The score uses clipped n-gram precision, the standard brevity penalty,
    optional additive smoothing, and effective order for short predictions.
    It is intentionally named ``sentence_bleu`` rather than claiming byte-for-
    byte compatibility with SacreBLEU or a corpus-level BLEU implementation.
    """

    normalizer: TextNormalizer = field(
        default_factory=lambda: TextNormalizer(lowercase=True, strip_punctuation=True)
    )
    max_order: int = 4
    smooth: float = 1.0
    effective_order: bool = True
    name: str = "sentence_bleu"

    def __post_init__(self) -> None:
        if isinstance(self.max_order, bool) or not isinstance(self.max_order, int):
            raise TypeError("BLEU max_order must be an integer")
        if not 1 <= self.max_order <= 8:
            raise ValueError("BLEU max_order must be between 1 and 8")
        if isinstance(self.smooth, bool) or not isinstance(self.smooth, (int, float)):
            raise TypeError("BLEU smooth must be a real number")
        if not math.isfinite(self.smooth) or self.smooth < 0:
            raise ValueError("BLEU smooth must be finite and non-negative")
        if type(self.effective_order) is not bool:
            raise TypeError("BLEU effective_order must be a boolean")

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = _strict_tokens(reference, self.normalizer)
        actual = _strict_tokens(prediction, self.normalizer)
        if not expected and not actual:
            return MetricValue(1.0, details={"orders": 0, "brevity_penalty": 1.0})
        if not actual or not expected:
            return MetricValue(0.0, details={"orders": 0, "brevity_penalty": 0.0})

        precisions: list[float] = []
        matches_by_order: list[int] = []
        totals_by_order: list[int] = []
        for order in range(1, self.max_order + 1):
            possible = len(actual) - order + 1
            if possible <= 0:
                if self.effective_order:
                    break
                precisions.append(0.0)
                matches_by_order.append(0)
                totals_by_order.append(0)
                continue
            expected_ngrams = _ngrams(expected, order) if len(expected) >= order else Counter()
            actual_ngrams = _ngrams(actual, order)
            matches = sum((expected_ngrams & actual_ngrams).values())
            if matches == 0 and self.smooth == 0:
                precision = 0.0
            else:
                precision = (matches + self.smooth) / (possible + self.smooth)
            precisions.append(precision)
            matches_by_order.append(matches)
            totals_by_order.append(possible)

        if not precisions or any(value == 0 for value in precisions):
            geometric_mean = 0.0
        else:
            geometric_mean = math.exp(
                sum(math.log(value) for value in precisions) / len(precisions)
            )
        brevity_penalty = math.exp(min(0.0, 1.0 - len(expected) / len(actual)))
        return MetricValue(
            brevity_penalty * geometric_mean,
            details={
                "orders": len(precisions),
                "matches_by_order": matches_by_order,
                "totals_by_order": totals_by_order,
                "precisions": precisions,
                "brevity_penalty": brevity_penalty,
                "reference_tokens": len(expected),
                "prediction_tokens": len(actual),
            },
        )


def _levenshtein_distance(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


@dataclass(frozen=True, slots=True)
class LevenshteinSimilarity:
    """Normalized character edit similarity in ``[0, 1]``."""

    normalizer: TextNormalizer = field(default_factory=TextNormalizer)
    name: str = "levenshtein_similarity"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        expected = self.normalizer(reference)
        actual = self.normalizer(prediction)
        scale = max(len(expected), len(actual))
        if scale == 0:
            return MetricValue(1.0, details={"distance": 0, "scale": 0})
        distance = _levenshtein_distance(expected, actual)
        return MetricValue(1.0 - distance / scale, details={"distance": distance, "scale": scale})
