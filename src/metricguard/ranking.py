"""Per-query retrieval metrics with explicit ranking and relevance semantics."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .models import MetricValue

RANKING_METRICS = (
    "average_precision_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank_at_k",
)


@dataclass(frozen=True)
class RankingMetric:
    """Evaluate an ordered unique document-ID list against relevance judgments.

    References map IDs to finite nonnegative relevance. Missing judgments mean
    zero relevance, not an error. Predictions are already ranked; score-tie
    handling belongs to the caller. All metrics return undefined when no relevant
    documents exist so EvaluationSuite's explicit undefined policy applies.

    NDCG uses *linear* gains with log2(rank + 1) discounts, not exponential gains.
    Binary metrics treat positive relevance as relevant. Precision divides by k
    even for short runs. Average precision divides by min(k, relevant documents).
    Reciprocal rank returns the inverse rank of the first relevant hit up to k.
    """

    name: str = "ndcg_at_k"
    k: int = 10

    def __post_init__(self) -> None:
        if self.name not in RANKING_METRICS:
            raise ValueError(f"unknown ranking metric: {self.name}")
        if isinstance(self.k, bool) or not isinstance(self.k, int) or self.k < 1:
            raise ValueError("k must be a positive integer")

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        """Evaluate one query; invalid judgments or duplicate predictions raise."""
        judgments = _judgments(reference)
        ranking = _ranking(prediction)
        relevant = sum(value > 0 for value in judgments.values())
        details: dict[str, Any] = {
            "k": self.k,
            "relevant": relevant,
            "retrieved": min(self.k, len(ranking)),
            "unjudged": sum(id not in judgments for id in ranking[: self.k]),
        }
        if not relevant:
            return MetricValue(None, reason="no positively judged documents", details=details)
        gains = [judgments.get(id, 0.0) for id in ranking[: self.k]]
        hits = sum(value > 0 for value in gains)
        details["hits"] = hits
        if self.name == "precision_at_k":
            score = hits / self.k
        elif self.name == "recall_at_k":
            score = hits / relevant
        elif self.name == "reciprocal_rank_at_k":
            score = next((1 / rank for rank, gain in enumerate(gains, 1) if gain > 0), 0.0)
        elif self.name == "average_precision_at_k":
            seen = 0
            precisions: list[float] = []
            for rank, gain in enumerate(gains, 1):
                if gain > 0:
                    seen += 1
                    precisions.append(seen / rank)
            score = math.fsum(precisions) / min(self.k, relevant)
        else:
            # Common scaling cancels in the ratio and avoids overflowing sums
            # for large but valid relevance values.
            scale = max(judgments.values())
            ideal = sorted(judgments.values(), reverse=True)[: self.k]
            numerator = math.fsum(
                (gain / scale) / math.log2(rank + 1) for rank, gain in enumerate(gains, 1)
            )
            denominator = math.fsum(
                (gain / scale) / math.log2(rank + 1) for rank, gain in enumerate(ideal, 1)
            )
            score = min(1.0, numerator / denominator)
            details["gain"] = "linear"
        return MetricValue(score, details=details)


def _judgments(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise TypeError("ranking reference must map document IDs to relevance values")
    result: dict[str, float] = {}
    for id, relevance in value.items():
        if not isinstance(id, str) or not id.strip():
            raise ValueError("judgment IDs must be nonempty strings")
        if isinstance(relevance, bool) or not isinstance(relevance, (int, float)):
            raise ValueError("relevance must be finite nonnegative numbers")
        try:
            numeric = float(relevance)
        except OverflowError as error:
            raise ValueError("relevance exceeds supported numeric range") from error
        if not math.isfinite(numeric) or numeric < 0:
            raise ValueError("relevance must be finite nonnegative numbers")
        result[id] = numeric
    return result


def _ranking(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError("ranking prediction must be an ordered list of document IDs")
    seen: set[str] = set()
    result: list[str] = []
    for id in value:
        if not isinstance(id, str) or not id.strip():
            raise ValueError("predicted IDs must be nonempty strings")
        if id in seen:
            raise ValueError(f"duplicate predicted document: {id!r}")
        seen.add(id)
        result.append(id)
    return tuple(result)
