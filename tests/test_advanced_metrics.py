import math

import pytest

from metricguard.advanced_metrics import LevenshteinSimilarity, RougeL, SentenceBleu
from metricguard.metrics import build_metric
from metricguard.normalizers import TextNormalizer


def test_rouge_l_pairwise_score_and_details() -> None:
    value = RougeL().evaluate("A quick brown fox", "quick fox")
    assert value.score == pytest.approx(2 * 1.0 * 0.5 / 1.5)
    assert value.details == {
        "lcs": 2,
        "precision": 1.0,
        "recall": 0.5,
        "reference_tokens": 4,
        "prediction_tokens": 2,
    }


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [("", "", 1.0), ("", "word", 0.0), ("word", "", 0.0), ("a", "b", 0.0)],
)
def test_rouge_l_empty_and_disjoint(reference: str, prediction: str, expected: float) -> None:
    assert RougeL().evaluate(reference, prediction).score == expected


def test_rouge_l_beta_and_normalization_are_explicit() -> None:
    metric = RougeL(normalizer=TextNormalizer(lowercase=False, strip_punctuation=False), beta=2.0)
    value = metric.evaluate("a b c", "a c")
    assert value.score == pytest.approx(5 * 1 * (2 / 3) / ((2 / 3) + 4))


def test_sentence_bleu_perfect_match_and_brevity_penalty() -> None:
    metric = SentenceBleu(max_order=4, smooth=1.0)
    assert metric.evaluate("the cat sat", "the cat sat").score == 1.0
    short = metric.evaluate("the cat sat on mat", "the cat")
    assert short.score == pytest.approx(math.exp(1 - 5 / 2))
    assert short.details["orders"] == 2


def test_sentence_bleu_smoothing_and_effective_order() -> None:
    assert SentenceBleu(max_order=1, smooth=1).evaluate("a", "b").score == 0.5
    assert SentenceBleu(max_order=1, smooth=0).evaluate("a", "b").score == 0.0
    assert (
        SentenceBleu(max_order=4, smooth=1, effective_order=False).evaluate("a", "a").score == 0.0
    )


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [("", "", 1.0), ("", "a", 0.0), ("a", "", 0.0)],
)
def test_sentence_bleu_empty_cases(reference: str, prediction: str, expected: float) -> None:
    assert SentenceBleu().evaluate(reference, prediction).score == expected


def test_levenshtein_similarity() -> None:
    value = LevenshteinSimilarity().evaluate("kitten", "sitting")
    assert value.score == pytest.approx(4 / 7)
    assert value.details == {"distance": 3, "scale": 7}
    assert LevenshteinSimilarity().evaluate("", "").score == 1.0


@pytest.mark.parametrize("metric", [RougeL(), SentenceBleu(), LevenshteinSimilarity()])
def test_text_metrics_do_not_coerce_non_strings(metric: object) -> None:
    with pytest.raises(TypeError, match="string"):
        metric.evaluate(1, "1")  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("config", "metric_type"),
    [
        ({"kind": "rouge_l", "beta": 2}, RougeL),
        (
            {"kind": "sentence_bleu", "max_order": 2, "smooth": 0.5},
            SentenceBleu,
        ),
        ({"kind": "levenshtein_similarity"}, LevenshteinSimilarity),
    ],
)
def test_advanced_metrics_are_available_from_factory(
    config: dict[str, object], metric_type: type[object]
) -> None:
    assert isinstance(build_metric(config), metric_type)


@pytest.mark.parametrize(
    "config",
    [
        {"kind": "rouge_l", "beta": True},
        {"kind": "rouge_l", "beta": 0},
        {"kind": "sentence_bleu", "max_order": True},
        {"kind": "sentence_bleu", "smooth": "1"},
        {"kind": "sentence_bleu", "effective_order": 1},
        {"kind": "levenshtein_similarity", "unknown": 1},
    ],
)
def test_advanced_factory_rejects_invalid_options(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_metric(config)


@pytest.mark.parametrize(
    "constructor",
    [
        lambda: RougeL(beta=float("nan")),
        lambda: RougeL(beta=-1),
        lambda: SentenceBleu(max_order=0),
        lambda: SentenceBleu(max_order=9),
        lambda: SentenceBleu(smooth=float("inf")),
        lambda: SentenceBleu(smooth=-1),
    ],
)
def test_advanced_metric_validation(constructor: object) -> None:
    with pytest.raises(ValueError):
        constructor()  # type: ignore[operator]
