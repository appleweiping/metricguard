from decimal import Decimal

import pytest

from metricguard.metrics import (
    CharacterF1,
    ExactMatch,
    NumericEquivalence,
    TokenF1,
    build_metric,
)
from metricguard.normalizers import TextNormalizer


def test_exact_match_respects_explicit_normalizer() -> None:
    metric = ExactMatch(TextNormalizer(lowercase=True, strip_punctuation=True))
    value = metric.evaluate("Hello, world!", "hello world")
    assert value.score == 1.0
    assert value.details["prediction"] == "hello world"


def test_token_f1_counts_duplicate_tokens() -> None:
    value = TokenF1().evaluate("red red blue", "red blue blue")
    assert value.score == pytest.approx(2 / 3)
    assert value.details["overlap"] == 2
    assert value.details["precision"] == pytest.approx(2 / 3)


@pytest.mark.parametrize(
    ("reference", "prediction", "expected"),
    [("", "", 1.0), ("", "answer", 0.0), ("a", "", 0.0), ("a", "b", 0.0)],
)
def test_token_f1_empty_and_disjoint_cases(
    reference: str, prediction: str, expected: float
) -> None:
    assert TokenF1().evaluate(reference, prediction).score == expected


def test_character_f1() -> None:
    assert CharacterF1().evaluate("book", "boot").score == pytest.approx(0.75)
    assert CharacterF1().evaluate("", "").score == 1.0
    assert CharacterF1().evaluate("x", "").score == 0.0
    assert CharacterF1().evaluate("x", "y").score == 0.0


def test_numeric_equivalence_exact_and_tolerant() -> None:
    exact = NumericEquivalence()
    tolerant = NumericEquivalence(absolute_tolerance=Decimal("0.01"))
    assert exact.evaluate("42", 42).score == 1.0
    assert exact.evaluate("42", "42.001").score == 0.0
    value = tolerant.evaluate("42", "42.001")
    assert value.score == 1.0
    assert value.details["difference"] == "0.001"


def test_numeric_relative_tolerance() -> None:
    metric = NumericEquivalence(relative_tolerance=Decimal("0.01"))
    assert metric.evaluate("100", "100.5").score == 1.0
    assert metric.evaluate("100", "102").score == 0.0


def test_numeric_percent_and_commas_are_opt_in() -> None:
    strict = NumericEquivalence()
    configured = NumericEquivalence(allow_percent=True, allow_commas=True)
    assert strict.evaluate("1,000", "1000").score is None
    assert configured.evaluate("1,000", 1000).score == 1.0
    assert configured.evaluate("25%", "0.25").score == 1.0


@pytest.mark.parametrize("bad", ["1,2,3", ",1", "1,,2", "1234,567", "1,23.00"])
def test_numeric_commas_require_standard_thousands_groups(bad: str) -> None:
    assert NumericEquivalence(allow_commas=True).evaluate(bad, "123").score is None


@pytest.mark.parametrize("valid", ["1,234", "+12,345.67", "-1,000e2"])
def test_numeric_valid_thousands_groups(valid: str) -> None:
    plain = valid.replace(",", "")
    assert NumericEquivalence(allow_commas=True).evaluate(valid, plain).score == 1.0


@pytest.mark.parametrize("bad", [True, float("nan"), float("inf"), "NaN", object()])
def test_numeric_invalid_values_are_undefined(bad: object) -> None:
    value = NumericEquivalence().evaluate(bad, "1")
    assert value.score is None
    assert "reference" in (value.reason or "")


def test_numeric_invalid_both_names_both_sides() -> None:
    value = NumericEquivalence().evaluate("nope", "still nope")
    assert value.reason == "unparseable reference and prediction"


def test_negative_tolerances_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        NumericEquivalence(absolute_tolerance=Decimal("-1"))


@pytest.mark.parametrize("value", [Decimal("NaN"), Decimal("Infinity"), Decimal("-Infinity")])
def test_non_finite_tolerances_rejected(value: Decimal) -> None:
    with pytest.raises(ValueError, match="finite"):
        NumericEquivalence(absolute_tolerance=value)


def test_direct_numeric_configuration_types_are_strict() -> None:
    with pytest.raises(TypeError, match="Decimal"):
        NumericEquivalence(absolute_tolerance=0.1)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="booleans"):
        NumericEquivalence(allow_percent=1)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config", "expected_type"),
    [
        ("exact_match", ExactMatch),
        ({"kind": "token_f1", "normalizer": {"lowercase": True}}, TokenF1),
        ({"kind": "character_f1"}, CharacterF1),
        ({"kind": "numeric_equivalence", "absolute_tolerance": 0.1}, NumericEquivalence),
    ],
)
def test_metric_factory(config: object, expected_type: type[object]) -> None:
    assert isinstance(build_metric(config), expected_type)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({}, "requires"),
        ({"kind": "unknown"}, "unknown metric"),
        ({"kind": "exact_match", "mystery": True}, "unknown exact_match"),
        ({"kind": "token_f1", "normalizer": {"mystery": True}}, "unknown normalizer"),
        ([], "must be a string or object"),
    ],
)
def test_metric_factory_rejects_bad_configuration(config: object, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_metric(config)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "config",
    [
        {"kind": "numeric_equivalence", "allow_percent": "false"},
        {"kind": "numeric_equivalence", "allow_commas": 1},
        {"kind": "exact_match", "normalizer": {"lowercase": "false"}},
        {"kind": "exact_match", "normalizer": {"unicode_form": 1}},
    ],
)
def test_metric_factory_rejects_non_strict_option_types(config: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        build_metric(config)


@pytest.mark.parametrize("value", ["NaN", "Infinity", "-Infinity", "invalid", None, True])
def test_metric_factory_rejects_invalid_tolerance(value: object) -> None:
    with pytest.raises(ValueError, match="finite decimal"):
        build_metric({"kind": "numeric_equivalence", "absolute_tolerance": value})
