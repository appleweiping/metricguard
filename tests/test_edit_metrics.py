import pytest

from metricguard import CharacterErrorRate, WordErrorRate, build_metric


def test_character_error_rate_reports_edit_distance() -> None:
    metric = CharacterErrorRate()
    value = metric.evaluate("kitten", "sitting")
    assert value.score == pytest.approx(3 / 6)
    assert value.details["distance"] == 3
    assert build_metric("character_error_rate").name == "character_error_rate"


def test_word_error_rate_normalizes_and_handles_empty_reference() -> None:
    metric = WordErrorRate()
    value = metric.evaluate("A small TEST.", "a small task")
    assert value.score == pytest.approx(1 / 3)
    assert value.details["reference_words"] == 3
    assert metric.evaluate("", "extra").score is None
    assert metric.evaluate("", "").score == 0
    assert (
        build_metric({"kind": "word_error_rate", "normalizer": {"lowercase": False}}).name
        == "word_error_rate"
    )
