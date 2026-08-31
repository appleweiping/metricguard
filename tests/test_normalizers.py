import pytest

from metricguard.normalizers import TextNormalizer


def test_default_normalizer_is_conservative() -> None:
    normalizer = TextNormalizer()
    assert normalizer("  Café\n  Noir!  ") == "Café Noir!"


def test_configurable_normalization_and_tokens() -> None:
    normalizer = TextNormalizer(lowercase=True, strip_punctuation=True)
    assert normalizer.tokenize(" Hello,  WORLD! ") == ("hello", "world")
    assert normalizer.to_dict()["lowercase"] is True


def test_unicode_equivalence() -> None:
    composed = "é"
    decomposed = "e\u0301"
    assert TextNormalizer(unicode_form="NFC")(composed) == TextNormalizer(unicode_form="NFC")(
        decomposed
    )


def test_empty_tokenization() -> None:
    assert TextNormalizer().tokenize(" \n\t ") == ()


def test_invalid_unicode_form() -> None:
    with pytest.raises(ValueError, match="unsupported Unicode"):
        TextNormalizer(unicode_form="ASCII")


@pytest.mark.parametrize("field", ["lowercase", "collapse_whitespace", "strip_punctuation"])
def test_boolean_options_are_strict(field: str) -> None:
    with pytest.raises(TypeError, match="boolean"):
        TextNormalizer(**{field: "false"})  # type: ignore[arg-type]


def test_non_string_values_are_not_implicitly_coerced() -> None:
    with pytest.raises(TypeError, match="string"):
        TextNormalizer()(None)
