"""Immutable geometry, identity and Unicode document assembly contracts."""

from dataclasses import FrozenInstanceError, replace

import pytest

from metricguard import (
    OcrDocumentResult,
    OcrLine,
    OcrPageImage,
    OcrPageInfo,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
)

IDENTITY = OcrProviderIdentity("test", "1", "en", "a" * 64, "b" * 64)
INFO = OcrPageInfo(0, "c" * 64, "image", 20, 10, "d" * 64, "authored")


def test_geometry_snapshots_union_and_unicode_page_spans():
    box = [1, 2, 3, 4]
    word = OcrWord("é", box)
    box[0] = 99
    words = [word, OcrWord("🙂", (5, 3, 2, 3))]
    line = OcrLine("é 🙂", words)
    words.clear()
    assert line.box == (1, 2, 6, 4)
    first = OcrPageResult(INFO, IDENTITY, [line], 0.0)
    blank = OcrPageResult(replace(INFO, page_index=1), IDENTITY, [])
    document = OcrDocumentResult([first, blank])
    assert document.text == "é 🙂\n\f\n"
    assert document.to_dict()["page_spans"] == [
        {"page_index": 0, "start": 0, "end": 3},
        {"page_index": 1, "start": 6, "end": 6},
    ]
    assert first.to_dict()["lines"][0]["box"] == [1, 2, 6, 4]
    assert len(document.digest) == 64
    assert document.digest == OcrDocumentResult((first, blank)).digest
    assert document.digest != replace(document, separator="|").digest
    with pytest.raises(FrozenInstanceError):
        first.text_angle = 5


@pytest.mark.parametrize(
    "box", [(1, 2), (-1, 0, 1, 1), (0, 0, 0, 1), (True, 0, 1, 1), (0, 0, float("inf"), 1)]
)
def test_reject_invalid_geometry(box):
    with pytest.raises(ValueError, match="box"):
        OcrWord("x", box)


def test_invalid_text_and_structure():
    for text in (42, "\ud800", " "):
        with pytest.raises(ValueError):
            OcrWord(text, (0, 0, 1, 1))
    for words in ([], ["x"], None):
        with pytest.raises(ValueError):
            OcrLine("x", words)
    with pytest.raises(ValueError, match="provider"):
        replace(IDENTITY, provider="")
    with pytest.raises(ValueError, match="digest"):
        replace(IDENTITY, implementation_sha256="A" * 64)
    with pytest.raises(ValueError, match="raster source"):
        replace(INFO, source_type="url")
    with pytest.raises(ValueError, match="20 million"):
        replace(INFO, width=10000, height=3000)
    with pytest.raises(ValueError, match="RGB bytes"):
        OcrPageImage(0, "c" * 64, "image", 1, 1, b"\x00", "test")
    for width in (True, 1.0, -1, "1"):
        with pytest.raises(ValueError, match="width"):
            OcrPageImage(0, "c" * 64, "image", width, 1, b"\x00" * 3, "test")


def test_reject_mismatched_results_and_documents():
    valid = OcrPageResult(INFO, IDENTITY, ())
    for changed in ({"page": None}, {"provider": None}, {"lines": ["x"]}):
        with pytest.raises(ValueError):
            replace(valid, **changed)
    for angle in (True, float("nan"), 181):
        with pytest.raises(ValueError, match="angle"):
            replace(valid, text_angle=angle)
    with pytest.raises(ValueError, match="outside"):
        replace(valid, lines=(OcrLine("x", (OcrWord("x", (19, 0, 2, 2)),)),))
    for pages, message in [
        ((), "at least"),
        ((None,), "OcrPageResult"),
        ((valid, valid), "source order"),
        (
            (valid, replace(valid, page=replace(INFO, page_index=1, source_sha256="e" * 64))),
            "share",
        ),
    ]:
        with pytest.raises(ValueError, match=message):
            OcrDocumentResult(pages)
    with pytest.raises(ValueError, match="separator"):
        OcrDocumentResult((valid,), separator="x" * 101)


def test_result_resource_limits():
    word = OcrWord("x", (0, 0, 1, 1))
    line = OcrLine("x", (word,))
    with pytest.raises(ValueError, match="line/word"):
        OcrPageResult(INFO, IDENTITY, (line,) * 20_001)
    with pytest.raises(ValueError, match="line/word"):
        OcrPageResult(INFO, IDENTITY, (OcrLine("x", (word,) * 100_001),))
    with pytest.raises(ValueError, match="4 MiB"):
        OcrPageResult(INFO, IDENTITY, (OcrLine("x" * (4 * 1024 * 1024 + 1), (word,)),))
    # A blank line string cannot hide an arbitrarily large word payload.
    large_word = OcrWord("x" * (4 * 1024 * 1024 + 1), (0, 0, 1, 1))
    with pytest.raises(ValueError, match="4 MiB"):
        OcrPageResult(INFO, IDENTITY, (OcrLine("", (large_word,)),))
    with pytest.raises(ValueError, match="finite"):
        OcrWord("x", (10**400, 0, 1, 1))
    with pytest.raises(ValueError, match="angle"):
        OcrPageResult(INFO, IDENTITY, (), 10**400)
