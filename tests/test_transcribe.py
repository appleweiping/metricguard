"""Document transaction boundary, gold-free provider contract and safe publication."""

import json
import os
from dataclasses import replace

import pytest
from PIL import Image

from metricguard import (
    OcrLine,
    OcrPageResult,
    OcrProviderIdentity,
    OcrWord,
    ocr_cli,
    ocr_pipeline,
    transcribe_document,
)
from metricguard.cli import main

IDENTITY = OcrProviderIdentity("authored-test", "1", "en", "a" * 64, "b" * 64)


class AuthoredBackend:
    def __init__(self):
        self.identity = IDENTITY
        self.calls = []

    def recognize(self, page):
        self.calls.append(page.info)
        text = "café 🙂" if page.page_index == 0 else ""
        lines = (OcrLine(text, (OcrWord(text, (0, 0, 1, 1)),)),) if text else ()
        return OcrPageResult(page.info, self.identity, lines)


@pytest.fixture
def pages(tmp_path):
    path = tmp_path / "two-pages.tiff"
    with Image.new("RGB", (4, 3), "red") as first, Image.new("RGB", (4, 3), "white") as second:
        first.save(path, save_all=True, append_images=[second])
    return path


def test_ordered_assembly_and_no_raster_retention(pages):
    backend = AuthoredBackend()
    document = transcribe_document(pages, backend)
    assert document.text == "café 🙂\n\f\n"
    assert [info.page_index for info in backend.calls] == [0, 1]
    assert document.to_dict()["page_spans"][1] == {"page_index": 1, "start": 9, "end": 9}
    assert "pixels" not in json.dumps(document.to_dict())
    assert "reference" not in json.dumps(document.to_dict())


def test_failure_closes_input_without_returning_partial_document(pages, monkeypatch):
    real_loader = ocr_pipeline.load_ocr_pages
    closed = []

    def traced_loader(*args, **kwargs):
        try:
            yield from real_loader(*args, **kwargs)
        finally:
            closed.append(True)

    monkeypatch.setattr(ocr_pipeline, "load_ocr_pages", traced_loader)
    backend = AuthoredBackend()
    recognize = backend.recognize

    def fail_second(page):
        if page.page_index == 1:
            raise ValueError("authored second-page failure")
        return recognize(page)

    backend.recognize = fail_second
    with pytest.raises(ValueError, match="second-page failure"):
        transcribe_document(pages, backend)
    assert len(backend.calls) == 1
    assert closed == [True]


def test_pin_identity_before_and_after_calls(pages):
    class ChangingBackend(AuthoredBackend):
        def recognize(self, page):
            result = super().recognize(page)
            self.identity = replace(IDENTITY, version="2")
            return result

    with pytest.raises(ValueError, match="during recognition"):
        transcribe_document(pages, ChangingBackend())

    class BeforeCall:
        accesses = 0

        @property
        def identity(self):
            self.accesses += 1
            return IDENTITY if self.accesses == 1 else replace(IDENTITY, version="2")

        def recognize(self, _page):
            pytest.fail("must not run changed provider")

    with pytest.raises(ValueError, match="before recognition"):
        transcribe_document(pages, BeforeCall())
    backend = AuthoredBackend()
    backend.identity = "name-only"
    with pytest.raises(ValueError, match="immutable provider identity"):
        transcribe_document(pages, backend)


def test_mismatched_page_provider_and_result_rejected(pages):
    backend = AuthoredBackend()
    for result in [None, OcrPageResult.__name__]:
        backend.recognize = lambda _page, value=result: value
        with pytest.raises(ValueError, match="different page"):
            transcribe_document(pages, backend)

    def wrong_source(page):
        return OcrPageResult(replace(page.info, source_sha256="c" * 64), IDENTITY, ())

    backend.recognize = wrong_source
    with pytest.raises(ValueError, match="different page"):
        transcribe_document(pages, backend)
    backend.recognize = lambda page: OcrPageResult(page.info, replace(IDENTITY, version="2"), ())
    with pytest.raises(ValueError, match="identity changed"):
        transcribe_document(pages, backend)


def test_document_text_and_word_budgets_include_all_pages(pages):
    assert (
        transcribe_document(pages, AuthoredBackend(), max_document_text_bytes=23).text
        == "café 🙂\n\f\n"
    )
    with pytest.raises(ValueError, match="text byte limit"):
        transcribe_document(pages, AuthoredBackend(), max_document_text_bytes=22)
    for limit in (True, 0, 64 * 1024 * 1024 + 1):
        with pytest.raises(ValueError, match="document text limit"):
            transcribe_document(pages, AuthoredBackend(), max_document_text_bytes=limit)
    backend = AuthoredBackend()
    backend.recognize = lambda page: OcrPageResult(
        page.info, IDENTITY, (OcrLine("", (OcrWord("hidden payload", (0, 0, 1, 1)),)),)
    )
    with pytest.raises(ValueError, match="text byte limit"):
        transcribe_document(pages, backend, max_document_text_bytes=1)
    words = (OcrWord("x", (0, 0, 1, 1)),) * 50_001
    backend.recognize = lambda page: OcrPageResult(page.info, IDENTITY, (OcrLine("x", words),))
    with pytest.raises(ValueError, match="100,000 words"):
        transcribe_document(pages, backend)


def test_cli_complete_stdout_and_atomic_output(pages, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(ocr_cli, "WindowsOcrBackend", lambda *_args, **_kwargs: AuthoredBackend())
    assert main(["transcribe", str(pages)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "complete"
    assert result["text"] == "café 🙂\n\f\n"
    target = tmp_path / "reports" / "ocr.json"
    assert main(["transcribe", str(pages), "--output", str(target)]) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == result
    assert list(target.parent.iterdir()) == [target]


def test_cli_aliases_rejected_before_provider_execution(pages, monkeypatch, tmp_path, capsys):
    def must_not_run(*_args, **_kwargs):
        pytest.fail("alias rejection must happen before native initialization")

    monkeypatch.setattr(ocr_cli, "WindowsOcrBackend", must_not_run)
    alias = tmp_path / "hardlink.tiff"
    os.link(pages, alias)
    original = pages.read_bytes()
    for output in (pages, alias):
        assert main(["transcribe", str(pages), "--output", str(output)]) == 2
        assert "alias" in capsys.readouterr().err
    assert pages.read_bytes() == original
    assert main(["transcribe", str(pages), "--output", str(tmp_path)]) == 2
    assert "must be a file" in capsys.readouterr().err


def test_cli_failure_preserves_existing_report_and_cleans_temp(
    pages, monkeypatch, tmp_path, capsys
):
    target = tmp_path / "result.json"
    target.write_text("previous report", encoding="utf-8")
    monkeypatch.setattr(ocr_cli, "WindowsOcrBackend", lambda *_args, **_kwargs: AuthoredBackend())
    assert main(["transcribe", str(pages), "--max-pages", "1", "--output", str(target)]) == 2
    assert "frame count" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "previous report"

    def denied(*_args):
        raise PermissionError("test publication failure")

    monkeypatch.setattr(ocr_cli.os, "replace", denied)
    assert main(["transcribe", str(pages), "--output", str(target)]) == 2
    assert "publication failure" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "previous report"
    assert not list(tmp_path.glob(".result.json.*.tmp"))
