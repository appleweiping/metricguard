from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from metricguard import (
    OcrDocumentPage,
    load_ocr_document_cases,
    run_document_ocr_benchmark,
)
from metricguard.cli import main


def test_document_benchmark_orders_pages_and_reports_each_document() -> None:
    pages = (
        OcrDocumentPage("d2-p0", "doc-2", 0, "second"),
        OcrDocumentPage("d1-p1", "doc-1", 1, "world"),
        OcrDocumentPage("d1-p0", "doc-1", 0, "hello"),
    )
    seen: list[str] = []

    def backend(page: OcrDocumentPage) -> str:
        seen.append(page.page_id)
        return page.reference

    report = run_document_ocr_benchmark(pages, backend)
    assert seen == ["d1-p0", "d1-p1", "d2-p0"]
    assert report.pages == 3
    assert report.document_count == 2
    assert [item.document_id for item in report.documents] == ["doc-1", "doc-2"]
    assert report.documents[0].report.exact_match_rate == 1.0
    assert report.to_dict()["page_report"]["cases"] == 3


def test_document_benchmark_rejects_bad_backend_and_duplicate_positions() -> None:
    page = OcrDocumentPage("p", "doc", 0, "text")
    with pytest.raises(TypeError, match="callable"):
        run_document_ocr_benchmark((page,), None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="at least one"):
        run_document_ocr_benchmark((), lambda _: "text")
    with pytest.raises(TypeError, match="return text"):
        run_document_ocr_benchmark((page,), lambda _: 1)  # type: ignore[return-value]
    with pytest.raises(ValueError, match="IDs"):
        run_document_ocr_benchmark((page, OcrDocumentPage("p", "doc", 1, "text")), lambda _: "text")
    with pytest.raises(ValueError, match="positions"):
        run_document_ocr_benchmark(
            (page, OcrDocumentPage("other", "doc", 0, "text")), lambda _: "text"
        )


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{}", "array"),
        ("[]", "no pages"),
        ('[{"id":"p","document_id":"d","page":true,"reference":"x"}]', "non-negative"),
        ('[{"id":"p","document_id":"d","page":0}]', "missing"),
        ('[{"id":"p","document_id":"d","page":0,"reference":"x","other":1}]', "unknown"),
    ],
)
def test_document_case_loader_rejects_invalid_manifests(
    tmp_path: Path, content: str, message: str
) -> None:
    path = tmp_path / "pages.json"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_ocr_document_cases(path)


def test_document_case_loader_resolves_images_and_rejects_duplicates(tmp_path: Path) -> None:
    image = tmp_path / "page.txt"
    image.write_text("hello", encoding="utf-8")
    path = tmp_path / "pages.jsonl"
    path.write_text(
        '{"id":"p1","document_id":"d","page":1,"reference":"world","image":"page.txt"}\n'
        '{"id":"p0","document_id":"d","page":0,"reference":"hello"}\n',
        encoding="utf-8",
    )
    pages = load_ocr_document_cases(path)
    assert pages[0].image == image
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        json.dumps(
            [
                {"id": "p", "document_id": "d", "page": 0, "reference": "x"},
                {"id": "p", "document_id": "d", "page": 1, "reference": "y"},
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicates id"):
        load_ocr_document_cases(duplicate)


def test_document_cli_runs_shell_free_backend(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    image = tmp_path / "page.txt"
    image.write_text("hello", encoding="utf-8")
    cases = tmp_path / "pages.json"
    cases.write_text(
        json.dumps(
            [
                {
                    "id": "p",
                    "document_id": "doc",
                    "page": 0,
                    "reference": "hello",
                    "image": "page.txt",
                }
            ]
        ),
        encoding="utf-8",
    )
    assert (
        main(
            [
                "ocr-document",
                str(cases),
                "--command",
                sys.executable,
                "--command=-c",
                "--command",
                "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
                "--command",
                "{image}",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["documents"] == 1 and payload["pages"] == 1
