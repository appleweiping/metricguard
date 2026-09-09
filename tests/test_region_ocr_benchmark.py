"""Independent fixture and negative-result checks; native quality is a separate opt-in run."""

import importlib.util
from pathlib import Path

import pytest

from metricguard.ocr_inputs import load_ocr_pages
from metricguard.ocr_types import OcrLine, OcrPageResult, OcrProviderIdentity, OcrWord

SCRIPT = Path(__file__).resolve().parents[1] / "benchmarks" / "verify_region_ocr.py"


@pytest.fixture(scope="module")
def benchmark():
    spec = importlib.util.spec_from_file_location("region_ocr_benchmark_fixture", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authored_layouts_have_expected_geometric_opportunities_and_negative_case(
    tmp_path, benchmark
):
    paths = benchmark.fixtures(tmp_path / "new")
    assert tuple(paths) == (
        "columns.png",
        "columns-and-blank.pdf",
        "bridged-columns.png",
        "single.png",
    )
    for name, (path, references) in paths.items():
        pages = list(load_ocr_pages(path))
        plans = [benchmark.layout_plan(page) for page in pages]
        assert len(pages) == len(references)
        assert all(plan.uncovered_pixels == 0 for plan in plans)
        if name == "columns.png":
            assert len(plans[0].regions) == 2 and plans[0].reason == "split"
            cut = plans[0].regions[0].box.right
            assert 400 < cut < 700
            assert plans[0].regions[1].box.left == cut
        elif name == "columns-and-blank.pdf":
            assert [len(plan.regions) for plan in plans] == [2, 1]
            assert plans[1].reason == "blank" and references[1] == ""
        else:
            assert len(plans[0].regions) == 1 and plans[0].reason == "no_split"
    with pytest.raises(FileExistsError):
        benchmark.fixtures(tmp_path / "new")


@pytest.mark.parametrize("hallucinate_blank", [False, True])
def test_fixture_benchmark_keeps_wrong_predictions_and_checks_actual_call_inventory(
    tmp_path, benchmark, monkeypatch, hallucinate_blank
):
    calls = []

    class DeliberatelyWrong:
        identity = OcrProviderIdentity("scripted-test", "1", "en-US", "a" * 64, "b" * 64)

        def __init__(self, language):
            assert language == "en-US"

        def recognize(self, page):
            assert set(page.__dataclass_fields__) == {
                "page_index",
                "source_sha256",
                "source_type",
                "width",
                "height",
                "pixels",
                "renderer",
            }
            calls.append(page.info)
            lines = (
                ()
                if not hallucinate_blank and page.pixels == b"\xff" * len(page.pixels)
                else (OcrLine("WRONG", (OcrWord("WRONG", (0, 0, 1, 1)),)),)
            )
            return OcrPageResult(page.info, self.identity, lines)

    monkeypatch.setattr(benchmark, "WindowsOcrBackend", DeliberatelyWrong)
    report = benchmark.verify(tmp_path / "new")
    assert len(calls) == 12
    assert report["provider"]["provider"] == "scripted-test"
    assert report["pdf_text_layer_characters"] == [0, 0]
    rows = report["page_evaluations"]
    assert len(rows) == 5
    assert sum(row["whole_recognition_calls"] for row in rows) == 5
    assert sum(row["region_recognition_calls"] for row in rows) == 7
    nonblank = [row for row in rows if row["reference"]]
    assert len(nonblank) == 4
    assert all(not row["region_scores"]["exact_normalized"] for row in nonblank)
    assert all(row["region_scores"]["cer"] > 0 for row in nonblank)
    negative = next(row for row in rows if row["fixture"] == "bridged-columns.png")
    assert negative["detector_reason"] == "no_split" and negative["regions"] == 1
    assert negative["whole_scores"] == negative["region_scores"]
    blank = next(row for row in rows if not row["reference"])
    assert blank["detector_reason"] == "blank"
    if hallucinate_blank:
        assert blank["region_text"] == "WRONG" and blank["status"] == "recognized"
        assert blank["region_scores"]["cer"] is None
        assert blank["region_scores"]["cer_undefined_reason"]
    else:
        assert blank["region_text"] == "" and blank["status"] == "recognized_empty"
    assert sum(item["pages"] for item in report["independent_checks"].values()) == 5
    assert sum(item["crops"] for item in report["independent_checks"].values()) == 7


def test_empty_reference_scores_are_explicit_and_no_quality_assertion_is_hidden(benchmark):
    blank = benchmark.scores("", "")
    assert blank["cer"] == blank["wer"] == 0
    hallucination = benchmark.scores("", "invented")
    assert hallucination["cer"] is None and hallucination["wer"] is None
    assert hallucination["cer_undefined_reason"]
    assert not hallucination["exact_raw"]


def test_counter_only_forwards_rgb_identity_to_backend(benchmark):
    class Spy:
        identity = OcrProviderIdentity("spy", "1", "en-US", "a" * 64, "b" * 64)

        def recognize(self, page):
            assert page == "supplied-page"
            return "received"

    counter = benchmark.CountingBackend(Spy())
    assert counter.identity == Spy.identity and counter.calls == 0
    assert counter.recognize("supplied-page") == "received"
    assert counter.calls == 1
