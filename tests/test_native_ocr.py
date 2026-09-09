"""Portable protocol/transport checks plus opt-in execution of the real recognizer."""

import copy
import json
import os
import sys
from dataclasses import replace

import pytest
from PIL import Image, ImageDraw, ImageFont

from metricguard import OcrPageImage, WindowsOcrBackend, transcribe_document
from metricguard import native_ocr as native

CAPABILITIES = {"os_version": "test-os", "max_dimension": 10000, "languages": ["en-US"]}
PROTOCOL = "metricguard.windows-ocr.v1"


@pytest.fixture
def fake_native(monkeypatch, tmp_path):
    helper = tmp_path / "fixed.ps1"
    helper.write_text("fixed helper", encoding="utf-8")
    monkeypatch.setattr(native, "_command", lambda: ("interpreter", str(helper)))
    response = {
        "protocol": PROTOCOL,
        "action": "recognize",
        "identity": copy.deepcopy(CAPABILITIES),
        "language": "en-US",
        "text_angle": None,
        "lines": [{"text": "hello", "words": [{"text": "hello", "box": [1, 2, 3, 4]}]}],
    }
    requests = []

    def exchange(_command, request, _timeout, _limit):
        requests.append(request)
        if request["action"] == "capabilities":
            return {"protocol": PROTOCOL, "action": "capabilities", "identity": CAPABILITIES}
        return response

    monkeypatch.setattr(native, "_exchange", exchange)
    return helper, response, requests


def raster():
    return OcrPageImage(0, "a" * 64, "image", 20, 10, b"\xff" * 600, "test-raster")


def test_native_protocol_geometry_identity_and_gold_free_request(fake_native):
    helper, _response, requests = fake_native
    backend = WindowsOcrBackend()
    initial = backend.identity
    result = backend.recognize(raster())
    assert result.text == "hello"
    assert result.page == raster().info
    assert result.lines[0].box == (1, 2, 3, 4)
    assert result.provider == initial
    assert backend.available_languages == ("en-US",)
    assert set(requests[-1]) == {"action", "language", "width", "height", "png"}
    assert requests[-1]["png"].startswith("iVBOR")
    assert WindowsOcrBackend(timeout=2).identity != initial
    helper.write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="helper changed"):
        backend.recognize(raster())


@pytest.mark.parametrize(
    "mutation,message",
    [
        ({"unexpected": 1}, "invalid fields"),
        ({"language": "en-GB"}, "identity changed"),
        ({"identity": {**CAPABILITIES, "os_version": "changed"}}, "identity changed"),
        ({"lines": {}}, "bounded array"),
        ({"lines": [None]}, "line has invalid"),
        ({"lines": [{"text": "x", "words": {}}]}, "line has invalid"),
        ({"lines": [{"text": "x", "words": [None]}]}, "word has invalid"),
        ({"lines": [{"text": "x", "words": [{"text": "x", "box": [0, 0, 21, 1]}]}]}, "outside"),
        ({"text_angle": float("inf")}, "angle"),
    ],
)
def test_reject_invalid_native_results(fake_native, mutation, message):
    _helper, response, _requests = fake_native
    response.update(mutation)
    with pytest.raises(ValueError, match=message):
        WindowsOcrBackend().recognize(raster())


def test_native_no_words_blank_and_count_limit(fake_native):
    _helper, response, _requests = fake_native
    backend = WindowsOcrBackend()
    response["lines"] = []
    assert backend.recognize(raster()).text == ""
    response["lines"] = [{"text": "x", "words": [None] * 100_001}]
    with pytest.raises(ValueError, match="word count"):
        backend.recognize(raster())
    with pytest.raises(ValueError, match="validated raster"):
        backend.recognize("not a raster")
    backend._capabilities["max_dimension"] = 1
    with pytest.raises(ValueError, match="dimension limit"):
        backend.recognize(raster())


@pytest.mark.parametrize(
    "kwargs",
    [
        {"timeout": True},
        {"timeout": 0},
        {"timeout": float("nan")},
        {"max_output_bytes": 0},
        {"max_output_bytes": True},
        {"language": ""},
    ],
)
def test_invalid_native_configuration_before_process(kwargs):
    with pytest.raises(ValueError):
        WindowsOcrBackend(**kwargs)


def test_unavailable_language_and_invalid_capability_fields(fake_native, monkeypatch):
    with pytest.raises(ValueError, match="not installed"):
        WindowsOcrBackend("not-installed")
    for capability in [
        None,
        {},
        {**CAPABILITIES, "languages": ["en-US", "en-US"]},
        {**CAPABILITIES, "max_dimension": True},
    ]:
        with pytest.raises(ValueError, match="capabilities"):
            native._capabilities(capability)
    monkeypatch.setattr(native, "_exchange", lambda *_args: {"unexpected": 1})
    with pytest.raises(ValueError, match="capabilities response"):
        WindowsOcrBackend()


def test_unavailable_platform_and_missing_powershell(monkeypatch, tmp_path):
    monkeypatch.setattr(native.sys, "platform", "linux")
    with pytest.raises(ValueError, match="unavailable on this platform"):
        native._command()
    monkeypatch.setattr(native.sys, "platform", "win32")
    monkeypatch.setenv("SYSTEMROOT", str(tmp_path))
    with pytest.raises(ValueError, match="PowerShell"):
        native._command()


@pytest.mark.parametrize(
    "payload,message",
    [
        (b"\xff", "invalid JSON"),
        (b'{"x":1,"x":2}', "invalid JSON"),
        (b'{"x":NaN}', "invalid JSON"),
        (b"[]", "protocol mismatch"),
        (b'{"protocol":"wrong","action":"capabilities"}', "protocol mismatch"),
    ],
)
def test_exchange_strict_json(monkeypatch, payload, message):
    monkeypatch.setattr(native, "_bounded_process", lambda *_args, **_kwargs: payload)
    with pytest.raises(ValueError, match=message):
        native._exchange(("fixed",), {"action": "capabilities"}, 1, 1024)


def test_exchange_accepts_only_matching_action(monkeypatch):
    payload = {"protocol": PROTOCOL, "action": "capabilities", "identity": CAPABILITIES}
    monkeypatch.setattr(
        native, "_bounded_process", lambda *_args, **_kwargs: json.dumps(payload).encode()
    )
    assert native._exchange(("fixed",), {"action": "capabilities"}, 1, 1024) == payload


def test_bounded_process_transports_large_input_without_shell_interpolation():
    data = b"$(); && \x00 input\n" * 10000
    result = native._bounded_process(
        (sys.executable, "-c", "import sys; sys.stdout.buffer.write(sys.stdin.buffer.read())"),
        data,
        timeout=10,
        limit=len(data),
    )
    assert result == data


@pytest.mark.parametrize("pipe", ["stdout", "stderr"])
def test_bounded_process_stops_output_flood(pipe):
    with pytest.raises(ValueError, match="byte limit"):
        native._bounded_process(
            (sys.executable, "-c", f"import sys; sys.{pipe}.buffer.write(b'x'*1000000)"),
            b"",
            timeout=10,
            limit=1024,
        )


def test_process_timeout_failure_and_no_secret_stderr(tmp_path):
    with pytest.raises(ValueError, match="timed out"):
        native._bounded_process(
            (sys.executable, "-c", "import time; time.sleep(5)"), b"", timeout=0.1, limit=10
        )
    with pytest.raises(ValueError, match="status 7") as error:
        native._bounded_process(
            (
                sys.executable,
                "-c",
                "import sys; sys.stderr.write('private-image-path'); sys.exit(7)",
            ),
            b"",
            timeout=10,
            limit=1024,
        )
    assert "private" not in str(error.value)
    with pytest.raises(ValueError, match="cannot start"):
        native._bounded_process((str(tmp_path / "missing-executable"),), b"", timeout=1, limit=10)
    with pytest.raises(ValueError, match="request exceeds"):
        native._bounded_process(
            ("not launched",), b"x" * (64 * 1024 * 1024 + 1), timeout=1, limit=10
        )


@pytest.mark.native_ocr
@pytest.mark.skipif(
    sys.platform != "win32" or os.environ.get("METRICGUARD_RUN_NATIVE_OCR") != "1",
    reason="explicit opt-in Windows recognizer integration; no network or model downloads",
)
def test_real_native_png_and_raster_only_two_page_pdf(tmp_path):
    import pypdfium2 as pdfium

    expected = ("METRIC GUARD 2026", "LOCAL OCR PAGE TWO")
    images = []
    for text in expected:
        image = Image.new("RGB", (1100, 220), "white")
        ImageDraw.Draw(image).text(
            (40, 60), text, fill="black", font=ImageFont.load_default(size=52)
        )
        images.append(image)
    try:
        png, pdf = tmp_path / "authored.png", tmp_path / "authored.pdf"
        images[0].save(png)
        images[0].save(pdf, save_all=True, append_images=images[1:], resolution=144)
    finally:
        for image in images:
            image.close()
    with pdfium.PdfDocument(pdf) as document:
        for index in range(len(document)):
            page = document[index]
            try:
                text_page = page.get_textpage()
                try:
                    assert text_page.get_text_range() == ""
                finally:
                    text_page.close()
            finally:
                page.close()
    backend = WindowsOcrBackend()
    png_result = transcribe_document(png, backend)
    pdf_result = transcribe_document(pdf, backend)
    assert png_result.text == expected[0]
    assert [page.text for page in pdf_result.pages] == list(expected)
    assert all(page.lines[0].words for page in pdf_result.pages)
    assert pdf_result.pages[0].provider == png_result.pages[0].provider
    assert replace(pdf_result.pages[0].page, page_index=0).source_type == "pdf"
