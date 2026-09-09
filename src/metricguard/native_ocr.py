"""Explicit Windows.Media.Ocr execution through a fixed, packaged local helper."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
import os
import subprocess  # nosec B404
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import Any, BinaryIO, cast

from .io import _strict_json_loads
from .ocr_inputs import _dependency
from .ocr_types import OcrLine, OcrPageImage, OcrPageResult, OcrProviderIdentity, OcrWord

_PROTOCOL = "metricguard.windows-ocr.v1"


def _bounded_process(argv: tuple[str, ...], data: bytes, *, timeout: float, limit: int) -> bytes:
    """Bound both output pipes while the fixed helper consumes a bounded request.

    The helper never launches children. This is not a sandbox for arbitrary
    commands or a process-tree supervisor for caller-supplied executables.
    """
    if len(data) > 64 * 1024 * 1024:
        raise ValueError("native OCR request exceeds 64 MiB")
    creation_flags = 0
    if sys.platform == "win32":
        creation_flags = subprocess.CREATE_NO_WINDOW
    try:
        process = subprocess.Popen(  # nosec B603
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=False,
            bufsize=0,
            creationflags=creation_flags,
        )
    except OSError as error:
        raise ValueError("cannot start native OCR helper") from error
    overflow = threading.Event()
    write_failed = threading.Event()
    output = bytearray()

    def drain(stream: BinaryIO, *, keep: bool) -> None:
        seen = 0
        try:
            while True:
                chunk = stream.read(min(65536, limit - seen + 1))
                if not chunk:
                    return
                seen += len(chunk)
                if seen > limit:
                    overflow.set()
                    with suppress(OSError):
                        process.kill()
                    return
                if keep:
                    output.extend(chunk)
        except OSError:
            write_failed.set()
        finally:
            stream.close()

    def feed() -> None:
        stream = cast(BinaryIO, process.stdin)
        try:
            view = memoryview(data)
            while view:
                written = stream.write(view[:65536])
                if not written:
                    raise OSError("closed helper input")
                view = view[written:]
        except OSError:
            write_failed.set()
        finally:
            stream.close()

    threads = (
        threading.Thread(
            target=drain, args=(cast(BinaryIO, process.stdout),), kwargs={"keep": True}, daemon=True
        ),
        threading.Thread(
            target=drain,
            args=(cast(BinaryIO, process.stderr),),
            kwargs={"keep": False},
            daemon=True,
        ),
        threading.Thread(target=feed, daemon=True),
    )
    deadline = time.monotonic() + timeout
    for thread in threads:
        thread.start()
    try:
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as error:
            raise ValueError("native OCR helper timed out") from error
    finally:
        if process.poll() is None:
            with suppress(OSError):
                process.kill()
            with suppress(subprocess.TimeoutExpired):
                process.wait(timeout=5)
        for thread in threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()) + 0.1)
    if overflow.is_set():
        raise ValueError("native OCR helper exceeded stdout/stderr byte limit")
    if any(thread.is_alive() for thread in threads):
        raise ValueError("native OCR helper pipes did not close before the deadline")
    if process.returncode != 0:
        raise ValueError(f"native OCR helper exited with status {process.returncode}")
    if write_failed.is_set():
        raise ValueError("native OCR helper transport failed")
    return bytes(output)


def _command() -> tuple[str, ...]:
    if sys.platform != "win32":
        raise ValueError("Windows native OCR is unavailable on this platform")
    executable = (
        Path(os.environ.get("SYSTEMROOT", r"C:\Windows"))
        / "System32/WindowsPowerShell/v1.0/powershell.exe"
    )
    if not executable.is_file():
        raise ValueError("Windows PowerShell 5.1 is required for native OCR")
    helper = Path(__file__).with_name("_windows_ocr.ps1")
    return (
        str(executable),
        "-NoProfile",
        "-NonInteractive",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(helper),
    )


def _exchange(
    command: tuple[str, ...], request: dict[str, Any], timeout: float, limit: int
) -> dict[str, Any]:
    data = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    raw = _bounded_process(command, data, timeout=timeout, limit=limit)
    try:
        result = _strict_json_loads(raw.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError) as error:
        raise ValueError("native OCR helper returned invalid JSON") from error
    if (
        not isinstance(result, dict)
        or result.get("protocol") != _PROTOCOL
        or result.get("action") != request["action"]
    ):
        raise ValueError("native OCR helper response protocol mismatch")
    return result


def _capabilities(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"os_version", "max_dimension", "languages"}:
        raise ValueError("native OCR capabilities have invalid fields")
    languages = value["languages"]
    if (
        not isinstance(value["os_version"], str)
        or not value["os_version"].strip()
        or type(value["max_dimension"]) is not int
        or not 1 <= value["max_dimension"] <= 100_000
        or not isinstance(languages, list)
        or len(languages) > 1000
        or any(not isinstance(tag, str) or not tag.strip() or len(tag) > 100 for tag in languages)
        or len(set(languages)) != len(languages)
    ):
        raise ValueError("native OCR capabilities contain invalid values")
    return {**value, "languages": sorted(languages)}


class WindowsOcrBackend:
    """Use already-installed Windows recognizer languages; never install/download them.

    Identity pins helper code, settings and OS-reported capabilities. Windows
    does not expose model-weight hashes: this is not byte-level model identity.
    """

    def __init__(
        self,
        language: str = "en-US",
        *,
        timeout: float = 30.0,
        max_output_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        if (
            type(timeout) not in (int, float)
            or not 0 < timeout <= 300
            or not math.isfinite(timeout)
        ):
            raise ValueError("native OCR timeout must be finite and in (0, 300]")
        if type(max_output_bytes) is not int or not 1 <= max_output_bytes <= 16 * 1024 * 1024:
            raise ValueError("native OCR output limit must be in [1, 16 MiB]")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("native OCR language must be an explicit language tag")
        self._command = _command()
        self._helper_hash = hashlib.sha256(Path(self._command[-1]).read_bytes()).hexdigest()
        self._timeout = float(timeout)
        self._limit = max_output_bytes
        self._language = language
        response = _exchange(self._command, {"action": "capabilities"}, self._timeout, self._limit)
        if set(response) != {"protocol", "action", "identity"}:
            raise ValueError("native OCR capabilities response has invalid fields")
        self._capabilities = _capabilities(response["identity"])
        if language not in self._capabilities["languages"]:
            raise ValueError("requested OCR language is not installed; no language was downloaded")

    @property
    def identity(self) -> OcrProviderIdentity:
        current_hash = hashlib.sha256(Path(self._command[-1]).read_bytes()).hexdigest()
        if current_hash != self._helper_hash:
            raise ValueError("native OCR helper changed after backend initialization")
        configuration = json.dumps(
            {
                "capabilities": self._capabilities,
                "language": self._language,
                "timeout": self._timeout,
                "max_output_bytes": self._limit,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return OcrProviderIdentity(
            "windows-media-ocr",
            self._capabilities["os_version"],
            self._language,
            self._helper_hash,
            hashlib.sha256(configuration.encode("utf-8")).hexdigest(),
        )

    @property
    def available_languages(self) -> tuple[str, ...]:
        return tuple(self._capabilities["languages"])

    def recognize(self, page: OcrPageImage) -> OcrPageResult:
        if not isinstance(page, OcrPageImage):
            raise ValueError("native OCR requires a validated raster page")
        identity = self.identity
        if max(page.width, page.height) > self._capabilities["max_dimension"]:
            raise ValueError("OCR raster exceeds the installed recognizer dimension limit")
        image_module = _dependency("PIL.Image")
        image = image_module.frombytes("RGB", (page.width, page.height), page.pixels)
        try:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            png = buffer.getvalue()
        finally:
            image.close()
        if len(png) > 32 * 1024 * 1024:
            raise ValueError("native OCR PNG exceeds 32 MiB")
        response = _exchange(
            self._command,
            {
                "action": "recognize",
                "language": identity.language,
                "width": page.width,
                "height": page.height,
                "png": base64.b64encode(png).decode("ascii"),
            },
            self._timeout,
            self._limit,
        )
        if set(response) != {"protocol", "action", "identity", "language", "lines", "text_angle"}:
            raise ValueError("native OCR response has invalid fields")
        if (
            _capabilities(response["identity"]) != self._capabilities
            or response["language"] != identity.language
            or self.identity != identity
        ):
            raise ValueError("native OCR provider identity changed during recognition")
        raw_lines = response["lines"]
        if not isinstance(raw_lines, list) or len(raw_lines) > 20_000:
            raise ValueError("native OCR lines must be a bounded array")
        lines = []
        words_seen = 0
        for line in raw_lines:
            if (
                not isinstance(line, dict)
                or set(line) != {"text", "words"}
                or not isinstance(line["words"], list)
            ):
                raise ValueError("native OCR line has invalid fields")
            words_seen += len(line["words"])
            if words_seen > 100_000:
                raise ValueError("native OCR word count exceeds its limit")
            words = []
            for word in line["words"]:
                if not isinstance(word, dict) or set(word) != {"text", "box"}:
                    raise ValueError("native OCR word has invalid fields")
                words.append(OcrWord(word["text"], word["box"]))
            lines.append(OcrLine(line["text"], tuple(words)))
        return OcrPageResult(page.info, identity, tuple(lines), response["text_angle"])
