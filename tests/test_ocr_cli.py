from __future__ import annotations

import json
import sys

from metricguard.cli import _parser, main


def test_ocr_cli_runs_and_writes_report(tmp_path, capsys) -> None:
    cases = tmp_path / "cases.jsonl"
    cases.write_text('{"id":"a","reference":"hello","prediction":"hello"}\n', encoding="utf-8")
    args = _parser().parse_args(["ocr", str(cases)])
    assert args.command == "ocr"
    assert main(["ocr", str(cases)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 1
    assert result["exact_match"]["mean"] == 1.0


def test_ocr_backend_cli_runs_shell_free_argv_template(tmp_path, capsys) -> None:
    image = tmp_path / "page.txt"
    image.write_text("hello world", encoding="utf-8")
    cases = tmp_path / "images.jsonl"
    cases.write_text('{"id":"a","image":"page.txt","reference":"hello world"}\n', encoding="utf-8")
    command = [
        sys.executable,
        "-c",
        "import pathlib,sys; print(pathlib.Path(sys.argv[1]).read_text())",
        "{image}",
    ]
    argv = ["ocr-backend", str(cases)]
    for token in command:
        argv.append(f"--command={token}" if token.startswith("-") else "--command")
        if not token.startswith("-"):
            argv.append(token)
    args = _parser().parse_args(argv)
    assert args.command == command
    assert main(argv) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["cases"] == 1
    assert result["exact_match"]["mean"] == 1.0
