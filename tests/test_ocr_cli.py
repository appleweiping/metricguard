from __future__ import annotations

import json

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
