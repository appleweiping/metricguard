import json
from pathlib import Path

import pytest

from metricguard.cli import main
from metricguard.io import CaseFormatError, load_cases, load_metric_config
from metricguard.metrics import ExactMatch
from metricguard.models import (
    AuditFinding,
    CaseResult,
    EvaluationCase,
    MetricValue,
    Severity,
    SuiteReport,
    UndefinedPolicy,
)
from metricguard.reporting import (
    comparison_to_dict,
    render_comparison_json,
    render_comparison_markdown,
    render_json,
    render_markdown,
    report_to_dict,
)
from metricguard.statistics import BootstrapConfig, paired_comparison
from metricguard.suite import EvaluationSuite


def write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_load_json_and_jsonl(tmp_path: Path) -> None:
    data = [{"id": "café", "reference": "yes", "prediction": "yes", "tags": ["unicode"]}]
    json_path = write(tmp_path / "cases.json", json.dumps(data, ensure_ascii=False))
    jsonl_path = write(tmp_path / "cases.jsonl", "\n" + json.dumps(data[0]) + "\n")
    assert load_cases(json_path)[0].case_id == "café"
    assert load_cases(jsonl_path)[0].tags == ("unicode",)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("{}", "must contain an array"),
        ("[1]", "must be an object"),
        ('[{"id":"x"}]', "missing"),
        ('[{"id":1,"reference":"a","prediction":"b"}]', "id must be"),
        ('[{"id":"x","reference":"a","prediction":"b","tags":1}]', "tags must"),
        ('[{"id":"x","reference":"a","prediction":"b","metadata":1}]', "metadata must"),
        ('[{"id":"x","reference":"a","prediction":"b","extra":1}]', "unknown fields"),
    ],
)
def test_load_cases_rejects_bad_shapes(tmp_path: Path, content: str, message: str) -> None:
    with pytest.raises(CaseFormatError, match=message):
        load_cases(write(tmp_path / "bad.json", content))


def test_load_cases_rejects_missing_and_malformed_files(tmp_path: Path) -> None:
    with pytest.raises(CaseFormatError, match="does not exist"):
        load_cases(tmp_path / "missing.json")
    with pytest.raises(CaseFormatError, match="invalid JSON"):
        load_cases(write(tmp_path / "bad.jsonl", '{"id":'))


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_load_cases_wraps_invalid_utf8(tmp_path: Path, suffix: str) -> None:
    source = tmp_path / f"invalid{suffix}"
    source.write_bytes(b"\xff")
    with pytest.raises(CaseFormatError, match="cannot read"):
        load_cases(source)


@pytest.mark.parametrize(
    "content",
    [
        '[{"id":"x","id":"y","reference":"a","prediction":"b"}]',
        '[{"id":"x","reference":NaN,"prediction":1}]',
        '[{"id":"x","reference":Infinity,"prediction":1}]',
    ],
)
def test_load_cases_rejects_ambiguous_or_nonstandard_json(tmp_path: Path, content: str) -> None:
    with pytest.raises(CaseFormatError):
        load_cases(write(tmp_path / "strict.json", content))


@pytest.mark.parametrize(
    "content",
    [
        '[{"id":" ","reference":"a","prediction":"b"}]',
        '[{"id":"x","reference":"a","prediction":"b","tags":["a","a"]}]',
    ],
)
def test_load_cases_wraps_model_validation(tmp_path: Path, content: str) -> None:
    with pytest.raises(CaseFormatError, match="case 1"):
        load_cases(write(tmp_path / "bad-model.json", content))


def test_load_metric_config(tmp_path: Path) -> None:
    path = write(tmp_path / "metric.json", '{"kind":"exact_match"}')
    assert load_metric_config(path)["kind"] == "exact_match"
    with pytest.raises(CaseFormatError, match="must be a JSON object"):
        load_metric_config(write(tmp_path / "array.json", "[]"))
    with pytest.raises(CaseFormatError, match="duplicate"):
        load_metric_config(write(tmp_path / "duplicate.json", '{"kind":"a","kind":"b"}'))
    invalid_utf8 = tmp_path / "invalid-utf8.json"
    invalid_utf8.write_bytes(b"\xff")
    with pytest.raises(CaseFormatError, match="cannot load"):
        load_metric_config(invalid_utf8)


def test_report_formats_are_stable() -> None:
    report = EvaluationSuite(
        [EvaluationCase("one", "yes", "yes")], undefined_policy=UndefinedPolicy.ERROR
    ).run(ExactMatch())
    payload = report_to_dict(report)
    assert payload["schema_version"] == 1
    assert payload["summary"]["mean_score"] == 1.0
    assert json.loads(render_json(report))["metric"] == "exact_match"
    markdown = render_markdown(report)
    assert "**PASS**" in markdown
    assert "`one`" in markdown
    assert "No contract findings" in markdown


def test_markdown_escapes_untrusted_fields() -> None:
    report = SuiteReport(
        "bad|metric`\nname",
        (
            CaseResult(
                "case|`\nrow",
                MetricValue(None, "bad|reason\nnext"),
                None,
                skipped=True,
            ),
        ),
        (
            AuditFinding(
                "rule|x",
                Severity.ERROR,
                "message|x\nnext",
                case_id="id|x",
            ),
        ),
    )
    markdown = render_markdown(report)
    assert "&#124;" in markdown
    assert "&#96;" in markdown
    assert "<br>" in markdown
    assert "bad|reason" not in markdown


def test_cli_list(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["list"]) == 0
    assert "token_f1" in capsys.readouterr().out


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--version"])
    assert error.value.code == 0
    assert capsys.readouterr().out == "metricguard 0.2.0\n"


def test_cli_run_markdown_and_json_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = write(
        tmp_path / "cases.jsonl",
        '{"id":"a","reference":"Hello, world!","prediction":"hello world"}\n',
    )
    config = write(
        tmp_path / "metric.json",
        '{"kind":"exact_match","normalizer":{"lowercase":true,"strip_punctuation":true}}',
    )
    assert main(["run", str(cases), "--metric-config", str(config), "--audit"]) == 0
    assert "Macro mean: 1.000000" in capsys.readouterr().out

    output = tmp_path / "report.json"
    assert (
        main(
            [
                "run",
                str(cases),
                "--metric",
                "token_f1",
                "--format",
                "json",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(output.read_text(encoding="utf-8"))["summary"]["case_count"] == 1


def test_cli_expected_error(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["run", str(tmp_path / "missing.json"), "--metric", "exact_match"]) == 2
    assert "metricguard: error:" in capsys.readouterr().err


def test_cli_symmetric_implies_audit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = write(tmp_path / "empty.json", "[]")
    assert (
        main(
            [
                "run",
                str(cases),
                "--metric",
                "exact_match",
                "--symmetric",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["findings"][0]["rule"] == "coverage"


def test_cli_invalid_tolerance_is_a_clean_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cases = write(tmp_path / "cases.json", "[]")
    config = write(
        tmp_path / "metric.json",
        '{"kind":"numeric_equivalence","absolute_tolerance":"NaN"}',
    )
    assert main(["run", str(cases), "--metric-config", str(config)]) == 2
    captured = capsys.readouterr()
    assert "finite decimal" in captured.err
    assert "Traceback" not in captured.err


def test_comparison_report_formats_are_stable() -> None:
    baseline = EvaluationSuite([EvaluationCase("one", "yes", "no")]).run(ExactMatch())
    candidate = EvaluationSuite([EvaluationCase("one", "yes", "yes")]).run(ExactMatch())
    comparison = paired_comparison(
        baseline, candidate, config=BootstrapConfig(samples=7, confidence=0.8)
    )
    payload = comparison_to_dict(comparison)
    assert payload["schema_version"] == 1
    assert payload["gate"]["passed"] is True
    assert payload["methods"]["p_value"] == "paired-sign-flip-monte-carlo-v1"
    assert payload["methods"]["p_value_samples"] == 7
    assert json.loads(render_comparison_json(comparison))["delta"]["point"] == 1.0
    markdown = render_comparison_markdown(comparison)
    assert "Regression gate: **PASS**" in markdown
    assert "80% paired-bootstrap interval" in markdown


def test_cli_compare_passes_and_fails_regression_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(
        tmp_path / "baseline.jsonl",
        '{"id":"a","reference":"yes","prediction":"no"}\n'
        '{"id":"b","reference":"yes","prediction":"yes"}\n',
    )
    candidate = write(
        tmp_path / "candidate.jsonl",
        '{"id":"b","reference":"yes","prediction":"yes"}\n'
        '{"id":"a","reference":"yes","prediction":"yes"}\n',
    )
    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "exact_match",
                "--samples",
                "31",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["delta"]["point"] == 0.5
    assert payload["gate"]["passed"] is True

    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "exact_match",
                "--samples",
                "7",
                "--minimum-delta",
                "0.75",
            ]
        )
        == 2
    )
    assert "Regression gate: **FAIL**" in capsys.readouterr().out


def test_cli_compare_rejects_dataset_drift(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(
        tmp_path / "baseline.jsonl", '{"id":"a","reference":"yes","prediction":"no"}\n'
    )
    candidate = write(
        tmp_path / "candidate.jsonl", '{"id":"a","reference":"no","prediction":"no"}\n'
    )
    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "exact_match",
                "--samples",
                "3",
            ]
        )
        == 2
    )
    assert "different references" in capsys.readouterr().err


def test_cli_compare_supports_lower_is_better_direction(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(
        tmp_path / "baseline.jsonl",
        '{"id":"a","reference":"yes","prediction":"yes"}\n',
    )
    candidate = write(
        tmp_path / "candidate.jsonl",
        '{"id":"a","reference":"yes","prediction":"no"}\n',
    )
    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "exact_match",
                "--samples",
                "7",
                "--direction",
                "lower",
                "--format",
                "json",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["delta"]["point"] == -1.0
    assert payload["improvement"]["point"] == 1.0
    assert payload["improvement"]["direction"] == "lower"


def test_cli_refuses_to_overwrite_any_input(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    baseline = write(
        tmp_path / "baseline.jsonl",
        '{"id":"a","reference":"yes","prediction":"yes"}\n',
    )
    candidate = write(
        tmp_path / "candidate.jsonl",
        '{"id":"a","reference":"yes","prediction":"no"}\n',
    )
    original_baseline = baseline.read_text(encoding="utf-8")
    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "exact_match",
                "--output",
                str(baseline),
            ]
        )
        == 2
    )
    assert baseline.read_text(encoding="utf-8") == original_baseline
    assert "output path must differ" in capsys.readouterr().err

    original_candidate = candidate.read_text(encoding="utf-8")
    assert (
        main(
            [
                "run",
                str(candidate),
                "--metric",
                "exact_match",
                "--output",
                str(candidate),
            ]
        )
        == 2
    )
    assert candidate.read_text(encoding="utf-8") == original_candidate
