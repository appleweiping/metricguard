from dataclasses import dataclass

import pytest

from metricguard.contracts import Contract, ContractAuditor
from metricguard.metrics import ExactMatch, TokenF1
from metricguard.models import (
    EvaluationCase,
    MetricValue,
    Severity,
    UndefinedPolicy,
)
from metricguard.suite import EvaluationSuite


def cases() -> tuple[EvaluationCase, ...]:
    return (
        EvaluationCase("same", "alpha", "alpha", tags=("smoke",)),
        EvaluationCase("different", "alpha", "beta"),
    )


def test_suite_rejects_metric_protocol_output_violation() -> None:
    class BadMetric:
        name = "bad"

        def evaluate(self, reference: object, prediction: object) -> object:
            del reference, prediction
            return 0.5

    with pytest.raises(TypeError, match="expected MetricValue"):
        EvaluationSuite(cases()).run(BadMetric())  # type: ignore[arg-type]


def test_case_validation() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        EvaluationCase(" ", "a", "b")
    with pytest.raises(ValueError, match="duplicate tags"):
        EvaluationCase("x", "a", "b", tags=("a", "a"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"case_id": 1},
        {"tags": "tag"},
        {"tags": ("tag", 1)},
        {"metadata": []},
        {"metadata": {1: "value"}},
    ],
)
def test_case_rejects_wrong_runtime_types(kwargs: dict[str, object]) -> None:
    values: dict[str, object] = {
        "case_id": "case",
        "reference": "a",
        "prediction": "b",
    }
    values.update(kwargs)
    with pytest.raises(TypeError):
        EvaluationCase(**values)  # type: ignore[arg-type]


def test_metric_value_validation() -> None:
    with pytest.raises(ValueError, match="finite"):
        MetricValue(float("nan"))
    with pytest.raises(ValueError, match="requires a reason"):
        MetricValue(None)
    assert MetricValue(0.5).defined
    assert not MetricValue(None, "missing").defined


def test_suite_summary() -> None:
    report = EvaluationSuite(cases()).run(ExactMatch())
    assert report.scored_count == 2
    assert report.skipped_count == 0
    assert report.mean_score == 0.5
    assert report.passed_contracts


def test_duplicate_case_ids_rejected() -> None:
    duplicate = EvaluationCase("same", "x", "x")
    with pytest.raises(ValueError, match="duplicate case IDs"):
        EvaluationSuite((*cases(), duplicate))


@dataclass(frozen=True)
class UndefinedMetric:
    name: str = "undefined"

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        return MetricValue(None, "not comparable")


@pytest.mark.parametrize(
    ("policy", "resolved", "skipped"),
    [
        (UndefinedPolicy.SKIP, None, True),
        (UndefinedPolicy.ZERO, 0.0, False),
        (UndefinedPolicy.ONE, 1.0, False),
    ],
)
def test_undefined_policies(policy: UndefinedPolicy, resolved: float | None, skipped: bool) -> None:
    report = EvaluationSuite(cases()[:1], undefined_policy=policy).run(UndefinedMetric())
    assert report.results[0].resolved_score == resolved
    assert report.results[0].skipped is skipped


def test_undefined_error_policy_names_case() -> None:
    with pytest.raises(ValueError, match="'same'"):
        EvaluationSuite(cases()[:1]).run(UndefinedMetric())


def test_suite_rejects_wrong_runtime_types() -> None:
    with pytest.raises(TypeError, match="EvaluationCase"):
        EvaluationSuite(["not-a-case"])  # type: ignore[list-item]
    with pytest.raises(TypeError, match="UndefinedPolicy"):
        EvaluationSuite(cases(), undefined_policy="skip")  # type: ignore[arg-type]


def test_contract_validation() -> None:
    with pytest.raises(ValueError, match="minimum"):
        Contract(minimum=2, maximum=1)
    with pytest.raises(ValueError, match="tolerance"):
        Contract(tolerance=-1)
    with pytest.raises(ValueError, match="identity_score"):
        Contract(identity_score=2)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": False},
        {"maximum": "1"},
        {"identity_score": True},
        {"tolerance": "0.1"},
        {"symmetric": "false"},
        {"deterministic": 1},
    ],
)
def test_contract_rejects_wrong_runtime_types(kwargs: dict[str, object]) -> None:
    with pytest.raises(TypeError):
        Contract(**kwargs)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="Contract"):
        ContractAuditor("default")  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tolerance": float("nan")},
        {"minimum": float("-inf")},
        {"maximum": float("inf")},
        {"identity_score": float("nan")},
    ],
)
def test_contract_rejects_non_finite_values(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="finite"):
        Contract(**kwargs)


def test_builtin_metric_passes_contract_audit() -> None:
    auditor = ContractAuditor(Contract(symmetric=True))
    report = EvaluationSuite(cases()).run(TokenF1(), auditor=auditor)
    assert report.findings == ()
    assert report.passed_contracts


@dataclass
class BrokenMetric:
    name: str = "broken"
    calls: int = 0

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        self.calls += 1
        if reference == prediction:
            return MetricValue(0.25)
        if self.calls % 2:
            return MetricValue(2.0)
        return MetricValue(-1.0)


@dataclass
class ChangesAfterFirstCall:
    name: str = "changes-after-first-call"
    calls: int = 0

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        self.calls += 1
        return MetricValue(0.0 if self.calls == 1 else 1.0)


def test_suite_audit_compares_determinism_with_first_observed_value() -> None:
    report = EvaluationSuite(cases()[:1]).run(
        ChangesAfterFirstCall(),
        auditor=ContractAuditor(),
    )
    assert report.results[0].raw.score == 0.0
    assert any(finding.rule == "deterministic" for finding in report.findings)


def test_contract_auditor_reports_multiple_properties() -> None:
    findings = ContractAuditor(Contract()).audit(BrokenMetric(), cases())
    rules = {finding.rule for finding in findings}
    assert {"bounded", "deterministic", "identity"}.issubset(rules)
    assert all(finding.severity is Severity.ERROR for finding in findings)


@dataclass(frozen=True)
class AsymmetricMetric:
    name: str = "asymmetric"

    def evaluate(self, reference: object, prediction: object) -> MetricValue:
        if reference == prediction:
            return MetricValue(1.0)
        return MetricValue(1.0 if str(reference) < str(prediction) else 0.0)


def test_contract_auditor_reports_asymmetry_when_requested() -> None:
    findings = ContractAuditor(Contract(symmetric=True)).audit(AsymmetricMetric(), cases())
    assert any(finding.rule == "symmetric" for finding in findings)


def test_empty_contract_audit_warns() -> None:
    findings = ContractAuditor().audit(ExactMatch(), ())
    assert len(findings) == 1
    assert findings[0].severity is Severity.WARNING
