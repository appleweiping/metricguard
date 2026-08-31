from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from metricguard.cli import main
from metricguard.metrics import ExactMatch, build_metric
from metricguard.models import MetricValue
from metricguard.registry import MetricPluginError, MetricRegistry


@dataclass
class PluginMetric:
    name: str = "plugin_score"

    def evaluate(self, reference: Any, prediction: Any) -> MetricValue:
        return MetricValue(0.25)


@dataclass
class FakeEntryPoint:
    name: str
    value: str
    loaded: object

    def load(self) -> object:
        return self.loaded


class FakeEntryPoints(list[FakeEntryPoint]):
    def select(self, *, group: str) -> FakeEntryPoints:
        assert group == "metricguard.metrics"
        return self


def test_registry_contains_sorted_builtins_and_builds_defensively() -> None:
    registry = MetricRegistry.with_builtins()
    assert registry.names == tuple(sorted(registry.names))
    assert "rouge_l" in registry.names
    config: dict[str, object] = {"kind": "exact_match"}
    assert isinstance(registry.build(config), ExactMatch)
    assert config == {"kind": "exact_match"}


def test_plugin_discovery_is_explicit(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_if_called() -> object:
        raise AssertionError("entry point discovery should be opt-in")

    monkeypatch.setattr("metricguard.registry.metadata.entry_points", fail_if_called)
    assert isinstance(build_metric("exact_match"), ExactMatch)


def test_plugin_discovery_and_build(monkeypatch: pytest.MonkeyPatch) -> None:
    entry_points = FakeEntryPoints(
        [FakeEntryPoint("plugin_score", "sample:factory", lambda options: PluginMetric())]
    )
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: entry_points)
    registry = MetricRegistry.with_builtins()
    assert registry.load_plugins() == ("plugin_score",)
    assert registry.build("plugin_score").evaluate("a", "b").score == 0.25


@pytest.mark.parametrize(
    "name", ["", " leading", "UPPER", "hyphen-name", "éclair", "1metric", "___"]
)
def test_registry_rejects_invalid_names(name: str) -> None:
    with pytest.raises(ValueError, match="lowercase"):
        MetricRegistry().register(name, lambda options: PluginMetric(name=name))


def test_registry_rejects_duplicates_and_invalid_factory_results() -> None:
    registry = MetricRegistry()
    registry.register("plugin_score", lambda options: PluginMetric())
    with pytest.raises(ValueError, match="already registered"):
        registry.register("plugin_score", lambda options: PluginMetric())
    registry.register("invalid", lambda options: object())  # type: ignore[arg-type]
    with pytest.raises(MetricPluginError, match="did not return"):
        registry.build("invalid")
    registry.register("wrong_name", lambda options: PluginMetric())
    with pytest.raises(MetricPluginError, match="returned a metric named"):
        registry.build("wrong_name")


def test_plugin_conflicts_and_load_failures_are_contextual(monkeypatch: pytest.MonkeyPatch) -> None:
    duplicate = FakeEntryPoints(
        [FakeEntryPoint("exact_match", "sample:factory", lambda options: PluginMetric())]
    )
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: duplicate)
    with pytest.raises(MetricPluginError, match="conflicts"):
        MetricRegistry.with_builtins().load_plugins()

    class BrokenEntryPoint(FakeEntryPoint):
        def load(self) -> object:
            raise RuntimeError("boom")

    broken = FakeEntryPoints([BrokenEntryPoint("broken", "sample:broken", object())])
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: broken)
    with pytest.raises(MetricPluginError, match=r"cannot load.*broken"):
        MetricRegistry.with_builtins().load_plugins()


def test_plugin_discovery_is_atomic_on_late_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    class BrokenEntryPoint(FakeEntryPoint):
        def load(self) -> object:
            raise RuntimeError("late failure")

    entries = FakeEntryPoints(
        [
            FakeEntryPoint("aaa_good", "sample:good", lambda options: PluginMetric("aaa_good")),
            BrokenEntryPoint("zzz_broken", "sample:broken", object()),
        ]
    )
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: entries)
    registry = MetricRegistry.with_builtins()
    original_names = registry.names
    with pytest.raises(MetricPluginError, match="late failure"):
        registry.load_plugins()
    assert registry.names == original_names


def test_invalid_or_duplicate_plugin_names_fail_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loads: list[str] = []

    class ObservedEntryPoint(FakeEntryPoint):
        def load(self) -> object:
            loads.append(self.name)
            return self.loaded

    invalid = FakeEntryPoints(
        [ObservedEntryPoint("Not_Valid", "sample:invalid", lambda options: PluginMetric())]
    )
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: invalid)
    with pytest.raises(MetricPluginError, match="invalid plugin metric name"):
        MetricRegistry.with_builtins().load_plugins()
    assert loads == []

    duplicate = FakeEntryPoints(
        [
            ObservedEntryPoint("same", "sample:first", lambda options: PluginMetric("same")),
            ObservedEntryPoint("same", "sample:second", lambda options: PluginMetric("same")),
        ]
    )
    monkeypatch.setattr("metricguard.registry.metadata.entry_points", lambda: duplicate)
    with pytest.raises(MetricPluginError, match="conflicts"):
        MetricRegistry.with_builtins().load_plugins()
    assert loads == []


def test_entry_point_enumeration_failure_is_contextual(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_discovery() -> object:
        raise RuntimeError("metadata unavailable")

    monkeypatch.setattr("metricguard.registry.metadata.entry_points", broken_discovery)
    with pytest.raises(MetricPluginError, match=r"cannot discover.*metadata unavailable"):
        MetricRegistry.with_builtins().load_plugins()


def test_cli_comparison_discovers_and_imports_plugins_once(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    discovery_calls = 0
    load_calls = 0

    class CountedEntryPoint(FakeEntryPoint):
        def load(self) -> object:
            nonlocal load_calls
            load_calls += 1
            return self.loaded

    entries = FakeEntryPoints(
        [CountedEntryPoint("plugin_score", "sample:factory", lambda options: PluginMetric())]
    )

    def discover() -> FakeEntryPoints:
        nonlocal discovery_calls
        discovery_calls += 1
        return entries

    monkeypatch.setattr("metricguard.registry.metadata.entry_points", discover)
    case = '{"id":"a","reference":"x","prediction":"y"}\n'
    baseline = tmp_path / "baseline.jsonl"
    candidate = tmp_path / "candidate.jsonl"
    baseline.write_text(case, encoding="utf-8")
    candidate.write_text(case, encoding="utf-8")
    assert (
        main(
            [
                "compare",
                str(baseline),
                str(candidate),
                "--metric",
                "plugin_score",
                "--load-plugins",
                "--samples",
                "3",
                "--format",
                "json",
            ]
        )
        == 0
    )
    assert discovery_calls == load_calls == 1
    assert '"metric": "plugin_score"' in capsys.readouterr().out
