"""Explicit metric registration and opt-in entry-point discovery."""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any, cast

from .metrics import BUILTIN_METRIC_NAMES, Metric, _build_builtin_metric

MetricFactory = Callable[[dict[str, Any]], Metric]
_METRIC_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")


class MetricPluginError(ValueError):
    """Raised when an explicitly requested metric plugin is invalid."""


class MetricRegistry:
    """Name-to-factory registry with deterministic, opt-in plugin loading."""

    def __init__(self) -> None:
        self._factories: dict[str, MetricFactory] = {}

    @classmethod
    def with_builtins(cls) -> MetricRegistry:
        """Return a new registry containing every dependency-free metric."""

        registry = cls()
        for name in BUILTIN_METRIC_NAMES:
            registry.register(name, _builtin_factory(name))
        return registry

    @property
    def names(self) -> tuple[str, ...]:
        """Registered metric names in stable lexical order."""

        return tuple(sorted(self._factories))

    def register(self, name: str, factory: MetricFactory, *, replace: bool = False) -> None:
        """Register a metric factory after validating its public name."""

        _validate_metric_name(name)
        if not callable(factory):
            raise TypeError("metric factory must be callable")
        if name in self._factories and not replace:
            raise ValueError(f"metric {name!r} is already registered")
        self._factories[name] = factory

    def build(self, config: str | Mapping[str, Any]) -> Metric:
        """Build a registered metric from a name or configuration mapping."""

        if isinstance(config, str):
            kind = config
            options: dict[str, Any] = {}
        elif isinstance(config, Mapping):
            options = dict(config)
            kind_value = options.pop("kind", None)
            if not isinstance(kind_value, str):
                raise ValueError("metric configuration requires a string 'kind'")
            kind = kind_value
        else:
            raise TypeError("metric configuration must be a string or object")
        factory = self._factories.get(kind)
        if factory is None:
            raise ValueError(f"unknown metric kind: {kind}")
        try:
            metric = factory(dict(options))
        except (MetricPluginError, TypeError, ValueError):
            raise
        except Exception as error:
            raise MetricPluginError(f"metric factory {kind!r} failed: {error}") from error
        if not isinstance(metric, Metric):
            raise MetricPluginError(
                f"metric factory {kind!r} did not return an object with name and evaluate"
            )
        if metric.name != kind:
            raise MetricPluginError(
                f"metric factory {kind!r} returned a metric named {metric.name!r}"
            )
        return metric

    def load_plugins(self, *, group: str = "metricguard.metrics") -> tuple[str, ...]:
        """Discover and register entry points only after an explicit call.

        Each entry point name is the metric name and its loaded object must be
        a callable accepting a fresh options dictionary.
        """

        if not isinstance(group, str) or not group:
            raise ValueError("entry-point group must be a non-empty string")
        try:
            discovered = metadata.entry_points()
            select = getattr(discovered, "select", None)
            if callable(select):
                selected = tuple(select(group=group))
            else:  # pragma: no cover - compatibility with older importlib-metadata APIs
                selected = tuple(cast(Any, discovered).get(group, ()))
            ordered = sorted(selected, key=lambda item: (item.name, item.value))
        except Exception as error:
            raise MetricPluginError(f"cannot discover metric plugins: {error}") from error

        validated: list[Any] = []
        reserved = set(self._factories)
        for entry_point in ordered:
            try:
                _validate_metric_name(entry_point.name)
            except (TypeError, ValueError) as error:
                raise MetricPluginError(
                    f"invalid plugin metric name {entry_point.name!r}: {error}"
                ) from error
            if entry_point.name in reserved:
                raise MetricPluginError(
                    f"plugin metric {entry_point.name!r} conflicts with a registered metric"
                )
            validated.append(entry_point)
            reserved.add(entry_point.name)

        pending: list[tuple[str, MetricFactory]] = []
        for entry_point in validated:
            try:
                factory = entry_point.load()
            except Exception as error:
                raise MetricPluginError(
                    f"cannot load metric plugin {entry_point.name!r}: {error}"
                ) from error
            if not callable(factory):
                raise MetricPluginError(
                    f"metric plugin {entry_point.name!r} must expose a callable factory"
                )
            pending.append((entry_point.name, factory))

        # Discovery is atomic: one broken entry point leaves the registry intact.
        for name, factory in pending:
            self._factories[name] = factory
        return tuple(name for name, _ in pending)


def _builtin_factory(name: str) -> MetricFactory:
    def factory(options: dict[str, Any]) -> Metric:
        return _build_builtin_metric(name, options)

    return factory


def _validate_metric_name(name: object) -> None:
    if not isinstance(name, str):
        raise TypeError("metric name must be a string")
    if _METRIC_NAME.fullmatch(name) is None:
        raise ValueError(
            "metric names must start with an ASCII lowercase letter and contain only "
            "ASCII lowercase letters, digits, and underscores"
        )
