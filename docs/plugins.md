# Metric plugin contract

Metric plugins use the `metricguard.metrics` Python entry-point group. The entry
point name is the metric's public configuration name. The loaded object must be a
factory that accepts a fresh `dict[str, Any]` of configuration options and returns
an object satisfying the `Metric` protocol.

```toml
[project.entry-points."metricguard.metrics"]
domain_score = "my_package.metrics:build_domain_score"
```

```python
def build_domain_score(options: dict[str, object]) -> DomainScore:
    threshold = options.pop("threshold", 0.5)
    if options:
        raise ValueError(f"unknown options: {sorted(options)}")
    return DomainScore(threshold=float(threshold))
```

The returned metric's `name` must equal the entry-point name. Conflicts with built-in
or previously registered names fail. Factories should validate unknown options and
their `evaluate` method must return `MetricValue`; the suite rejects other return
types before aggregation. Scores must be finite or explicitly undefined.

Discovery imports installed third-party code and is therefore opt-in:

```python
metric = build_metric(
    {"kind": "domain_score", "threshold": 0.7},
    load_plugins=True,
)
```

The equivalent CLI switch is `--load-plugins`. Use `metricguard list --plugins` to
inspect the resulting registry. Install and review plugins as executable code, not
as passive data files. Discovery is not a sandbox: imports and factory calls run in
the MetricGuard process with the user's permissions. Registration is atomic, so a
failed discovery does not leave a partially updated registry, although Python-side
effects from code already imported cannot be undone.

Factories receive a fresh options dictionary. A plugin used for paired comparison
must be deterministic across repeated calls and must document whether higher or
lower values are better; pass the corresponding CLI `--direction`. MetricGuard
uses the same metric configuration on baseline and candidate to prevent accidental
same-name/different-option comparisons.
