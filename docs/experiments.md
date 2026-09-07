# Metric experiment matrices

`ExperimentMatrix` evaluates several named metric/case combinations while
preserving declared order. Each cell can use its own undefined-value policy and
worker count; optional JSONL caches are isolated by a stable name digest.

```python
from metricguard import ExperimentMatrix, ExperimentSpec, build_metric

matrix = ExperimentMatrix(
    [
        ExperimentSpec("exact", build_metric("exact_match"), cases),
        ExperimentSpec("rouge", build_metric("rouge_l"), cases),
    ]
)
results = matrix.run(cache_dir=".metric-cache", max_workers=2)
print(ExperimentMatrix.means(results))
```

The matrix coordinates evaluation; it does not make custom metrics thread-safe.
Use the metric's documented concurrency contract and keep cache directories
private to one experiment configuration.
