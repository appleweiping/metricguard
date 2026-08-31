# MetricGuard benchmarks

`benchmark_metrics.py` measures dependency-free metric throughput and deterministic
paired-comparison resampling cost on generated text. It uses a fixed seed, reports machine and
Python metadata, and emits JSON so results can be archived or compared in CI.

Run after installing the development environment:

```bash
python benchmarks/benchmark_metrics.py --cases 5000 --samples 500
```

This is a workload benchmark, not a promise that timings transfer between machines.
Run from an otherwise idle system, record at least three trials, and compare the
median with identical arguments. The script intentionally keeps generated inputs in
memory so filesystem cache behavior does not dominate metric timings.
