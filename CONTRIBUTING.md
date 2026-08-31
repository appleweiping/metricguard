# Contributing

Bug reports should include the smallest case file, metric configuration, expected
behavior, actual behavior, Python version, and MetricGuard version.

For code changes:

1. Open an issue for changes to metric semantics or report schemas.
2. Create a focused branch and add a regression test first.
3. Run `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest`.
4. Document undefined cases and compatibility effects in the pull request.
5. Disclose material automated assistance and personally review every change.

Do not add a metric whose edge cases cannot be stated precisely. New runtime
dependencies require a design discussion.
