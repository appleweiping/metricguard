## Summary

Describe the user-visible change and why it is needed.

## Verification

- [ ] Tests cover metric contracts, invalid/non-finite inputs, determinism, and comparison alignment.
- [ ] `ruff check .`, `ruff format --check .`, `mypy src`, and `pytest` pass.
- [ ] Wheel and sdist pass integrity and isolated-install smoke checks.
- [ ] Statistical assumptions, benchmark evidence, docs, and changelog are updated when behavior changes.
- [ ] No private evaluation data or generated build artifacts are included.

## Compatibility and risk

Describe score semantics, optimization direction, statistical method, plugin, performance, and migration implications.
