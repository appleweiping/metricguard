# Releasing

MetricGuard follows semantic versioning. Releases come only from a clean, reviewed `main` commit with passing checks.

1. Update version exports, `pyproject.toml`, `CHANGELOG.md`, and `CITATION.cff`.
2. Run lint, format, strict typing, tests, statistical/benchmark smoke, and build checks.
3. Inspect both distributions; install the wheel in an empty environment and run a suite plus paired comparison.
4. Add curated notes at `docs/releases/vX.Y.Z.md` when appropriate; the workflow falls back to generated notes when
   that file is absent.
5. Before the first release, a repository administrator must enable GitHub's immutable releases setting. Create a
   protected `vX.Y.Z` tag at the reviewed commit.
6. Automation records SHA-256 checksums and provenance, then publishes every asset in the release creation operation
   so repository-level immutability can lock them.
7. PyPI publication requires a separate approved dispatch from that tag through a trusted publisher.

Never replace published artifacts or silently change a statistical method under an existing version.
