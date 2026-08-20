# Quality tooling

This package owns deterministic development-quality gates. It may inspect code
and tool output, but it must not contain portfolio, data or model logic.

- `ruff_baseline.py` records the historical Ruff debt as stable fingerprints
  and rejects only diagnostics introduced beyond that baseline.
- `test_suites.py` validates the ordered pytest suite policy used during
  collection before tests are physically reorganized.
- `test_catalog.py` attributes every tracked test file to one domain and suite,
  then records case counts, network boundary, duration and measured outcome.
- `dependencies.py` renders and compares the pip/Conda compatibility views from
  the single dependency source in `pyproject.toml`.
- `config_schemas.py` classifies maintained JSON configs and rejects unknown
  keys or structural drift recursively before execution.
- `code_inventory.py` maps tracked Python entrypoints, imports, command edges
  and reverse readers without importing application modules.

This package is the first strict Mypy scope. A later package may join the
`files` list only after its own strict check is green without broad ignores.
