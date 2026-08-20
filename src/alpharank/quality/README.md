# Quality tooling

This package owns deterministic development-quality gates. It may inspect code
and tool output, but it must not contain portfolio, data or model logic.

- `ruff_baseline.py` records the historical Ruff debt as stable fingerprints
  and rejects only diagnostics introduced beyond that baseline.
