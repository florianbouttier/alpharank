# Quality commands

Read-only development checks live here. They may create an explicit report only
when the caller supplies an output path; they do not alter source files.

- `check_ruff_baseline.py` compares the repository-wide Ruff diagnostics with
  the reviewed historical-debt baseline.
- `check_dependencies.py` verifies or explicitly regenerates the pip and Conda
  views from canonical `pyproject.toml` metadata.
- `check_config_schemas.py` validates every maintained JSON config against its
  strict versioned family schema.
- `check_error_handling.py` rejects library `print()` calls, bare handlers and
  broad catches outside an explicit logged process boundary.
- `run_ci_checks.py` runs the exact static, documentation and logical Pytest
  groups. `--group ci` reproduces the clean-checkout CI gate; `--group all` is
  the artifact-aware local pre-publication gate.
