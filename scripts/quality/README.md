# Quality commands

Read-only development checks live here. They may create an explicit report only
when the caller supplies an output path; they do not alter source files.

- `check_ruff_baseline.py` compares the repository-wide Ruff diagnostics with
  the reviewed historical-debt baseline.
