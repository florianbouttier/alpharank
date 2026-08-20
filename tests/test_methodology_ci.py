from __future__ import annotations

from pathlib import Path


def test_methodology_ci_matrix_covers_both_repositories() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/methodology-validation.yml").read_text(encoding="utf-8")

    for required in (
        "target: [alpharank, portfolio]",
        "run_ci_checks.py --group static",
        "run_ci_checks.py --group unit",
        "run_ci_checks.py --group integration",
        "run_ci_checks.py --group replay",
        "run_ci_checks.py --group network",
        "run_ci_checks.py --group production",
        "make test",
        "npm run build",
    ):
        assert required in workflow
