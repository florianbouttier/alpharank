from __future__ import annotations

from pathlib import Path


def test_methodology_ci_matrix_covers_both_repositories() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/methodology-validation.yml").read_text(encoding="utf-8")

    for required in (
        "target: [alpharank, portfolio]",
        "run_ci_checks.py --group ci",
        "make test",
        "npm run build",
    ):
        assert required in workflow

    assert "python -m pytest -q -p no:cacheprovider" not in workflow
