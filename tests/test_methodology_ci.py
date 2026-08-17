from __future__ import annotations

from pathlib import Path


def test_methodology_ci_matrix_covers_both_repositories() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/methodology-validation.yml").read_text(
        encoding="utf-8"
    )

    for required in (
        "target: [alpharank, portfolio]",
        "python -m pytest -q -p no:cacheprovider",
        "tests/test_future_mutation_invariance.py",
        "tests/test_recomputable_replay.py",
        "make test",
        "npm run build",
        "scripts/validate_markdown_links.py",
    ):
        assert required in workflow
