from __future__ import annotations

from pathlib import Path


def test_alpharank_ci_is_independent_from_portfolio_checkout() -> None:
    root = Path(__file__).resolve().parents[1]
    workflow = (root / ".github/workflows/methodology-validation.yml").read_text(encoding="utf-8")

    alpharank_job, portfolio_job = workflow.split("  portfolio-integration:", maxsplit=1)

    assert "  alpharank:" in alpharank_job
    assert "run_ci_checks.py --group ci" in alpharank_job
    assert "Checkout Portfolio sibling" not in alpharank_job
    assert "matrix.target" not in workflow

    assert "Portfolio cross-repository integration" in portfolio_job
    assert "Checkout Portfolio sibling" in portfolio_job
    assert "make test" in portfolio_job
    assert "npm run build" in portfolio_job

    assert "python -m pytest -q -p no:cacheprovider" not in workflow
