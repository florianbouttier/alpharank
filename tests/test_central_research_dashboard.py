"""Checks for the centralized AlphaRank research dashboard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts/experiments/render_central_research_dashboard.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "render_central_research_dashboard",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_payload_has_complete_monthly_shap_and_portfolio_coverage() -> None:
    module = _load_module()
    payload, sources = module.build_payload()

    assert payload["meta"]["test_months"] == 172
    assert payload["meta"]["shap_months"] == 172
    assert payload["meta"]["shap_rows"] == 1_200
    assert payload["meta"]["shap_features"] == 185
    assert len({row["holding_month"] for row in payload["holdings"]}) == 172
    assert all(path.exists() for path in sources)


def test_render_writes_one_html_and_auditable_manifest(
    tmp_path: Path,
) -> None:
    module = _load_module()
    report, manifest_path = module.render(tmp_path)
    html = report.read_text(encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert html.count('class="page') == 8
    assert "__PAYLOAD__" not in html
    assert "__SOURCES__" not in html
    assert "SHAP filtré par mois de test" in html
    assert "Le backtest ne réentraîne pas tous les mois" in html
    assert "Pourquoi une cible 6 mois mais une détention 1 mois" in html
    assert 'id="bt-start"' in html
    assert 'id="advanced-rows"' in html
    assert 'id="rolling-chart"' in html
    assert "Information ratio" in html
    assert "VaR / CVaR 95 %" in html
    assert "3 paires/15 variables au fold 1" in html
    assert "37/185 au fold 15" in html
    assert manifest["semantics"]["historical_retraining"] == (
        "once per outer fold"
    )
    assert manifest["counts"]["shap_months"] == 172
    assert len(manifest["sources"]) >= 20
