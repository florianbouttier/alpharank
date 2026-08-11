"""Checks for the centralized AlphaRank research dashboard."""

from __future__ import annotations

import importlib.util
import math
import gzip
import json
from collections import defaultdict
from pathlib import Path

import pytest


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
    assert payload["meta"]["shap_rows"] == 76_534
    assert payload["meta"]["shap_overview_rows"] == 1_200
    assert payload["meta"]["shap_features"] == 185
    assert payload["meta"]["attribution_rows"] == 5_408
    monthly_shap_rows = [row["rows"] for row in payload["shap_month_counts"]]
    assert sum(monthly_shap_rows) == 76_534
    assert min(monthly_shap_rows) == 361
    assert max(monthly_shap_rows) == 497
    assert len({row["holding_month"] for row in payload["holdings"]}) == 172
    assert len(payload["folds"]) == 15
    assert payload["folds"][0]["train_rows"] == 21_340
    assert payload["folds"][-1]["test_rows"] == 1_959
    assert len(payload["period_metrics"]) == 14_878
    assert payload["lineage"]["historical"]["snapshot_id"] == "20260713_201639"
    assert payload["lineage"]["historical"]["legacy_series_rows_verified"] == 172
    assert payload["lineage"]["historical"]["legacy_series_maximum_error"] == 0.0
    assert payload["lineage"]["historical"]["spy_series_maximum_error"] == 0.0
    assert payload["lineage"]["historical"]["benchmark"]["price_column"] == (
        "adjusted_close"
    )
    assert payload["lineage"]["legacy_current"]["snapshot_id"] == (
        "20260727_221253"
    )
    assert payload["lineage"]["live"]["snapshot_id"] == "20260727_221253"
    latest_manifest = json.loads(
        (PROJECT_ROOT / "data/open_source/official/manifests/latest_run.json").read_text(
            encoding="utf-8"
        )
    )
    latest_snapshot = (
        Path(latest_manifest["official_dir"]).parent
        / latest_manifest["published_output_snapshot"]
    )
    assert payload["lineage"]["latest"]["price_max_date"] == module._maximum_date(
        latest_snapshot / "US_Finalprice.parquet", "date"
    )
    assert all(path.exists() for path in sources)

    return_columns = {
        "alpha_top5_equal": "alpha_top5_return",
        "alpha_top10_equal": "alpha_top10_return",
        "Combined_Frequency": "legacy_return",
        "SPY total return": "spy_return",
    }
    monthly_contributions: dict[tuple[str, str], float] = defaultdict(float)
    log_contributions: dict[str, float] = defaultdict(float)
    for row in payload["attribution"]:
        monthly_contributions[(row["s"], row["m"])] += row["v"]
        log_contributions[row["s"]] += row["l"]
    monthly_lookup = {row["holding_month"]: row for row in payload["monthly"]}
    assert set(log_contributions) == set(return_columns)
    for (strategy, month), contribution in monthly_contributions.items():
        assert contribution == pytest.approx(
            monthly_lookup[month][return_columns[strategy]],
            abs=1e-12,
        )
    for strategy, column in return_columns.items():
        returns = [row[column] for row in payload["monthly"]]
        expected_cagr = math.prod(1.0 + value for value in returns) ** (
            12.0 / len(returns)
        ) - 1.0
        attributed_cagr = math.expm1(
            12.0 / len(returns) * log_contributions[strategy]
        )
        assert attributed_cagr == pytest.approx(expected_cagr, abs=1e-12)

    start = "2012-01-01"
    end = "2024-12-01"
    historical = [
        row["legacy_return"]
        for row in payload["monthly"]
        if start <= row["holding_month"] <= end
    ]
    current = [
        row["legacy_return"]
        for row in payload["legacy_current_monthly"]
        if start <= row["holding_month"] <= end
    ]
    historical_stats = module.advanced_performance_statistics(historical)
    current_stats = module.advanced_performance_statistics(current)
    assert historical_stats["cagr"] == pytest.approx(0.15864285194318084)
    assert current_stats["cagr"] == pytest.approx(0.1640331962393613)


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
    assert "SHAP par ensemble de test et par mois" in html
    assert "Le backtest ne réentraîne pas tous les mois" in html
    assert "Deux algorithmes distincts, un même contrat de portefeuille" in html
    assert "Lineage des trois contextes" in html
    assert "même snapshot figé du 13 juillet 2026" in html
    assert "Les 15 ensembles train → calibration → test" in html
    assert 'id="fold-select"' in html
    assert 'id="shap-fold"' in html
    assert "Pourquoi une cible 6 mois mais une détention 1 mois" in html
    assert 'id="bt-start"' in html
    assert 'id="advanced-rows"' in html
    assert 'id="rolling-chart"' in html
    assert "Information ratio" in html
    assert "VaR / CVaR 95 %" in html
    assert "3 paires/15 variables au fold 1" in html
    assert "37/185 au fold 15" in html
    assert "Deux snapshots Legacy, une convention SPY explicite" in html
    assert "SPY price return" in html
    assert "D'où vient le CAGR ?" in html
    assert 'id="attr-waterfall"' in html
    assert "Effet composé" in html
    assert manifest["semantics"]["historical_retraining"] == (
        "once per outer fold"
    )
    assert manifest["semantics"]["cagr_attribution"].startswith(
        "exact additive log-return allocation"
    )
    assert manifest["counts"]["shap_months"] == 172
    assert manifest["counts"]["shap_rows"] == 76_534
    assert manifest["shap_sidecars"]["rows"] == 76_534
    assert manifest["shap_sidecars"]["months"] == 172
    sidecars = sorted((report.parent / "shap").glob("*.json.gz"))
    assert len(sidecars) == 172
    with gzip.open(sidecars[0], "rt", encoding="utf-8") as handle:
        first_month = json.load(handle)
    expected_rows = manifest["shap_sidecars"]["files"][0]["rows"]
    assert len(first_month["rows"]) == expected_rows
    assert manifest["data_lineage"]["historical"]["status"] == (
        "same_snapshot_verified"
    )
    assert len(manifest["sources"]) >= 20
