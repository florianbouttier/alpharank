from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

from alpharank.reporting import sec_quality_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEC_SCRIPT = PROJECT_ROOT / "scripts/open_source/build_sec_quality_dashboard.py"
CENTRAL_SCRIPT = PROJECT_ROOT / "scripts/experiments/render_central_research_dashboard.py"


def _load_sec_script():
    spec = importlib.util.spec_from_file_location("sec_quality_dashboard", SEC_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_large_dashboards_keep_calculation_and_rendering_separate() -> None:
    sec_dashboard = _load_sec_script()

    assert sec_dashboard._build_quarterly_presence is (sec_quality_data._build_quarterly_presence)
    assert len(SEC_SCRIPT.read_text(encoding="utf-8").splitlines()) < 1_000
    assert len(CENTRAL_SCRIPT.read_text(encoding="utf-8").splitlines()) < 1_000


def test_sec_html_renderer_consumes_precalculated_tables() -> None:
    sec_dashboard = _load_sec_script()
    overview = pl.DataFrame(
        {
            "metric": ["revenue"],
            "metric_label": ["Chiffre d'affaires"],
            "hole_count": [2],
            "hole_pct": [10.0],
            "tickers_with_holes": [1],
            "total_tickers": [10],
            "zero_coverage_tickers": [0],
            "first_date": ["2024-01-01"],
            "last_date": ["2024-12-31"],
        }
    )
    empty = pl.DataFrame()

    html = sec_dashboard._render_dashboard_html(
        overview=overview,
        kpi_hole_summary=empty,
        sector_gap_summary=empty,
        ticker_gap_summary=empty,
        ticker_metric_holes=empty,
        quarterly_holes=empty,
        share_anomaly_summary=empty,
        missing=empty,
        deep_dive_tickers=[],
    )

    assert "Audit SEC des fondamentaux" in html
    assert "2" in html
    assert "Chiffre d'affaires" in html
