from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import polars as pl

from alpharank.reporting import sec_quality_data

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SEC_AUDIT_SCRIPT = PROJECT_ROOT / "scripts/open_source/reporting/build_sec_quality_dashboard.py"


def _load_sec_audit_script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("sec_quality_audit", SEC_AUDIT_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load static SEC audit script: {SEC_AUDIT_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_static_sec_audit_keeps_calculation_and_rendering_separate() -> None:
    sec_audit = _load_sec_audit_script()

    assert sec_audit._build_quarterly_presence is sec_quality_data._build_quarterly_presence
    assert len(SEC_AUDIT_SCRIPT.read_text(encoding="utf-8").splitlines()) < 1_000


def test_static_sec_html_consumes_precalculated_tables() -> None:
    sec_audit = _load_sec_audit_script()
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

    html = sec_audit._render_dashboard_html(
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
