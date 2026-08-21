from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "open_source" / "build_sec_core_kpi_yearly_report.py"
SPEC = importlib.util.spec_from_file_location("build_sec_core_kpi_yearly_report", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_worst_year_brief = MODULE._build_worst_year_brief
render_worst_year_brief_markdown = MODULE._render_worst_year_brief_markdown


def test_build_worst_year_brief_selects_worst_year_and_top_tickers() -> None:
    yearly_summary = pl.DataFrame(
        {
            "metric": ["revenue", "revenue", "net_income", "epsActual"],
            "metric_label": ["Chiffre d'affaires", "Chiffre d'affaires", "Résultat net", "EPS publié"],
            "fiscal_year": [2022, 2023, 2023, 2022],
            "expected_quarters": [100, 100, 100, 100],
            "present_quarters": [98, 95, 97, 96],
            "missing_quarters": [2, 5, 3, 4],
            "ticker_count": [25, 25, 25, 25],
            "missing_pct": [2.0, 5.0, 3.0, 4.0],
            "fill_pct": [98.0, 95.0, 97.0, 96.0],
        }
    )
    top_tickers = pl.DataFrame(
        {
            "metric": ["revenue", "revenue", "net_income", "epsActual"],
            "metric_label": ["Chiffre d'affaires", "Chiffre d'affaires", "Résultat net", "EPS publié"],
            "fiscal_year": [2023, 2023, 2023, 2022],
            "ticker": ["AAA.US", "BBB.US", "CCC.US", "DDD.US"],
            "ticker_code": ["AAA", "BBB", "CCC", "DDD"],
            "missing_quarters": [2, 1, 1, 2],
        }
    )

    brief = build_worst_year_brief(yearly_summary=yearly_summary, top_tickers=top_tickers)

    revenue_row = brief.filter(pl.col("metric") == "revenue").row(0, named=True)
    assert revenue_row["fiscal_year"] == 2023
    assert revenue_row["missing_quarters"] == 5
    assert revenue_row["top_tickers"] == "AAA, BBB"

    eps_row = brief.filter(pl.col("metric") == "epsActual").row(0, named=True)
    assert eps_row["fiscal_year"] == 2022
    assert eps_row["missing_pct"] == 4.0


def test_render_worst_year_brief_markdown_includes_publishable_lines() -> None:
    brief = pl.DataFrame(
        {
            "metric": ["epsActual"],
            "metric_label": ["EPS publié"],
            "fiscal_year": [2022],
            "missing_quarters": [87],
            "missing_pct": [3.3957845433],
            "fill_pct": [96.6042154567],
            "top_tickers": ["BBY, LDOS, RAD, STX"],
        }
    )

    markdown = render_worst_year_brief_markdown(brief)

    assert "EPS publié" in markdown
    assert "2022" in markdown
    assert "87 trous" in markdown
    assert "3.40% manquants" in markdown
    assert "BBY, LDOS, RAD, STX" in markdown
