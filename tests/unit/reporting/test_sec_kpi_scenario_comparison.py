from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


MODULE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "open_source" / "build_sec_kpi_scenario_comparison.py"
SPEC = importlib.util.spec_from_file_location("build_sec_kpi_scenario_comparison", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

build_metric_ranking = MODULE._build_metric_ranking
build_global_summary = MODULE._build_global_summary


def test_metric_ranking_orders_each_kpi_by_missing_pct() -> None:
    scenarios = pl.DataFrame(
        {
            "scenario": ["baseline", "candidate", "baseline", "candidate"],
            "metric": ["revenue", "revenue", "epsActual", "epsActual"],
            "metric_label": ["Chiffre d'affaires", "Chiffre d'affaires", "EPS publié", "EPS publié"],
            "fiscal_year": [2023, 2023, 2022, 2022],
            "missing_quarters": [90, 72, 70, 64],
            "missing_pct": [3.5, 2.8, 2.73, 2.5],
            "fill_pct": [96.5, 97.2, 97.27, 97.5],
            "top_tickers": ["A, B", "C, D", "E, F", "G, H"],
        }
    )

    ranked = build_metric_ranking(scenarios)

    revenue_best = ranked.filter(pl.col("metric") == "revenue").sort("metric_rank").row(0, named=True)
    eps_best = ranked.filter(pl.col("metric") == "epsActual").sort("metric_rank").row(0, named=True)
    assert revenue_best["scenario"] == "candidate"
    assert eps_best["scenario"] == "candidate"


def test_global_summary_uses_worst_kpi_pct_as_ranking_key() -> None:
    scenarios = pl.DataFrame(
        {
            "scenario": ["baseline", "baseline", "candidate", "candidate"],
            "metric": ["revenue", "epsActual", "revenue", "epsActual"],
            "metric_label": ["Chiffre d'affaires", "EPS publié", "Chiffre d'affaires", "EPS publié"],
            "fiscal_year": [2023, 2022, 2023, 2022],
            "missing_quarters": [90, 70, 72, 64],
            "missing_pct": [3.5, 2.73, 2.8, 2.5],
            "fill_pct": [96.5, 97.27, 97.2, 97.5],
            "top_tickers": ["A, B", "C, D", "E, F", "G, H"],
        }
    )

    summary = build_global_summary(scenarios).sort("global_rank")

    assert summary.row(0, named=True)["scenario"] == "candidate"
    assert summary.row(0, named=True)["worst_metric_pct"] == 2.8
