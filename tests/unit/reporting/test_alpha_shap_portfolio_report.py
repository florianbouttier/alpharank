from __future__ import annotations

from datetime import date
import importlib.util
from pathlib import Path

import polars as pl


MODULE_PATH = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "experiments"
    / "render_alpha_shap_portfolio_report.py"
)
SPEC = importlib.util.spec_from_file_location(
    "render_alpha_shap_portfolio_report",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_feature_lexicon_decodes_inverted_ema_pair_precisely() -> None:
    lexicon = MODULE._feature_lexicon(
        "relative_ema_ratio_95_50_z_month"
    )

    assert lexicon["ema_numerator_span_days"] == 95
    assert lexicon["ema_denominator_span_days"] == 50
    assert lexicon["span_order"] == "numérateur plus lent"
    assert lexicon["transformation"] == "z-score mensuel"
    assert (
        lexicon["exact_definition"]
        == "(relative_ema_ratio_95_50 - "
        "mean_month(relative_ema_ratio_95_50)) / "
        "std_sample_month(relative_ema_ratio_95_50)"
    )


def test_sort_catalog_uses_descending_mean_absolute_shap() -> None:
    catalog = pl.DataFrame(
        {
            "importance_rank": [1, 2, 3],
            "feature": ["low", "highest", "middle"],
            "mean_abs_shap": [0.01, 0.30, 0.20],
        }
    )

    ordered = MODULE._sort_catalog(catalog)

    assert ordered["feature"].to_list() == ["highest", "middle", "low"]
    assert ordered["importance_rank"].to_list() == [1, 2, 3]


def test_monthly_portfolios_align_legacy_top5_and_top10() -> None:
    allocation_holdings = pl.DataFrame(
        {
            "holding_month": [date(2020, 2, 1)] * 3,
            "strategy": [
                "alpha_top5_equal",
                "alpha_top10_equal",
                "alpha_top10_equal",
            ],
            "ticker": ["A.US", "A.US", "B.US"],
            "portfolio_weight": [1.0, 0.5, 0.5],
            "selection_rank": [1, 1, 2],
            "score": [0.8, 0.8, 0.7],
            "calibrated_probability": [0.6, 0.6, 0.5],
            "future_return_1m": [0.1, 0.1, -0.1],
            "sector": ["Tech", "Tech", "Finance"],
        }
    )
    legacy_holdings = pl.DataFrame(
        {
            "portfolio_model": ["Combined_Frequency"],
            "year_month": [date(2020, 2, 1)],
            "ticker": ["L.US"],
            "weight_normalized": [1.0],
            "dr": [0.03],
            "Sector": ["Health"],
        }
    )
    allocation_monthly = pl.DataFrame(
        {
            "holding_month": [date(2020, 2, 1), date(2020, 2, 1)],
            "strategy": ["alpha_top5_equal", "alpha_top10_equal"],
            "net_return": [0.1, 0.0],
            "legacy_return": [0.03, 0.03],
            "benchmark_return": [0.02, 0.02],
        }
    )

    portfolios, returns = MODULE._monthly_portfolios(
        allocation_holdings=allocation_holdings,
        legacy_holdings=legacy_holdings,
        allocation_monthly=allocation_monthly,
    )

    assert set(portfolios["portfolio"].to_list()) == {
        "Legacy publié",
        "Alpha Top 5 égal",
        "Alpha Top 10 égal",
    }
    assert returns.height == 1
    assert returns["alpha_top5_return"][0] == 0.1
    assert returns["alpha_top10_return"][0] == 0.0
