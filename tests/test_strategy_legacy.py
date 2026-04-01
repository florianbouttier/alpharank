from __future__ import annotations

import pandas as pd

from alpharank.data.processing import IndexDataManager
from alpharank.strategy.legacy import StrategyLearner


def test_return_from_training_exports_selected_optuna_caps() -> None:
    fit = pd.DataFrame(
        {
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist() * 2,
            "ticker": ["AAA.US"] * 4 + ["BBB.US"] * 4,
            "date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"] * 2),
            "n_long": [100] * 8,
            "n_short": [20] * 8,
            "n_asset": [30] * 8,
            "dr": [1.02, 1.01, 1.03, 1.01, 1.01, 1.02, 0.99, 1.04],
            "quantile_mtr": [0.9, 0.9, 0.9, 0.9, 0.8, 0.8, 0.8, 0.8],
            "mtr": [1.1] * 8,
        }
    )
    stocks_filter = pd.DataFrame(
        {
            "year_month": pd.period_range("2024-12", periods=4, freq="M").tolist() * 2,
            "ticker": ["AAA.US"] * 4 + ["BBB.US"] * 4,
        }
    )
    sector = pd.DataFrame({"ticker": ["AAA.US", "BBB.US"], "Sector": ["Technology", "Technology"]})
    index_prices = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-01-31", "2025-02-28", "2025-03-31", "2025-04-30"]),
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )
    components = pd.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US"] * 4,
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist() * 2,
        }
    )
    monthly_returns = pd.DataFrame(
        {
            "year_month": pd.period_range("2025-01", periods=4, freq="M").tolist(),
            "monthly_return": [0.0, 0.01, 0.0, 0.02],
        }
    )
    index = IndexDataManager(index_prices, components, monthly_returns_df=monthly_returns, backend="polars")

    output = StrategyLearner.return_from_training(
        df_fiting=fit,
        stocks_filter=stocks_filter,
        sector=sector,
        index=index,
        alpha=2,
        temp=12,
        mode="mean",
        params={"n_asset": 5, "n_max_per_sector": 2},
        backend="polars",
    )

    aggregated = output["aggregated"]
    detailed = output["detailed"]

    assert {"selected_model", "selected_n_asset", "selected_n_max_per_sector"}.issubset(aggregated.columns)
    assert {"selected_model", "selected_n_asset", "selected_n_max_per_sector"}.issubset(detailed.columns)
    assert aggregated["selected_n_asset"].unique().tolist() == [5]
    assert aggregated["selected_n_max_per_sector"].unique().tolist() == [2]
    assert detailed["selected_n_asset"].unique().tolist() == [5]
    assert detailed["selected_n_max_per_sector"].unique().tolist() == [2]
    assert aggregated["selected_model"].str.contains(r"\|asset=5\|sector=2").all()
