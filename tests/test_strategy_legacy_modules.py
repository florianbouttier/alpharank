from __future__ import annotations

import pandas as pd
import pytest

from alpharank.strategy.legacy import ModelEvaluator, StrategyLearner


def test_legacy_aggregation_and_month_selection_keep_expected_holdings() -> None:
    month = pd.Period("2025-01", freq="M")
    model_outputs = [
        {
            "detailed": pd.DataFrame(
                {
                    "year_month": [month, month],
                    "ticker": ["AAA.US", "BBB.US"],
                    "dr": [1.02, 1.01],
                    "Sector": ["Technology", "Health Care"],
                }
            )
        },
        {
            "detailed": pd.DataFrame(
                {
                    "year_month": [month, month],
                    "ticker": ["AAA.US", "CCC.US"],
                    "dr": [1.02, 1.02],
                    "Sector": ["Technology", "Industrials"],
                }
            )
        },
    ]

    aggregated = StrategyLearner.aggregate_portfolios(
        model_outputs,
        mode="frequency",
    )
    holdings = StrategyLearner.get_portfolio_at_month(aggregated, month=month)
    holdings_by_ticker = holdings.set_index("ticker").sort_index()

    assert holdings_by_ticker.index.tolist() == ["AAA.US", "BBB.US", "CCC.US"]
    assert holdings_by_ticker["n_models"].tolist() == [2, 1, 1]
    assert holdings_by_ticker["weight_normalized"].tolist() == [0.5, 0.25, 0.25]
    assert aggregated["aggregated"]["monthly_return"].tolist() == pytest.approx([0.0175], abs=1e-12)


def test_legacy_artifact_facade_keeps_comparison_contract() -> None:
    months = pd.period_range("2024-01", periods=24, freq="M")
    models = {
        "Legacy": pd.DataFrame(
            {
                "year_month": months,
                "monthly_return": [0.01] * len(months),
                "n": [10] * len(months),
            }
        )
    }

    artifacts = ModelEvaluator.compare_models(models)

    assert len(artifacts) == 9
    metrics, cumulative_returns = artifacts[:2]
    assert metrics.index.tolist() == ["Legacy"]
    assert metrics.loc["Legacy", "Number of Stocks (Avg)"] == 10.0
    assert cumulative_returns.columns.tolist() == ["Legacy"]
    assert isinstance(cumulative_returns.index, pd.DatetimeIndex)
