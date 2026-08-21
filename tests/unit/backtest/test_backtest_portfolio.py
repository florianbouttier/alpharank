from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n


def test_select_top_n_can_rank_on_custom_score_column() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "year_month": [date(2024, 1, 1)] * 3,
            "prediction": [0.9, 0.8, 0.1],
            "selection_score": [0.1, 0.8, 0.7],
        }
    )

    selected = select_top_n(frame, top_n=1, score_col="selection_score")

    assert selected.get_column("ticker").to_list() == ["B"]


def test_backtest_portfolio_uses_shared_equal_weight_simulation() -> None:
    selections = pl.DataFrame(
        {
            "year_month": [date(2024, 1, 1)] * 2,
            "decision_month": [date(2024, 1, 1)] * 2,
            "holding_month": [date(2024, 2, 1)] * 2,
            "ticker": ["A", "B"],
            "future_return": [0.10, -0.02],
            "benchmark_future_return": [0.01, 0.01],
            "target_label": [1.0, 0.0],
        }
    )
    monthly = compute_monthly_portfolio_returns(selections)
    assert monthly["portfolio_return"][0] == 0.04
    assert monthly["benchmark_return"][0] == 0.01
    assert monthly["active_return"][0] == 0.03
    assert monthly["hit_rate"][0] == 0.5
