from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl

from alpharank.multihorizon.confirmation import (
    cost_sensitivity,
    deflated_sharpe_statistics,
    holdings_and_concentration,
    meta_walk_forward_selection,
    moving_block_indices,
    paired_block_bootstrap,
)


def _months(start_year: int, count: int) -> list[date]:
    return [
        date(start_year + index // 12, index % 12 + 1, 1)
        for index in range(count)
    ]


def test_moving_block_indices_preserve_consecutive_circular_blocks() -> None:
    indices = moving_block_indices(
        10,
        block_months=4,
        rng=np.random.default_rng(7),
    )
    assert len(indices) == 10
    for block_start in (0, 4):
        block = indices[block_start : block_start + 4]
        assert np.all((block[1:] - block[:-1]) % 10 == 1)


def test_paired_bootstrap_and_cost_sensitivity_are_deterministic() -> None:
    monthly = pl.DataFrame(
        {
            "net_return": [0.02, -0.01, 0.03, 0.00] * 6,
            "gross_return": [0.021, -0.009, 0.031, 0.001] * 6,
            "turnover": [1.0] * 24,
            "benchmark_return": [0.01, -0.01, 0.01, 0.00] * 6,
            "legacy_return": [0.015, -0.01, 0.02, 0.00] * 6,
        }
    )
    first = paired_block_bootstrap(
        monthly,
        comparator_columns={"Legacy": "legacy_return"},
        samples=100,
        block_months=4,
        seed=3,
    )
    second = paired_block_bootstrap(
        monthly,
        comparator_columns={"Legacy": "legacy_return"},
        samples=100,
        block_months=4,
        seed=3,
    )
    assert first.equals(second)
    costs = cost_sensitivity(monthly, cost_bps_values=(0, 50))
    assert costs.filter(pl.col("cost_bps") == 0)["model_total_return"][0] > costs.filter(
        pl.col("cost_bps") == 50
    )["model_total_return"][0]


def test_deflated_sharpe_penalizes_multiple_trials() -> None:
    returns = np.asarray([0.02, -0.01, 0.03, 0.01] * 36)
    few = deflated_sharpe_statistics(returns, trials=1)
    many = deflated_sharpe_statistics(returns, trials=162)
    assert (
        many["deflated_sharpe_probability"]
        < few["deflated_sharpe_probability"]
    )


def test_meta_selection_uses_only_trailing_history(tmp_path) -> None:
    months = _months(2010, 48)
    for label, base_return in (("a", 0.02), ("b", 0.01)):
        directory = tmp_path / label / "classification_h01"
        directory.mkdir(parents=True)
        pl.DataFrame(
            {
                "decision_month": months * 3,
                "holding_month": months * 3,
                "top_n": [5] * 48 + [10] * 48 + [20] * 48,
                "n_positions": [5] * 144,
                "gross_return": [base_return] * 144,
                "net_return": [base_return] * 144,
                "benchmark_return": [0.005] * 144,
                "turnover": [0.2] * 144,
                "transaction_cost": [0.0] * 144,
                "legacy_return": [0.008] * 144,
            }
        ).write_csv(directory / "trading_monthly.csv")

    choices, monthly, summary = meta_walk_forward_selection(
        {"a": tmp_path / "a", "b": tmp_path / "b"},
        horizons=(1,),
        methods=("classification",),
        top_n_values=(5,),
        lookback_months=12,
    )

    assert choices.height > 0
    assert all(
        history_end < evaluation_start
        for history_end, evaluation_start in choices.select(
            "history_end", "evaluation_start"
        ).iter_rows()
    )
    assert choices["selected_feature_mode"].unique().to_list() == ["a"]
    assert monthly.height == summary["months"][0]


def test_holdings_concentration_uses_top_scores_and_sector_map(tmp_path) -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 3 + [date(2020, 2, 1)] * 3,
            "ticker": ["A", "B", "C"] * 2,
            "score": [0.9, 0.8, 0.1, 0.7, 0.6, 0.5],
        }
    )
    general_path = tmp_path / "general.parquet"
    pl.DataFrame(
        {
            "ticker": ["A", "B", "C"],
            "GicSector": ["Tech", "Finance", "Tech"],
            "Sector": ["Technology", "Financial", "Technology"],
            "GicIndustry": ["Software", "Banks", "Hardware"],
            "Industry": ["Software", "Banks", "Hardware"],
        }
    ).write_parquet(general_path)

    holdings, ticker, sector, concentration = holdings_and_concentration(
        predictions,
        general_path=general_path,
        top_n=2,
    )

    assert holdings["ticker"].to_list() == ["A", "B", "A", "B"]
    assert ticker["selected_months"].max() == 2
    assert sector["average_monthly_weight"].sum() == 1.0
    assert concentration["unique_tickers"][0] == 2
