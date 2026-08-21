from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.comparison import subperiod_metric_grid
from alpharank.portfolio.performance import advanced_performance_statistics


def test_subperiod_grid_uses_the_shared_performance_engine() -> None:
    monthly = pl.DataFrame(
        {
            "holding_month": [
                date(2025, 3, 1),
                date(2025, 1, 1),
                date(2025, 2, 1),
            ],
            "strategy": [0.03, 0.01, -0.02],
            "benchmark": [0.01, 0.005, -0.01],
        }
    )
    fields = ("total_return", "cagr", "max_drawdown")

    grid = subperiod_metric_grid(
        monthly,
        strategy_columns=("strategy", "benchmark"),
        benchmark_column="benchmark",
        metric_fields=fields,
    )
    expected = advanced_performance_statistics(
        [0.01, -0.02, 0.03],
        benchmark_returns=[0.005, -0.01, 0.01],
    )

    assert len(grid) == 6
    full_period = grid["2025-01-01|2025-03-01"][0]
    assert full_period == pytest.approx([expected[field] for field in fields])


def test_subperiod_grid_rejects_an_incomplete_comparison_contract() -> None:
    monthly = pl.DataFrame(
        {
            "holding_month": [date(2025, 1, 1)],
            "strategy": [0.01],
        }
    )

    with pytest.raises(
        ValueError,
        match="Comparison grid is missing columns: benchmark",
    ):
        subperiod_metric_grid(
            monthly,
            strategy_columns=("strategy",),
            benchmark_column="benchmark",
            metric_fields=("cagr",),
        )
