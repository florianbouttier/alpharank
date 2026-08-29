from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.comparison import (
    subperiod_metric_grid,
    subperiod_portfolio_metric_grid,
)
from alpharank.portfolio.performance import (
    advanced_performance_statistics,
    portfolio_period_statistics,
)


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


def test_portfolio_period_grid_centralizes_performance_cost_and_composition_kpi() -> None:
    months = [date(2025, 1, 1), date(2025, 2, 1), date(2025, 3, 1)]
    strategy = _portfolio_monthly(
        months,
        returns=[0.01, -0.02, 0.03],
        turnovers=[0.5, 0.2, 0.4],
        costs=[0.0005, 0.0002, 0.0004],
        positions=[5, 6, 7],
    )
    benchmark = _portfolio_monthly(
        months,
        returns=[0.005, -0.01, 0.01],
        turnovers=[0.0, 0.0, 0.0],
        costs=[0.0, 0.0, 0.0],
        positions=[0, 0, 0],
    )
    fields = (
        "cagr",
        "max_drawdown",
        "annualized_turnover",
        "total_transaction_cost",
        "average_positions",
    )

    grid = subperiod_portfolio_metric_grid(
        {"strategy": strategy, "SPY": benchmark},
        benchmark_strategy="SPY",
        strategy_order=("strategy", "SPY"),
        metric_fields=fields,
    )
    expected = portfolio_period_statistics(
        [0.01, -0.02, 0.03],
        benchmark_returns=[0.005, -0.01, 0.01],
        turnovers=[0.5, 0.2, 0.4],
        transaction_costs=[0.0005, 0.0002, 0.0004],
        position_counts=[5, 6, 7],
        maximum_position_weights=[0.2, 0.2, 0.2],
        maximum_sector_weights=[0.4, 0.4, 0.4],
    )

    assert len(grid) == 6
    assert grid["2025-01-01|2025-03-01"][0] == pytest.approx([expected[field] for field in fields])

    _assert_year_boundary_grid(fields)


def _assert_year_boundary_grid(fields: tuple[str, ...]) -> None:
    """Prove that the cube contains cumulative and isolated annual windows."""

    boundary_months = [
        date(2024, 8, 1),
        date(2024, 12, 1),
        date(2025, 1, 1),
        date(2025, 12, 1),
        date(2026, 1, 1),
        date(2026, 7, 1),
    ]
    boundary_strategy = _portfolio_monthly(
        boundary_months,
        returns=[0.01] * 6,
        turnovers=[0.2] * 6,
        costs=[0.0002] * 6,
        positions=[5] * 6,
    )
    boundary_benchmark = _portfolio_monthly(
        boundary_months,
        returns=[0.005] * 6,
        turnovers=[0.0] * 6,
        costs=[0.0] * 6,
        positions=[0] * 6,
    )
    annual_grid = subperiod_portfolio_metric_grid(
        {"strategy": boundary_strategy, "SPY": boundary_benchmark},
        benchmark_strategy="SPY",
        strategy_order=("strategy", "SPY"),
        metric_fields=fields,
        calendar_year_boundaries_only=True,
    )

    assert set(annual_grid) == {
        "2024-08-01|2024-12-01",
        "2024-08-01|2025-12-01",
        "2024-08-01|2026-07-01",
        "2025-01-01|2025-12-01",
        "2025-01-01|2026-07-01",
        "2026-01-01|2026-07-01",
    }


def _portfolio_monthly(
    months: list[date],
    *,
    returns: list[float],
    turnovers: list[float],
    costs: list[float],
    positions: list[int],
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "holding_month": months,
            "net_return": returns,
            "turnover": turnovers,
            "transaction_cost": costs,
            "n_positions": positions,
            "maximum_position_weight": [0.2] * len(months),
            "maximum_sector_weight": [0.4] * len(months),
        }
    )
