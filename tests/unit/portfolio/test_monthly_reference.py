from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.portfolio.daily import (
    DailyFrontierPeriod,
    combine_daily_sleeves,
    daily_period_analysis,
    daily_strategy_frontier,
)
from alpharank.portfolio.daily.monthly_reference import build_monthly_reference_ledger
from alpharank.portfolio.performance import PerformanceCalendar, advanced_performance_statistics


def _reference_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    months = [date(2026, 2, 1), date(2026, 3, 1), date(2026, 4, 1)]
    rows = [
        {
            "strategy": name,
            "holding_month": month,
            "net_return": value,
            "transaction_cost": 0.001 if name != "SPY" else 0.0,
            "turnover": 1.0 if name != "SPY" else 0.0,
            "n_positions": 5 if name != "SPY" else 1,
        }
        for name, values in [
            ("Alpha", [0.1, -0.05, 0.02]),
            ("Beta", [-0.02, 0.08, 0.01]),
            ("Gamma", [0.01, -0.01, 0.04]),
            ("SPY", [0.02, -0.01, 0.03]),
        ]
        for month, value in zip(months, values, strict=True)
    ]
    calendar = pl.DataFrame(
        {
            "strategy": ["SPY"] * 7,
            "valuation_date": [
                date(2026, 1, 30),
                date(2026, 2, 2),
                date(2026, 2, 27),
                date(2026, 3, 2),
                date(2026, 3, 31),
                date(2026, 4, 1),
                date(2026, 4, 30),
            ],
        }
    )
    return pl.from_dicts(rows), calendar


def test_monthly_reference_has_only_sealed_month_end_observations_and_explicit_risk() -> None:
    monthly, calendar = _reference_inputs()
    ledger = build_monthly_reference_ledger(monthly, calendar, benchmark_strategy="SPY")
    alpha = ledger.filter(pl.col("strategy") == "Alpha").sort("valuation_date")
    assert alpha["wealth"].to_list() == pytest.approx([1, 1.1, 1.045, 1.0659])
    assert alpha["valuation_date"].to_list() == [
        date(2026, 1, 30),
        date(2026, 2, 27),
        date(2026, 3, 31),
        date(2026, 4, 30),
    ]
    view = daily_period_analysis(
        ledger,
        start=date(2026, 2, 1),
        end=date(2026, 4, 30),
        benchmark_strategy="SPY",
        frequency="monthly",
    )
    assert view["frequency"] == "monthly" and view["periods_per_year"] == 12
    metrics = view["series"]["Alpha"]["metrics"]
    expected = advanced_performance_statistics(
        [0.1, -0.05, 0.02],
        benchmark_returns=[0.02, -0.01, 0.03],
        calendar=PerformanceCalendar(12, 90 / 365.25, "initial_capital"),
    )
    for key in ["total_return", "annualized_volatility", "max_drawdown", "cagr"]:
        assert metrics[key] == pytest.approx(expected[key])
    assert "positive_month_rate" in metrics and "positive_day_rate" not in metrics


def test_monthly_reference_composer_and_frontier_reuse_the_same_monthly_returns() -> None:
    monthly, calendar = _reference_inputs()
    ledger = build_monthly_reference_ledger(monthly, calendar, benchmark_strategy="SPY")
    mixed = combine_daily_sleeves(ledger, {"Alpha": 0.5, "Beta": 0.5})
    assert mixed["wealth"][-1] == pytest.approx(np.prod(1 + np.array([0.04, 0.015, 0.015])))
    frontier = daily_strategy_frontier(
        ledger,
        ("Alpha", "Beta", "Gamma"),
        period=DailyFrontierPeriod(date(2026, 2, 1), date(2026, 4, 30), "SPY", frequency="monthly"),
    )
    assert len(frontier["points"]) == 168
    assert all(point["annualized_volatility"] is not None for point in frontier["points"])
    assert frontier["recommendations"]


@pytest.mark.parametrize("failure", ["duplicate", "missing_strategy_month", "invalid_return"])
def test_monthly_reference_rejects_unsealed_or_inconsistent_rows(failure: str) -> None:
    monthly, calendar = _reference_inputs()
    if failure == "duplicate":
        monthly = pl.concat([monthly, monthly.head(1)])
    elif failure == "missing_strategy_month":
        monthly = monthly.filter(
            ~((pl.col("strategy") == "Alpha") & (pl.col("holding_month") == date(2026, 3, 1)))
        )
    else:
        monthly = monthly.with_columns(pl.lit(float("nan")).alias("net_return"))
    with pytest.raises(ValueError):
        build_monthly_reference_ledger(monthly, calendar, benchmark_strategy="SPY")
