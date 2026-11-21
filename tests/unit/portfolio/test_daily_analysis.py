from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.daily import (
    DailyFrontierPeriod,
    build_daily_ledger,
    combine_daily_sleeves,
    daily_period_analysis,
    daily_strategy_frontier,
)
from alpharank.portfolio.performance import PerformanceCalendar, advanced_performance_statistics


def _inputs() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    days = [date(2026, 1, 30), date(2026, 2, 2), date(2026, 2, 3), date(2026, 2, 27)]
    prices = pl.DataFrame(
        {
            "ticker": ["A", "A", "A", "A", "B", "B", "B", "B"],
            "date": days * 2,
            "adjusted_close": [100.0, 120.0, 90.0, 150.0, 100.0, 80.0, 120.0, 100.0],
        }
    )
    holdings = pl.DataFrame(
        {
            "strategy": ["Alpha", "Alpha"],
            "ticker": ["A", "B"],
            "holding_month": [date(2026, 2, 1)] * 2,
            "decision_month": [date(2026, 1, 1)] * 2,
            "target_weight": [0.5, 0.5],
            "realized_return": [0.5, 0.0],
        }
    )
    monthly = pl.DataFrame(
        {
            "strategy": ["Alpha", "SPY"],
            "holding_month": [date(2026, 2, 1)] * 2,
            "net_return": [0.249, 0.1],
            "transaction_cost": [0.001, 0.0],
            "turnover": [1.0, 0.0],
            "n_positions": [2, 1],
        }
    )
    benchmark = pl.DataFrame({"date": days, "adjusted_close": [100.0, 101.0, 99.0, 110.0]})
    return (
        holdings,
        monthly,
        pl.concat([prices, benchmark.with_columns(pl.lit("SPY").alias("ticker"))], how="diagonal"),
    )


def test_daily_valuation_preserves_monthly_endpoint_cost_and_intra_month_drift() -> None:
    holdings, monthly, prices = _inputs()
    result = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
    alpha = result.ledger.filter(pl.col("strategy") == "Alpha").sort("valuation_date")
    assert alpha["wealth"].to_list() == pytest.approx([1.0, 0.999, 1.049, 1.249])
    assert result.monthly_reconciliation["absolute_error"].max() < 1e-12
    assert result.coverage["missing_ticker_dates"] == 0


def test_daily_dates_are_free_and_use_the_close_before_the_first_included_return() -> None:
    holdings, monthly, prices = _inputs()
    result = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
    view = daily_period_analysis(
        result.ledger, start=date(2026, 2, 3), end=date(2026, 2, 28), benchmark_strategy="SPY"
    )
    assert view["anchor_date"] == "2026-02-02"
    assert view["end_date"] == "2026-02-27"
    row = view["series"]["Alpha"]
    assert row["points"][0]["wealth"] == 1.0
    assert row["metrics"]["total_return"] == pytest.approx(1.249 / 0.999 - 1.0)
    assert row["points"][0]["relative_wealth"] == 1.0


def test_missing_daily_prices_remain_gaps_and_do_not_produce_fictitious_risk() -> None:
    holdings, monthly, prices = _inputs()
    prices = prices.filter(~((pl.col("ticker") == "A") & (pl.col("date") == date(2026, 2, 3))))
    result = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
    view = daily_period_analysis(
        result.ledger, start=date(2026, 2, 1), end=date(2026, 2, 28), benchmark_strategy="SPY"
    )
    row = view["series"]["Alpha"]
    assert row["metrics"]["total_return"] == pytest.approx(0.249)
    assert row["metrics"]["annualized_volatility"] is None
    assert row["metrics"]["max_drawdown"] is None
    assert row["points"][2]["wealth"] is None
    assert row["coverage"]["missing_closes"] == 1


def test_monthly_return_mismatch_fails_instead_of_rescaling_daily_prices() -> None:
    holdings, monthly, prices = _inputs()
    monthly = monthly.with_columns(
        pl.when(pl.col("strategy") == "Alpha")
        .then(0.3)
        .otherwise(pl.col("net_return"))
        .alias("net_return")
    )
    with pytest.raises(ValueError, match="monthly"):
        build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")


def test_later_prices_do_not_change_an_earlier_selected_daily_period() -> None:
    holdings, monthly, prices = _inputs()
    baseline = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
    mutated = baseline.ledger.with_columns(
        pl.when(pl.col("valuation_date") > date(2026, 2, 3))
        .then(pl.col("wealth") * 2)
        .otherwise(pl.col("wealth"))
        .alias("wealth")
    )
    baseline_view = daily_period_analysis(
        baseline.ledger, start=date(2026, 2, 1), end=date(2026, 2, 3), benchmark_strategy="SPY"
    )
    changed_view = daily_period_analysis(
        mutated, start=date(2026, 2, 1), end=date(2026, 2, 3), benchmark_strategy="SPY"
    )
    assert baseline_view == changed_view


def test_daily_annualization_uses_252_and_initial_capital_without_changing_monthly_default() -> (
    None
):
    values = [-0.1, 0.05, 0.02]
    monthly = advanced_performance_statistics(values, benchmark_returns=values)
    daily = advanced_performance_statistics(
        values,
        benchmark_returns=values,
        calendar=PerformanceCalendar(252, 30 / 365.25, "initial_capital"),
    )
    assert daily["total_return"] == monthly["total_return"]
    assert daily["annualized_volatility"] == pytest.approx(
        monthly["annualized_volatility"] * (252 / 12) ** 0.5
    )
    assert daily["max_drawdown"] == pytest.approx(-0.1)
    assert monthly["max_drawdown"] == 0.0


def test_calendar_year_is_complete_when_january_first_is_a_closed_market_day() -> None:
    days = [date(2024, 12, 31), date(2025, 1, 2), date(2025, 12, 31), date(2026, 1, 2)]
    ledger = pl.DataFrame(
        {
            "strategy": ["Alpha"] * 4 + ["SPY"] * 4,
            "valuation_date": days * 2,
            "wealth": [1.0, 1.1, 1.2, 1.3, 1.0, 1.05, 1.1, 1.15],
        }
    )
    view = daily_period_analysis(
        ledger, start=date(2025, 1, 1), end=date(2025, 12, 31), benchmark_strategy="SPY"
    )
    assert view["series"]["Alpha"]["metrics"]["complete_years"] == 1.0
    assert view["series"]["Alpha"]["metrics"]["years_beating_benchmark"] == 1.0


def test_daily_composer_uses_monthly_sleeves_not_daily_rebalanced_returns() -> None:
    holdings, monthly, prices = _inputs()
    ledger = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY").ledger
    mix = combine_daily_sleeves(ledger, {"Alpha": 0.5, "SPY": 0.5})
    assert mix["wealth"].to_list() == pytest.approx([1.0, 1.0045, 1.0195, 1.1745])


def test_zero_weight_sleeve_does_not_introduce_a_spurious_missing_price() -> None:
    holdings, monthly, prices = _inputs()
    prices = prices.filter(~((pl.col("ticker") == "A") & (pl.col("date") == date(2026, 2, 3))))
    ledger = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY").ledger
    mix = combine_daily_sleeves(ledger, {"Alpha": 0.0, "SPY": 1.0})
    assert mix["wealth"].to_list() == pytest.approx([1.0, 1.01, 0.99, 1.1])


def test_daily_frontier_retains_the_canonical_constraints_and_exact_date_window() -> None:
    holdings, monthly, prices = _inputs()
    ledger = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY").ledger
    beta = ledger.filter(pl.col("strategy") == "Alpha").with_columns(
        pl.lit("Beta").alias("strategy")
    )
    gamma = ledger.filter(pl.col("strategy") == "Alpha").with_columns(
        pl.lit("Gamma").alias("strategy")
    )
    ledger = pl.concat([ledger, beta, gamma])
    result = daily_strategy_frontier(
        ledger,
        ("Alpha", "Beta", "Gamma"),
        period=DailyFrontierPeriod(date(2026, 2, 3), date(2026, 2, 28), "SPY"),
    )
    assert len(result["points"]) == 168
    assert result["recommendations"]


def test_missing_month_end_is_not_fabricated_from_the_sealed_monthly_return() -> None:
    holdings, monthly, prices = _inputs()
    prices = prices.filter(~((pl.col("ticker") == "A") & (pl.col("date") == date(2026, 2, 27))))
    result = build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
    alpha = result.ledger.filter(pl.col("strategy") == "Alpha")
    assert alpha["wealth"][-1] is None
    assert (
        result.monthly_reconciliation.filter(pl.col("strategy") == "Alpha")["status"][0]
        == "missing_daily_endpoint"
    )
    view = daily_period_analysis(
        result.ledger, start=date(2026, 2, 1), end=date(2026, 2, 28), benchmark_strategy="SPY"
    )
    assert view["series"]["Alpha"]["metrics"]["total_return"] is None


@pytest.mark.parametrize("failure", ["duplicate", "weight", "negative_price"])
def test_daily_valuation_rejects_invalid_selected_inputs(failure: str) -> None:
    holdings, monthly, prices = _inputs()
    if failure == "duplicate":
        prices = pl.concat([prices, prices.head(1)])
    elif failure == "weight":
        holdings = holdings.with_columns(pl.lit(0.4).alias("target_weight"))
    else:
        prices = prices.with_columns(
            pl.when(pl.col("ticker") == "A")
            .then(-1.0)
            .otherwise(pl.col("adjusted_close"))
            .alias("adjusted_close")
        )
    with pytest.raises(ValueError):
        build_daily_ledger(holdings, monthly, prices, benchmark_ticker="SPY")
