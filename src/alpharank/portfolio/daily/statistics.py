"""Source-backed daily statistics on a single explicitly bounded close calendar."""

from __future__ import annotations

import math
from datetime import date

import numpy as np
import polars as pl
from numpy.typing import NDArray

from alpharank.portfolio.daily.contracts import OBSERVATIONS_PER_YEAR, ObservationFrequency
from alpharank.portfolio.performance import (
    ADVANCED_METRIC_FIELDS,
    PerformanceCalendar,
    advanced_performance_statistics,
    performance_statistics,
)


def finite_metric(value: float) -> float | None:
    """Serialize an undefined metric as null, never zero or a JSON NaN."""
    return float(value) if math.isfinite(value) else None


def close_window_metrics(
    wealth: NDArray[np.float64],
    benchmark: NDArray[np.float64],
    days: tuple[date, ...],
    *,
    frequency: ObservationFrequency = "daily",
) -> dict[str, float | None]:
    """Risk requires the complete declared grid; endpoint growth may survive interior gaps."""
    years = (days[-1] - days[0]).days / 365.25
    calendar = PerformanceCalendar(OBSERVATIONS_PER_YEAR[frequency], years, "initial_capital")
    if np.isfinite(wealth).all() and np.isfinite(benchmark).all():
        result = advanced_performance_statistics(
            wealth[1:] / wealth[:-1] - 1,
            benchmark_returns=benchmark[1:] / benchmark[:-1] - 1,
            calendar=calendar,
        )
        if len(wealth) < 3:
            result["annualized_volatility"] = float("nan")
    else:
        names = (
            *ADVANCED_METRIC_FIELDS,
            "total_return",
            "cagr",
            "annualized_volatility",
            "sharpe",
            "max_drawdown",
            "positive_month_rate",
        )
        result = dict.fromkeys(names, float("nan"))
        if np.isfinite(wealth[[0, -1]]).all():
            growth = performance_statistics(
                [wealth[-1] / wealth[0] - 1], calendar=calendar, sharpe_convention="legacy"
            )
            result.update({key: growth[key] for key in ["total_return", "cagr"]})
    if frequency == "daily":
        result["positive_day_rate"] = result.pop("positive_month_rate")
    return {key: finite_metric(value) for key, value in result.items()}


def operation_metrics(frame: pl.DataFrame, years: float) -> dict[str, float | None]:
    """Use actual included fee/rebalance dates; no synthetic initial turnover."""
    result: dict[str, float | None] = {}
    for column, total_key, annual_key in [
        ("turnover", "total_turnover", "annualized_turnover"),
        ("transaction_cost", "total_transaction_cost", "annualized_transaction_cost"),
    ]:
        if column in frame.columns:
            total = float(frame[column].sum())
            result.update({total_key: total, annual_key: total / years})
    months = frame["holding_month"].n_unique() if "holding_month" in frame.columns else 0
    turnover = result.get("total_turnover")
    result["average_monthly_turnover"] = (
        turnover / months if turnover is not None and months else None
    )
    for column, prefix in [
        ("n_positions", "positions"),
        ("maximum_position_weight", "maximum_position_weight"),
        ("maximum_sector_weight", "maximum_sector_weight"),
    ]:
        if column not in frame.columns or frame[column].null_count() or frame.is_empty():
            average = minimum = maximum = None
        else:
            values = frame[column].to_numpy().astype(float)
            average = float(np.mean(values))
            minimum = float(np.min(values))
            maximum = float(np.max(values))
        if prefix == "positions":
            result.update(
                average_positions=average, minimum_positions=minimum, maximum_positions=maximum
            )
        else:
            result["average_" + prefix] = average
            result[
                "maximum_single_name_weight"
                if column == "maximum_position_weight"
                else "maximum_sector_weight"
            ] = maximum
    return result


def daily_year_views(
    frame: pl.DataFrame,
    benchmark: NDArray[np.float64],
    *,
    start: date,
    end: date,
    frequency: ObservationFrequency = "daily",
) -> tuple[list[dict[str, float | int | bool | None]], list[dict[str, float | int | None]]]:
    """Calendar-year and cumulative cards use the exact active interval, never a full-year substitute."""
    days = tuple(frame["valuation_date"].to_list())
    wealth = frame["wealth"].to_numpy().astype(float)
    annual: list[dict[str, float | int | bool | None]] = []
    cumulative: list[dict[str, float | int | None]] = []
    for year in range(start.year, end.year + 1):
        indices = [i for i, day in enumerate(days) if day.year == year and day >= start and i > 0]
        if not indices:
            continue
        first, last = indices[0] - 1, indices[-1]
        metrics = close_window_metrics(
            wealth[first : last + 1],
            benchmark[first : last + 1],
            days[first : last + 1],
            frequency=frequency,
        )
        spy = close_window_metrics(
            benchmark[first : last + 1],
            benchmark[first : last + 1],
            days[first : last + 1],
            frequency=frequency,
        )
        total, reference = metrics["total_return"], spy["total_return"]
        annual.append(
            {
                "year": year,
                "return": total,
                "benchmark_return": reference,
                "excess_return": total - reference
                if total is not None and reference is not None
                else None,
                "partial": start > date(year, 1, 1) or end < date(year, 12, 31),
                **{key: metrics[key] for key in ["cagr", "annualized_volatility", "max_drawdown"]},
            }
        )
        following = close_window_metrics(
            wealth[first:], benchmark[first:], days[first:], frequency=frequency
        )
        cumulative.append(
            {
                "year": year,
                **{
                    key: following[key] for key in ["cagr", "annualized_volatility", "max_drawdown"]
                },
            }
        )
    return annual, cumulative


def year_summary(annual: list[dict[str, float | int | bool | None]]) -> dict[str, float | None]:
    """Partial years are excluded, but missing complete years invalidate comparisons."""
    returns, excess = [], []
    complete = [row for row in annual if not row["partial"]]
    for row in complete:
        value, benchmark = row["return"], row["benchmark_return"]
        if (
            isinstance(value, (int, float))
            and isinstance(benchmark, (int, float))
            and math.isfinite(value)
            and math.isfinite(benchmark)
        ):
            returns.append(float(value))
            excess.append(float(value - benchmark))
    available = bool(returns) and len(returns) == len(complete)
    return {
        "complete_years": float(len(complete)),
        "unavailable_years": float(len(complete) - len(returns)),
        "years_beating_benchmark": float(sum(value > 0 for value in excess)) if available else None,
        "worst_year_return": min(returns) if available else None,
        "worst_year_excess": min(excess) if available else None,
    }
