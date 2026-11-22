"""Prepare a daily window and chart projections using only included observations."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl

from alpharank.portfolio.daily.contracts import (
    OBSERVATIONS_PER_YEAR,
    DailyPeriodView,
    DailyPoint,
    DailySeriesView,
    ObservationFrequency,
)
from alpharank.portfolio.daily.statistics import (
    close_window_metrics,
    daily_year_views,
    finite_metric,
    operation_metrics,
    year_summary,
)


def _window_days(ledger: pl.DataFrame, start: date, end: date) -> tuple[date, ...]:
    if end < start:
        raise ValueError("The end date must not precede the start date.")
    calendar = sorted(ledger["valuation_date"].unique().to_list())
    if not calendar or end < calendar[0] or start > calendar[-1]:
        raise ValueError("The requested dates lie outside the daily source coverage.")
    earlier = [day for day in calendar if day < start]
    anchor = earlier[-1] if earlier else calendar[0]
    days = tuple(day for day in calendar if anchor <= day <= end)
    if len(days) < 2:
        raise ValueError("The selected interval contains no realized trading-session return.")
    return days


def _series_view(
    frame: pl.DataFrame,
    reference: pl.DataFrame,
    *,
    start: date,
    end: date,
    frequency: ObservationFrequency,
) -> DailySeriesView:
    days = tuple(frame["valuation_date"].to_list())
    if days != tuple(reference["valuation_date"].to_list()):
        raise ValueError(
            "Every daily strategy must retain the complete benchmark calendar, including gaps."
        )
    raw = frame["wealth"].to_numpy().astype(float)
    spy_raw = reference["wealth"].to_numpy().astype(float)
    wealth = raw / raw[0]
    benchmark = spy_raw / spy_raw[0]
    metrics = close_window_metrics(raw, spy_raw, days, frequency=frequency)
    metrics.update(operation_metrics(frame.slice(1), (days[-1] - days[0]).days / 365.25))
    annual, cumulative = daily_year_views(frame, spy_raw, start=start, end=end, frequency=frequency)
    metrics.update(year_summary(annual))
    points: list[DailyPoint] = []
    peak = 1.0
    for index, day in enumerate(days):
        value = float(wealth[index])
        peak = max(peak, value) if np.isfinite(value) and np.isfinite(peak) else float("nan")
        points.append(
            {
                "date": day.isoformat(),
                "wealth": finite_metric(value),
                "drawdown": finite_metric(value / peak - 1),
                "relative_wealth": finite_metric(value / benchmark[index]),
            }
        )
    return {
        "points": points,
        "metrics": metrics,
        "annual": annual,
        "cumulative": cumulative,
        "coverage": {
            "observations": len(days) - 1,
            "missing_closes": int(np.count_nonzero(~np.isfinite(raw))),
            "estimated_closes": int(frame["estimated_close"].sum())
            if "estimated_close" in frame.columns
            else 0,
        },
    }


def daily_period_analysis(
    ledger: pl.DataFrame,
    *,
    start: date,
    end: date,
    benchmark_strategy: str,
    frequency: ObservationFrequency = "daily",
) -> DailyPeriodView:
    """Measure a declared observation grid, anchored at the preceding close.

    Closed-market boundaries resolve to observed sessions and are returned
    explicitly. Daily is the default; monthly is reserved for a separately
    captured sealed-return reference. No strategy is dropped to conceal a gap.
    """
    if frequency not in OBSERVATIONS_PER_YEAR:
        raise ValueError("Unknown observation frequency.")
    required = {"strategy", "valuation_date", "wealth"}
    if not required.issubset(ledger.columns) or ledger.schema["valuation_date"] != pl.Date:
        raise ValueError("Daily analysis requires strategy, native valuation_date and wealth.")
    if ledger.select("strategy", "valuation_date").is_duplicated().any():
        raise ValueError("Duplicate strategy/day in the daily ledger.")
    days = _window_days(ledger, start, end)
    available_days = sorted(ledger["valuation_date"].unique().to_list())
    bounded_start = max(start, available_days[0] + timedelta(days=1))
    bounded_end = min(end, available_days[-1])
    selected = ledger.filter(pl.col("valuation_date").is_between(days[0], days[-1])).sort(
        "valuation_date"
    )
    reference = selected.filter(pl.col("strategy") == benchmark_strategy)
    if reference.height != len(days) or reference["wealth"].null_count():
        raise ValueError("The benchmark lacks a required observed daily close.")
    series = {
        str(key[0]): _series_view(
            frame, reference, start=bounded_start, end=bounded_end, frequency=frequency
        )
        for key, frame in selected.partition_by("strategy", as_dict=True).items()
    }
    return {
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "start_date": start.isoformat() if frequency == "monthly" else days[1].isoformat(),
        "end_date": days[-1].isoformat(),
        "anchor_date": days[0].isoformat(),
        "frequency": frequency,
        "periods_per_year": OBSERVATIONS_PER_YEAR[frequency],
        "series": series,
    }
