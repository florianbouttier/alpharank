"""Observe sealed monthly portfolio returns without imputing any daily security price."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import polars as pl


def _validate_reference(monthly: pl.DataFrame, benchmark_strategy: str) -> None:
    required = {"strategy", "holding_month", "net_return", "transaction_cost"}
    if not required.issubset(monthly.columns) or monthly.schema["holding_month"] != pl.Date:
        raise ValueError(
            "The monthly reference requires native months and sealed portfolio returns."
        )
    if monthly.is_empty() or monthly.select("strategy", "holding_month").is_duplicated().any():
        raise ValueError("The monthly reference requires non-empty unique strategy/month keys.")
    if any(monthly[name].null_count() for name in required):
        raise ValueError("The sealed monthly reference cannot contain null keys, returns or costs.")
    if monthly.filter(pl.col("holding_month").dt.day() != 1).height:
        raise ValueError("Sealed holding months must use their first calendar day.")
    for field in ["net_return", "transaction_cost"]:
        if (
            not monthly.schema[field].is_numeric()
            or not np.isfinite(monthly[field].to_numpy()).all()
        ):
            raise ValueError("Non-finite sealed monthly field: " + field)
    if monthly.filter((pl.col("net_return") <= -1) | (pl.col("transaction_cost") < 0)).height:
        raise ValueError("Monthly reference wealth must be positive and costs non-negative.")
    reference = monthly.filter(pl.col("strategy") == benchmark_strategy).sort("holding_month")
    months = reference["holding_month"].to_list()
    if not months:
        raise ValueError("The monthly reference benchmark is missing.")
    expected = 12 * (months[-1].year - months[0].year) + months[-1].month - months[0].month + 1
    if len(months) != expected or any(
        frame.sort("holding_month")["holding_month"].to_list() != months
        for frame in monthly.partition_by("strategy")
    ):
        raise ValueError("Every reference strategy must retain the full shared monthly calendar.")


def build_monthly_reference_ledger(
    monthly: pl.DataFrame, daily_calendar: pl.DataFrame, *, benchmark_strategy: str
) -> pl.DataFrame:
    """Keep only actual month-end dates, with wealth compounded from sealed net returns.

    This research-only fallback inherits archived terminal-return assumptions.
    It does not create daily prices, fix a shareholder consideration, rerank a
    portfolio or claim that monthly risk measures the unobserved daily path.
    """
    _validate_reference(monthly, benchmark_strategy)
    if (
        not {"strategy", "valuation_date"}.issubset(daily_calendar.columns)
        or daily_calendar.schema["valuation_date"] != pl.Date
    ):
        raise ValueError("The monthly reference needs the explicit benchmark session calendar.")
    calendar = (
        daily_calendar.filter(pl.col("strategy") == benchmark_strategy)
        .select("valuation_date")
        .with_columns(pl.col("valuation_date").dt.truncate("1mo").alias("holding_month"))
        .group_by("holding_month")
        .agg(pl.col("valuation_date").max())
    )
    dates = dict(calendar.select("holding_month", "valuation_date").iter_rows())
    first_month = monthly["holding_month"].min()
    if not isinstance(first_month, date):
        raise ValueError("The monthly reference requires at least one holding month.")
    previous_month = (first_month - timedelta(days=1)).replace(day=1)
    if previous_month not in dates:
        raise ValueError("The benchmark calendar lacks the initial monthly anchor.")
    operations = [
        name
        for name in ["turnover", "n_positions", "maximum_position_weight", "maximum_sector_weight"]
        if name in monthly.columns
    ]
    rows = (
        monthly.select("strategy", "holding_month", "net_return", "transaction_cost", *operations)
        .with_columns(
            pl.col("net_return").cast(pl.Float64), pl.col("transaction_cost").cast(pl.Float64)
        )
        .join(calendar, on="holding_month", how="left", validate="m:1")
        .sort("strategy", "holding_month")
        .with_columns(
            (1 + pl.col("net_return")).alias("month_factor"),
            pl.col("net_return").alias("sealed_monthly_return"),
        )
        .with_columns(pl.col("month_factor").cum_prod().over("strategy").alias("wealth"))
        .drop("net_return")
    )
    if rows["valuation_date"].null_count():
        raise ValueError("The benchmark calendar lacks a sealed monthly endpoint.")
    anchors = (
        rows.group_by("strategy", maintain_order=True)
        .first()
        .with_columns(
            pl.lit(dates[previous_month]).alias("valuation_date"),
            pl.lit(1.0).alias("month_factor"),
            pl.lit(1.0).alias("wealth"),
            *[
                pl.lit(0.0).alias(name)
                for name in ["transaction_cost", "turnover"]
                if name in rows.columns
            ],
        )
    )
    return pl.concat([anchors, rows]).sort("strategy", "valuation_date")
