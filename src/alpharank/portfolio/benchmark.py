from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import polars as pl


@dataclass(frozen=True)
class BenchmarkConvention:
    identifier: str
    label: str
    price_column: str
    includes_distributions: bool


SPY_TOTAL_RETURN = BenchmarkConvention(
    identifier="spy_total_return_adjusted_close",
    label="SPY total return",
    price_column="adjusted_close",
    includes_distributions=True,
)
SPY_PRICE_RETURN = BenchmarkConvention(
    identifier="spy_price_return_close",
    label="SPY price return",
    price_column="close",
    includes_distributions=False,
)
BENCHMARK_CONVENTIONS = {
    convention.identifier: convention
    for convention in (SPY_TOTAL_RETURN, SPY_PRICE_RETURN)
}


def benchmark_convention(identifier: str) -> BenchmarkConvention:
    try:
        return BENCHMARK_CONVENTIONS[identifier]
    except KeyError as exc:
        raise ValueError(
            f"Unknown benchmark convention {identifier!r}; expected one of "
            f"{sorted(BENCHMARK_CONVENTIONS)}."
        ) from exc


def completed_through_month(
    prices: pl.DataFrame,
    *,
    date_column: str = "date",
) -> date:
    if date_column not in prices.columns:
        raise ValueError(f"Benchmark prices are missing: {date_column}")
    maximum = prices.select(pl.col(date_column).cast(pl.Date, strict=False).max()).item()
    if maximum is None:
        raise ValueError("Benchmark prices contain no valid date.")
    return date(
        maximum.year - (maximum.month == 1),
        12 if maximum.month == 1 else maximum.month - 1,
        1,
    )


def monthly_benchmark_returns(
    prices: pl.DataFrame,
    *,
    convention: BenchmarkConvention = SPY_TOTAL_RETURN,
    date_column: str = "date",
) -> pl.DataFrame:
    """Build explicit month-end benchmark returns from one price series."""

    required = {date_column, convention.price_column}
    missing = sorted(required - set(prices.columns))
    if missing:
        raise ValueError(f"Benchmark prices are missing: {', '.join(missing)}")
    if "ticker" in prices.columns and prices["ticker"].drop_nulls().n_unique() > 1:
        raise ValueError("Benchmark prices must contain at most one ticker.")
    monthly = (
        prices.select(
            pl.col(date_column).cast(pl.Date, strict=False).alias("date"),
            pl.col(convention.price_column).cast(pl.Float64).alias("price"),
        )
        .filter(pl.col("date").is_not_null() & pl.col("price").is_not_null())
        .with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
        .sort("date")
        .group_by("year_month", maintain_order=True)
        .agg(
            pl.col("price").last().alias("month_end_price"),
            pl.col("date").last().alias("month_end_date"),
        )
        .sort("year_month")
        .with_columns(
            pl.col("month_end_price").pct_change().alias("monthly_return"),
            pl.lit(convention.identifier).alias("benchmark_id"),
            pl.lit(convention.label).alias("benchmark_label"),
            pl.lit(convention.price_column).alias("benchmark_price_column"),
            pl.lit(convention.includes_distributions).alias(
                "benchmark_includes_distributions"
            ),
        )
    )
    return monthly
