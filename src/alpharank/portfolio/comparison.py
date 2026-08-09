from __future__ import annotations

from collections.abc import Mapping

import polars as pl


def align_return_series(
    series: Mapping[str, pl.DataFrame],
    *,
    month_column: str = "holding_month",
    return_column: str = "net_return",
    how: str = "inner",
) -> pl.DataFrame:
    """Align named monthly return series on one explicit calendar."""

    if how not in {"inner", "full"}:
        raise ValueError("how must be 'inner' or 'full'.")
    aligned: pl.DataFrame | None = None
    for name, frame in series.items():
        missing = [column for column in (month_column, return_column) if column not in frame.columns]
        if missing:
            raise ValueError(f"Series {name!r} is missing columns: {', '.join(missing)}")
        current = (
            frame.select(
                pl.col(month_column).cast(pl.Date),
                pl.col(return_column).cast(pl.Float64).alias(name),
            )
            .unique(month_column)
            .sort(month_column)
        )
        aligned = current if aligned is None else aligned.join(current, on=month_column, how=how)
    if aligned is None:
        return pl.DataFrame(schema={month_column: pl.Date})
    return aligned.sort(month_column)


def reference_monthly_series(
    source: pl.DataFrame,
    *,
    strategy: str,
    return_column: str,
    benchmark_column: str = "benchmark_return",
) -> pl.DataFrame:
    """Represent Legacy or SPY as a canonical monthly comparison series."""

    required = {"decision_month", "holding_month", return_column, benchmark_column}
    missing = sorted(required - set(source.columns))
    if missing:
        raise ValueError(f"Reference source is missing: {', '.join(missing)}")
    return (
        source.select(
            "decision_month",
            "holding_month",
            pl.col(return_column).cast(pl.Float64).alias("net_return"),
            pl.col(benchmark_column).cast(pl.Float64).alias("benchmark_return"),
        )
        .unique("holding_month")
        .with_columns(
            pl.lit(strategy).alias("strategy"),
            pl.col("net_return").alias("gross_return"),
            pl.lit(0.0).alias("turnover"),
            pl.lit(0.0).alias("transaction_cost"),
            (pl.col("net_return") - pl.col("benchmark_return")).alias("active_return"),
            (
                (1.0 + pl.col("net_return")) / (1.0 + pl.col("benchmark_return")) - 1.0
            ).alias("relative_return"),
            pl.lit(0).cast(pl.Int64).alias("n_positions"),
            pl.lit(0.0).alias("maximum_position_weight"),
            pl.lit(0.0).alias("maximum_sector_weight"),
            pl.lit(0).cast(pl.Int64).alias("sector_count"),
        )
        .select(
            "strategy",
            "decision_month",
            "holding_month",
            "gross_return",
            "turnover",
            "transaction_cost",
            "net_return",
            "benchmark_return",
            "active_return",
            "relative_return",
            "n_positions",
            "maximum_position_weight",
            "maximum_sector_weight",
            "sector_count",
        )
        .sort("holding_month")
    )
