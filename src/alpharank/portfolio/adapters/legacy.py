from __future__ import annotations

import polars as pl


def legacy_detailed_to_holdings(
    detailed: pl.DataFrame,
    *,
    strategy: str,
    benchmark_monthly: pl.DataFrame,
    holding_month_column: str = "year_month",
    realized_return_column: str = "dr",
    weight_column: str = "weight_normalized",
    benchmark_month_column: str = "year_month",
    benchmark_return_column: str = "monthly_return",
) -> pl.DataFrame:
    """Adapt a finalized Legacy basket to the common holdings contract."""

    required = {holding_month_column, "ticker", realized_return_column, weight_column}
    missing = sorted(required - set(detailed.columns))
    if missing:
        raise ValueError(f"Legacy detailed frame is missing: {', '.join(missing)}")
    benchmark = benchmark_monthly.select(
        pl.col(benchmark_month_column).cast(pl.Date).alias("holding_month"),
        pl.col(benchmark_return_column).cast(pl.Float64).alias("benchmark_return"),
    ).unique("holding_month")
    result = (
        detailed.select(
            pl.col(holding_month_column).cast(pl.Date).alias("holding_month"),
            pl.col("ticker").cast(pl.Utf8),
            pl.col(realized_return_column).cast(pl.Float64).alias("realized_return"),
            pl.col(weight_column).cast(pl.Float64).alias("target_weight"),
            *(
                [pl.col("Sector").cast(pl.Utf8).alias("sector")]
                if "Sector" in detailed.columns
                else []
            ),
            *(
                [pl.col("n_models")]
                if "n_models" in detailed.columns
                else []
            ),
        )
        .with_columns(
            pl.col("holding_month").dt.offset_by("-1mo").alias("decision_month"),
            pl.lit(strategy).alias("strategy"),
        )
        .join(benchmark, on="holding_month", how="left")
        .with_columns(
            (
                pl.col("target_weight")
                / pl.col("target_weight").sum().over(["strategy", "holding_month"])
            ).alias("target_weight")
        )
        .sort(["strategy", "decision_month", "ticker"])
    )
    return result
