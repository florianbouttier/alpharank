from __future__ import annotations

import polars as pl


def select_top_n(predictions: pl.DataFrame, top_n: int) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions.with_columns(pl.lit(None).alias("rank")).head(0)

    ranked = predictions.with_columns(
        pl.col("prediction").rank(method="ordinal", descending=True).over("year_month").alias("rank")
    )
    return ranked.filter(pl.col("rank") <= pl.lit(top_n)).sort(["year_month", "rank"])


def compute_monthly_portfolio_returns(selections: pl.DataFrame) -> pl.DataFrame:
    if selections.is_empty():
        return pl.DataFrame(
            schema={
                "year_month": pl.Date,
                "decision_month": pl.Date,
                "holding_month": pl.Date,
                "portfolio_return": pl.Float64,
                "benchmark_return": pl.Float64,
                "active_return": pl.Float64,
                "hit_rate": pl.Float64,
                "n_positions": pl.Int64,
            }
        )

    monthly = (
        selections.group_by("holding_month")
        .agg(
            pl.col("decision_month").min().alias("decision_month"),
            pl.mean("future_return").alias("portfolio_return"),
            pl.mean("benchmark_future_return").alias("benchmark_return"),
            pl.mean("target_label").alias("hit_rate"),
            pl.len().alias("n_positions"),
        )
        .sort("holding_month")
        .with_columns(
            pl.col("benchmark_return").fill_null(0.0).alias("benchmark_return"),
            pl.col("hit_rate").fill_null(0.0).alias("hit_rate"),
        )
        .with_columns((pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return"))
        .with_columns(pl.col("holding_month").alias("year_month"))
        .select(
            [
                "year_month",
                "decision_month",
                "holding_month",
                "portfolio_return",
                "benchmark_return",
                "active_return",
                "hit_rate",
                "n_positions",
            ]
        )
    )

    return monthly
