from __future__ import annotations

import polars as pl

from alpharank.portfolio.simulation import simulate_weighted_portfolio


def select_top_n(predictions: pl.DataFrame, top_n: int, score_col: str = "prediction") -> pl.DataFrame:
    if predictions.is_empty():
        return predictions.with_columns(pl.lit(None).alias("rank")).head(0)
    if score_col not in predictions.columns:
        raise ValueError(f"Missing score column for selection: {score_col}")

    ranked = predictions.with_columns(
        pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("rank")
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

    holdings = selections.with_columns(
        pl.lit("Portfolio").alias("strategy"),
        (1.0 / pl.len().over("holding_month")).alias("target_weight"),
        pl.col("future_return").alias("realized_return"),
        pl.col("benchmark_future_return")
        .fill_null(pl.col("benchmark_future_return").drop_nulls().first().over("holding_month"))
        .fill_null(0.0)
        .alias("benchmark_return"),
    )
    simulated = simulate_weighted_portfolio(
        holdings.select(
            "strategy",
            "decision_month",
            "holding_month",
            "ticker",
            "target_weight",
            "realized_return",
            "benchmark_return",
        ),
        transaction_cost_bps=0.0,
        causal_timing_policy="legacy_month_only",
    )
    hit_rate = selections.group_by("holding_month").agg(
        pl.mean("target_label").fill_null(0.0).alias("hit_rate")
    )
    monthly = (
        simulated.join(hit_rate, on="holding_month", how="left")
        .with_columns(
            pl.col("net_return").alias("portfolio_return"),
            pl.col("holding_month").alias("year_month"),
        )
        .select(
            "year_month",
            "decision_month",
            "holding_month",
            "portfolio_return",
            "benchmark_return",
            "active_return",
            "hit_rate",
            "n_positions",
        )
        .sort("holding_month")
    )

    return monthly
