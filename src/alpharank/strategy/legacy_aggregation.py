"""Portfolio aggregation owned by the Legacy strategy."""

from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd

from alpharank.data.processing import IndexDataManager
from alpharank.portfolio.adapters.legacy import legacy_detailed_to_holdings
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.utils.frame_backend import (
    Backend,
    ensure_backend_name,
    normalize_year_month_to_period,
    normalize_year_month_to_timestamp,
    require_polars,
    to_pandas,
    to_polars,
)

try:
    import polars as pl
except ImportError:  # pragma: no cover - optional dependency
    pl = None


def aggregate_portfolios(
    optuna_outputs: List[Dict[str, Any]],
    mode: str = "equal",
    index: IndexDataManager | None = None,
    union_mode: bool = True,
    backend: Backend = "polars",
) -> Dict[str, pd.DataFrame]:
    """Aggregate multiple Legacy portfolio outputs into one portfolio."""

    if not optuna_outputs:
        raise ValueError("optuna_outputs list cannot be empty")
    if mode not in ["equal", "frequency"]:
        raise ValueError(f"mode must be 'equal' or 'frequency', got '{mode}'")

    backend_name = ensure_backend_name(backend, default="polars")
    n_models = len(optuna_outputs)
    if backend_name != "polars":
        raise ValueError("Pandas backend is disabled for StrategyLearner.aggregate_portfolios.")

    require_polars()
    all_detailed_pl = []
    for model_index, output in enumerate(optuna_outputs):
        detailed_key = "detailed" if "detailed" in output else "detailled"
        df = output[detailed_key]
        if isinstance(df, pd.DataFrame):
            pldf = to_polars(normalize_year_month_to_timestamp(df, col="year_month"))
        else:
            pldf = df.clone()
            if "year_month" in pldf.columns:
                ym_dtype = pldf.schema.get("year_month")
                if ym_dtype == pl.Date:
                    pass
                elif ym_dtype == pl.Datetime:
                    pldf = pldf.with_columns(
                        pl.col("year_month").dt.truncate("1mo").alias("year_month")
                    )
                else:
                    pldf = pldf.with_columns(
                        pl.col("year_month")
                        .cast(pl.Utf8)
                        .str.strptime(pl.Date, format="%Y-%m-%d", strict=False)
                        .alias("year_month")
                    )
        all_detailed_pl.append(pldf.with_columns(pl.lit(model_index).alias("source_model")))
    detailed_models = pl.concat(all_detailed_pl, how="vertical_relaxed")
    if not union_mode:
        valid_keys = (
            detailed_models.group_by(["year_month", "ticker"])
            .agg(pl.col("source_model").n_unique().alias("_models"))
            .filter(pl.col("_models") == n_models)
            .select(["year_month", "ticker"])
        )
        detailed_models = detailed_models.join(valid_keys, on=["year_month", "ticker"], how="inner")

    detailed_holdings = (
        detailed_models.group_by(["year_month", "ticker"])
        .agg(
            pl.col("dr").mean().alias("dr"),
            pl.col("source_model").n_unique().alias("n_models"),
            pl.col("Sector").first().alias("Sector"),
        )
        .with_columns(
            (pl.lit(1.0) if mode == "equal" else (pl.col("n_models") / n_models)).alias("weight")
        )
        .with_columns(
            (pl.col("weight") / pl.col("weight").sum().over("year_month")).alias(
                "weight_normalized"
            )
        )
    )
    if index is not None:
        benchmark_for_engine = to_polars(
            normalize_year_month_to_timestamp(
                index.monthly_returns[["year_month", "monthly_return"]],
                col="year_month",
            )
        ).with_columns(pl.col("monthly_return").fill_null(0.0))
    else:
        benchmark_for_engine = (
            detailed_holdings.select("year_month")
            .unique()
            .with_columns(pl.lit(0.0).alias("monthly_return"))
        )
    engine_holdings = legacy_detailed_to_holdings(
        detailed_holdings.with_columns((pl.col("dr") - 1.0).alias("dr")),
        strategy=f"Legacy_{mode}",
        benchmark_monthly=benchmark_for_engine,
    ).filter(pl.col("realized_return").is_not_null())
    engine_monthly = simulate_weighted_portfolio(
        engine_holdings,
        transaction_cost_bps=0.0,
        causal_timing_policy="legacy_month_only",
        # Keep the explicitly named historical compatibility behavior.
        validate=False,
    )
    model_counts = (
        detailed_holdings.filter(pl.col("dr").is_not_null())
        .with_columns(pl.col("year_month").cast(pl.Date))
        .group_by("year_month")
        .agg(pl.col("n_models").mean().alias("avg_models_per_stock"))
    )
    aggregated = engine_monthly.select(
        pl.col("holding_month").alias("year_month"),
        pl.col("net_return").alias("monthly_return"),
        pl.col("n_positions").cast(pl.UInt32).alias("n"),
    ).join(model_counts, on="year_month", how="left")
    if index is not None:
        index_returns = normalize_year_month_to_timestamp(
            index.monthly_returns[["year_month", "monthly_return"]],
            col="year_month",
        )
        index_returns_pl = (
            to_polars(index_returns)
            .with_columns(pl.col("year_month").cast(pl.Date))
            .rename({"monthly_return": "monthly_return_index"})
        )
        aggregated = aggregated.join(index_returns_pl, on="year_month", how="left").with_columns(
            ((1 + pl.col("monthly_return")) / (1 + pl.col("monthly_return_index"))).alias(
                "monthly_return_vs_index"
            )
        )

    detailed_output = detailed_holdings.select(
        [
            "year_month",
            "ticker",
            "dr",
            "n_models",
            "Sector",
            "weight",
            "weight_normalized",
        ]
    ).with_columns((pl.col("dr") - 1).alias("dr"))
    return {
        "detailed": normalize_year_month_to_period(to_pandas(detailed_output), "year_month"),
        "aggregated": normalize_year_month_to_period(to_pandas(aggregated), "year_month"),
    }
