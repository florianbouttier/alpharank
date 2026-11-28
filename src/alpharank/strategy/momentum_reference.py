"""Score a price-only 12–1 reference on an existing decision-time universe."""

from __future__ import annotations

import polars as pl

from alpharank.utils.frame_backend import normalize_polars_date_columns

MOMENTUM_REFERENCE_POLICY = "momentum_12_1_same_universe_v1"
MOMENTUM_SCORE = "momentum_12_1"
MOMENTUM_EVIDENCE = (
    "momentum_start_date",
    "momentum_end_date",
    "momentum_start_price",
    "momentum_end_price",
    MOMENTUM_SCORE,
    "momentum_status",
)


def score_momentum_12_1(
    candidates: pl.DataFrame, prices: pl.DataFrame, market_prices: pl.DataFrame
) -> pl.DataFrame:
    """Add P[t−1] / P[t−12] − 1 to the original candidate keys."""
    keys = ["decision_month", "ticker"]
    invalid = candidates.filter(
        pl.col("decision_month").is_null()
        | (pl.col("decision_month").dt.day() != 1)
        | pl.col("ticker").is_null()
        | pl.struct(keys).is_duplicated()
    )
    if not invalid.is_empty():
        raise ValueError("Momentum requires unique ticker and first-of-month decision keys.")

    prepared = candidates.with_columns(
        pl.col("decision_month").dt.offset_by("-12mo").alias("_start_month"),
        pl.col("decision_month").dt.offset_by("-1mo").alias("_end_month"),
    )
    months = pl.concat(
        [
            prepared.select(pl.col(column).alias("month"))
            for column in ("_start_month", "_end_month")
        ]
    ).unique()
    market_ends = (
        normalize_polars_date_columns(market_prices.select("date"), ("date",))
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .join(months, on="month", how="semi")
        .group_by("month")
        .agg(pl.col("date").max())
    )
    closes = normalize_polars_date_columns(
        prices.select("ticker", "date", pl.col("adjusted_close").cast(pl.Float64)), ("date",)
    )
    result = _join_endpoint(prepared, market_ends, closes, "start")
    result = _join_endpoint(result, market_ends, closes, "end")
    start_price = pl.col("momentum_start_price")
    end_price = pl.col("momentum_end_price")
    score = end_price / start_price - 1.0
    status = (
        pl.when(pl.any_horizontal(pl.col("momentum_start_date", "momentum_end_date").is_null()))
        .then(pl.lit("missing_market_month_end"))
        .when(~(start_price.is_finite() & (start_price > 0.0)).fill_null(False))
        .then(pl.lit("missing_start_price"))
        .when(~(end_price.is_finite() & (end_price > 0.0) & score.is_finite()).fill_null(False))
        .then(pl.lit("missing_end_price"))
        .otherwise(pl.lit("eligible"))
    )
    return (
        result.with_columns(status.alias("momentum_status"))
        .with_columns(
            pl.when(pl.col("momentum_status") == "eligible").then(score).alias(MOMENTUM_SCORE)
        )
        .drop("_start_month", "_end_month")
        .sort(keys)
    )


def _join_endpoint(
    candidates: pl.DataFrame, market_ends: pl.DataFrame, closes: pl.DataFrame, endpoint: str
) -> pl.DataFrame:
    date_column = f"momentum_{endpoint}_date"
    anchored = candidates.join(
        market_ends.rename({"month": f"_{endpoint}_month", "date": date_column}),
        on=f"_{endpoint}_month",
        how="left",
        validate="m:1",
    )
    prices = closes.rename({"date": date_column, "adjusted_close": f"momentum_{endpoint}_price"})
    keys = ["ticker", date_column]
    requested = anchored.select(keys).unique().drop_nulls()
    return anchored.join(
        prices.join(requested, on=keys, how="semi"), on=keys, how="left", validate="m:1"
    )
