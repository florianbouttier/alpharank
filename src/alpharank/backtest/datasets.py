from __future__ import annotations

from datetime import date, datetime
from typing import Iterable, List, Tuple

import polars as pl

from alpharank.backtest.data_loading import find_existing_column


NUMERIC_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
    pl.Float32,
    pl.Float64,
}


def _parse_start_month(start_month: str) -> date:
    return datetime.strptime(start_month, "%Y-%m").date()


def prepare_constituents_monthly(constituents: pl.DataFrame) -> pl.DataFrame:
    ticker_col = find_existing_column(constituents, ["ticker", "Ticker"])
    date_col = find_existing_column(constituents, ["date", "Date"])

    if ticker_col is None or date_col is None:
        return pl.DataFrame(schema={"ticker": pl.Utf8, "year_month": pl.Date})

    prepared = (
        constituents.select(
            pl.col(ticker_col).cast(pl.Utf8).str.to_uppercase().alias("ticker_raw"),
            pl.col(date_col).cast(pl.Date, strict=False).alias("date"),
        )
        .with_columns(
            pl.when(pl.col("ticker_raw").str.ends_with(".US"))
            .then(pl.col("ticker_raw"))
            .otherwise(pl.col("ticker_raw").str.replace_all(r"\\.", "-") + pl.lit(".US"))
            .alias("ticker")
        )
        .with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
        .select(["ticker", "year_month"])
        .unique()
    )

    return prepared


def _drop_sparse_features(
    df: pl.DataFrame,
    feature_cols: Iterable[str],
    max_missing_ratio: float,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    feature_list = list(feature_cols)
    if not feature_list:
        return df, [], []

    missing_stats = df.select([pl.col(c).is_null().mean().alias(c) for c in feature_list]).to_dicts()[0]

    kept_features = [c for c in feature_list if missing_stats.get(c, 1.0) <= max_missing_ratio]
    dropped_features = [c for c in feature_list if c not in kept_features]

    return df, kept_features, dropped_features


def _impute_feature_nulls(df: pl.DataFrame, feature_cols: Iterable[str]) -> pl.DataFrame:
    exprs = []
    for col in feature_cols:
        exprs.append(
            pl.col(col)
            .fill_null(pl.col(col).median().over("year_month"))
            .fill_null(pl.col(col).median())
            .fill_null(0.0)
            .alias(col)
        )
    return df.with_columns(exprs)


def build_model_frame(
    monthly_prices: pl.DataFrame,
    technical_features: pl.DataFrame,
    fundamental_features: pl.DataFrame,
    index_monthly: pl.DataFrame,
    constituents: pl.DataFrame,
    start_month: str,
    missing_feature_threshold: float,
) -> Tuple[pl.DataFrame, List[str], List[str]]:
    base = monthly_prices.select(["ticker", "year_month", "monthly_return"])

    frame = (
        base.join(technical_features, on=["ticker", "year_month"], how="left")
        .join(fundamental_features, on=["ticker", "year_month"], how="left")
        .sort(["ticker", "year_month"])
        .with_columns(pl.col("monthly_return").shift(-1).over("ticker").alias("future_return"))
    )

    constituents_monthly = prepare_constituents_monthly(constituents)
    if not constituents_monthly.is_empty():
        frame = frame.join(constituents_monthly, on=["ticker", "year_month"], how="inner")

    benchmark = (
        index_monthly.select(["year_month", "index_monthly_return"])
        .sort("year_month")
        .with_columns(pl.col("index_monthly_return").shift(-1).alias("benchmark_future_return"))
        .select(["year_month", "benchmark_future_return"])
    )
    frame = frame.join(benchmark, on="year_month", how="left").with_columns(
        (pl.col("future_return") - pl.col("benchmark_future_return")).alias("future_excess_return"),
        pl.when((pl.col("benchmark_future_return") + 1.0).abs() > 1e-12)
        .then((1.0 + pl.col("future_return")) / (1.0 + pl.col("benchmark_future_return")) - 1.0)
        .otherwise(None)
        .alias("future_relative_return"),
    )

    start_date = _parse_start_month(start_month)
    frame = frame.filter(pl.col("year_month") >= pl.lit(start_date))

    excluded = {
        "ticker",
        "year_month",
        "monthly_return",
        "future_return",
        "benchmark_future_return",
        "future_excess_return",
        "future_relative_return",
    }
    candidate_features = [
        col for col in frame.columns if col not in excluded and frame.schema.get(col) in NUMERIC_DTYPES
    ]

    if not candidate_features:
        raise ValueError("No numeric features available after joins.")

    frame = frame.with_columns(
        [
            pl.col(col).cast(pl.Float64, strict=False).alias(col)
            for col in candidate_features
        ]
    )

    frame = frame.with_columns(
        [
            pl.when(pl.col(col).is_finite()).then(pl.col(col)).otherwise(None).alias(col)
            for col in candidate_features
        ]
    )

    frame, kept_features, dropped_features = _drop_sparse_features(
        frame,
        candidate_features,
        max_missing_ratio=missing_feature_threshold,
    )

    if not kept_features:
        raise ValueError(
            "All candidate features were dropped due to missing ratio. "
            "Increase --missing-feature-threshold or adjust feature engineering."
        )

    frame = _impute_feature_nulls(frame, kept_features)

    return frame, kept_features, dropped_features
