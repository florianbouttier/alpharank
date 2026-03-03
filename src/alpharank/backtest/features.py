from __future__ import annotations

import polars as pl

from alpharank.backtest.data_loading import find_existing_column


def _resolve_price_column(df: pl.DataFrame) -> str:
    column = find_existing_column(df, ["adjusted_close", "close", "adj_close"])
    if column is None:
        raise ValueError(
            "No supported price column found. Expected one of: adjusted_close, close, adj_close"
        )
    return column


def compute_monthly_stock_prices(final_price: pl.DataFrame) -> pl.DataFrame:
    price_col = _resolve_price_column(final_price)

    monthly = (
        final_price.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(price_col).cast(pl.Float64).alias("close"),
        )
        .sort(["ticker", "date"])
        .with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
        .group_by(["ticker", "year_month"])
        .agg(
            pl.col("date").last().alias("date"),
            pl.col("close").last().alias("last_close"),
        )
        .sort(["ticker", "year_month"])
        .with_columns(pl.col("last_close").pct_change().over("ticker").alias("monthly_return"))
    )

    return monthly


def compute_monthly_index_returns(sp500_price: pl.DataFrame) -> pl.DataFrame:
    price_col = _resolve_price_column(sp500_price)

    monthly = (
        sp500_price.select(
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(price_col).cast(pl.Float64).alias("index_close"),
        )
        .sort("date")
        .with_columns(pl.col("date").dt.truncate("1mo").alias("year_month"))
        .group_by("year_month")
        .agg(
            pl.col("date").last().alias("date"),
            pl.col("index_close").last().alias("index_close"),
        )
        .sort("year_month")
        .with_columns(pl.col("index_close").pct_change().alias("index_monthly_return"))
    )

    return monthly


def compute_technical_features(monthly_prices: pl.DataFrame) -> pl.DataFrame:
    base = monthly_prices.sort(["ticker", "year_month"])

    gains = pl.when(pl.col("monthly_return") > 0).then(pl.col("monthly_return")).otherwise(0.0)
    losses = pl.when(pl.col("monthly_return") < 0).then(-pl.col("monthly_return")).otherwise(0.0)

    technical = (
        base.with_columns(
            pl.col("monthly_return").shift(1).over("ticker").alias("ret_lag_1"),
            pl.col("monthly_return").shift(2).over("ticker").alias("ret_lag_2"),
            pl.col("monthly_return").rolling_mean(window_size=3).over("ticker").alias("ret_mean_3m"),
            pl.col("monthly_return").rolling_mean(window_size=6).over("ticker").alias("ret_mean_6m"),
            pl.col("monthly_return").rolling_std(window_size=3).over("ticker").alias("ret_vol_3m"),
            pl.col("monthly_return").rolling_std(window_size=6).over("ticker").alias("ret_vol_6m"),
            pl.col("monthly_return").rolling_sum(window_size=3).over("ticker").alias("mom_3m"),
            pl.col("monthly_return").rolling_sum(window_size=6).over("ticker").alias("mom_6m"),
            pl.col("last_close").ewm_mean(span=3, adjust=False).over("ticker").alias("ema_3"),
            pl.col("last_close").ewm_mean(span=12, adjust=False).over("ticker").alias("ema_12"),
            pl.col("last_close").rolling_max(window_size=12).over("ticker").alias("rolling_high_12m"),
            pl.col("last_close").rolling_min(window_size=12).over("ticker").alias("rolling_low_12m"),
            gains.rolling_mean(window_size=6).over("ticker").alias("avg_gain_6m"),
            losses.rolling_mean(window_size=6).over("ticker").alias("avg_loss_6m"),
        )
        .with_columns(
            (pl.col("ema_3") / pl.col("ema_12")).alias("ema_ratio_3_12"),
            (pl.col("last_close") / pl.col("rolling_high_12m") - 1.0).alias("dist_to_12m_high"),
            (pl.col("last_close") / pl.col("rolling_low_12m") - 1.0).alias("dist_to_12m_low"),
            (
                100.0
                - (100.0 / (1.0 + (pl.col("avg_gain_6m") / (pl.col("avg_loss_6m") + pl.lit(1e-12)))))
            ).alias("rsi_6m")
        )
        .select(
            "ticker",
            "year_month",
            "ret_lag_1",
            "ret_lag_2",
            "ret_mean_3m",
            "ret_mean_6m",
            "ret_vol_3m",
            "ret_vol_6m",
            "mom_3m",
            "mom_6m",
            "ema_ratio_3_12",
            "dist_to_12m_high",
            "dist_to_12m_low",
            "rsi_6m",
        )
    )

    return technical
