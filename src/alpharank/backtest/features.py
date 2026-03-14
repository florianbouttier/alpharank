from __future__ import annotations

from itertools import chain

import polars as pl

from alpharank.backtest.config import TechnicalFeatureConfig
from alpharank.backtest.data_loading import find_existing_column


_EPSILON = 1e-12


def _resolve_price_column(df: pl.DataFrame) -> str:
    column = find_existing_column(df, ["adjusted_close", "close", "adj_close"])
    if column is None:
        raise ValueError(
            "No supported price column found. Expected one of: adjusted_close, close, adj_close"
        )
    return column


def _safe_ratio(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return (
        pl.when(
            numerator.is_not_null()
            & denominator.is_not_null()
            & (denominator.abs() > pl.lit(_EPSILON))
        )
        .then(numerator / denominator)
        .otherwise(None)
    )


def _signed_return_component(return_expr: pl.Expr, *, positive: bool) -> pl.Expr:
    if positive:
        return (
            pl.when(return_expr.is_null())
            .then(None)
            .when(return_expr > 0.0)
            .then(return_expr)
            .otherwise(0.0)
        )

    return (
        pl.when(return_expr.is_null())
        .then(None)
        .when(return_expr < 0.0)
        .then(-return_expr)
        .otherwise(0.0)
    )


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


def compute_technical_features(
    monthly_prices: pl.DataFrame,
    config: TechnicalFeatureConfig | None = None,
) -> pl.DataFrame:
    feature_config = config or TechnicalFeatureConfig()
    base = (
        monthly_prices.sort(["ticker", "year_month"])
        .with_columns(
            _signed_return_component(pl.col("monthly_return"), positive=True).alias("_gain"),
            _signed_return_component(pl.col("monthly_return"), positive=False).alias("_loss"),
        )
    )

    ema_spans = sorted(set(feature_config.price_to_ema_spans) | set(chain.from_iterable(feature_config.ema_pairs)))
    range_windows = sorted(set(feature_config.range_windows) | {window for window, _ in feature_config.stochastic_windows})
    volatility_windows = sorted(
        set(feature_config.volatility_windows) | set(chain.from_iterable(feature_config.volatility_ratio_pairs))
    )

    base_exprs: list[pl.Expr] = []
    for window in ema_spans:
        base_exprs.append(pl.col("last_close").ewm_mean(span=window, adjust=False).over("ticker").alias(f"_ema_{window}"))
    for window in feature_config.rsi_windows:
        base_exprs.extend(
            [
                pl.col("_gain").rolling_mean(window_size=window).over("ticker").alias(f"_avg_gain_{window}m"),
                pl.col("_loss").rolling_mean(window_size=window).over("ticker").alias(f"_avg_loss_{window}m"),
            ]
        )
    for window in feature_config.bollinger_windows:
        base_exprs.extend(
            [
                pl.col("last_close").rolling_mean(window_size=window).over("ticker").alias(f"_price_sma_{window}m"),
                pl.col("last_close").rolling_std(window_size=window).over("ticker").alias(f"_price_std_{window}m"),
            ]
        )
    for window in range_windows:
        base_exprs.extend(
            [
                pl.col("last_close").rolling_max(window_size=window).over("ticker").alias(f"_rolling_high_{window}m"),
                pl.col("last_close").rolling_min(window_size=window).over("ticker").alias(f"_rolling_low_{window}m"),
            ]
        )
    for window in volatility_windows:
        base_exprs.append(
            pl.col("monthly_return").rolling_std(window_size=window).over("ticker").alias(f"_volatility_{window}m")
        )

    technical = base.with_columns(base_exprs)

    stochastic_exprs: list[pl.Expr] = []
    for window, smoothing in feature_config.stochastic_windows:
        high_col = pl.col(f"_rolling_high_{window}m")
        low_col = pl.col(f"_rolling_low_{window}m")
        stochastic_exprs.append(
            (
                100.0
                * _safe_ratio(
                    pl.col("last_close") - low_col,
                    high_col - low_col,
                )
            )
            .rolling_mean(window_size=smoothing)
            .over("ticker")
            .alias(f"_stoch_d_{window}_{smoothing}")
        )

    if stochastic_exprs:
        technical = technical.with_columns(stochastic_exprs)

    feature_exprs: list[pl.Expr] = []
    feature_columns: list[str] = []

    for window in feature_config.roc_windows:
        name = f"price_roc_{window}m"
        feature_columns.append(name)
        feature_exprs.append(
            (_safe_ratio(pl.col("last_close"), pl.col("last_close").shift(window).over("ticker")) - 1.0).alias(name)
        )

    for short_window, long_window in feature_config.ema_pairs:
        name = f"ema_ratio_{short_window}_{long_window}"
        feature_columns.append(name)
        feature_exprs.append(_safe_ratio(pl.col(f"_ema_{short_window}"), pl.col(f"_ema_{long_window}")).alias(name))

    for window in feature_config.price_to_ema_spans:
        name = f"price_to_ema_{window}"
        feature_columns.append(name)
        feature_exprs.append((_safe_ratio(pl.col("last_close"), pl.col(f"_ema_{window}")) - 1.0).alias(name))

    for window in feature_config.rsi_windows:
        rs = _safe_ratio(pl.col(f"_avg_gain_{window}m"), pl.col(f"_avg_loss_{window}m"))
        name = f"rsi_{window}m"
        feature_columns.append(name)
        feature_exprs.append((100.0 - (100.0 / (1.0 + rs))).alias(name))

    for short_window, long_window in feature_config.rsi_ratio_pairs:
        short_rsi = 100.0 - (
            100.0
            / (
                1.0
                + _safe_ratio(
                    pl.col(f"_avg_gain_{short_window}m"),
                    pl.col(f"_avg_loss_{short_window}m"),
                )
            )
        )
        long_rsi = 100.0 - (
            100.0
            / (
                1.0
                + _safe_ratio(
                    pl.col(f"_avg_gain_{long_window}m"),
                    pl.col(f"_avg_loss_{long_window}m"),
                )
            )
        )
        name = f"rsi_ratio_{short_window}_{long_window}"
        feature_columns.append(name)
        feature_exprs.append(_safe_ratio(short_rsi, long_rsi).alias(name))

    for window in feature_config.bollinger_windows:
        sma_col = pl.col(f"_price_sma_{window}m")
        std_col = pl.col(f"_price_std_{window}m")
        upper_band = sma_col + 2.0 * std_col
        lower_band = sma_col - 2.0 * std_col
        band_width = upper_band - lower_band

        percent_b_name = f"bollinger_percent_b_{window}m"
        feature_columns.append(percent_b_name)
        feature_exprs.append(_safe_ratio(pl.col("last_close") - lower_band, band_width).alias(percent_b_name))

        bandwidth_name = f"bollinger_bandwidth_{window}m"
        feature_columns.append(bandwidth_name)
        feature_exprs.append(_safe_ratio(band_width, sma_col).alias(bandwidth_name))

    for window, smoothing in feature_config.stochastic_windows:
        name = f"stoch_d_{window}_{smoothing}"
        feature_columns.append(name)
        feature_exprs.append(pl.col(f"_stoch_d_{window}_{smoothing}").alias(name))

    for window in feature_config.range_windows:
        high_col = pl.col(f"_rolling_high_{window}m")
        low_col = pl.col(f"_rolling_low_{window}m")

        high_name = f"dist_to_{window}m_high"
        feature_columns.append(high_name)
        feature_exprs.append((_safe_ratio(pl.col("last_close"), high_col) - 1.0).alias(high_name))

        low_name = f"dist_to_{window}m_low"
        feature_columns.append(low_name)
        feature_exprs.append((_safe_ratio(pl.col("last_close"), low_col) - 1.0).alias(low_name))

        range_position_name = f"range_position_{window}m"
        feature_columns.append(range_position_name)
        feature_exprs.append(_safe_ratio(pl.col("last_close") - low_col, high_col - low_col).alias(range_position_name))

    for window in feature_config.volatility_windows:
        name = f"volatility_{window}m"
        feature_columns.append(name)
        feature_exprs.append(pl.col(f"_volatility_{window}m").alias(name))

    for short_window, long_window in feature_config.volatility_ratio_pairs:
        name = f"volatility_ratio_{short_window}_{long_window}"
        feature_columns.append(name)
        feature_exprs.append(
            _safe_ratio(pl.col(f"_volatility_{short_window}m"), pl.col(f"_volatility_{long_window}m")).alias(name)
        )

    technical = technical.with_columns(feature_exprs)

    return technical.select(["ticker", "year_month", *feature_columns])
