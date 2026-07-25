from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from alpharank.backtest.config import FundamentalFeatureConfig, TechnicalFeatureConfig
from alpharank.backtest.data_loading import RawDataBundle, find_existing_column, load_raw_data
from alpharank.backtest.datasets import prepare_constituents_monthly
from alpharank.backtest.features import (
    compute_monthly_index_returns,
    compute_monthly_stock_prices,
    compute_technical_features,
)
from alpharank.backtest.fundamentals import build_monthly_fundamental_features


RELATIVE_EMA_SHORT_SPANS = (5, 10, 20, 40, 60, 80, 100)
RELATIVE_EMA_LONG_SPANS = (60, 90, 120, 180, 260, 360, 400)
RELATIVE_EMA_PAIRS = tuple(
    (short, long)
    for short in RELATIVE_EMA_SHORT_SPANS
    for long in RELATIVE_EMA_LONG_SPANS
    if long > short
)


@dataclass(frozen=True)
class ResearchFrame:
    frame: pl.DataFrame
    feature_columns: tuple[str, ...]
    input_paths: dict[str, Path]
    relative_ema_pairs: tuple[tuple[int, int], ...]


def _month_number(column: str) -> pl.Expr:
    return pl.col(column).dt.year() * 12 + pl.col(column).dt.month()


def _resolve_price_column(frame: pl.DataFrame) -> str:
    column = find_existing_column(frame, ("adjusted_close", "close", "adj_close"))
    if column is None:
        raise ValueError("No supported close column was found.")
    return column


def _apply_exclusions(raw: RawDataBundle, excluded: Sequence[str]) -> RawDataBundle:
    excluded_values = [str(value).upper() for value in excluded]

    def filt(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter(~pl.col("ticker").cast(pl.Utf8).str.to_uppercase().is_in(excluded_values))

    return RawDataBundle(
        final_price=filt(raw.final_price),
        income_statement=filt(raw.income_statement),
        balance_sheet=filt(raw.balance_sheet),
        cash_flow=filt(raw.cash_flow),
        earnings=filt(raw.earnings),
        constituents=raw.constituents,
        sp500_price=raw.sp500_price,
        source_paths=raw.source_paths,
    )


def _relative_daily_features(
    final_price: pl.DataFrame,
    sp500_price: pl.DataFrame,
    pairs: Sequence[tuple[int, int]],
) -> tuple[pl.DataFrame, list[str]]:
    stock_price = _resolve_price_column(final_price)
    index_price = _resolve_price_column(sp500_price)
    index = (
        sp500_price.select(
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(index_price).cast(pl.Float64).alias("_index_close"),
        )
        .drop_nulls()
        .sort("date")
        .unique("date", keep="last", maintain_order=True)
    )
    daily = (
        final_price.select(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("date").cast(pl.Date, strict=False).alias("date"),
            pl.col(stock_price).cast(pl.Float64).alias("_stock_close"),
        )
        .drop_nulls()
        .join(index, on="date", how="inner")
        .with_columns((pl.col("_stock_close") / pl.col("_index_close")).alias("_relative_close"))
        .sort(["ticker", "date"])
    )
    spans = sorted({value for pair in pairs for value in pair})
    daily = daily.with_columns(
        [
            pl.col("_relative_close").ewm_mean(span=span, adjust=False).over("ticker").alias(f"_rel_ema_{span}")
            for span in spans
        ]
    )
    feature_columns: list[str] = []
    expressions: list[pl.Expr] = []
    for short, long in pairs:
        name = f"relative_ema_ratio_{short}_{long}"
        feature_columns.append(name)
        expressions.append(
            pl.when(pl.col(f"_rel_ema_{long}").abs() > 1e-12)
            .then(pl.col(f"_rel_ema_{short}") / pl.col(f"_rel_ema_{long}"))
            .otherwise(None)
            .alias(name)
        )
    return (
        daily.with_columns(expressions)
        .with_columns(pl.col("date").dt.truncate("1mo").alias("decision_month"))
        .group_by(["ticker", "decision_month"])
        .agg([pl.col(column).last().alias(column) for column in feature_columns])
        .sort(["ticker", "decision_month"]),
        feature_columns,
    )


def _add_cross_sectional_relative_ema_features(
    frame: pl.DataFrame,
    base_columns: Sequence[str],
) -> tuple[pl.DataFrame, list[str]]:
    expressions: list[pl.Expr] = []
    added: list[str] = []
    for column in base_columns:
        rank_name = f"{column}_rank_month"
        z_name = f"{column}_z_month"
        top_name = f"{column}_top_quartile"
        bottom_name = f"{column}_bottom_quartile"
        rank = pl.col(column).rank(method="average").over("decision_month") / pl.len().over("decision_month")
        mean = pl.col(column).mean().over("decision_month")
        std = pl.col(column).std().over("decision_month")
        expressions.extend(
            [
                rank.alias(rank_name),
                pl.when(std.abs() > 1e-12).then((pl.col(column) - mean) / std).otherwise(None).alias(z_name),
                (rank >= 0.75).cast(pl.Int8).alias(top_name),
                (rank <= 0.25).cast(pl.Int8).alias(bottom_name),
            ]
        )
        added.extend((rank_name, z_name, top_name, bottom_name))
    return frame.with_columns(expressions), list(base_columns) + added


def _add_regime_features(frame: pl.DataFrame, index_monthly: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    index_features = index_monthly.sort("year_month")
    expressions: list[pl.Expr] = []
    names: list[str] = []
    for horizon in (1, 3, 6, 12, 24):
        roc_name = f"spy_roc_{horizon}m"
        names.append(roc_name)
        expressions.append((pl.col("index_close") / pl.col("index_close").shift(horizon) - 1.0).alias(roc_name))
    for horizon in (3, 6, 12, 24):
        vol_name = f"spy_volatility_{horizon}m"
        names.append(vol_name)
        expressions.append(
            (pl.col("index_monthly_return").rolling_std(horizon) * (12.0**0.5)).alias(vol_name)
        )
    index_features = index_features.with_columns(expressions).select(
        pl.col("year_month").alias("decision_month"),
        *names,
    )
    breadth = (
        frame.group_by("decision_month")
        .agg(
            (pl.col("monthly_return") > 0.0).mean().alias("market_breadth_positive"),
            pl.col("monthly_return").std().alias("market_cross_sectional_dispersion"),
        )
    )
    names.extend(("market_breadth_positive", "market_cross_sectional_dispersion"))
    return frame.join(index_features, on="decision_month", how="left").join(
        breadth, on="decision_month", how="left"
    ), names


def _future_list(column: str, horizon: int) -> pl.Expr:
    return pl.concat_list([pl.col(column).shift(-step).over("ticker") for step in range(1, horizon + 1)])


def _add_multihorizon_targets(
    frame: pl.DataFrame,
    index_monthly: pl.DataFrame,
    horizons: Iterable[int],
) -> pl.DataFrame:
    benchmark = index_monthly.select(
        pl.col("year_month").alias("decision_month"),
        pl.col("index_close").alias("_benchmark_close"),
        pl.col("index_monthly_return").alias("_benchmark_monthly_return"),
    )
    result = frame.join(benchmark, on="decision_month", how="left").sort(["ticker", "decision_month"])
    benchmark_by_month = benchmark.sort("decision_month")
    for horizon in horizons:
        end_month = pl.col("decision_month").shift(-horizon).over("ticker")
        stock_return = pl.col("last_close").shift(-horizon).over("ticker") / pl.col("last_close") - 1.0
        valid_stock = (_month_number("_target_end_month") - _month_number("decision_month")) == horizon
        result = result.with_columns(end_month.alias("_target_end_month")).with_columns(
            pl.when(valid_stock).then(stock_return).otherwise(None).alias(f"future_return_{horizon}m"),
            pl.when(valid_stock)
            .then(_future_list("monthly_return", horizon).list.std() * (12.0**0.5))
            .otherwise(None)
            .alias(f"future_volatility_{horizon}m"),
        )
        benchmark_h = benchmark_by_month.with_columns(
            (
                pl.col("_benchmark_close").shift(-horizon) / pl.col("_benchmark_close") - 1.0
            ).alias(f"benchmark_future_return_{horizon}m")
        ).select("decision_month", f"benchmark_future_return_{horizon}m")
        result = result.join(benchmark_h, on="decision_month", how="left")
        relative = (
            (1.0 + pl.col(f"future_return_{horizon}m"))
            / (1.0 + pl.col(f"benchmark_future_return_{horizon}m"))
            - 1.0
        )
        result = result.with_columns(relative.alias(f"future_excess_return_{horizon}m"))
        result = result.with_columns(
            (
                pl.col(f"future_excess_return_{horizon}m").rank(method="average").over("decision_month")
                / pl.col(f"future_excess_return_{horizon}m").count().over("decision_month")
            ).alias(f"future_excess_rank_{horizon}m"),
            (pl.col(f"future_excess_return_{horizon}m") > 0.0)
            .cast(pl.Int8)
            .alias(f"future_positive_label_{horizon}m"),
        )
        future_excess_monthly = pl.concat_list(
            [
                (
                    pl.col("monthly_return").shift(-step).over("ticker")
                    - pl.col("_benchmark_monthly_return").shift(-step).over("ticker")
                )
                for step in range(1, horizon + 1)
            ]
        )
        result = result.with_columns(
            pl.when(valid_stock)
            .then(
                future_excess_monthly.list.eval(
                    pl.when(pl.element() < 0.0).then(pl.element() ** 2).otherwise(0.0)
                )
                .list.mean()
                .sqrt()
                .mul(12.0**0.5)
            )
            .otherwise(None)
            .alias(f"future_downside_{horizon}m")
        ).drop("_target_end_month")
    return result.drop(["_benchmark_close", "_benchmark_monthly_return"])


def _append_legacy_labels(frame: pl.DataFrame, legacy_path: Path) -> pl.DataFrame:
    raw_legacy = pl.read_parquet(legacy_path).filter(
        pl.col("portfolio_model") == "Combined_Frequency"
    )
    legacy_months = (
        raw_legacy.select(pl.col("year_month").cast(pl.Date).alias("holding_month"))
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_label_available"))
    )
    legacy = (
        raw_legacy
        .with_columns(
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("ticker").cast(pl.Utf8),
        )
        .select("holding_month", "ticker", "n_models", "weight_normalized")
        .unique(["holding_month", "ticker"])
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )
    return (
        frame.with_columns(pl.col("decision_month").dt.offset_by("1mo").alias("holding_month"))
        .join(legacy_months, on="holding_month", how="left")
        .join(legacy, on=["holding_month", "ticker"], how="left")
        .with_columns(
            pl.col("legacy_label_available").fill_null(0).cast(pl.Int8),
            pl.col("legacy_selected").fill_null(0).cast(pl.Int8),
            pl.col("n_models").fill_null(0).cast(pl.Int8).alias("legacy_n_models"),
            pl.col("weight_normalized").fill_null(0.0).alias("legacy_weight_normalized"),
        )
        .drop(["n_models", "weight_normalized"])
    )


def build_research_frame(
    *,
    data_dir: Path,
    legacy_detailed_returns_path: Path,
    horizons: Sequence[int],
    start_month: str,
    excluded_tickers: Sequence[str],
    relative_ema_pairs: Sequence[tuple[int, int]] | None = None,
) -> ResearchFrame:
    """Build the raw, non-imputed, point-in-time multi-horizon panel."""

    raw = _apply_exclusions(load_raw_data(data_dir), excluded_tickers)
    monthly_prices = compute_monthly_stock_prices(raw.final_price)
    index_monthly = compute_monthly_index_returns(raw.sp500_price)
    technical_config = TechnicalFeatureConfig(
        roc_windows=(1, 3, 6, 12, 24, 36),
        ema_pairs=((2, 6), (3, 6), (3, 12), (6, 12), (6, 18), (12, 24), (12, 36), (24, 36)),
        price_to_ema_spans=(3, 6, 12, 24, 36),
        rsi_windows=(3, 6, 12, 24),
        rsi_ratio_pairs=((3, 12), (6, 24)),
        bollinger_windows=(6, 12, 24),
        stochastic_windows=((6, 3), (12, 3)),
        range_windows=(6, 12, 24),
        volatility_windows=(3, 6, 12, 24, 36),
        volatility_ratio_pairs=((3, 12), (6, 24), (12, 36)),
    )
    technical = compute_technical_features(monthly_prices, config=technical_config)
    fundamentals = build_monthly_fundamental_features(
        monthly_prices=monthly_prices,
        balance_sheet=raw.balance_sheet,
        income_statement=raw.income_statement,
        cash_flow=raw.cash_flow,
        earnings=raw.earnings,
        config=FundamentalFeatureConfig(quarterly_growth_lags=(1, 4, 12)),
    )
    selected_relative_pairs = tuple(relative_ema_pairs or RELATIVE_EMA_PAIRS)
    relative, relative_base = _relative_daily_features(
        raw.final_price,
        raw.sp500_price,
        selected_relative_pairs,
    )
    constituents = prepare_constituents_monthly(raw.constituents).rename({"year_month": "decision_month"})
    frame = (
        monthly_prices.rename({"year_month": "decision_month", "date": "decision_asof_date"})
        .join(
            technical.rename({"year_month": "decision_month"}),
            on=["ticker", "decision_month"],
            how="left",
        )
        .join(
            fundamentals.rename({"year_month": "decision_month"}),
            on=["ticker", "decision_month"],
            how="left",
        )
        .join(relative, on=["ticker", "decision_month"], how="left")
        .join(constituents, on=["ticker", "decision_month"], how="inner")
        .filter(
            pl.col("decision_month")
            >= pl.lit(datetime.strptime(start_month, "%Y-%m").date())
        )
    )
    frame, relative_features = _add_cross_sectional_relative_ema_features(frame, relative_base)
    frame, regime_features = _add_regime_features(frame, index_monthly)
    frame = _add_multihorizon_targets(frame, index_monthly, horizons)
    frame = _append_legacy_labels(frame, legacy_detailed_returns_path)
    identity = {
        "ticker",
        "decision_month",
        "holding_month",
        "decision_asof_date",
        "last_close",
        "monthly_return",
        "legacy_selected",
        "legacy_n_models",
        "legacy_weight_normalized",
        "legacy_label_available",
    }
    target_prefixes = (
        "future_",
        "benchmark_future_",
    )
    numeric = {
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
    feature_columns = [
        column
        for column in frame.columns
        if column not in identity
        and not column.startswith(target_prefixes)
        and frame.schema.get(column) in numeric
    ]
    frame = frame.with_columns(
        (
            pl.col("earnings_yield").is_not_null()
            & (pl.col("earnings_yield") > 0.01)
        )
        .cast(pl.Int8)
        .alias("legacy_eligibility_proxy")
        if "earnings_yield" in frame.columns
        else pl.lit(1).cast(pl.Int8).alias("legacy_eligibility_proxy")
    )
    if "legacy_eligibility_proxy" not in feature_columns:
        feature_columns.append("legacy_eligibility_proxy")
    return ResearchFrame(
        frame=frame.sort(["decision_month", "ticker"]),
        feature_columns=tuple(feature_columns),
        input_paths=dict(raw.source_paths),
        relative_ema_pairs=selected_relative_pairs,
    )
