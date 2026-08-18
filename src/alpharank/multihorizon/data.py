from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
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
from alpharank.data.price_eligibility import (
    MonthlyPriceEligibilityPolicy,
    build_monthly_price_eligibility,
)


RELATIVE_EMA_SHORT_SPANS = (5, 10, 20, 40, 60, 80, 100)
RELATIVE_EMA_LONG_SPANS = (60, 90, 120, 180, 260, 360, 400)
RELATIVE_EMA_PAIRS = tuple(
    (short, long)
    for short in RELATIVE_EMA_SHORT_SPANS
    for long in RELATIVE_EMA_LONG_SPANS
    if long > short
)
TARGET_STATUS_VALUES = (
    "evaluable",
    "terminal_event_resolved",
    "provisional_last_observation",
    "horizon_pending",
    "benchmark_target_unavailable",
    "ticker_target_unavailable",
    "terminal_event_unresolved",
)
TRAINABLE_TARGET_STATUSES = frozenset(
    {"evaluable", "terminal_event_resolved", "provisional_last_observation"}
)


class TargetCensoringError(RuntimeError):
    """Raised when mature labels would be dropped through survival filtering."""


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
    *,
    target_prices: pl.DataFrame | None = None,
    mature_target_gap_policy: str = "fail_closed",
) -> pl.DataFrame:
    if mature_target_gap_policy not in {
        "fail_closed",
        "provisional_last_observation_v1",
    }:
        raise ValueError(
            f"Unsupported mature target gap policy: {mature_target_gap_policy!r}"
        )
    benchmark = index_monthly.select(
        pl.col("year_month").alias("decision_month"),
        pl.col("index_close").alias("_benchmark_close"),
        pl.col("index_monthly_return").alias("_benchmark_monthly_return"),
    )
    result = frame.sort(["ticker", "decision_month"])
    benchmark_by_month = benchmark.sort("decision_month")
    price_panel = target_prices if target_prices is not None else frame
    price_month_column = (
        "year_month" if "year_month" in price_panel.columns else "decision_month"
    )
    target_panel = (
        price_panel.select(
            pl.col("ticker").cast(pl.String),
            pl.col(price_month_column).cast(pl.Date).alias("decision_month"),
            (
                pl.col("date").cast(pl.Date, strict=False)
                if "date" in price_panel.columns
                else pl.col(price_month_column).cast(pl.Date)
            ).alias("price_observed_date"),
            pl.col("last_close").cast(pl.Float64),
            pl.col("monthly_return").cast(pl.Float64),
        )
        .join(benchmark, on="decision_month", how="left")
        .sort(["ticker", "decision_month"])
    )
    for horizon in horizons:
        end_month = pl.col("decision_month").shift(-horizon).over("ticker")
        stock_return = pl.col("last_close").shift(-horizon).over("ticker") / pl.col("last_close") - 1.0
        valid_stock = (_month_number("_target_end_month") - _month_number("decision_month")) == horizon
        stock_targets = target_panel.with_columns(
            end_month.alias("_target_end_month")
        ).with_columns(
            pl.when(valid_stock).then(stock_return).otherwise(None).alias(f"future_return_{horizon}m"),
            pl.when(valid_stock)
            .then(_future_list("monthly_return", horizon).list.std() * (12.0**0.5))
            .otherwise(None)
            .alias(f"future_volatility_{horizon}m"),
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
        stock_targets = (
            stock_targets.with_columns(
                pl.when(valid_stock)
                .then(
                    future_excess_monthly.list.eval(
                        pl.when(pl.element() < 0.0)
                        .then(pl.element() ** 2)
                        .otherwise(0.0)
                    )
                    .list.mean()
                    .sqrt()
                    .mul(12.0**0.5)
                )
                .otherwise(None)
                .alias(f"future_downside_{horizon}m")
            )
            .select(
                "ticker",
                "decision_month",
                f"future_return_{horizon}m",
                f"future_volatility_{horizon}m",
                f"future_downside_{horizon}m",
                pl.when(valid_stock)
                .then(
                    pl.col("price_observed_date")
                    .shift(-horizon)
                    .over("ticker")
                )
                .otherwise(None)
                .alias(f"future_return_observed_end_date_{horizon}m"),
            )
        )
        if mature_target_gap_policy == "provisional_last_observation_v1":
            requests = target_panel.select(
                "ticker",
                "decision_month",
                pl.col("last_close").alias("_provisional_start_close"),
                pl.col("decision_month")
                .dt.offset_by(f"{horizon}mo")
                .alias("_provisional_scheduled_month"),
            ).sort(["ticker", "_provisional_scheduled_month"])
            observations = target_panel.select(
                "ticker",
                pl.col("decision_month").alias("_provisional_observed_month"),
                pl.col("price_observed_date").alias("_provisional_observed_date"),
                pl.col("last_close").alias("_provisional_observed_close"),
            ).sort(["ticker", "_provisional_observed_month"])
            provisional = (
                requests.join_asof(
                    observations,
                    left_on="_provisional_scheduled_month",
                    right_on="_provisional_observed_month",
                    by="ticker",
                    strategy="backward",
                    check_sortedness=False,
                )
                .with_columns(
                    pl.when(
                        pl.col("_provisional_observed_month")
                        < pl.col("_provisional_scheduled_month")
                    )
                    .then(
                        pl.col("_provisional_observed_close")
                        / pl.col("_provisional_start_close")
                        - 1.0
                    )
                    .otherwise(None)
                    .alias("_provisional_return")
                )
                .select(
                    "ticker",
                    "decision_month",
                    "_provisional_return",
                    "_provisional_observed_date",
                )
            )
            stock_targets = (
                stock_targets.join(
                    provisional,
                    on=["ticker", "decision_month"],
                    how="left",
                    validate="1:1",
                )
                .with_columns(
                    pl.coalesce(
                        f"future_return_{horizon}m",
                        "_provisional_return",
                    ).alias(f"future_return_{horizon}m"),
                    pl.coalesce(
                        f"future_return_observed_end_date_{horizon}m",
                        "_provisional_observed_date",
                    ).alias(f"future_return_observed_end_date_{horizon}m"),
                    pl.when(pl.col(f"future_return_{horizon}m").is_not_null())
                    .then(pl.lit("observed_horizon_return"))
                    .when(pl.col("_provisional_return").is_not_null())
                    .then(pl.lit("provisional_last_observation"))
                    .otherwise(pl.lit("unresolved_missing_return"))
                    .alias(f"return_resolution_{horizon}m"),
                )
                .drop("_provisional_return", "_provisional_observed_date")
            )
        else:
            stock_targets = stock_targets.with_columns(
                pl.when(pl.col(f"future_return_{horizon}m").is_not_null())
                .then(pl.lit("observed_horizon_return"))
                .otherwise(pl.lit("unresolved_missing_return"))
                .alias(f"return_resolution_{horizon}m")
            )
        result = result.join(
            stock_targets,
            on=["ticker", "decision_month"],
            how="left",
            validate="1:1",
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
    return result


def mask_targets_after_completed_month(
    frame: pl.DataFrame,
    *,
    horizons: Iterable[int],
    completed_through_month: date,
) -> pl.DataFrame:
    """Remove labels whose return window extends beyond the completed calendar.

    The final decision month may still be scored. Its future labels remain null
    until the corresponding holding period is complete.
    """

    result = frame
    for horizon in sorted(set(horizons)):
        target_columns = [
            column
            for column in result.columns
            if column.endswith(f"_{horizon}m")
            and (
                column.startswith("future_")
                or column.startswith("benchmark_future_")
            )
        ]
        if not target_columns:
            continue
        target_end_month = pl.col("decision_month").dt.offset_by(f"{horizon}mo")
        result = result.with_columns(
            [
                pl.when(target_end_month <= pl.lit(completed_through_month))
                .then(pl.col(column))
                .otherwise(None)
                .alias(column)
                for column in target_columns
            ]
        )
    return result


def classify_training_target_status(
    frame: pl.DataFrame,
    *,
    horizons: Iterable[int],
    completed_through_month: date,
) -> pl.DataFrame:
    """Classify every target row without treating missingness as survival.

    A terminal event is considered resolved only when its return has already
    been incorporated in the target and the row carries explicit resolution
    lineage. An event identifier with no usable target remains fail-closed.
    """

    result = frame
    for horizon in sorted(set(horizons)):
        target = f"future_excess_return_{horizon}m"
        benchmark = f"benchmark_future_return_{horizon}m"
        if target not in result.columns or benchmark not in result.columns:
            raise ValueError(
                f"Cannot classify H{horizon} target without {target!r} and {benchmark!r}."
            )
        status_column = f"target_status_{horizon}m"
        resolution_column = f"return_resolution_{horizon}m"
        event_column = f"terminal_event_id_{horizon}m"
        resolution = (
            pl.col(resolution_column).cast(pl.String)
            if resolution_column in result.columns
            else pl.lit(None, dtype=pl.String)
        )
        event_id = (
            pl.col(event_column).cast(pl.String)
            if event_column in result.columns
            else pl.lit(None, dtype=pl.String)
        )
        target_end_month = pl.col("decision_month").dt.offset_by(f"{horizon}mo")
        result = result.with_columns(
            pl.when(target_end_month > pl.lit(completed_through_month))
            .then(pl.lit("horizon_pending"))
            .when(pl.col(benchmark).is_null())
            .then(pl.lit("benchmark_target_unavailable"))
            .when(
                pl.col(target).is_not_null()
                & (resolution == pl.lit("resolved_terminal_event"))
            )
            .then(pl.lit("terminal_event_resolved"))
            .when(
                pl.col(target).is_not_null()
                & (resolution == pl.lit("provisional_last_observation"))
            )
            .then(pl.lit("provisional_last_observation"))
            .when(pl.col(target).is_not_null())
            .then(pl.lit("evaluable"))
            .when(event_id.is_not_null())
            .then(pl.lit("terminal_event_unresolved"))
            .otherwise(pl.lit("ticker_target_unavailable"))
            .alias(status_column)
        )
    return result


def target_censoring_counts(
    frame: pl.DataFrame,
    *,
    method: str,
    horizon: int,
) -> dict[str, int]:
    """Return a complete, zero-filled target-status census."""

    if method == "teacher":
        evaluable = frame.filter(pl.col("legacy_selected").is_not_null()).height
        return {
            "evaluable": evaluable,
            "terminal_event_resolved": 0,
            "provisional_last_observation": 0,
            "horizon_pending": 0,
            "benchmark_target_unavailable": 0,
            "ticker_target_unavailable": frame.height - evaluable,
            "terminal_event_unresolved": 0,
        }
    status_column = f"target_status_{horizon}m"
    if status_column not in frame.columns:
        raise ValueError(f"Target status column is missing: {status_column}")
    observed = {
        str(row[status_column]): int(row["len"])
        for row in frame.group_by(status_column).len().to_dicts()
    }
    unknown = sorted(set(observed) - set(TARGET_STATUS_VALUES))
    if unknown:
        raise ValueError(f"Unknown target censoring statuses: {unknown}")
    return {status: observed.get(status, 0) for status in TARGET_STATUS_VALUES}


def provisional_target_journal(
    frame: pl.DataFrame,
    *,
    horizons: Iterable[int],
) -> pl.DataFrame:
    """List every carried-last-observation target for manual event review."""

    parts: list[pl.DataFrame] = []
    for horizon in sorted(set(horizons)):
        status = f"target_status_{horizon}m"
        if status not in frame.columns:
            continue
        rows = frame.filter(
            pl.col(status) == "provisional_last_observation"
        ).select(
            "ticker",
            "decision_month",
            "decision_asof_date",
            pl.lit(horizon).cast(pl.Int32).alias("horizon_months"),
            pl.col("decision_month")
            .dt.offset_by(f"{horizon}mo")
            .alias("scheduled_target_end_month"),
            pl.col(f"future_return_observed_end_date_{horizon}m").alias(
                "last_observed_price_date"
            ),
            pl.col(f"future_return_{horizon}m").alias("provisional_stock_return"),
            pl.col(f"benchmark_future_return_{horizon}m").alias(
                "benchmark_horizon_return"
            ),
            pl.col(f"future_excess_return_{horizon}m").alias(
                "provisional_excess_return"
            ),
            pl.lit("price_series_ended_before_scheduled_horizon").alias(
                "audit_reason"
            ),
            pl.lit("pending_manual_terminal_event_review").alias(
                "manual_review_status"
            ),
            pl.lit("provisional_last_observation_v1").alias("resolution_policy"),
        )
        parts.append(rows)
    if not parts:
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "decision_month": pl.Date,
                "decision_asof_date": pl.Date,
                "horizon_months": pl.Int32,
                "scheduled_target_end_month": pl.Date,
                "last_observed_price_date": pl.Date,
                "provisional_stock_return": pl.Float64,
                "benchmark_horizon_return": pl.Float64,
                "provisional_excess_return": pl.Float64,
                "audit_reason": pl.String,
                "manual_review_status": pl.String,
                "resolution_policy": pl.String,
            }
        )
    return pl.concat(parts, how="diagonal_relaxed").sort(
        ["decision_month", "ticker", "horizon_months"]
    )


def require_resolved_training_targets(
    frame: pl.DataFrame,
    *,
    method: str,
    horizon: int,
    context: str,
) -> dict[str, int]:
    """Reject mature missing labels instead of silently selecting survivors."""

    counts = target_censoring_counts(frame, method=method, horizon=horizon)
    unresolved = {
        status: counts[status]
        for status in (
            "benchmark_target_unavailable",
            "ticker_target_unavailable",
            "terminal_event_unresolved",
        )
        if counts[status]
    }
    if unresolved:
        details = ", ".join(
            f"{status}={count}" for status, count in unresolved.items()
        )
        raise TargetCensoringError(
            f"Unresolved mature training targets in {context}: {details}. "
            "Resolve source data or a sourced terminal event; never drop these rows."
        )
    return counts


def filter_trainable_targets(
    frame: pl.DataFrame, *, method: str, horizon: int
) -> pl.DataFrame:
    """Filter only after the unresolved-mature fail-closed check has passed."""

    require_resolved_training_targets(
        frame,
        method=method,
        horizon=horizon,
        context="target panel",
    )
    if method == "teacher":
        return frame.filter(pl.col("legacy_selected").is_not_null())
    return frame.filter(
        pl.col(f"target_status_{horizon}m").is_in(TRAINABLE_TARGET_STATUSES)
    )


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
    minimum_monthly_price_observations: int = 1,
    minimum_monthly_median_dollar_volume: float = 0.0,
    maximum_monthly_ohlc_violation_rate: float = 1.0,
    mature_target_gap_policy: str = "fail_closed",
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
    price_eligibility = build_monthly_price_eligibility(
        raw.final_price,
        policy=MonthlyPriceEligibilityPolicy(
            policy_id="research_config",
            minimum_observations=minimum_monthly_price_observations,
            minimum_median_dollar_volume=minimum_monthly_median_dollar_volume,
            maximum_ohlc_violation_rate=maximum_monthly_ohlc_violation_rate,
        ),
    )
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
        .join(
            price_eligibility,
            on=["ticker", "decision_month"],
            how="left",
        )
        .filter(
            pl.col("decision_month")
            >= pl.lit(datetime.strptime(start_month, "%Y-%m").date())
        )
        .filter(pl.col("price_eligible").fill_null(False))
        .drop(
            [
                "price_observations",
                "median_dollar_volume",
                "ohlc_violation_rate",
                "price_eligible",
            ]
        )
    )
    frame, relative_features = _add_cross_sectional_relative_ema_features(frame, relative_base)
    frame, regime_features = _add_regime_features(frame, index_monthly)
    frame = _add_multihorizon_targets(
        frame,
        index_monthly,
        horizons,
        target_prices=monthly_prices,
        mature_target_gap_policy=mature_target_gap_policy,
    )
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
