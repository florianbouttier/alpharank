from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl


LEGACY_OPTUNA_MODELS = (
    "Legacy_Optuna_11",
    "Legacy_Optuna_12",
    "Legacy_Optuna_21",
    "Legacy_Optuna_22",
)
RELATIVE_EMA_SUFFIXES = (
    "",
    "_rank_month",
    "_z_month",
    "_top_quartile",
    "_bottom_quartile",
)


def load_legacy_winner_schedule(legacy_path: Path) -> pl.DataFrame:
    """Return one exact winning EMA pair per Legacy path and holding month."""

    return (
        pl.read_parquet(legacy_path)
        .filter(pl.col("portfolio_model").is_in(LEGACY_OPTUNA_MODELS))
        .select(
            pl.col("portfolio_model"),
            pl.col("year_month").cast(pl.Date).alias("holding_month"),
            pl.col("n_short").cast(pl.Int32),
            pl.col("n_long").cast(pl.Int32),
        )
        .drop_nulls()
        .unique(["portfolio_model", "holding_month", "n_short", "n_long"])
        .sort(["holding_month", "portfolio_model"])
    )


def legacy_winning_pairs(
    legacy_path: Path,
    *,
    cutoff_holding_month: date | None = None,
) -> tuple[tuple[int, int], ...]:
    schedule = load_legacy_winner_schedule(legacy_path)
    if cutoff_holding_month is not None:
        schedule = schedule.filter(pl.col("holding_month") <= cutoff_holding_month)
    return tuple(
        sorted(
            {
                (int(short), int(long))
                for short, long in schedule.select("n_short", "n_long").iter_rows()
            }
        )
    )


def relative_ema_feature_columns(
    pairs: Iterable[tuple[int, int]],
) -> tuple[str, ...]:
    return tuple(
        f"relative_ema_ratio_{short}_{long}{suffix}"
        for short, long in pairs
        for suffix in RELATIVE_EMA_SUFFIXES
    )


def point_in_time_fold_features(
    *,
    all_features: Sequence[str],
    legacy_path: Path,
    train_decision_cutoff: date,
    include_non_relative_features: bool,
) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    """Select only winner pairs observable by the end of the outer train fold."""

    # Legacy output is indexed by the holding month, one month after decision.
    cutoff_holding_month = train_decision_cutoff.replace(day=1)
    pairs = legacy_winning_pairs(
        legacy_path,
        cutoff_holding_month=cutoff_holding_month,
    )
    exact = set(relative_ema_feature_columns(pairs))
    selected = [
        column
        for column in all_features
        if column in exact
        or (
            include_non_relative_features
            and not column.startswith("relative_ema_ratio_")
        )
    ]
    if not selected:
        raise ValueError(
            f"No point-in-time Legacy EMA winner was available by {train_decision_cutoff}."
        )
    return tuple(selected), pairs


def add_active_legacy_oracle_features(
    frame: pl.DataFrame,
    *,
    legacy_path: Path,
    available_pairs: Sequence[tuple[int, int]],
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Add the four exact EMA signals active in Legacy at each decision month."""

    result = frame
    schedule = load_legacy_winner_schedule(legacy_path).with_columns(
        pl.col("holding_month").dt.offset_by("-1mo").alias("decision_month")
    )
    feature_columns: list[str] = []
    for model in LEGACY_OPTUNA_MODELS:
        path_name = model.removeprefix("Legacy_Optuna_")
        short_col = f"_legacy_active_short_{path_name}"
        long_col = f"_legacy_active_long_{path_name}"
        path_schedule = (
            schedule.filter(pl.col("portfolio_model") == model)
            .select(
                "decision_month",
                pl.col("n_short").alias(short_col),
                pl.col("n_long").alias(long_col),
            )
            .unique("decision_month")
        )
        result = result.join(path_schedule, on="decision_month", how="left")
        expressions: list[pl.Expr] = []
        for suffix in RELATIVE_EMA_SUFFIXES:
            output_name = f"legacy_active_{path_name}{suffix or '_raw'}"
            feature_columns.append(output_name)
            choices = [
                pl.when(
                    (pl.col(short_col) == short) & (pl.col(long_col) == long)
                ).then(pl.col(f"relative_ema_ratio_{short}_{long}{suffix}"))
                for short, long in available_pairs
            ]
            expressions.append(pl.coalesce(choices).alias(output_name))
        result = result.with_columns(expressions).drop([short_col, long_col])
    return result, tuple(feature_columns)
