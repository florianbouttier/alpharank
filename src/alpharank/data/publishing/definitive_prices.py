from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from alpharank.data.warehouse.staging import (
    PRICE_COLUMNS,
    PRICE_METADATA_COLUMNS,
    PRICE_VALUE_COLUMNS,
)
from alpharank.data.warehouse.staging import stage_yahoo_prices as stage_yahoo_prices


@dataclass(frozen=True)
class DefinitivePriceResult:
    frame: pl.DataFrame
    audit: pl.DataFrame
    current_row_count: int
    carried_forward_row_count: int
    unresolved_row_count: int


def bootstrap_definitive_prices(previous_validated: pl.DataFrame) -> pl.DataFrame:
    """Adapt the retained validated lineage to the DEF price contract."""

    required = {"ticker", "date", *PRICE_VALUE_COLUMNS, *PRICE_METADATA_COLUMNS}
    missing = sorted(required - set(previous_validated.columns))
    if missing:
        raise ValueError(f"Previous validated DEF price columns are missing: {missing}")
    frame = previous_validated.select(list(PRICE_COLUMNS)).sort(["ticker", "date"])
    _require_unique_price_keys(frame, layer="previous DEF")
    return frame


def build_definitive_prices(
    *,
    staged_current: pl.DataFrame,
    previous_definitive: pl.DataFrame,
    requested_tickers: Sequence[str],
    freeze_previous_prefix_tickers: Sequence[str] = (),
) -> DefinitivePriceResult:
    """Resolve exact ticker/date keys while retaining the selected RAW origin."""

    current = bootstrap_definitive_prices(staged_current)
    previous = bootstrap_definitive_prices(previous_definitive)
    requested = sorted(
        {f"{str(ticker).upper().removesuffix('.US')}.US" for ticker in requested_tickers}
    )
    frozen_prefix_tickers = sorted(
        {
            f"{str(ticker).upper().removesuffix('.US')}.US"
            for ticker in freeze_previous_prefix_tickers
        }
    )
    previous = previous.filter(pl.col("ticker").is_in(requested))
    previous_prefix_ends = (
        previous.filter(pl.col("ticker").is_in(frozen_prefix_tickers))
        .group_by("ticker")
        .agg(pl.col("date").max().alias("__previous_prefix_end"))
    )

    current = current.with_columns(pl.lit(True).alias("__current_present")).rename(
        {column: f"current__{column}" for column in (*PRICE_VALUE_COLUMNS, *PRICE_METADATA_COLUMNS)}
    )
    previous = previous.with_columns(pl.lit(True).alias("__previous_present")).rename(
        {column: f"previous__{column}" for column in (*PRICE_VALUE_COLUMNS, *PRICE_METADATA_COLUMNS)}
    )
    joined = previous.join(
        current,
        on=["ticker", "date"],
        how="full",
        coalesce=True,
    ).join(
        previous_prefix_ends,
        on="ticker",
        how="left",
    ).with_columns(
        pl.col("__current_present").fill_null(False),
        pl.col("__previous_present").fill_null(False),
    )
    current_valid = (
        pl.col("__current_present")
        & pl.col("current__adjusted_close").is_not_null()
        & (pl.col("current__adjusted_close") > 0.0)
    )
    previous_valid = (
        pl.col("__previous_present")
        & pl.col("previous__adjusted_close").is_not_null()
        & (pl.col("previous__adjusted_close") > 0.0)
    )
    frozen_previous_prefix = (
        pl.col("ticker").is_in(frozen_prefix_tickers)
        & pl.col("__previous_prefix_end").is_not_null()
        & (pl.col("date") <= pl.col("__previous_prefix_end"))
    )
    current_selectable = current_valid & ~frozen_previous_prefix
    joined = joined.with_columns(
        current_valid.alias("__current_valid"),
        previous_valid.alias("__previous_valid"),
        frozen_previous_prefix.alias("__frozen_previous_prefix"),
        current_selectable.alias("__current_selectable"),
    ).with_columns(
        pl.when(pl.col("__frozen_previous_prefix") & pl.col("__previous_valid"))
        .then(pl.lit("carried_forward_incomplete_ticker_prefix"))
        .when(pl.col("__frozen_previous_prefix") & ~pl.col("__previous_valid"))
        .then(pl.lit("unresolved_new_key_in_incomplete_ticker_prefix"))
        .when(pl.col("__current_selectable"))
        .then(pl.lit("current_raw"))
        .when(~pl.col("__current_present") & pl.col("__previous_valid"))
        .then(pl.lit("carried_forward_missing_current_raw"))
        .when(pl.col("__current_present") & ~pl.col("__current_valid") & pl.col("__previous_valid"))
        .then(pl.lit("carried_forward_invalid_current_raw"))
        .when(pl.col("__current_present"))
        .then(pl.lit("unresolved_invalid_current_raw"))
        .otherwise(pl.lit("unresolved_missing_current_raw"))
        .alias("selection_reason")
    )

    resolved = joined.filter(
        pl.col("__current_selectable") | pl.col("__previous_valid")
    )
    frame_expressions: list[pl.Expr] = [pl.col("date"), pl.col("ticker")]
    for column in (*PRICE_VALUE_COLUMNS, *PRICE_METADATA_COLUMNS):
        frame_expressions.append(
            pl.when(pl.col("__current_selectable"))
            .then(pl.col(f"current__{column}"))
            .otherwise(pl.col(f"previous__{column}"))
            .alias(column)
        )
    frame = resolved.select(frame_expressions).select(list(PRICE_COLUMNS)).sort(
        ["ticker", "date"]
    )
    _require_unique_price_keys(frame, layer="DEF")

    audit = joined.filter(pl.col("selection_reason") != "current_raw").select(
        "ticker",
        "date",
        "selection_reason",
        pl.col("current__ingestion_run_id").alias("current_raw_run_id"),
        pl.col("previous__ingestion_run_id").alias("selected_previous_raw_run_id"),
        pl.col("current__adjusted_close").alias("current_adjusted_close"),
        pl.col("previous__adjusted_close").alias("selected_previous_adjusted_close"),
    ).sort(["ticker", "date"])
    carried_count = audit.filter(pl.col("selection_reason").str.starts_with("carried_forward")).height
    unresolved_count = audit.filter(pl.col("selection_reason").str.starts_with("unresolved")).height
    return DefinitivePriceResult(
        frame=frame,
        audit=audit,
        current_row_count=frame.height - carried_count,
        carried_forward_row_count=carried_count,
        unresolved_row_count=unresolved_count,
    )


def _require_unique_price_keys(frame: pl.DataFrame, *, layer: str) -> None:
    duplicate_count = frame.select(pl.struct(["ticker", "date"]).is_duplicated().sum()).item()
    if duplicate_count:
        raise ValueError(
            f"{layer} price frame contains {duplicate_count} duplicate ticker/date rows"
        )
