"""Reconcile frozen rebalance intentions with the first market opening."""

from __future__ import annotations

import hashlib
import json
from datetime import time
from pathlib import Path

import polars as pl

from alpharank.portfolio.contracts import validate_holdings
from alpharank.portfolio.execution import NEW_YORK

OPENING_EXECUTION_POLICY_ID = "frozen_intents_first_market_open_v1"
_INTENT_KEYS = ("strategy", "decision_month", "holding_month", "ticker")
_OPEN_OBSERVED = "opening_quote_observed"


def evaluate_opening_intents(
    holdings: pl.DataFrame,
    prices: pl.DataFrame,
    benchmark_prices: pl.DataFrame,
    pre_execution_blocks: pl.DataFrame,
) -> pl.DataFrame:
    """Keep every intended position, even if its market-opening execution is blocked."""
    validate_holdings(holdings)
    if holdings.is_empty():
        raise ValueError("Opening execution requires at least one frozen intention.")
    rank = pl.col("selection_rank") if "selection_rank" in holdings else pl.lit(None)
    intents = holdings.select(*_INTENT_KEYS, "target_weight", rank.alias("selection_rank"))
    scheduled = intents.join(_market_openings(benchmark_prices), on="holding_month", how="left")
    with_quotes = scheduled.join(
        _opening_prices(prices), on=["ticker", "scheduled_open_date"], how="left", validate="m:1"
    )
    evidence = _known_suspensions(scheduled, pre_execution_blocks)
    return (
        with_quotes.join(evidence, on=["ticker", "scheduled_open_date"], how="left", validate="m:1")
        .with_columns(
            pl.when(pl.col("scheduled_open_at").is_null())
            .then(pl.lit("market_open_unavailable"))
            .when(pl.col("terminal_event_id").is_not_null())
            .then(pl.lit("known_pre_open_suspension"))
            .when(~pl.col("benchmark_open_usable").fill_null(False))
            .then(pl.lit("benchmark_open_unavailable"))
            .when(~_positive_price("opening_price"))
            .then(pl.lit("opening_quote_unavailable"))
            .when(~pl.col("opening_adjustment_usable").fill_null(False))
            .then(pl.lit("opening_adjustment_unavailable"))
            .otherwise(pl.lit(_OPEN_OBSERVED))
            .alias("execution_status")
        )
        .sort(_INTENT_KEYS)
    )


def _positive_price(column: str) -> pl.Expr:
    value = pl.col(column).cast(pl.Float64, strict=False)
    return (value.is_finite() & (value > 0.0)).fill_null(False)


def _opening_prices(prices: pl.DataFrame) -> pl.DataFrame:
    return prices.select(
        pl.col("ticker").cast(pl.String),
        pl.col("date").cast(pl.String).str.to_date(strict=False).alias("scheduled_open_date"),
        pl.col("open").cast(pl.Float64, strict=False).alias("opening_price"),
        (_positive_price("close") & _positive_price("adjusted_close")).alias(
            "opening_adjustment_usable"
        ),
    ).filter(pl.col("scheduled_open_date").is_not_null())


def _market_openings(prices: pl.DataFrame) -> pl.DataFrame:
    if prices["ticker"].drop_nulls().n_unique() != 1:
        raise ValueError("Opening execution requires exactly one benchmark ticker.")
    sessions = _opening_prices(prices)
    if sessions["scheduled_open_date"].is_duplicated().any():
        raise ValueError("Opening execution requires one benchmark quote per session.")
    return (
        sessions.with_columns(
            pl.col("scheduled_open_date").dt.truncate("1mo").alias("holding_month")
        )
        .sort("scheduled_open_date")
        .group_by("holding_month")
        .agg(
            pl.col("scheduled_open_date").first(),
            (_positive_price("opening_price") & pl.col("opening_adjustment_usable"))
            .first()
            .alias("benchmark_open_usable"),
        )
        .with_columns(
            pl.col("scheduled_open_date")
            .dt.combine(time(9, 30))
            .dt.replace_time_zone(NEW_YORK.key)
            .dt.convert_time_zone("UTC")
            .alias("scheduled_open_at")
        )
    )


def _known_suspensions(scheduled: pl.DataFrame, blocks: pl.DataFrame) -> pl.DataFrame:
    keys = ["ticker", "scheduled_open_date"]
    if blocks.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "scheduled_open_date": pl.Date,
                "terminal_event_id": pl.String,
                "terminal_effective_date": pl.Date,
                "terminal_known_at": pl.Datetime(time_zone="UTC"),
            }
        )
    return (
        scheduled.select(*keys, "scheduled_open_at")
        .unique()
        .join(blocks.filter(~pl.col("entry_allowed")), on="ticker", how="inner")
        .filter(
            (pl.col("known_at") < pl.col("scheduled_open_at"))
            & (pl.col("effective_date") <= pl.col("scheduled_open_date"))
        )
        .sort("known_at", "terminal_event_id")
        .unique(subset=keys, keep="last")
        .select(
            *keys,
            "terminal_event_id",
            pl.col("effective_date").alias("terminal_effective_date"),
            pl.col("known_at").alias("terminal_known_at"),
        )
    )


def record_opening_execution(
    journal: pl.DataFrame,
    destination: Path,
    *,
    composition_id: str,
    terminal_registry_sha256: str,
) -> tuple[Path, Path]:
    """Persist the full intent and execution status before permitting valuation."""
    journal_path = destination / "opening_execution_journal.parquet"
    receipt_path = destination / "opening_execution_receipt.json"
    journal.write_parquet(journal_path)
    outcomes = dict(journal.group_by("execution_status").len().iter_rows())
    blocked = journal.height - outcomes.get(_OPEN_OBSERVED, 0)
    receipt = {
        "policy_id": OPENING_EXECUTION_POLICY_ID,
        "composition_id": composition_id,
        "terminal_registry_sha256": terminal_registry_sha256,
        "status": "blocked_unavailable_opening" if blocked else "all_openings_observed",
        "intent_count": journal.height,
        "execution_status_counts": outcomes,
        "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
    }
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if blocked:
        raise RuntimeError(
            f"Opening execution blocked for {blocked} intentions. See {receipt_path}"
        )
    return journal_path, receipt_path


def validate_opening_execution(journal: pl.DataFrame, holdings: pl.DataFrame) -> None:
    """Disallow any valuation that substitutes a later quote or changes the frozen position."""
    contract = ("target_weight", "selection_rank", "execution_at", "execution_price_unadjusted")
    expected = journal.select(
        *_INTENT_KEYS,
        "target_weight",
        "selection_rank",
        pl.col("scheduled_open_at").alias("execution_at"),
        pl.col("opening_price").alias("execution_price_unadjusted"),
    ).sort(_INTENT_KEYS)
    if not expected.equals(holdings.select(*_INTENT_KEYS, *contract).sort(_INTENT_KEYS)):
        raise RuntimeError("Valued execution differs from the frozen intentions or first opening.")
