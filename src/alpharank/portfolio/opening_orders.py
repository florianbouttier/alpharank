"""Reconcile frozen rebalance intentions with the first market opening."""

from __future__ import annotations

import hashlib
import json
from datetime import date, time, timedelta
from pathlib import Path
from typing import cast

import polars as pl

from alpharank.portfolio.allocation import CASH_WEIGHT_KEY
from alpharank.portfolio.contracts import REBALANCE_OPEN_POLICY_ID, validate_holdings
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
    schedule = _market_openings(benchmark_prices).select(
        "holding_month", "scheduled_open_date", "scheduled_open_at", "benchmark_open_usable"
    )
    scheduled = intents.join(schedule, on="holding_month", how="left")
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
        pl.when(
            _positive_price("open") & _positive_price("close") & _positive_price("adjusted_close")
        )
        .then(pl.col("open") * pl.col("adjusted_close") / pl.col("close"))
        .alias("opening_price_adjusted"),
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
            pl.col("scheduled_open_date").last().alias("last_market_session"),
            pl.col("opening_price_adjusted").first().alias("benchmark_adjusted_open"),
            (_positive_price("opening_price") & pl.col("opening_adjustment_usable"))
            .first()
            .alias("benchmark_open_usable"),
        )
        .with_columns(
            _market_timestamp("scheduled_open_date", time(9, 30)).alias("scheduled_open_at")
        )
    )


def _market_timestamp(column: str, clock: time) -> pl.Expr:
    return (
        pl.col(column)
        .dt.combine(clock)
        .dt.replace_time_zone(NEW_YORK.key)
        .dt.convert_time_zone("UTC")
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


_VALUED_EXITS = ("observed_open_to_open", "resolved_terminal_event")


def evaluate_rebalance_exits(
    opening_journal: pl.DataFrame,
    prices: pl.DataFrame,
    benchmark_prices: pl.DataFrame,
    terminal_events: pl.DataFrame,
    *,
    reviewed_successor_openings: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Value the outgoing assets at the next market open, or record why a mark is unavailable."""
    openings = _market_openings(benchmark_prices)
    last_market_month = cast(date | None, openings["holding_month"].max())
    clock = _join_rebalance_clock(opening_journal, openings)
    quotes = _opening_prices(prices).select(
        "ticker",
        pl.col("scheduled_open_date").alias("exit_date"),
        pl.col("opening_price").alias("exit_price_unadjusted"),
        pl.col("opening_price_adjusted").alias("exit_price_adjusted"),
    )
    exits = _join_terminal_evidence(clock, terminal_events).join(
        quotes, on=["ticker", "exit_date"], how="left", validate="m:1"
    )
    if reviewed_successor_openings is not None:
        exits = _join_reviewed_successor_marks(exits, prices, reviewed_successor_openings)
    return exits.with_columns(
        _exit_status(last_market_month, review_successors=reviewed_successor_openings is not None)
    ).sort(_INTENT_KEYS)


def _join_rebalance_clock(journal: pl.DataFrame, openings: pl.DataFrame) -> pl.DataFrame:
    return (
        journal.join(
            openings.select("holding_month", "benchmark_adjusted_open"),
            on="holding_month",
            how="left",
            validate="m:1",
        )
        .join(
            openings.select(
                pl.col("holding_month").dt.offset_by("1mo"),
                _market_timestamp("last_market_session", time(16)).alias("signal_cutoff_at"),
            ),
            on="holding_month",
            how="left",
            validate="m:1",
        )
        .join(
            openings.select(
                pl.col("holding_month").dt.offset_by("-1mo"),
                pl.col("scheduled_open_date").alias("exit_date"),
                pl.col("scheduled_open_at").alias("exit_at"),
                pl.col("benchmark_adjusted_open").alias("benchmark_adjusted_exit"),
            ),
            on="holding_month",
            how="left",
            validate="m:1",
        )
    )


def _join_terminal_evidence(clock: pl.DataFrame, events: pl.DataFrame) -> pl.DataFrame:
    evidence = (
        clock.select("ticker", "holding_month", "scheduled_open_date", "exit_date")
        .unique()
        .join(events, on="ticker", how="inner")
        .filter(pl.col("effective_date").is_between("scheduled_open_date", "exit_date"))
        .select(
            "ticker",
            "holding_month",
            pl.col("terminal_event_id").alias("exit_event_id"),
            pl.col("event_type").alias("exit_event_type"),
            pl.col("effective_date").alias("exit_event_date"),
            pl.col("known_at").alias("exit_event_known_at"),
            pl.col("source").alias("exit_event_source"),
            pl.col("source_url").alias("exit_event_source_url"),
            pl.col("price_vintage_id").alias("exit_event_price_vintage_id"),
            pl.coalesce("cash_per_share", "recovery_per_share")
            .add(pl.col("distribution_per_share").fill_null(0.0))
            .alias("terminal_cash_per_share"),
        )
    )
    return clock.join(evidence, on=["ticker", "holding_month"], how="left", validate="m:1")


def _join_reviewed_successor_marks(
    exits: pl.DataFrame, prices: pl.DataFrame, evidence: pl.DataFrame
) -> pl.DataFrame:
    candidates = evidence["successor_ticker"].unique().to_list()
    volume = pl.col("volume").cast(pl.Float64, strict=False) if "volume" in prices else pl.lit(None)
    quotes = prices.filter(pl.col("ticker").is_in(candidates)).select(
        pl.col("ticker").alias("successor_ticker"),
        pl.col("date").cast(pl.String).str.to_date(strict=False).alias("exit_date"),
        pl.col("open").cast(pl.Float64, strict=False).alias("successor_open"),
        volume.alias("successor_volume"),
    )
    matched = evidence.join(quotes, on=["successor_ticker", "exit_date"], how="left", validate="m:1")
    verified = (
        _positive_price("successor_open")
        & ((pl.col("successor_open") - pl.col("reviewed_successor_open")).abs() <= 1e-5)
        & (pl.col("successor_volume") == pl.col("reviewed_successor_volume"))
    )
    return exits.join(matched, on=["exit_event_id", "exit_date"], how="left", validate="m:1").with_columns(
        pl.when(verified)
        .then(pl.col("successor_open") * pl.col("successor_ratio"))
        .alias("terminal_stock_per_share")
    )


def _exit_status(last_market_month: date | None, *, review_successors: bool) -> pl.Expr:
    terminal_terms_known = (
        pl.col("exit_event_date").is_between("scheduled_open_date", "exit_date", closed="none")
        & (pl.col("exit_event_known_at") < pl.col("exit_at"))
        & pl.col("terminal_cash_per_share").is_finite()
        & (pl.col("terminal_cash_per_share") >= 0)
    )
    terminal_cash = terminal_terms_known & pl.col("exit_event_type").is_in(
        ["cash_merger", "bankruptcy", "delisting"]
    )
    approved_stock, observed_stock = pl.lit(False), pl.lit(False)
    if review_successors:
        approved_stock = (
            terminal_terms_known
            & (pl.col("exit_event_type") == "stock_merger")
            & pl.col("successor_ticker").is_not_null()
        )
        observed_stock = approved_stock & _positive_price("terminal_stock_per_share")
    return (
        pl.when(pl.col("exit_at").is_null() & (pl.col("holding_month") == last_market_month))
        .then(pl.lit("pending_next_market_open"))
        .when(pl.col("exit_at").is_null() | pl.col("signal_cutoff_at").is_null())
        .then(pl.lit("missing_market_boundary"))
        .when(pl.col("execution_status") != _OPEN_OBSERVED)
        .then(pl.lit("opening_execution_blocked"))
        .when(
            ~(
                _positive_price("benchmark_adjusted_open")
                & _positive_price("benchmark_adjusted_exit")
            )
        )
        .then(pl.lit("benchmark_exit_unavailable"))
        .when(terminal_cash | observed_stock)
        .then(pl.lit("resolved_terminal_event"))
        .when(approved_stock)
        .then(pl.lit("terminal_successor_open_unavailable"))
        .when(pl.col("exit_event_id").is_not_null())
        .then(pl.lit("unsupported_terminal_exit"))
        .when(~_positive_price("exit_price_adjusted"))
        .then(pl.lit("exit_quote_unavailable"))
        .otherwise(pl.lit("observed_open_to_open"))
        .alias("valuation_status")
    )


def record_rebalance_exits(exits: pl.DataFrame, destination: Path) -> Path:
    """Keep pending and refused exit evidence before any performance is calculated."""
    journal = destination / "rebalance_valuation_journal.parquet"
    exits.write_parquet(journal)
    pending = "pending_next_market_open"
    unavailable = exits.filter(~pl.col("valuation_status").is_in([*_VALUED_EXITS, pending]))
    if unavailable.height:
        raise RuntimeError(
            f"Rebalance valuation blocked for {unavailable.height} intentions. See {journal}"
        )
    mature_months = (
        exits.filter(pl.col("valuation_status") != pending)
        .select("holding_month")
        .unique()
        .sort("holding_month")
    )
    if mature_months.is_empty():
        raise RuntimeError(f"Rebalance valuation has no completed holding period. See {journal}")
    if mature_months.filter(
        pl.col("holding_month") != pl.col("holding_month").shift(1).dt.offset_by("1mo")
    ).height:
        raise RuntimeError(
            f"Rebalance valuation calendar has an interior missing month. See {journal}"
        )
    return journal


def value_rebalance_holdings(intents: pl.DataFrame, exits: pl.DataFrame) -> pl.DataFrame:
    """Attach validated price marks and cash settlements to the frozen holdings."""
    terminal = pl.col("valuation_status") == "resolved_terminal_event"
    successor_columns: list[pl.Expr] = []
    settlement = pl.col("terminal_cash_per_share")
    stock, successor_ticker, successor_open = pl.lit(False), pl.lit(None, dtype=pl.String), pl.lit(None, dtype=pl.Float64)
    if "terminal_stock_per_share" in exits.columns:
        stock = terminal & pl.col("terminal_stock_per_share").is_not_null()
        successor_ticker, successor_open = pl.col("successor_ticker"), pl.col("successor_open")
        settlement = settlement + pl.col("terminal_stock_per_share").fill_null(0.0)
        successor_columns.append(
            pl.when(stock).then(pl.col("terminal_cash_per_share") / settlement)
            .otherwise(pl.lit(0.0)).alias("ending_cash_fraction")
        )
    matured = exits.filter(pl.col("valuation_status").is_in(_VALUED_EXITS))
    valuations = matured.select(
        *_INTENT_KEYS,
        pl.when(terminal)
        .then(settlement / pl.col("opening_price") - 1)
        .otherwise(pl.col("exit_price_adjusted") / pl.col("opening_price_adjusted") - 1)
        .alias("realized_return"),
        (pl.col("benchmark_adjusted_exit") / pl.col("benchmark_adjusted_open") - 1).alias(
            "benchmark_return"
        ),
        pl.col("opening_price").alias("execution_price_unadjusted"),
        pl.col("opening_price_adjusted").alias("execution_price_adjusted"),
        pl.col("scheduled_open_at").alias("execution_at"),
        pl.col("signal_cutoff_at"),
        pl.col("signal_cutoff_at").alias("feature_max_asof_at"),
        (pl.col("scheduled_open_at") + timedelta(microseconds=1)).alias(
            "first_return_observation_at"
        ),
        pl.col("exit_at").alias("holding_return_end_at"),
        pl.col("exit_at").alias("scheduled_holding_end_at"),
        pl.lit(0).alias("holding_observation_gap_calendar_days"),
        pl.lit(REBALANCE_OPEN_POLICY_ID).alias("execution_policy_id"),
        pl.col("valuation_status").alias("return_resolution"),
        pl.when(stock)
        .then(successor_ticker)
        .when(terminal)
        .then(pl.lit(CASH_WEIGHT_KEY))
        .otherwise(pl.col("ticker"))
        .alias("ending_ticker"),
        *successor_columns,
        pl.col("exit_event_id").alias("terminal_event_id"),
        pl.col("exit_event_type").alias("terminal_event_type"),
        pl.col("exit_event_date").alias("terminal_effective_date"),
        pl.col("exit_event_known_at").alias("terminal_event_known_at"),
        pl.col("exit_event_source").alias("terminal_event_source"),
        pl.col("exit_event_source_url").alias("terminal_event_source_url"),
        pl.col("exit_event_price_vintage_id").alias("terminal_price_vintage_id"),
        settlement.alias("terminal_value_per_share"),
        pl.when(stock).then(successor_ticker).alias("terminal_successor_ticker"),
        pl.when(stock).then(successor_open).alias("terminal_successor_end_price"),
        pl.when(stock).then(pl.lit("reviewed_successor_and_cash_at_rebalance"))
        .when(terminal).then(pl.lit("reviewed_cash_at_rebalance")).alias("return_resolution_reason"),
        pl.when(terminal)
        .then(pl.lit("reviewed_terminal_event_resolved"))
        .alias("manual_review_status"),
    )
    return intents.drop("realized_return", "benchmark_return").join(
        valuations, on=_INTENT_KEYS, how="inner", validate="1:1"
    )
