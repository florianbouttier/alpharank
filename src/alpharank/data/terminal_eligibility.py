from __future__ import annotations

from dataclasses import dataclass

import polars as pl


TERMINAL_ENTRY_POLICY_ID = "reviewed_terminal_entry_gate_v1"


@dataclass(frozen=True)
class TerminalEntryGateResult:
    eligible: pl.DataFrame
    blocked: pl.DataFrame


def apply_terminal_entry_gate(
    candidates: pl.DataFrame,
    entry_blocks: pl.DataFrame,
    *,
    holding_month_column: str = "year_month",
    ticker_column: str = "ticker",
) -> TerminalEntryGateResult:
    """Reject known post-terminal entries before portfolio ranking."""

    required_candidates = {holding_month_column, ticker_column}
    missing_candidates = sorted(required_candidates.difference(candidates.columns))
    if missing_candidates:
        raise ValueError(
            f"Terminal entry candidates are missing columns: {missing_candidates}"
        )
    required_blocks = {
        "ticker",
        "terminal_event_id",
        "blocked_from_holding_month",
        "entry_block_rule",
    }
    missing_blocks = sorted(required_blocks.difference(entry_blocks.columns))
    if missing_blocks:
        raise ValueError(f"Terminal entry blocks are missing columns: {missing_blocks}")
    if candidates.is_empty() or entry_blocks.is_empty():
        return TerminalEntryGateResult(
            eligible=candidates.clone(),
            blocked=candidates.head(0),
        )

    normalized_blocks = entry_blocks.select(
        pl.col("ticker").cast(pl.String).alias("_terminal_entry_ticker"),
        pl.col("terminal_event_id").cast(pl.String),
        pl.col("blocked_from_holding_month")
        .cast(pl.Date, strict=False)
        .alias("_terminal_blocked_from_month"),
        pl.col("entry_block_rule").cast(pl.String),
    )
    duplicate_tickers = normalized_blocks.group_by("_terminal_entry_ticker").agg(
        pl.len().alias("event_count")
    ).filter(pl.col("event_count") > 1)
    if duplicate_tickers.height:
        raise ValueError(
            "Multiple terminal entry blocks exist for one ticker: "
            f"{duplicate_tickers['_terminal_entry_ticker'].to_list()}"
        )

    indexed = candidates.with_row_index("_terminal_entry_row_id").with_columns(
        pl.col(ticker_column).cast(pl.String).alias("_terminal_entry_ticker"),
        pl.col(holding_month_column)
        .cast(pl.Date, strict=False)
        .dt.truncate("1mo")
        .alias("_terminal_holding_month"),
    )
    if indexed.select(pl.col("_terminal_holding_month").is_null().any()).item():
        raise ValueError("Terminal entry candidates contain an invalid holding month.")
    joined = indexed.join(
        normalized_blocks,
        on="_terminal_entry_ticker",
        how="left",
        validate="m:1",
    ).with_columns(
        (
            pl.col("terminal_event_id").is_not_null()
            & (
                pl.col("_terminal_holding_month")
                >= pl.col("_terminal_blocked_from_month")
            )
        ).alias("terminal_entry_blocked")
    )
    blocked = joined.filter(pl.col("terminal_entry_blocked")).sort(
        [holding_month_column, ticker_column]
    )
    eligible_ids = joined.filter(~pl.col("terminal_entry_blocked")).select(
        "_terminal_entry_row_id"
    )
    eligible = (
        indexed.join(
            eligible_ids,
            on="_terminal_entry_row_id",
            how="semi",
        )
        .sort("_terminal_entry_row_id")
        .drop(
            "_terminal_entry_row_id",
            "_terminal_entry_ticker",
            "_terminal_holding_month",
        )
    )
    return TerminalEntryGateResult(eligible=eligible, blocked=blocked)
