from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from alpharank.portfolio.allocation import equal_weights, select_ranked_candidates
from alpharank.portfolio.terminal_returns import resolve_terminal_shareholder_returns


def boosting_predictions_to_holdings(
    predictions: pl.DataFrame,
    *,
    strategy: str,
    top_n: int,
    score_column: str = "score",
    sector_column: str | None = None,
    maximum_names_per_sector: int | None = None,
    tie_breaker_columns: Sequence[str] = (),
    retention_rank: int | None = None,
    realized_return_column: str = "future_return_1m",
    benchmark_return_column: str = "benchmark_future_return_1m",
    terminal_events: pl.DataFrame | None = None,
    terminal_successor_prices: pl.DataFrame | None = None,
    terminal_price_vintage_id: str | None = None,
    terminal_starting_price_column: str = "last_close",
) -> pl.DataFrame:
    """Adapt out-of-sample scores without conditioning selection on future returns.

    Missing realized stock or benchmark returns remain visible on selected rows
    so downstream simulation can apply an explicit missing-return policy. They
    must never cause a lower-ranked candidate to enter the portfolio.
    """

    if retention_rank is not None:
        if retention_rank < top_n or maximum_names_per_sector is not None:
            raise ValueError("Rank retention requires rank >= top_n and no sector cap.")
        predictions = _retain_ranked_predictions(
            predictions, top_n, retention_rank, score_column, tie_breaker_columns
        )
    parts: list[pl.DataFrame] = []
    for month in predictions.partition_by("decision_month", maintain_order=True):
        selected = select_ranked_candidates(
            month,
            top_n=top_n,
            score_column=score_column,
            sector_column=sector_column,
            maximum_names_per_sector=maximum_names_per_sector,
            # Preserve the historical stable input-order tie rule. New callers
            # can request ticker tie-breaking at their signal boundary.
            tie_breaker_columns=tie_breaker_columns,
        )
        if selected.is_empty():
            continue
        parts.append(
            _selected_holdings(selected, strategy, realized_return_column, benchmark_return_column)
        )
    if not parts:
        return pl.DataFrame(
            schema={
                "strategy": pl.Utf8,
                "decision_month": pl.Date,
                "holding_month": pl.Date,
                "ticker": pl.Utf8,
                "target_weight": pl.Float64,
                "realized_return": pl.Float64,
                "benchmark_return": pl.Float64,
                "selection_rank": pl.Int64,
            }
        )
    holdings = pl.concat(parts, how="diagonal_relaxed").sort(
        ["strategy", "decision_month", "selection_rank"]
    )
    if terminal_events is not None:
        if terminal_price_vintage_id is None:
            raise ValueError(
                "terminal_price_vintage_id is required with terminal events."
            )
        holdings = resolve_terminal_shareholder_returns(
            holdings,
            terminal_events=terminal_events,
            successor_prices=terminal_successor_prices,
            price_vintage_id=terminal_price_vintage_id,
            starting_price_column=terminal_starting_price_column,
        ).holdings
    return holdings


def _selected_holdings(
    selected: pl.DataFrame, strategy: str, realized_column: str, benchmark_column: str
) -> pl.DataFrame:
    return selected.with_columns(
        pl.lit(strategy).alias("strategy"),
        pl.col("decision_month").dt.offset_by("1mo").alias("holding_month"),
        pl.Series("target_weight", equal_weights(selected.height)),
        pl.col(realized_column).cast(pl.Float64).alias("realized_return"),
        pl.col(benchmark_column).cast(pl.Float64).alias("benchmark_return"),
        pl.int_range(1, selected.height + 1, eager=True).alias("selection_rank"),
    )


def _retain_ranked_predictions(
    predictions: pl.DataFrame,
    top_n: int,
    retention_rank: int,
    score_column: str,
    tie_breaker_columns: Sequence[str],
) -> pl.DataFrame:
    """Retain incumbent candidates by rank, then fill vacant slots by score."""
    incumbents: list[str] = []
    selected_months: list[pl.DataFrame] = []
    months = predictions.sort("decision_month", maintain_order=True).partition_by("decision_month")
    for month in months:
        if month.height < top_n or month["ticker"].is_duplicated().any():
            raise ValueError("Rank buffering requires a full basket of unique eligible candidates.")
        ranked = select_ranked_candidates(
            month,
            top_n=retention_rank,
            score_column=score_column,
            tie_breaker_columns=tie_breaker_columns,
        )
        incumbent = pl.col("ticker").is_in(incumbents)
        retained = ranked.filter(incumbent)
        new = ranked.filter(~incumbent).head(top_n - retained.height)
        incumbents = retained["ticker"].to_list() + new["ticker"].to_list()
        selected_months.append(ranked.filter(pl.col("ticker").is_in(incumbents)))
    return pl.concat(selected_months) if selected_months else predictions
