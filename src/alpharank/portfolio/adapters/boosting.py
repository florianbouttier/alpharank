from __future__ import annotations

import polars as pl

from alpharank.portfolio.allocation import equal_weights, select_ranked_candidates
from alpharank.portfolio.terminal_returns import resolve_terminal_shareholder_returns


def boosting_predictions_to_holdings(
    predictions: pl.DataFrame,
    *,
    strategy: str,
    top_n: int,
    score_column: str = "score",
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

    parts: list[pl.DataFrame] = []
    for month in predictions.partition_by("decision_month", maintain_order=True):
        selected = select_ranked_candidates(
            month,
            top_n=top_n,
            score_column=score_column,
            # Preserve the historical stable input-order tie rule. New callers
            # can request ticker tie-breaking at their signal boundary.
            tie_breaker_columns=(),
        )
        if selected.is_empty():
            continue
        weights = equal_weights(selected.height)
        parts.append(
            selected.with_columns(
                pl.lit(strategy).alias("strategy"),
                pl.col("decision_month").dt.offset_by("1mo").alias("holding_month"),
                pl.Series("target_weight", weights),
                pl.col(realized_return_column).cast(pl.Float64).alias("realized_return"),
                pl.col(benchmark_return_column).cast(pl.Float64).alias("benchmark_return"),
                pl.int_range(1, selected.height + 1, eager=True).alias("selection_rank"),
            )
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
