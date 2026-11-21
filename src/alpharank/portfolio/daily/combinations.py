"""Daily paths of strategy sleeves with explicit monthly or entry-only rebalance."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray


def _sleeve_frames(
    ledger: pl.DataFrame, allocations: Mapping[str, float]
) -> tuple[list[pl.DataFrame], NDArray[np.float64]]:
    if not allocations or not all(math.isfinite(w) and w >= 0 for w in allocations.values()):
        raise ValueError("Strategy sleeves require finite non-negative allocation weights.")
    if not math.isclose(sum(allocations.values()), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Strategy sleeve allocations must sum to one.")
    positive = {name: weight for name, weight in allocations.items() if weight > 0}
    frames = [ledger.filter(pl.col("strategy") == name).sort("valuation_date") for name in positive]
    if frames[0].is_empty() or any(
        f["valuation_date"].to_list() != frames[0]["valuation_date"].to_list() for f in frames
    ):
        raise ValueError(
            "Every selected sleeve must share the full daily calendar, including gaps."
        )
    return frames, np.asarray(list(positive.values()), dtype=float)


def _monthly_path(frames: list[pl.DataFrame], weights: NDArray[np.float64]) -> NDArray[np.float64]:
    if "sealed_monthly_return" not in frames[0].columns or "month_factor" not in frames[0].columns:
        raise ValueError("Monthly sleeve rebalancing requires sealed monthly anchors.")
    factors = np.column_stack([frame["month_factor"].to_numpy() for frame in frames]).astype(float)
    returns = np.column_stack(
        [frame["sealed_monthly_return"].to_numpy() for frame in frames]
    ).astype(float)
    months = frames[0]["holding_month"].to_list()
    path = np.empty(len(months), dtype=float)
    capital, current = 1.0, months[0]
    for index, month in enumerate(months):
        if month != current:
            capital *= 1 + float(np.dot(returns[index - 1], weights))
            current = month
        path[index] = capital * float(np.dot(factors[index], weights))
    return path


def combine_daily_sleeves(
    ledger: pl.DataFrame,
    allocations: Mapping[str, float],
    *,
    label: str = "Portefeuille composé",
    rebalance: Literal["monthly", "entry"] = "monthly",
) -> pl.DataFrame:
    """Combine whole sleeves, never daily-rebalance their constituent securities.

    A historical mix resets sleeve weights monthly and preserves their recorded
    net returns/costs. Entry-only follow-up keeps the original sleeve shares.
    No additional transfer cost or netting of duplicate securities is assumed.
    """
    frames, weights = _sleeve_frames(ledger, allocations)
    if rebalance == "monthly":
        path = _monthly_path(frames, weights)
    elif rebalance == "entry":
        path = (
            np.column_stack([frame["wealth"].to_numpy() for frame in frames]).astype(float)
            @ weights
        )
    else:
        raise ValueError("Unknown sleeve rebalance frequency.")
    result = frames[0].with_columns(
        pl.lit(label).alias("strategy"), pl.Series("wealth", path).fill_nan(None)
    )
    if rebalance == "monthly":
        for column in ["month_factor", "sealed_monthly_return"]:
            values = (
                np.column_stack([frame[column].to_numpy() for frame in frames]).astype(float)
                @ weights
            )
            result = result.with_columns(pl.Series(column, values).fill_nan(None))
    if "daily_return" in result.columns:
        result = result.with_columns(
            (pl.col("wealth") / pl.col("wealth").shift(1) - 1).alias("daily_return")
        )
    for column in ["transaction_cost", "turnover"]:
        if all(column in frame.columns for frame in frames):
            values = (
                np.column_stack([frame[column].to_numpy() for frame in frames]).astype(float)
                @ weights
            )
            result = result.with_columns(pl.Series(column, values))
    for column in ["n_positions", "maximum_position_weight", "maximum_sector_weight"]:
        if column in result.columns:
            result = result.with_columns(pl.lit(None, dtype=pl.Float64).alias(column))
    return result


def daily_correlations(
    ledger: pl.DataFrame, strategies: tuple[str, ...], *, benchmark_strategy: str
) -> dict[str, object]:
    """Pearson on same-date daily returns and active returns; no pairwise gap deletion."""
    names = tuple(dict.fromkeys((*strategies, benchmark_strategy)))
    frames = [ledger.filter(pl.col("strategy") == name).sort("valuation_date") for name in names]
    if not frames or any(frame.height != frames[0].height for frame in frames):
        raise ValueError("Correlations require a complete shared date calendar.")
    wealth = np.column_stack([frame["wealth"].to_numpy() for frame in frames]).astype(float)
    returns = wealth[1:] / wealth[:-1] - 1
    active = returns - returns[:, [-1]]

    def correlation_matrix(values: NDArray[np.float64]) -> list[list[float | None]]:
        matrix = []
        for first in values.T:
            row = []
            for second in values.T:
                valid = np.isfinite(first).all() and np.isfinite(second).all() and len(first) > 1
                correlation = (
                    float(np.corrcoef(first, second)[0, 1])
                    if valid and np.std(first) > 0 and np.std(second) > 0
                    else None
                )
                row.append(correlation)
            matrix.append(row)
        return matrix

    return {
        "strategies": list(names),
        "returns": correlation_matrix(returns),
        "active_returns": correlation_matrix(active),
    }
