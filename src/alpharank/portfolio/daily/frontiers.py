"""Evaluate the existing constrained strategy simplex on an exact daily window."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import polars as pl

from alpharank.portfolio.daily.combinations import combine_daily_sleeves
from alpharank.portfolio.daily.statistics import close_window_metrics
from alpharank.portfolio.frontiers import (
    FRONTIER_METRIC_FIELDS,
    select_strategy_frontier,
    strategy_simplex_weights,
)


@dataclass(frozen=True, slots=True)
class DailyFrontierPeriod:
    """Bounded period and declared sleeve execution policy for a frontier."""

    start: date
    end: date
    benchmark_strategy: str
    rebalance: Literal["monthly", "entry"] = "monthly"


def daily_strategy_frontier(
    ledger: pl.DataFrame, strategies: tuple[str, ...], *, period: DailyFrontierPeriod
) -> dict[str, object]:
    """Post-hoc frontier uses the same daily source/period/costs, with gaps left ineligible."""
    calendar = sorted(ledger["valuation_date"].unique().to_list())
    earlier = [day for day in calendar if day < period.start]
    anchor = earlier[-1] if earlier else calendar[0]
    mask = pl.col("valuation_date").is_between(anchor, period.end)
    reference = ledger.filter((pl.col("strategy") == period.benchmark_strategy) & mask).sort(
        "valuation_date"
    )
    days = tuple(reference["valuation_date"].to_list())
    if len(days) < 2:
        raise ValueError("Frontier period contains no daily return.")
    benchmark = reference["wealth"].to_numpy().astype(float)
    points, rows = [], []
    for weights in strategy_simplex_weights(len(strategies)):
        mix = combine_daily_sleeves(
            ledger, dict(zip(strategies, weights, strict=True)), rebalance=period.rebalance
        )
        wealth = mix.filter(mask).sort("valuation_date")["wealth"].to_numpy().astype(float)
        metrics = close_window_metrics(wealth, benchmark, days)
        points.append(
            {"weights": weights.tolist(), **{key: metrics[key] for key in FRONTIER_METRIC_FIELDS}}
        )
        rows.append(tuple(metrics[key] for key in FRONTIER_METRIC_FIELDS))
    selected = select_strategy_frontier(tuple(rows))
    references = {}
    for name in (*strategies, period.benchmark_strategy):
        values = (
            ledger.filter((pl.col("strategy") == name) & mask)
            .sort("valuation_date")["wealth"]
            .to_numpy()
            .astype(float)
        )
        references[name] = close_window_metrics(values, benchmark, days)
    return {
        "strategy_order": list(strategies),
        "points": points,
        "volatility_frontier_indices": list(selected.volatility_frontier_indices),
        "drawdown_frontier_indices": list(selected.drawdown_frontier_indices),
        "recommendations": selected.recommendations,
        "reference_metrics": references,
    }
