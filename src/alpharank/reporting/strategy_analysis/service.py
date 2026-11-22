"""Read-only request orchestration; all financial calculations stay in portfolio."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

import polars as pl

from alpharank.portfolio.daily import (
    DailyFrontierPeriod,
    combine_daily_sleeves,
    daily_correlations,
    daily_period_analysis,
    daily_strategy_frontier,
)
from alpharank.portfolio.daily.contracts import ObservationFrequency
from alpharank.reporting.strategy_analysis.coverage import (
    monthly_fallback_window,
    require_complete_months,
    window_price_gaps,
    window_valuation_reconciliation,
)
from alpharank.reporting.strategy_analysis.dataset import AnalysisDataset


@dataclass(frozen=True, slots=True)
class AnalysisQuery:
    """One explicitly dated query against a captured source cohort."""

    start: date
    end: date
    source: str = "history"
    sleeves: tuple[str, ...] = ()
    frontier: bool = False


def query_analysis(dataset: AnalysisDataset, query: AnalysisQuery) -> dict[str, Any]:
    """Resolve a cohort and delegate an exact-period comparison to the common engine."""
    if query.source not in dataset.ledgers:
        raise ValueError("Unknown daily data source: " + query.source)
    bounds = dataset.sources[query.source]
    if query.start.isoformat() < bounds["min_date"] or query.end.isoformat() > bounds["max_date"]:
        raise ValueError(
            "La fenêtre dépasse la couverture de cette source : "
            + bounds["min_date"]
            + " au "
            + bounds["max_date"]
            + "."
        )
    ledger = dataset.ledgers[query.source]
    frequency: ObservationFrequency = "monthly" if query.source == "monthly_reference" else "daily"
    if frequency == "monthly":
        require_complete_months(query.start, query.end)
    policy: Literal["monthly", "entry"] = "entry" if query.source == "recent" else "monthly"
    allowed = tuple(dataset.payload["portfolio_composer"]["strategy_order"])
    if len(set(query.sleeves)) != len(query.sleeves) or not set(query.sleeves).issubset(allowed):
        raise ValueError("Unknown or duplicate portfolio sleeve.")
    if query.sleeves:
        allocations = dict.fromkeys(query.sleeves, 1.0 / len(query.sleeves))
        mix = combine_daily_sleeves(ledger, allocations, rebalance=policy)
        evaluated = pl.concat([ledger, mix], how="diagonal_relaxed")
    else:
        evaluated = ledger
    view: dict[str, Any] = dict(
        daily_period_analysis(
            evaluated,
            start=query.start,
            end=query.end,
            benchmark_strategy=dataset.benchmark_strategy,
            frequency=frequency,
        )
    )
    if query.sleeves:
        view["correlations"] = daily_correlations(
            ledger.filter(
                pl.col("valuation_date").is_between(
                    date.fromisoformat(view["anchor_date"]), date.fromisoformat(view["end_date"])
                )
            ),
            query.sleeves,
            benchmark_strategy=dataset.benchmark_strategy,
        )
        view["correlations"]["frequency"] = frequency
    if query.frontier:
        view["frontier"] = _frontier_view(
            dataset,
            ledger,
            DailyFrontierPeriod(
                query.start, query.end, dataset.benchmark_strategy, policy, frequency
            ),
        )
    available, gaps = (
        (False, [])
        if query.source == "recent"
        else window_price_gaps(
            dataset,
            start=date.fromisoformat(view["anchor_date"]),
            end=date.fromisoformat(view["end_date"]),
        )
    )
    view["price_gap_audit_available"], view["price_gaps"] = available, gaps
    view["monthly_fallback"] = monthly_fallback_window(
        query.start,
        query.end,
        enabled=query.source != "recent" and "monthly_reference" in dataset.ledgers,
    )
    view["source"] = query.source
    view["valuation_policy"] = bounds.get("valuation_policy", "observed_only")
    view["valuation_reconciliation"] = (
        window_valuation_reconciliation(
            dataset, start=date.fromisoformat(view["anchor_date"]), end=query.end
        )
        if query.source == "history"
        else []
    )
    return view


def _frontier_view(
    dataset: AnalysisDataset, ledger: pl.DataFrame, period: DailyFrontierPeriod
) -> dict[str, Any]:
    policy = dataset.payload.get("strategy_frontiers")
    if not policy:
        raise ValueError(
            "Cette version n'archive pas de politique de frontière ; aucune n'est inventée."
        )
    return daily_strategy_frontier(ledger, tuple(policy["strategy_order"]), period=period)
