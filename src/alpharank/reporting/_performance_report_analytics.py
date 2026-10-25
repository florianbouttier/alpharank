from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import polars as pl

from alpharank.portfolio.combinations import (
    EqualWeightCombinationGrid,
    equal_weight_strategy_combination_grid,
)
from alpharank.portfolio.comparison import subperiod_portfolio_metric_grid
from alpharank.portfolio.frontiers import (
    FRONTIER_ADDITIONAL_COST,
    FRONTIER_MAXIMUM_WEIGHT,
    FRONTIER_REBALANCE_FREQUENCY,
    FRONTIER_WEIGHT_STEP,
    StrategyFrontierGrid,
    constrained_strategy_frontier_grid,
)

REPORT_FRONTIER_STRATEGY_ORDER = (
    "Legacy · Frequency",
    "Boosting · Top 5",
    "Boosting tendance · Top 5",
)


def build_report_analytics(
    monthly: pl.DataFrame,
    *,
    strategy_order: Sequence[str],
    benchmark_strategy: str,
    performance_metric_fields: Sequence[str],
    composer_metric_fields: Sequence[str],
) -> tuple[
    dict[str, pl.DataFrame],
    dict[str, list[list[float | None]]],
    EqualWeightCombinationGrid,
    StrategyFrontierGrid,
]:
    """Precompute every browser-selectable strategy and allocation metric."""

    series = {
        strategy: monthly.filter(pl.col("strategy") == strategy).sort("holding_month")
        for strategy in strategy_order
    }
    metric_windows = subperiod_portfolio_metric_grid(
        series,
        benchmark_strategy=benchmark_strategy,
        strategy_order=strategy_order,
        metric_fields=performance_metric_fields,
        calendar_year_boundaries_only=True,
    )
    composer = equal_weight_strategy_combination_grid(
        series,
        benchmark_strategy=benchmark_strategy,
        strategy_order=[strategy for strategy in strategy_order if strategy != benchmark_strategy],
        metric_fields=composer_metric_fields,
    )
    frontiers = constrained_strategy_frontier_grid(
        series,
        benchmark_strategy=benchmark_strategy,
        strategy_order=REPORT_FRONTIER_STRATEGY_ORDER,
        window_keys=tuple(metric_windows),
    )
    return series, metric_windows, composer, frontiers


def strategy_frontier_payload(grid: StrategyFrontierGrid) -> dict[str, Any]:
    """Serialize a frontier grid without introducing browser-side finance."""

    return {
        "strategy_order": list(grid.strategy_order),
        "benchmark_strategy": grid.benchmark_strategy,
        "weights": grid.weights.tolist(),
        "metric_fields": list(grid.metric_fields),
        "windows": {
            key: {
                "months": window.months,
                "candidate_metrics": [list(row) for row in window.candidate_metrics],
                "volatility_frontier_indices": list(window.volatility_frontier_indices),
                "drawdown_frontier_indices": list(window.drawdown_frontier_indices),
                "recommendations": window.recommendations,
                "reference_metrics": {
                    strategy: list(row) for strategy, row in window.reference_metrics.items()
                },
            }
            for key, window in grid.windows.items()
        },
        "anchor_windows": list(grid.anchor_windows),
        "policy": _frontier_policy(),
    }


def frontier_methodology_card() -> dict[str, Any]:
    """Describe the fixed frontier contract beside the interactive charts."""

    return {
        "title": "Frontières efficientes multi-stratégies",
        "status": "Diagnostic post-hoc, non promu",
        "summary": (
            "Trois poches distinctes sont combinées sous contraintes pour lire les "
            "compromis rendement/volatilité et rendement/drawdown."
        ),
        "pseudo_code": [
            "énumérer les poids longs par pas de 5 %, plafonnés à 70 % par poche",
            "rééquilibrer mensuellement les trois rendements nets sans levier",
            "calculer tous les KPI avec le moteur commun face au SPY",
            "retenir les points non dominés et trois profils descriptifs par fenêtre",
        ],
    }


def _frontier_policy() -> dict[str, Any]:
    return {
        "method": "constrained_monthly_strategy_sleeves",
        "rebalance_frequency": FRONTIER_REBALANCE_FREQUENCY,
        "weight_step": FRONTIER_WEIGHT_STEP,
        "maximum_strategy_weight": FRONTIER_MAXIMUM_WEIGHT,
        "long_only": True,
        "leverage": False,
        "input_returns": "net_return après les frais propres à chaque stratégie",
        "additional_inter_sleeve_cost": FRONTIER_ADDITIONAL_COST,
        "selection": "diagnostic post-hoc sur chaque fenêtre, sans promotion",
        "history_note": (
            "La demande 2010-aujourd'hui commence effectivement en août 2011, "
            "premier mois commun observable."
        ),
    }
