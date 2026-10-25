from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.portfolio.frontiers import (
    FRONTIER_METRIC_FIELDS,
    constrained_strategy_frontier_grid,
)
from alpharank.portfolio.performance import advanced_performance_statistics


def test_frontier_grid_respects_simplex_and_canonical_metrics() -> None:
    months = [date(2024, month, 1) for month in range(1, 7)]
    returns = {
        "Legacy": [0.02, -0.01, 0.03, 0.01, -0.02, 0.04],
        "Boosting": [0.04, -0.04, 0.05, -0.01, 0.01, 0.06],
        "Trend": [0.01, 0.01, 0.015, 0.005, 0.01, 0.02],
        "SPY": [0.01, -0.005, 0.02, 0.005, -0.01, 0.025],
    }
    series = {
        strategy: pl.DataFrame({"holding_month": months, "net_return": strategy_returns})
        for strategy, strategy_returns in returns.items()
    }
    window_key = "2024-01-01|2024-06-01"

    grid = constrained_strategy_frontier_grid(
        series,
        benchmark_strategy="SPY",
        strategy_order=("Legacy", "Boosting", "Trend"),
        window_keys=(window_key,),
    )

    assert grid.weights.shape[1] == 3
    assert np.allclose(grid.weights.sum(axis=1), 1.0)
    assert np.all(grid.weights >= 0.0)
    assert np.all(grid.weights <= 0.70)
    assert np.allclose(grid.weights / 0.05, np.round(grid.weights / 0.05))

    window = grid.windows[window_key]
    candidate_index = next(
        index
        for index, weights in enumerate(grid.weights.tolist())
        if weights == pytest.approx([0.3, 0.3, 0.4])
    )
    expected = advanced_performance_statistics(
        np.asarray(returns["Legacy"]) * 0.3
        + np.asarray(returns["Boosting"]) * 0.3
        + np.asarray(returns["Trend"]) * 0.4,
        benchmark_returns=returns["SPY"],
    )
    row = window.candidate_metrics[candidate_index]
    for field, value in zip(FRONTIER_METRIC_FIELDS, row, strict=True):
        assert value == pytest.approx(expected[field])


def test_frontiers_are_non_dominated_and_profiles_point_to_candidates() -> None:
    months = [date(2023, month, 1) for month in range(1, 13)]
    series = {
        "Legacy": _series(months, [0.012, -0.008, 0.018, 0.006] * 3),
        "Boosting": _series(months, [0.03, -0.04, 0.04, 0.01] * 3),
        "Trend": _series(months, [0.008, 0.007, 0.009, 0.006] * 3),
        "SPY": _series(months, [0.01, -0.015, 0.02, 0.005] * 3),
    }
    key = "2023-01-01|2023-12-01"
    grid = constrained_strategy_frontier_grid(
        series,
        benchmark_strategy="SPY",
        strategy_order=("Legacy", "Boosting", "Trend"),
        window_keys=(key,),
    )
    window = grid.windows[key]
    cagr_index = FRONTIER_METRIC_FIELDS.index("cagr")
    volatility_index = FRONTIER_METRIC_FIELDS.index("annualized_volatility")
    drawdown_index = FRONTIER_METRIC_FIELDS.index("max_drawdown")

    _assert_non_dominated(
        window.candidate_metrics,
        window.volatility_frontier_indices,
        return_index=cagr_index,
        risk_index=volatility_index,
        absolute_risk=False,
    )
    _assert_non_dominated(
        window.candidate_metrics,
        window.drawdown_frontier_indices,
        return_index=cagr_index,
        risk_index=drawdown_index,
        absolute_risk=True,
    )
    assert set(window.recommendations) == {
        "minimum_volatility",
        "maximum_sharpe",
        "maximum_calmar",
    }
    assert all(0 <= index < len(grid.weights) for index in window.recommendations.values())


def _series(months: list[date], returns: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"holding_month": months, "net_return": returns})


def _assert_non_dominated(
    metrics: tuple[tuple[float | None, ...], ...],
    frontier_indices: tuple[int, ...],
    *,
    return_index: int,
    risk_index: int,
    absolute_risk: bool,
) -> None:
    for candidate_index in frontier_indices:
        candidate_return = metrics[candidate_index][return_index]
        candidate_risk = metrics[candidate_index][risk_index]
        assert candidate_return is not None and candidate_risk is not None
        risk = abs(candidate_risk) if absolute_risk else candidate_risk
        for other_index, other in enumerate(metrics):
            if other_index == candidate_index:
                continue
            other_return = other[return_index]
            other_risk_value = other[risk_index]
            if other_return is None or other_risk_value is None:
                continue
            other_risk = abs(other_risk_value) if absolute_risk else other_risk_value
            assert not (
                other_risk <= risk
                and other_return >= candidate_return
                and (other_risk < risk or other_return > candidate_return)
            )
