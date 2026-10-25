from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from typing import Mapping, Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from alpharank.portfolio.comparison import align_return_series
from alpharank.portfolio.performance import advanced_performance_statistics

FRONTIER_WEIGHT_STEP = 0.05
FRONTIER_MAXIMUM_WEIGHT = 0.70
FRONTIER_REBALANCE_FREQUENCY = "monthly"
FRONTIER_ADDITIONAL_COST = 0.0
FRONTIER_METRIC_FIELDS = (
    "cagr",
    "annualized_volatility",
    "max_drawdown",
    "sharpe",
    "calmar",
)
FRONTIER_PROFILE_NAMES = (
    "minimum_volatility",
    "maximum_sharpe",
    "maximum_calmar",
)


@dataclass(frozen=True, slots=True)
class StrategyFrontierWindow:
    """Canonical metrics and Pareto selections for one inclusive window."""

    months: int
    candidate_metrics: tuple[tuple[float | None, ...], ...]
    volatility_frontier_indices: tuple[int, ...]
    drawdown_frontier_indices: tuple[int, ...]
    recommendations: dict[str, int]
    reference_metrics: dict[str, tuple[float | None, ...]]


@dataclass(frozen=True, slots=True)
class StrategyFrontierGrid:
    """Long-only strategy allocations evaluated on canonical monthly returns."""

    strategy_order: tuple[str, ...]
    benchmark_strategy: str
    months: tuple[date, ...]
    weights: NDArray[np.float64]
    metric_fields: tuple[str, ...]
    windows: dict[str, StrategyFrontierWindow]
    anchor_windows: tuple[dict[str, str | int], ...]


def constrained_strategy_frontier_grid(
    series_by_strategy: Mapping[str, pl.DataFrame],
    *,
    benchmark_strategy: str,
    strategy_order: Sequence[str],
    window_keys: Sequence[str],
    weight_step: float = FRONTIER_WEIGHT_STEP,
    maximum_weight: float = FRONTIER_MAXIMUM_WEIGHT,
    month_column: str = "holding_month",
    return_column: str = "net_return",
) -> StrategyFrontierGrid:
    """Evaluate a deterministic constrained simplex across historical windows."""

    strategies = tuple(strategy_order)
    _validate_contract(
        series_by_strategy,
        benchmark_strategy=benchmark_strategy,
        strategy_order=strategies,
        weight_step=weight_step,
        maximum_weight=maximum_weight,
    )
    aligned = align_return_series(
        {strategy: series_by_strategy[strategy] for strategy in (*strategies, benchmark_strategy)},
        month_column=month_column,
        return_column=return_column,
        how="inner",
    )
    months = tuple(aligned[month_column].to_list())
    if not months:
        raise ValueError("Strategy frontiers require a non-empty common calendar.")
    base_returns = np.column_stack(
        [aligned[strategy].to_numpy() for strategy in strategies]
    ).astype(float, copy=False)
    benchmark_returns = aligned[benchmark_strategy].to_numpy().astype(float, copy=False)
    if not np.isfinite(base_returns).all() or not np.isfinite(benchmark_returns).all():
        raise ValueError("Strategy frontiers require finite common-calendar returns.")

    weights = _simplex_weights(
        width=len(strategies),
        weight_step=weight_step,
        maximum_weight=maximum_weight,
    )
    candidate_returns = base_returns @ weights.T
    month_indices = {month.isoformat(): index for index, month in enumerate(months)}
    windows = {
        key: _window_metrics(
            key,
            month_indices=month_indices,
            candidate_returns=candidate_returns,
            base_returns=base_returns,
            benchmark_returns=benchmark_returns,
            strategy_order=strategies,
            benchmark_strategy=benchmark_strategy,
        )
        for key in window_keys
    }
    return StrategyFrontierGrid(
        strategy_order=strategies,
        benchmark_strategy=benchmark_strategy,
        months=months,
        weights=weights,
        metric_fields=FRONTIER_METRIC_FIELDS,
        windows=windows,
        anchor_windows=_anchor_windows(tuple(windows)),
    )


def _validate_contract(
    series_by_strategy: Mapping[str, pl.DataFrame],
    *,
    benchmark_strategy: str,
    strategy_order: tuple[str, ...],
    weight_step: float,
    maximum_weight: float,
) -> None:
    if len(strategy_order) < 2:
        raise ValueError("Strategy frontiers require at least two strategy sleeves.")
    if len(set(strategy_order)) != len(strategy_order):
        raise ValueError("Strategy frontier sleeves must be unique.")
    if benchmark_strategy in strategy_order:
        raise ValueError("The benchmark cannot be an optimized strategy sleeve.")
    missing = sorted(set((*strategy_order, benchmark_strategy)) - set(series_by_strategy))
    if missing:
        raise ValueError("Strategy frontier grid is missing: " + ", ".join(missing))
    units = round(1.0 / weight_step)
    if not math.isclose(units * weight_step, 1.0, abs_tol=1e-12):
        raise ValueError("weight_step must divide 100% exactly.")
    maximum_units = round(maximum_weight / weight_step)
    if not math.isclose(maximum_units * weight_step, maximum_weight, abs_tol=1e-12):
        raise ValueError("maximum_weight must be a multiple of weight_step.")
    if maximum_weight <= 0.0 or maximum_weight > 1.0:
        raise ValueError("maximum_weight must be in (0, 1].")
    if maximum_weight * len(strategy_order) < 1.0:
        raise ValueError("maximum_weight makes the simplex infeasible.")


def _simplex_weights(
    *,
    width: int,
    weight_step: float,
    maximum_weight: float,
) -> NDArray[np.float64]:
    total_units = round(1.0 / weight_step)
    maximum_units = round(maximum_weight / weight_step)
    allocations: list[tuple[int, ...]] = []

    def visit(prefix: tuple[int, ...], remaining: int) -> None:
        if len(prefix) == width - 1:
            if 0 <= remaining <= maximum_units:
                allocations.append((*prefix, remaining))
            return
        for units in range(min(maximum_units, remaining) + 1):
            visit((*prefix, units), remaining - units)

    visit((), total_units)
    if not allocations:
        raise ValueError("No allocation satisfies the strategy frontier constraints.")
    return np.round(np.asarray(allocations, dtype=float) * weight_step, decimals=12)


def _window_metrics(
    key: str,
    *,
    month_indices: dict[str, int],
    candidate_returns: NDArray[np.float64],
    base_returns: NDArray[np.float64],
    benchmark_returns: NDArray[np.float64],
    strategy_order: tuple[str, ...],
    benchmark_strategy: str,
) -> StrategyFrontierWindow:
    start_month, end_month = key.split("|")
    if start_month not in month_indices or end_month not in month_indices:
        raise ValueError(f"Unknown strategy frontier window: {key}")
    start = month_indices[start_month]
    end = month_indices[end_month] + 1
    if end <= start:
        raise ValueError(f"Invalid strategy frontier window: {key}")
    benchmark = benchmark_returns[start:end]
    candidate_metrics = tuple(
        _metric_row(candidate_returns[start:end, index], benchmark)
        for index in range(candidate_returns.shape[1])
    )
    references = {
        strategy: _metric_row(base_returns[start:end, index], benchmark)
        for index, strategy in enumerate(strategy_order)
    }
    references[benchmark_strategy] = _metric_row(benchmark, benchmark)
    cagr_index = FRONTIER_METRIC_FIELDS.index("cagr")
    volatility_index = FRONTIER_METRIC_FIELDS.index("annualized_volatility")
    drawdown_index = FRONTIER_METRIC_FIELDS.index("max_drawdown")
    return StrategyFrontierWindow(
        months=end - start,
        candidate_metrics=candidate_metrics,
        volatility_frontier_indices=_pareto_indices(
            candidate_metrics,
            return_index=cagr_index,
            risk_index=volatility_index,
            absolute_risk=False,
        ),
        drawdown_frontier_indices=_pareto_indices(
            candidate_metrics,
            return_index=cagr_index,
            risk_index=drawdown_index,
            absolute_risk=True,
        ),
        recommendations=_recommendations(candidate_metrics),
        reference_metrics=references,
    )


def _metric_row(
    returns: NDArray[np.float64],
    benchmark: NDArray[np.float64],
) -> tuple[float | None, ...]:
    metrics = advanced_performance_statistics(returns, benchmark_returns=benchmark)
    return tuple(_finite_or_none(metrics[field]) for field in FRONTIER_METRIC_FIELDS)


def _pareto_indices(
    rows: tuple[tuple[float | None, ...], ...],
    *,
    return_index: int,
    risk_index: int,
    absolute_risk: bool,
) -> tuple[int, ...]:
    valid: list[tuple[int, float, float]] = []
    for index, row in enumerate(rows):
        annual_return = row[return_index]
        risk_value = row[risk_index]
        if annual_return is None or risk_value is None:
            continue
        risk = abs(risk_value) if absolute_risk else risk_value
        valid.append((index, annual_return, risk))
    frontier = [
        candidate
        for candidate in valid
        if not any(
            other_risk <= candidate[2]
            and other_return >= candidate[1]
            and (other_risk < candidate[2] or other_return > candidate[1])
            for other_index, other_return, other_risk in valid
            if other_index != candidate[0]
        )
    ]
    return tuple(index for index, _, _ in sorted(frontier, key=lambda row: (row[2], row[1])))


def _recommendations(
    rows: tuple[tuple[float | None, ...], ...],
) -> dict[str, int]:
    cagr = FRONTIER_METRIC_FIELDS.index("cagr")
    volatility = FRONTIER_METRIC_FIELDS.index("annualized_volatility")
    sharpe = FRONTIER_METRIC_FIELDS.index("sharpe")
    calmar = FRONTIER_METRIC_FIELDS.index("calmar")
    return {
        "minimum_volatility": _best_index(
            rows,
            required=(volatility, cagr),
            key=lambda row: (
                _required_value(row, volatility),
                -_required_value(row, cagr),
            ),
        ),
        "maximum_sharpe": _best_index(
            rows,
            required=(sharpe, volatility),
            key=lambda row: (
                -_required_value(row, sharpe),
                _required_value(row, volatility),
            ),
        ),
        "maximum_calmar": _best_index(
            rows,
            required=(calmar, volatility),
            key=lambda row: (
                -_required_value(row, calmar),
                _required_value(row, volatility),
            ),
        ),
    }


def _best_index(
    rows: tuple[tuple[float | None, ...], ...],
    *,
    required: tuple[int, ...],
    key: Callable[[tuple[float | None, ...]], tuple[float, ...]],
) -> int:
    candidates = [
        (index, row)
        for index, row in enumerate(rows)
        if all(row[field] is not None for field in required)
    ]
    if not candidates:
        raise ValueError("No finite strategy frontier candidate is available.")
    return min(candidates, key=lambda item: (key(item[1]), item[0]))[0]


def _required_value(row: tuple[float | None, ...], index: int) -> float:
    value = row[index]
    if value is None:
        raise ValueError("Required frontier metric is unavailable.")
    return value


def _anchor_windows(window_keys: tuple[str, ...]) -> tuple[dict[str, str | int], ...]:
    if not window_keys:
        return ()
    starts = sorted({key.split("|")[0] for key in window_keys})
    ends = sorted({key.split("|")[1] for key in window_keys})
    latest_end = ends[-1]
    first_start = starts[0]
    anchors: list[dict[str, str | int]] = []
    for requested_year in (2010, 2015, 2020, 2023, 2025):
        target = first_start if requested_year == 2010 else f"{requested_year}-01-01"
        key = f"{target}|{latest_end}"
        if key not in window_keys:
            continue
        anchors.append(
            {
                "requested_start_year": requested_year,
                "effective_start": target,
                "end": latest_end,
                "window_key": key,
            }
        )
    return tuple(anchors)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None
