from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence

import numpy as np
import polars as pl
from numpy.typing import NDArray

from alpharank.portfolio.comparison import align_return_series, subperiod_metric_grid

MAX_EQUAL_WEIGHT_STRATEGIES = 10
EQUAL_WEIGHT_REBALANCE_FREQUENCY = "monthly"
EQUAL_WEIGHT_ADDITIONAL_COST = 0.0


@dataclass(frozen=True, slots=True)
class EqualWeightCombinationGrid:
    """Precomputed equal-weight strategy sleeves and their canonical KPI."""

    strategy_order: tuple[str, ...]
    combination_masks: tuple[int, ...]
    months: tuple[date, ...]
    monthly_returns: np.ndarray
    metric_fields: tuple[str, ...]
    metric_windows: dict[str, list[list[float | None]]]


def equal_weight_strategy_combination_grid(
    series_by_strategy: dict[str, pl.DataFrame],
    *,
    benchmark_strategy: str,
    strategy_order: Sequence[str],
    metric_fields: Sequence[str],
    month_column: str = "holding_month",
    return_column: str = "net_return",
) -> EqualWeightCombinationGrid:
    """Build every non-empty monthly equal-weight combination of strategy sleeves.

    Input sleeve returns are assumed to already include each strategy's own
    transaction costs. The combination adds no separate inter-sleeve cost.
    """

    candidates = tuple(strategy_order)
    _validate_combination_contract(
        series_by_strategy,
        benchmark_strategy=benchmark_strategy,
        strategy_order=candidates,
    )
    aligned = align_return_series(
        {strategy: series_by_strategy[strategy] for strategy in (*candidates, benchmark_strategy)},
        month_column=month_column,
        return_column=return_column,
        how="inner",
    )
    months = tuple(aligned[month_column].to_list())
    if not months:
        raise ValueError("Equal-weight combinations require a non-empty common calendar.")
    base_returns = np.column_stack(
        [aligned[strategy].to_numpy() for strategy in candidates]
    ).astype(float, copy=False)
    benchmark_returns = aligned[benchmark_strategy].to_numpy().astype(float, copy=False)
    if not np.isfinite(base_returns).all() or not np.isfinite(benchmark_returns).all():
        raise ValueError("Equal-weight combinations require finite common-calendar returns.")

    masks = tuple(range(1, 1 << len(candidates)))
    memberships = _membership_matrix(masks, width=len(candidates))
    weights = memberships / memberships.sum(axis=1, keepdims=True)
    monthly_returns = base_returns @ weights.T
    combination_columns = [f"combination_{mask}" for mask in masks]
    monthly_wide = pl.DataFrame(
        {
            month_column: months,
            benchmark_strategy: benchmark_returns,
            **{
                column: monthly_returns[:, index]
                for index, column in enumerate(combination_columns)
            },
        }
    )
    windows = subperiod_metric_grid(
        monthly_wide,
        strategy_columns=combination_columns,
        benchmark_column=benchmark_strategy,
        metric_fields=metric_fields,
        month_column=month_column,
        calendar_year_boundaries_only=True,
    )
    return EqualWeightCombinationGrid(
        strategy_order=candidates,
        combination_masks=masks,
        months=months,
        monthly_returns=monthly_returns,
        metric_fields=tuple(metric_fields),
        metric_windows=windows,
    )


def _validate_combination_contract(
    series_by_strategy: dict[str, pl.DataFrame],
    *,
    benchmark_strategy: str,
    strategy_order: tuple[str, ...],
) -> None:
    if not strategy_order:
        raise ValueError("At least one strategy sleeve is required.")
    if len(strategy_order) > MAX_EQUAL_WEIGHT_STRATEGIES:
        raise ValueError(
            "Equal-weight combination grids support at most "
            f"{MAX_EQUAL_WEIGHT_STRATEGIES} strategy sleeves."
        )
    if len(set(strategy_order)) != len(strategy_order):
        raise ValueError("Strategy sleeves must be unique.")
    if benchmark_strategy in strategy_order:
        raise ValueError("The benchmark cannot be a strategy sleeve.")
    missing = sorted(set((*strategy_order, benchmark_strategy)) - set(series_by_strategy))
    if missing:
        raise ValueError("Combination grid is missing strategies: " + ", ".join(missing))


def _membership_matrix(masks: tuple[int, ...], *, width: int) -> NDArray[np.float64]:
    mask_values = np.asarray(masks, dtype=np.uint16)[:, None]
    bit_values = np.left_shift(np.uint16(1), np.arange(width, dtype=np.uint16))[None, :]
    membership: NDArray[np.bool_] = np.bitwise_and(mask_values, bit_values) != 0
    return membership.astype(np.float64)
