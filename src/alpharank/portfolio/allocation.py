from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl


CASH_WEIGHT_KEY = "__CASH__"


def portfolio_turnover(previous: Mapping[str, float], current: Mapping[str, float]) -> float:
    """One-way turnover using half the L1 change, including residual cash."""

    previous_complete = _weights_with_residual_cash(previous, empty_is_cash=True)
    current_complete = _weights_with_residual_cash(current, empty_is_cash=False)
    names = set(previous_complete) | set(current_complete)
    return 0.5 * sum(
        abs(
            float(current_complete.get(name, 0.0))
            - float(previous_complete.get(name, 0.0))
        )
        for name in names
    )


def drifted_weights_after_returns(
    target_weights: Mapping[str, float],
    realized_returns: Mapping[str, float],
) -> dict[str, float]:
    """Derive end-of-period weights before the next rebalance."""

    complete = _weights_with_residual_cash(target_weights, empty_is_cash=False)
    values: dict[str, float] = {}
    for name, weight in complete.items():
        realized = 0.0 if name == CASH_WEIGHT_KEY else float(realized_returns[name])
        if not np.isfinite(realized) or realized < -1.0:
            raise ValueError(f"Invalid realized return for drifted weight: {name}={realized}")
        values[name] = float(weight) * (1.0 + realized)
    total = sum(values.values())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError("Portfolio wealth must remain positive after realized returns.")
    return {name: value / total for name, value in values.items() if value != 0.0}


def _weights_with_residual_cash(
    weights: Mapping[str, float], *, empty_is_cash: bool
) -> dict[str, float]:
    normalized = {str(name): float(weight) for name, weight in weights.items()}
    if not normalized and empty_is_cash:
        return {CASH_WEIGHT_KEY: 1.0}
    if any(not np.isfinite(weight) or weight < 0.0 for weight in normalized.values()):
        raise ValueError("Portfolio weights must be finite and non-negative.")
    invested = sum(normalized.values())
    if invested > 1.0 + 1e-9:
        raise ValueError("Portfolio weights cannot exceed one including cash.")
    normalized[CASH_WEIGHT_KEY] = normalized.get(CASH_WEIGHT_KEY, 0.0) + max(
        0.0, 1.0 - invested
    )
    return normalized


def equal_weights(size: int) -> np.ndarray:
    if size < 0:
        raise ValueError("size must be non-negative.")
    if size == 0:
        return np.asarray([], dtype=float)
    return np.full(size, 1.0 / size, dtype=float)


def select_ranked_candidates(
    frame: pl.DataFrame,
    *,
    top_n: int,
    score_column: str = "score",
    sector_column: str | None = None,
    maximum_names_per_sector: int | None = None,
    tie_breaker_columns: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Select a deterministic top-N, optionally respecting a sector name cap."""

    if top_n <= 0:
        raise ValueError("top_n must be positive.")
    if score_column not in frame.columns:
        raise ValueError(f"Missing score column: {score_column}")
    tie_breakers = (
        ["ticker"]
        if tie_breaker_columns is None and "ticker" in frame.columns
        else list(tie_breaker_columns or ())
    )
    sort_columns = [score_column, *tie_breakers]
    descending = [True, *([False] * len(tie_breakers))]
    ordered = frame.sort(sort_columns, descending=descending)
    if maximum_names_per_sector is None:
        return ordered.head(top_n)
    if sector_column is None or sector_column not in ordered.columns:
        raise ValueError("A sector column is required when a sector name cap is set.")
    selected_indices: list[int] = []
    counts: dict[str, int] = {}
    for index, sector in enumerate(ordered[sector_column].to_list()):
        normalized_sector = str(sector)
        if counts.get(normalized_sector, 0) >= maximum_names_per_sector:
            continue
        selected_indices.append(index)
        counts[normalized_sector] = counts.get(normalized_sector, 0) + 1
        if len(selected_indices) == top_n:
            break
    if len(selected_indices) < min(top_n, ordered.height):
        raise ValueError("Not enough names to satisfy maximum_names_per_sector.")
    return ordered[selected_indices]


def capped_inverse_risk_weights(
    risk: Sequence[float],
    *,
    maximum_weight: float,
    floor_quantile: float = 0.20,
) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    if values.size == 0:
        return values
    if maximum_weight * values.size < 1.0 - 1e-12:
        raise ValueError("maximum_weight is infeasible for the number of assets.")
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size == 0:
        base = np.ones(values.size, dtype=float)
    else:
        floor = max(float(np.quantile(finite_positive, floor_quantile)), 1e-8)
        clean = np.where(
            np.isfinite(values) & (values > 0.0),
            np.maximum(values, floor),
            np.nanmedian(finite_positive),
        )
        base = 1.0 / clean
    weights = np.zeros(values.size, dtype=float)
    active = np.ones(values.size, dtype=bool)
    remaining = 1.0
    while np.any(active):
        proposed = remaining * base[active] / base[active].sum()
        active_indices = np.flatnonzero(active)
        capped = proposed > maximum_weight + 1e-12
        if not np.any(capped):
            weights[active_indices] = proposed
            break
        capped_indices = active_indices[capped]
        weights[capped_indices] = maximum_weight
        remaining -= maximum_weight * len(capped_indices)
        active[capped_indices] = False
    return weights / weights.sum()


def constrained_inverse_risk_weights(
    risk: Sequence[float],
    sectors: Sequence[str],
    *,
    maximum_weight: float,
    maximum_sector_weight: float,
    floor_quantile: float = 0.20,
) -> np.ndarray:
    values = np.asarray(risk, dtype=float)
    sector_values = np.asarray(sectors, dtype=object)
    if values.size != sector_values.size:
        raise ValueError("risk and sectors must have the same length.")
    if values.size == 0:
        return values
    finite_positive = values[np.isfinite(values) & (values > 0.0)]
    if finite_positive.size:
        floor = max(float(np.quantile(finite_positive, floor_quantile)), 1e-8)
        clean = np.where(
            np.isfinite(values) & (values > 0.0),
            np.maximum(values, floor),
            np.nanmedian(finite_positive),
        )
        base = 1.0 / clean
    else:
        base = np.ones(values.size, dtype=float)
    weights = np.zeros(values.size, dtype=float)
    remaining = 1.0
    unique_sectors = list(dict.fromkeys(sector_values.tolist()))
    for _ in range(values.size + len(unique_sectors) + 2):
        stock_capacity = maximum_weight - weights
        sector_capacity = {
            sector: maximum_sector_weight - float(weights[sector_values == sector].sum())
            for sector in unique_sectors
        }
        active = np.asarray(
            [
                stock_capacity[index] > 1e-12
                and sector_capacity[sector_values[index]] > 1e-12
                for index in range(values.size)
            ]
        )
        if remaining <= 1e-12:
            break
        if not np.any(active):
            raise ValueError("Portfolio constraints are infeasible.")
        proportions = np.zeros(values.size, dtype=float)
        proportions[active] = base[active] / base[active].sum()
        allocation = remaining
        allocation = min(
            allocation,
            *[
                stock_capacity[index] / proportions[index]
                for index in np.flatnonzero(active)
                if proportions[index] > 0.0
            ],
        )
        for sector in unique_sectors:
            sector_share = float(proportions[sector_values == sector].sum())
            if sector_share > 0.0:
                allocation = min(allocation, sector_capacity[sector] / sector_share)
        if allocation <= 1e-12:
            raise ValueError("Portfolio constraints cannot absorb remaining weight.")
        weights += allocation * proportions
        remaining -= allocation
    if remaining > 1e-8:
        raise ValueError("Portfolio constraints are infeasible.")
    return weights / weights.sum()
