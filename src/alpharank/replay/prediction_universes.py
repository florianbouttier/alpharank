"""Prediction universes applied before common-replay Boosting ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.replay.trend_eligibility import (
    CausalTrendEligibilityRegistry,
    filter_predictions_to_causal_trend_universe,
)
from alpharank.strategy.legacy_valuation import (
    build_legacy_valuation_registry,
    filter_predictions_to_legacy_valuation_universe,
)


@dataclass(frozen=True, slots=True)
class PredictionUniverseFrames:
    """Completed/live prediction sets and their eligibility audit."""

    completed: tuple[tuple[str, pl.DataFrame], ...]
    live: tuple[tuple[str, pl.DataFrame], ...]
    valuation_registry: pl.DataFrame


def build_prediction_universes(
    *,
    snapshot_dir: Path,
    completed: pl.DataFrame,
    live: pl.DataFrame,
    include_legacy_valuation_universe: bool,
    trend_registry: CausalTrendEligibilityRegistry | None,
) -> PredictionUniverseFrames:
    """Build every requested pre-ranking universe without changing native rows."""

    completed_sets: list[tuple[str, pl.DataFrame]] = [("native", completed)]
    live_sets: list[tuple[str, pl.DataFrame]] = [("native", live)]
    valuation_registry = pl.DataFrame()
    if include_legacy_valuation_universe:
        valuation_registry = build_legacy_valuation_registry(
            snapshot_dir=snapshot_dir,
            candidates=completed,
        )
        completed_sets.append(
            (
                "legacy_valuation",
                filter_predictions_to_legacy_valuation_universe(completed, valuation_registry),
            )
        )
    if trend_registry is not None:
        completed_sets.append(
            (
                "causal_trend",
                filter_predictions_to_causal_trend_universe(completed, trend_registry),
            )
        )
        live_sets.append(
            (
                "causal_trend",
                filter_predictions_to_causal_trend_universe(live, trend_registry),
            )
        )
    return PredictionUniverseFrames(
        completed=tuple(completed_sets),
        live=tuple(live_sets),
        valuation_registry=valuation_registry,
    )


def build_boosting_holdings(
    prediction_sets: Sequence[tuple[str, pl.DataFrame]],
    *,
    top_n_values: Sequence[int],
) -> pl.DataFrame:
    """Rank each prediction universe separately into equal-weight holdings."""

    return pl.concat(
        [
            boosting_predictions_to_holdings(
                frame,
                strategy=_strategy_name(universe, top_n),
                top_n=top_n,
            )
            for universe, frame in prediction_sets
            for top_n in top_n_values
        ],
        how="diagonal_relaxed",
    )


def costed_strategy_names(
    top_n_values: Sequence[int],
    *,
    include_legacy_valuation_universe: bool,
    include_causal_trend_universe: bool,
) -> list[str]:
    """Return the exact strategy labels subject to simulated turnover costs."""

    universes = ["native"]
    if include_legacy_valuation_universe:
        universes.append("legacy_valuation")
    if include_causal_trend_universe:
        universes.append("causal_trend")
    return [
        *[_strategy_name(universe, value) for universe in universes for value in top_n_values],
        "Legacy",
    ]


def _strategy_name(universe: str, top_n: int) -> str:
    suffixes = {
        "native": "",
        "legacy_valuation": " | Legacy PE universe",
        "causal_trend": " | Causal trend",
    }
    try:
        return f"Boosting Top {top_n}{suffixes[universe]}"
    except KeyError as exc:
        raise ValueError(f"Unknown Boosting prediction universe: {universe}.") from exc
