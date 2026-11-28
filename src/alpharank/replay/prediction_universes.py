"""Prediction universes applied before common-replay Boosting ranking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.artifacts import write_common_portfolio_artifacts
from alpharank.portfolio.benchmark import SPY_TOTAL_RETURN
from alpharank.portfolio.simulation import simulate_weighted_portfolio
from alpharank.replay.trend_eligibility import (
    CausalTrendEligibilityRegistry,
    filter_predictions_to_causal_trend_universe,
)
from alpharank.strategy.legacy_valuation import (
    build_legacy_valuation_registry,
    filter_predictions_to_legacy_valuation_universe,
)
from alpharank.strategy.momentum_reference import (
    MOMENTUM_EVIDENCE,
    MOMENTUM_REFERENCE_POLICY,
    MOMENTUM_SCORE,
    score_momentum_12_1,
)

BENCHMARK_METADATA = {
    "id": SPY_TOTAL_RETURN.identifier,
    "label": SPY_TOTAL_RETURN.label,
    "price_column": SPY_TOTAL_RETURN.price_column,
    "includes_distributions": SPY_TOTAL_RETURN.includes_distributions,
}


def momentum_reference_manifest(top_n_values: Sequence[int], cost_bps: float) -> dict[str, object]:
    return {
        "policy_id": MOMENTUM_REFERENCE_POLICY,
        "status": "research_diagnostic",
        "publication_eligible": False,
        "promotion_eligible": False,
        "top_n_values": list(top_n_values),
        "transaction_cost_bps": cost_bps,
        "execution_timing_policy": "legacy_month_only",
    }


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


def write_momentum_reference_artifacts(
    output_dir: Path,
    snapshot_dir: Path,
    candidates: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    input_hashes: Mapping[str, str],
    top_n_values: Sequence[int],
    transaction_cost_bps: float,
) -> dict[str, Path]:
    """Write a diagnostic comparison on the sealed native Boosting universe."""
    prices = snapshot_dir / "US_Finalprice.parquet"
    market = snapshot_dir / "SP500Price.parquet"
    for key, path in (("final_price", prices), ("sp500_price", market)):
        if sha256_path(path) != input_hashes.get(key):
            raise ValueError(f"Momentum snapshot hash mismatch for {path.name}.")
    scored = score_momentum_12_1(
        candidates,
        pl.read_parquet(prices, columns=["ticker", "date", "adjusted_close"]),
        pl.read_parquet(market, columns=["date"]),
    )
    holdings, monthly = replay_momentum_reference(
        scored, benchmark, top_n_values=top_n_values, transaction_cost_bps=transaction_cost_bps
    )
    artifacts = write_common_portfolio_artifacts(
        output_dir=output_dir,
        holdings=holdings,
        monthly_returns=monthly,
        prefix="momentum_reference",
        benchmark_metadata=BENCHMARK_METADATA,
    )
    journal = output_dir / "momentum_reference_candidates.parquet"
    scored.select("decision_month", "ticker", *MOMENTUM_EVIDENCE).write_parquet(journal)
    return {
        **{f"momentum_{key}": path for key, path in artifacts.items()},
        "momentum_candidates": journal,
    }


def replay_momentum_reference(
    scored: pl.DataFrame,
    benchmark: pl.DataFrame,
    *,
    top_n_values: Sequence[int],
    transaction_cost_bps: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Rank past-only scores, then verify selected outcomes and common calendar."""
    coverage = scored.group_by("decision_month").agg(
        (pl.col("momentum_status") == "eligible").sum().alias("eligible")
    )
    insufficient = coverage.filter(pl.col("eligible") < max(top_n_values))
    if scored.is_empty() or insufficient.height:
        raise ValueError(
            f"Insufficient Momentum lookback coverage: {insufficient.head(10).to_dicts()}"
        )
    eligible = scored.filter(pl.col("momentum_status") == "eligible").with_columns(
        pl.col(MOMENTUM_SCORE).alias("score")
    )
    holdings = pl.concat(
        [
            boosting_predictions_to_holdings(
                eligible,
                strategy=f"Momentum 12-1 Top {top_n}",
                top_n=top_n,
                tie_breaker_columns=("ticker",),
            )
            for top_n in top_n_values
        ]
    )
    unsupported = holdings.filter(pl.col("target_status_1m").ne_missing("evaluable"))
    if unsupported.height:
        examples = unsupported.select("holding_month", "ticker", "target_status_1m").head(10)
        raise ValueError(
            f"Momentum selected unsupported realized-return statuses: {examples.to_dicts()}"
        )
    monthly = simulate_replay_strategies(holdings, transaction_cost_bps=transaction_cost_bps)
    _require_momentum_benchmark(monthly, benchmark)
    return holdings, monthly


def _require_momentum_benchmark(monthly: pl.DataFrame, benchmark: pl.DataFrame) -> None:
    expected = benchmark.select("holding_month", pl.col("net_return").alias("spy_return"))
    for frame in monthly.partition_by("strategy", maintain_order=True):
        matched = frame.join(expected, on="holding_month", how="inner", validate="1:1")
        if frame.height != matched.height or expected.height != matched.height:
            raise ValueError("Momentum reference does not cover the exact common calendar.")
        mismatch = matched.filter(
            ~pl.col("spy_return").is_finite().fill_null(False)
            | ((pl.col("benchmark_return") - pl.col("spy_return")).abs() > 1e-12)
        )
        if mismatch.height:
            raise ValueError("Momentum reference does not use the common SPY total return.")


def simulate_replay_strategies(
    holdings: pl.DataFrame, *, transaction_cost_bps: float
) -> pl.DataFrame:
    """Replay strategies with the common historical month-only execution contract."""
    return pl.concat(
        [
            simulate_weighted_portfolio(
                frame,
                transaction_cost_bps=transaction_cost_bps,
                causal_timing_policy="legacy_month_only",
            )
            for frame in holdings.partition_by("strategy", maintain_order=True)
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
