"""Build an optional research sleeve from existing Legacy and Boosting decisions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path

import polars as pl

from alpharank.governance_contracts.common import sha256_path
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.replay.sector_allocation import (
    attach_point_in_time_sector_classifications,
    build_market_signal_cutoffs,
)

COMPLEMENTARY_SLEEVE_POLICY = "boosting_ex_legacy_frequency_v1"
ALLOCATION_KEYS = ["decision_month", "ticker"]


@dataclass(frozen=True, slots=True)
class ComplementarySleeveConfig:
    """Equal-weight sleeve size and optional point-in-time sector cap."""

    top_n: int
    maximum_names_per_sector: int | None = None
    sector_history_path: Path | None = None

    def __post_init__(self) -> None:
        if self.top_n < 1:
            raise ValueError("Complementary top_n must be positive.")
        cap = self.maximum_names_per_sector
        if cap is not None and cap < 1:
            raise ValueError("Complementary sector cap must be positive.")
        if (cap is None) != (self.sector_history_path is None):
            raise ValueError(
                "Complementary sector cap and sector history must be provided together."
            )

    def manifest(self, legacy_detailed: Path) -> dict[str, object]:
        """Describe the inputs and the research-only allocation policy."""
        sector_history = self.sector_history_path
        return {
            "policy_id": COMPLEMENTARY_SLEEVE_POLICY,
            "status": "research_diagnostic",
            "publication_eligible": False,
            "promotion_eligible": False,
            "top_n": self.top_n,
            "maximum_names_per_sector": self.maximum_names_per_sector,
            "sector_cutoff": "last_observed_session_date_00:00_America/New_York"
            if sector_history
            else None,
            "legacy_detailed_path": str(legacy_detailed.resolve()),
            "legacy_detailed_sha256": sha256_path(legacy_detailed),
            "sector_history_path": str(sector_history.resolve()) if sector_history else None,
            "sector_history_sha256": sha256_path(sector_history) if sector_history else None,
        }


@dataclass(frozen=True, slots=True)
class ComplementarySleeve:
    """Selected baskets and the candidate and monthly eligibility journals."""

    holdings: pl.DataFrame
    candidates: pl.DataFrame
    monthly_audit: pl.DataFrame


def write_complementary_sleeve_artifacts(
    output_dir: Path,
    config: ComplementarySleeveConfig,
    *,
    completed: pl.DataFrame,
    live: pl.DataFrame,
    legacy_detailed_path: Path,
    market_prices_path: Path,
) -> dict[str, Path]:
    """Build diagnostics from the native completed and score-only partitions."""
    predictions = pl.concat(
        [
            completed.with_columns(pl.lit("completed_replay").alias("prediction_partition")),
            live.with_columns(pl.lit("score_only_live").alias("prediction_partition")),
        ],
        how="diagonal_relaxed",
    )
    legacy = pl.read_parquet(
        legacy_detailed_path, columns=["portfolio_model", "year_month", "ticker"]
    )
    market = (
        pl.read_parquet(market_prices_path, columns=["date"])
        if config.sector_history_path is not None
        else None
    )
    sleeve = build_complementary_boosting_sleeve(predictions, legacy, config, market_prices=market)
    frames = {
        "complementary_holdings": sleeve.holdings,
        "complementary_eligibility": sleeve.candidates,
        "complementary_monthly_audit": sleeve.monthly_audit,
    }
    paths = {key: output_dir / f"boosting_{key}.parquet" for key in frames}
    for key, frame in frames.items():
        frame.write_parquet(paths[key])
    return paths


def build_complementary_boosting_sleeve(
    predictions: pl.DataFrame,
    legacy_detailed: pl.DataFrame,
    config: ComplementarySleeveConfig,
    *,
    market_prices: pl.DataFrame | None = None,
) -> ComplementarySleeve:
    """Exclude same-decision Legacy Frequency positions before ranking existing scores."""
    _validate_allocation_keys(predictions, "Boosting predictions")
    legacy = _legacy_frequency_keys(legacy_detailed)
    marked = _candidate_eligibility(predictions, legacy, config, market_prices=market_prices)
    eligible = marked.filter(pl.col("eligibility_reason") == "eligible")
    audit = _allocation_audit(marked, eligible, config)
    feasible_months = audit.filter(pl.col("status") == "selected").select("decision_month")
    holdings = boosting_predictions_to_holdings(
        eligible.join(feasible_months, on="decision_month", how="semi"),
        strategy=f"Boosting Top {config.top_n} | Complementary to Legacy",
        top_n=config.top_n,
        sector_column="Sector" if config.maximum_names_per_sector is not None else None,
        maximum_names_per_sector=config.maximum_names_per_sector,
        tie_breaker_columns=("ticker",),
    )
    evidence = (
        "prediction_partition",
        "decision_at",
        "Sector",
        "classification_id",
        "sector_known_at_selected",
    )
    journal = marked.select(
        *ALLOCATION_KEYS,
        "score",
        "eligibility_reason",
        *[name for name in evidence if name in marked.columns],
    ).sort(ALLOCATION_KEYS)
    return ComplementarySleeve(holdings=holdings, candidates=journal, monthly_audit=audit)


def _validate_allocation_keys(frame: pl.DataFrame, label: str) -> None:
    if not set(ALLOCATION_KEYS).issubset(frame.columns):
        raise ValueError(f"{label} must contain decision_month and ticker.")
    if frame.select(pl.any_horizontal(pl.col(ALLOCATION_KEYS).is_null()).any()).item():
        raise ValueError(f"{label} contains null decision keys.")
    if frame.select(pl.struct(ALLOCATION_KEYS).n_unique()).item() != frame.height:
        raise ValueError(f"{label} contains duplicate decision keys.")


def _legacy_frequency_keys(legacy_detailed: pl.DataFrame) -> pl.DataFrame:
    required = {"portfolio_model", "year_month", "ticker"}
    if not required.issubset(legacy_detailed.columns):
        raise ValueError(
            "Legacy detailed holdings must expose portfolio_model, year_month and ticker."
        )
    frequency = legacy_detailed.filter(pl.col("portfolio_model") == "Combined_Frequency")
    keys = frequency.select(
        pl.col("year_month").cast(pl.Date).dt.offset_by("-1mo").alias("decision_month"),
        "ticker",
    )
    _validate_allocation_keys(keys, "Legacy Frequency holdings")
    return keys


def _candidate_eligibility(
    predictions: pl.DataFrame,
    legacy: pl.DataFrame,
    config: ComplementarySleeveConfig,
    *,
    market_prices: pl.DataFrame | None,
) -> pl.DataFrame:
    marked = (
        predictions.join(
            legacy.select("decision_month")
            .unique()
            .with_columns(pl.lit(True).alias("_has_legacy")),
            on="decision_month",
            how="left",
            validate="m:1",
        )
        .join(
            legacy.with_columns(pl.lit(True).alias("_legacy_selected")),
            on=ALLOCATION_KEYS,
            how="left",
            validate="1:1",
        )
        .with_columns(pl.col("_has_legacy", "_legacy_selected").fill_null(False))
    )
    sector_missing = pl.lit(False)
    if config.sector_history_path is not None:
        if market_prices is None:
            raise ValueError("Market prices are required for point-in-time sector classifications.")
        history = pl.read_parquet(config.sector_history_path)
        unknown_times = history.select(
            pl.any_horizontal(
                pl.col("observed_at", "effective_at")
                .cast(pl.Datetime(time_zone="UTC"), strict=False)
                .is_null()
            ).any()
        ).item()
        if unknown_times:
            raise ValueError(
                "Complementary sector history requires known observation and effect times."
            )
        marked = attach_point_in_time_sector_classifications(
            marked,
            history,
            build_market_signal_cutoffs(market_prices, local_cutoff=time(0, 0)),
        )
        sector_missing = pl.col("Sector").is_null()
    return marked.with_columns(
        pl.when(~pl.col("_has_legacy"))
        .then(pl.lit("blocked_missing_same_decision_legacy"))
        .when(pl.col("_legacy_selected"))
        .then(pl.lit("excluded_legacy_frequency"))
        .when(~pl.col("score").is_finite().fill_null(False))
        .then(pl.lit("excluded_non_finite_score"))
        .when(sector_missing)
        .then(pl.lit("excluded_missing_point_in_time_sector"))
        .otherwise(pl.lit("eligible"))
        .alias("eligibility_reason")
    )


def _allocation_audit(
    marked: pl.DataFrame, eligible: pl.DataFrame, config: ComplementarySleeveConfig
) -> pl.DataFrame:
    audit = marked.group_by("decision_month").agg(
        pl.len().alias("candidate_count"),
        pl.col("_has_legacy").first().alias("has_same_decision_legacy"),
        pl.col("_legacy_selected").sum().alias("legacy_overlap_count"),
        (pl.col("eligibility_reason") == "eligible").sum().alias("eligible_count"),
    )
    cap = config.maximum_names_per_sector
    if cap is None:
        audit = audit.with_columns(pl.col("eligible_count").alias("sector_capacity"))
    else:
        capacity = eligible.group_by("decision_month", "Sector").len().group_by("decision_month")
        audit = audit.join(
            capacity.agg(pl.col("len").clip(upper_bound=cap).sum().alias("sector_capacity")),
            on="decision_month",
            how="left",
            validate="1:1",
        ).with_columns(pl.col("sector_capacity").fill_null(0))
    return (
        audit.with_columns(
            pl.when(~pl.col("has_same_decision_legacy"))
            .then(pl.lit("blocked_missing_same_decision_legacy"))
            .when(pl.col("eligible_count") < config.top_n)
            .then(pl.lit("blocked_insufficient_eligible_candidates"))
            .when(pl.col("sector_capacity") < config.top_n)
            .then(pl.lit("blocked_sector_cap"))
            .otherwise(pl.lit("selected"))
            .alias("status")
        )
        .with_columns(
            pl.when(pl.col("status") == "selected")
            .then(pl.lit(config.top_n))
            .otherwise(0)
            .alias("selected_count")
        )
        .sort("decision_month")
    )
