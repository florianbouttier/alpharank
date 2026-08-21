"""Ex-ante SEC fundamental coverage policy."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from alpharank.data.contracts.point_in_time import join_point_in_time_attributes


@dataclass(frozen=True)
class FundamentalCoverageResult:
    annotated: pl.DataFrame
    eligible: pl.DataFrame
    coverage_by_year: pl.DataFrame


def load_fundamental_coverage_policy(path: Path) -> dict[str, Any]:
    policy = json.loads(path.read_text(encoding="utf-8"))
    validate_fundamental_coverage_policy(policy)
    return policy


def validate_fundamental_coverage_policy(policy: Mapping[str, Any]) -> None:
    if policy.get("required_source") != "SEC":
        raise ValueError("Official fundamental coverage must require SEC data.")
    if policy.get("missing_action") != "exclude_ex_ante":
        raise ValueError("Missing SEC fundamentals must use exclude_ex_ante.")
    if list(policy.get("fallback_sources") or []):
        raise ValueError("Official SEC coverage cannot use fallback sources.")
    if not str(policy.get("policy_id") or "").strip():
        raise ValueError("Fundamental coverage policy requires policy_id.")


def apply_missing_fundamentals_policy(
    candidates: pl.DataFrame,
    sec_availability: pl.DataFrame,
    *,
    policy: Mapping[str, Any],
    ticker_column: str = "ticker",
    decision_time_column: str = "decision_at",
) -> FundamentalCoverageResult:
    """Exclude missing SEC coverage using information available at decision time."""

    validate_fundamental_coverage_policy(policy)
    required_availability = {
        ticker_column,
        "available_at",
        "fundamental_set_id",
        "source",
    }
    missing = sorted(required_availability - set(sec_availability.columns))
    if missing:
        raise ValueError("SEC availability is missing: " + ", ".join(missing))
    non_sec = sec_availability.filter(pl.col("source") != "SEC")
    if not non_sec.is_empty():
        raise ValueError("Official fundamental availability must be SEC-only.")

    resolved = join_point_in_time_attributes(
        candidates,
        sec_availability,
        entity_column=ticker_column,
        decision_time_column=decision_time_column,
        effective_time_column="available_at",
        attribute_columns=("fundamental_set_id", "source"),
    )
    available = pl.col("fundamental_set_id").is_not_null()
    annotated = resolved.with_columns(
        available.alias("fundamentals_eligible"),
        pl.when(available)
        .then(pl.lit("sec_available"))
        .otherwise(pl.lit("missing_sec_excluded_ex_ante"))
        .alias("fundamental_coverage_status"),
        pl.lit(str(policy["policy_id"])).alias("fundamental_coverage_policy_id"),
    )
    leaked = annotated.filter(
        pl.col("available_at_selected").is_not_null()
        & (pl.col("available_at_selected") > pl.col(decision_time_column))
    )
    if not leaked.is_empty():
        raise RuntimeError("Future SEC availability reached an earlier decision.")

    report = (
        annotated.with_columns(
            pl.col(decision_time_column).dt.year().alias("decision_year")
        )
        .group_by("decision_year")
        .agg(
            pl.len().alias("candidate_count"),
            pl.col("fundamentals_eligible").sum().alias("sec_available_count"),
            (~pl.col("fundamentals_eligible")).sum().alias("missing_sec_count"),
            pl.col(ticker_column)
            .filter(~pl.col("fundamentals_eligible"))
            .n_unique()
            .alias("missing_sec_ticker_count"),
        )
        .with_columns(
            (
                pl.col("sec_available_count")
                / pl.col("candidate_count").cast(pl.Float64)
            ).alias("sec_coverage_rate")
        )
        .sort("decision_year")
    )
    return FundamentalCoverageResult(
        annotated=annotated.sort([decision_time_column, ticker_column]),
        eligible=annotated.filter("fundamentals_eligible").sort(
            [decision_time_column, ticker_column]
        ),
        coverage_by_year=report,
    )
