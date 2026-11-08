"""Causal one-name-per-sector allocation applied after Boosting scores."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import polars as pl

from alpharank.data.contracts.sector_history import resolve_point_in_time_sectors
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings

SECTOR_CHAMPION_POLICY_ID = "boosting_sector_champions_top5_v1"
SECTOR_CHAMPION_TOP_N = 5
MAXIMUM_NAMES_PER_SECTOR = 1
NEW_YORK = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


@dataclass(frozen=True, slots=True)
class SectorChampionAllocation:
    """Causal candidate ledger, selected holdings and monthly allocation audit."""

    candidates: pl.DataFrame
    holdings: pl.DataFrame
    audit: pl.DataFrame


def build_sector_champion_allocation(
    predictions: pl.DataFrame,
    sector_history: pl.DataFrame,
    market_prices: pl.DataFrame,
    *,
    strategy: str,
    top_n: int = SECTOR_CHAMPION_TOP_N,
) -> SectorChampionAllocation:
    """Select the best scored name in each sector with point-in-time data only."""

    _validate_predictions(predictions)
    cutoffs = build_market_signal_cutoffs(market_prices)
    candidates = _attach_sector_eligibility(predictions, sector_history, cutoffs)
    holdings_parts: list[pl.DataFrame] = []
    audit_rows: list[dict[str, object]] = []
    for month in candidates.partition_by("decision_month", maintain_order=True):
        eligible = month.filter(pl.col("sector_allocation_eligible"))
        distinct_sectors = eligible["Sector"].n_unique()
        decision_month = month["decision_month"][0]
        selected = pl.DataFrame()
        status = "blocked_insufficient_distinct_sectors"
        if distinct_sectors >= top_n:
            selected = boosting_predictions_to_holdings(
                eligible,
                strategy=strategy,
                top_n=top_n,
                sector_column="Sector",
                maximum_names_per_sector=MAXIMUM_NAMES_PER_SECTOR,
            )
            _require_one_name_per_sector(selected)
            holdings_parts.append(selected)
            status = "selected"
        audit_rows.append(
            {
                "policy_id": SECTOR_CHAMPION_POLICY_ID,
                "strategy": strategy,
                "decision_month": decision_month,
                "signal_cutoff_at": month["decision_at"][0],
                "candidate_count": month.height,
                "eligible_candidate_count": eligible.height,
                "missing_sector_candidate_count": month.height - eligible.height,
                "distinct_sector_count": distinct_sectors,
                "selected_count": selected.height,
                "maximum_names_per_sector": MAXIMUM_NAMES_PER_SECTOR,
                "missing_sector_tickers": month.filter(
                    ~pl.col("sector_allocation_eligible")
                )["ticker"].sort().to_list(),
                "status": status,
            }
        )
    holdings = (
        pl.concat(holdings_parts, how="diagonal_relaxed")
        if holdings_parts
        else _empty_holdings(predictions)
    )
    audit = pl.from_dicts(audit_rows, infer_schema_length=None).sort("decision_month")
    return SectorChampionAllocation(candidates=candidates, holdings=holdings, audit=audit)


def build_market_signal_cutoffs(market_prices: pl.DataFrame) -> pl.DataFrame:
    """Resolve each decision month to the final observed market close in New York."""

    if "date" not in market_prices.columns:
        raise ValueError("Market prices are missing the date column.")
    sessions = (
        market_prices.select(
            pl.col("date").cast(pl.String).str.to_date(strict=False).alias("session_date")
        )
        .drop_nulls("session_date")
        .unique()
        .with_columns(pl.col("session_date").dt.truncate("1mo").alias("decision_month"))
        .group_by("decision_month")
        .agg(pl.col("session_date").max())
        .sort("decision_month")
    )
    rows = [
        {
            "decision_month": row["decision_month"],
            "decision_at": _market_close_at(row["session_date"]),
        }
        for row in sessions.to_dicts()
    ]
    return pl.from_dicts(rows, infer_schema_length=None)


def _attach_sector_eligibility(
    predictions: pl.DataFrame,
    sector_history: pl.DataFrame,
    cutoffs: pl.DataFrame,
) -> pl.DataFrame:
    decisions = (
        predictions.select("decision_month", "ticker")
        .unique()
        .join(cutoffs, on="decision_month", how="left", validate="m:1")
    )
    if decisions["decision_at"].null_count():
        missing_months = decisions.filter(pl.col("decision_at").is_null())["decision_month"]
        raise ValueError(f"No market signal cutoff for decision months: {missing_months.unique()}")
    resolved = resolve_point_in_time_sectors(decisions, sector_history).select(
        "decision_month",
        "ticker",
        "decision_at",
        "Sector",
        "classification_id",
        "sector_known_at_selected",
    )
    return (
        predictions.join(
            resolved,
            on=["decision_month", "ticker"],
            how="left",
            validate="1:1",
        )
        .with_columns(
            pl.col("Sector").is_not_null().alias("sector_allocation_eligible"),
            pl.when(pl.col("Sector").is_not_null())
            .then(pl.lit("point_in_time_sector_available"))
            .otherwise(pl.lit("excluded_missing_point_in_time_sector"))
            .alias("sector_allocation_reason"),
            pl.lit(SECTOR_CHAMPION_POLICY_ID).alias("sector_allocation_policy_id"),
        )
        .sort(["decision_month", "ticker"])
    )


def _validate_predictions(predictions: pl.DataFrame) -> None:
    required = {
        "decision_month",
        "ticker",
        "score",
        "future_return_1m",
        "benchmark_future_return_1m",
    }
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError("Sector allocation predictions are missing: " + ", ".join(missing))
    duplicates = predictions.group_by("decision_month", "ticker").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("Sector allocation predictions contain duplicate month/ticker keys.")


def _require_one_name_per_sector(holdings: pl.DataFrame) -> None:
    violations = (
        holdings.group_by("strategy", "decision_month", "Sector")
        .len()
        .filter(pl.col("len") > MAXIMUM_NAMES_PER_SECTOR)
    )
    if not violations.is_empty():
        raise RuntimeError("Sector champion allocation selected more than one name per sector.")


def _market_close_at(session_date: object) -> datetime:
    if isinstance(session_date, datetime):
        session_date = session_date.date()
    if not isinstance(session_date, date):
        raise ValueError("Market session dates must be valid dates.")
    return datetime.combine(session_date, time(16, 0), tzinfo=NEW_YORK).astimezone(UTC)


def _empty_holdings(predictions: pl.DataFrame) -> pl.DataFrame:
    return boosting_predictions_to_holdings(
        predictions.head(0),
        strategy="sector_champions_empty",
        top_n=SECTOR_CHAMPION_TOP_N,
    )
