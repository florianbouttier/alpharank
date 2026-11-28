"""Explicit, immutable SEC sector profile for Legacy research."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import polars as pl

from alpharank.data.contracts.sector_history import resolve_point_in_time_sectors
from alpharank.data.warehouse.sector_history import read_verified_sec_sector_history
from alpharank.utils.frame_backend import to_pandas

LEGACY_SEC_SECTOR_PROFILE_ID = "legacy_sec_sector_at_close_v1"
PROFILE_COLUMN = "legacy_sector_candidate_profile"


def sector_candidate_run_context(profile_manifest: Path | None) -> dict[str, object]:
    if profile_manifest is None:
        return {}
    return {
        "legacy_sector_candidate_profile": LEGACY_SEC_SECTOR_PROFILE_ID,
        "legacy_sector_candidate_source": str(profile_manifest.resolve()),
        "legacy_sector_candidate_promotion_eligible": False,
    }


def legacy_sector_output_roots(
    output_dir: Path, checkpoints_dir: Path | None, profile_manifest: Path | None
) -> tuple[Path, Path | None]:
    if profile_manifest is None:
        return output_dir, checkpoints_dir
    candidate_root = "legacy_sector_candidates"
    return (
        output_dir / candidate_root,
        checkpoints_dir / candidate_root if checkpoints_dir else None,
    )


def prepare_legacy_sector_learning(
    general: pl.DataFrame,
    candidates: pl.DataFrame,
    market_prices: pl.DataFrame,
    run_dir: Path,
    profile_manifest: Path | None,
) -> pd.DataFrame:
    """Keep the historical input or pre-resolve a sealed sector candidate before learning."""
    if profile_manifest is None:
        return to_pandas(general.select("ticker", "Sector"))
    raw_profile = profile_manifest.read_bytes()
    profile = json.loads(raw_profile)
    if profile.get("policy_id") != LEGACY_SEC_SECTOR_PROFILE_ID:
        raise ValueError("Unknown Legacy sector candidate profile.")
    first_decision_month = date.fromisoformat(profile["first_decision_month"])
    if first_decision_month.day != 1:
        raise ValueError("Legacy sector first_decision_month must be a month start.")
    release = profile_manifest.parent / profile["sector_release"]["path"]
    calendar = profile_manifest.parent / profile["session_calendar"]["path"]
    release_hash = _verify_hash(
        release / "manifest.json", profile["sector_release"]["manifest_sha256"]
    )
    calendar_hash = _verify_hash(calendar, profile["session_calendar"]["sha256"])
    source = profile["session_calendar"]["source"]
    if not isinstance(source, str) or not source.strip():
        raise ValueError("Legacy sector candidate exchange calendar requires source attribution.")
    selected = build_candidate_sector_map(
        candidates,
        read_verified_sec_sector_history(release),
        market_prices,
        pl.read_parquet(calendar),
    )
    in_window = pl.col("year_month").dt.offset_by("-1mo").cast(pl.Date) >= first_decision_month
    selected = selected.with_columns(
        (in_window & pl.col("sector_constraint_enabled")).alias("sector_constraint_enabled"),
        pl.when(in_window)
        .then(pl.col("sector_constraint_reason"))
        .otherwise(pl.lit("excluded_before_profile_availability"))
        .alias("sector_constraint_reason"),
    )
    warmup = selected.filter(~in_window)
    in_scope = selected.filter(in_window)
    journal = run_dir / "legacy_sector_candidate_keys.parquet"
    selected.write_parquet(journal)
    metadata = {
        "policy_id": LEGACY_SEC_SECTOR_PROFILE_ID,
        "first_decision_month": first_decision_month.isoformat(),
        "excluded_warmup_months": warmup["year_month"].n_unique(),
        "excluded_warmup_rows": warmup.height,
        "promotion_eligible": False,
        "profile_manifest_sha256": hashlib.sha256(raw_profile).hexdigest(),
        "sector_release": str(release.resolve()),
        "sector_release_manifest_sha256": release_hash,
        "session_calendar": str(calendar.resolve()),
        "session_calendar_sha256": calendar_hash,
        "session_calendar_source": source,
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "candidate_rows": selected.height,
        "unclassified_rows": in_scope["Sector"].null_count(),
    }
    (run_dir / "legacy_sector_candidate_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if (
        in_scope.is_empty()
        or in_scope.group_by("year_month")
        .agg(pl.col("Sector").count().alias("count"))
        .filter(pl.col("count") == 0)
        .height
    ):
        raise ValueError(f"A Legacy sector candidate month has no known sector. See {journal}")
    return to_pandas(selected)


def build_candidate_sector_map(
    candidates: pl.DataFrame,
    history: pl.DataFrame,
    market_prices: pl.DataFrame,
    exchange_calendar: pl.DataFrame,
) -> pl.DataFrame:
    """Project each signal at its exchange close and key output by Legacy holding month."""
    keys = candidates.select(
        pl.col("year_month").cast(pl.Date).alias("decision_month"), "ticker"
    ).unique()
    cutoff = _exchange_monthly_closes(
        keys.select("decision_month").unique(), market_prices, exchange_calendar
    )
    resolved = resolve_point_in_time_sectors(keys.join(cutoff, on="decision_month"), history)
    return resolved.select(
        pl.col("decision_month").dt.offset_by("1mo").cast(pl.Datetime("ns")).alias("year_month"),
        "ticker",
        "decision_at",
        "Sector",
        "sector_known_at_selected",
        "classification_id",
        pl.col("Sector").is_not_null().alias("sector_constraint_enabled"),
        pl.when(pl.col("Sector").is_not_null())
        .then(pl.lit("known_at_exchange_decision_close"))
        .otherwise(pl.lit("excluded_unclassified_at_exchange_decision_close"))
        .alias("sector_constraint_reason"),
        pl.lit(LEGACY_SEC_SECTOR_PROFILE_ID).alias(PROFILE_COLUMN),
    ).sort("year_month", "ticker")


def _exchange_monthly_closes(
    required_months: pl.DataFrame, prices: pl.DataFrame, calendar: pl.DataFrame
) -> pl.DataFrame:
    close_dtype = calendar.schema["close_at"]
    if not isinstance(close_dtype, pl.Datetime) or close_dtype.time_zone is None:
        raise ValueError("Exchange close_at must be timezone-aware.")
    observed = _session_dates(prices).group_by("decision_month").agg(pl.col("date").max())
    official = (
        _session_dates(calendar)
        .with_columns(calendar["close_at"])
        .with_columns(pl.col("close_at").dt.convert_time_zone("UTC"))
    )
    if (
        official["date"].is_duplicated().any()
        or official.filter(
            pl.col("date").is_null()
            | pl.col("close_at").is_null()
            | (
                pl.col("close_at").dt.convert_time_zone("America/New_York").dt.date()
                != pl.col("date")
            )
        ).height
    ):
        raise ValueError("Exchange calendar contains invalid or duplicate session closes.")
    latest = (
        official.sort("date")
        .group_by("decision_month")
        .agg(
            pl.col("date").last().alias("exchange_session"),
            pl.col("close_at").last().alias("decision_at"),
        )
    )
    matched = required_months.join(observed, on="decision_month", how="left").join(
        latest, on="decision_month", how="left"
    )
    invalid = matched.filter(
        pl.col("decision_at").is_null() | pl.col("date").ne_missing(pl.col("exchange_session"))
    )
    if invalid.height:
        raise ValueError(
            f"Exchange and SPY month-end sessions disagree: {invalid.head(4).to_dicts()}"
        )
    return matched.select("decision_month", "decision_at")


def _session_dates(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(pl.col("date").cast(pl.String).str.to_date(strict=False)).with_columns(
        pl.col("date").dt.truncate("1mo").alias("decision_month")
    )


def _verify_hash(path: Path, expected: str) -> str:
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise ValueError(f"Legacy sector candidate input hash mismatch: {path.name}.")
    return observed
