from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Sequence

import polars as pl

from alpharank.data.prices.contracts import (
    PRODUCTION_PRICE_GATE_POLICY,
    PriceCandidateMode,
    PriceGatePolicy,
)
from alpharank.data.security_identity import (
    apply_security_identity_policy,
    assert_security_identity_compliance,
)


@dataclass(frozen=True)
class PriceGateResult:
    report: dict[str, object]
    daily_return_revisions: pl.DataFrame
    transition_factor_findings: pl.DataFrame
    historical_key_removals: pl.DataFrame


@dataclass(frozen=True, slots=True)
class ActiveVintageAudit:
    active_tickers: tuple[str, ...]
    active_non_yahoo_rows: int
    mixed_active_tickers: tuple[str, ...]
    global_vintage_ids: tuple[str, ...]
    current_resolution_tickers: frozenset[str]
    carried_active_rows: int
    carried_active_tickers: int
    missing_active_tickers: tuple[str, ...]


def audit_price_candidate(
    *,
    previous_prices: pl.DataFrame | None,
    candidate_prices: pl.DataFrame,
    candidate_lineage: pl.DataFrame,
    active_tickers: Sequence[str],
    expected_eodhd_keys: pl.DataFrame | None = None,
    expected_through: str,
    policy: PriceGatePolicy = PRODUCTION_PRICE_GATE_POLICY,
    active_resolution_vintage_id: str | None = None,
    candidate_mode: PriceCandidateMode = PriceCandidateMode.FRESH_ACTIVE_VINTAGE,
    observed_active_tickers: Sequence[str] = (),
) -> PriceGateResult:
    assert_security_identity_compliance(
        candidate_prices,
        ticker_column="ticker",
        date_column="date",
    )
    assert_security_identity_compliance(
        candidate_lineage,
        ticker_column="ticker",
        date_column="date",
    )
    candidate = _normalize_prices(candidate_prices)
    lineage = _normalize_lineage(candidate_lineage)
    _require_unique(candidate, label="candidate prices")
    _require_unique(lineage, label="candidate price lineage")
    if (
        candidate.select(["ticker", "date"])
        .join(lineage.select(["ticker", "date"]), on=["ticker", "date"], how="anti")
        .height
    ):
        raise RuntimeError("Candidate price lineage does not cover every selected price row")
    if (
        lineage.select(["ticker", "date"])
        .join(candidate.select(["ticker", "date"]), on=["ticker", "date"], how="anti")
        .height
    ):
        raise RuntimeError("Candidate price lineage contains keys absent from selected prices")

    active_audit = _audit_active_vintages(
        lineage=lineage,
        active_tickers=active_tickers,
        active_resolution_vintage_id=active_resolution_vintage_id,
        candidate_mode=candidate_mode,
        observed_active_tickers=observed_active_tickers,
    )
    active = list(active_audit.active_tickers)

    transition_findings = _transition_factor_findings(lineage, policy=policy)
    recent_cutoff = date.fromisoformat(expected_through) - timedelta(
        days=policy.recent_mutable_calendar_days
    )
    previous_identity = (
        apply_security_identity_policy(
            previous_prices,
            ticker_column="ticker",
            date_column="date",
        )
        if previous_prices is not None and not previous_prices.is_empty()
        else None
    )
    identity_normalized_previous = (
        previous_identity.frame if previous_identity is not None else previous_prices
    )
    revisions = (
        _daily_return_revisions(
            _normalize_prices(identity_normalized_previous),
            candidate,
            storage_tolerance=policy.historical_return_revision_threshold,
        )
        if previous_prices is not None and not previous_prices.is_empty()
        else _empty_revision_frame()
    )
    historical_revisions = revisions.filter(pl.col("date") < recent_cutoff)
    material_old_revisions = historical_revisions.filter(
        pl.col("absolute_daily_return_difference") > policy.historical_return_revision_threshold
    )
    return_availability_changes = historical_revisions.filter(
        pl.col("previous_daily_return").is_null() | pl.col("candidate_daily_return").is_null()
    )
    historical_key_removals = pl.DataFrame(
        schema={"ticker": pl.String, "date": pl.Date, "adjusted_close": pl.Float64}
    )
    if previous_prices is not None and not previous_prices.is_empty():
        previous = _normalize_prices(identity_normalized_previous)
        historical_key_removals = (
            previous.filter(
                (pl.col("date") < recent_cutoff) & pl.col("adjusted_close").is_not_null()
            )
            .join(
                candidate.select("ticker", "date"),
                on=["ticker", "date"],
                how="anti",
            )
            .select("ticker", "date", "adjusted_close")
            .sort(["ticker", "date"])
        )
    removed_old_keys = historical_key_removals.height

    missing_seed_keys = 0
    if expected_eodhd_keys is not None and not expected_eodhd_keys.is_empty():
        seed_identity = apply_security_identity_policy(
            expected_eodhd_keys,
            ticker_column="ticker",
            date_column="date",
        )
        seed_keys = seed_identity.frame.select(
            pl.col("ticker").cast(pl.String).str.to_uppercase(),
            pl.col("date").cast(pl.Date, strict=False),
        ).unique()
        inactive_seed_keys = seed_keys.filter(~pl.col("ticker").is_in(active))
        missing_seed_keys = inactive_seed_keys.join(
            candidate.select("ticker", "date"), on=["ticker", "date"], how="anti"
        ).height

    blocking_reasons: list[str] = []
    if active_audit.missing_active_tickers:
        blocking_reasons.append("missing_active_full_yahoo_vintage")
    missing_current_resolution = (
        sorted(set(active) - active_audit.current_resolution_tickers)
        if active_resolution_vintage_id is not None
        else []
    )
    requires_fresh_vintage = candidate_mode == PriceCandidateMode.FRESH_ACTIVE_VINTAGE
    if (
        requires_fresh_vintage
        and active_resolution_vintage_id is None
        and active_audit.mixed_active_tickers
    ):
        blocking_reasons.append("mixed_active_yahoo_vintages")
    if (
        requires_fresh_vintage
        and active_resolution_vintage_id is None
        and len(active_audit.global_vintage_ids) != 1
    ):
        blocking_reasons.append("active_universe_not_one_global_yahoo_vintage")
    if active_resolution_vintage_id is not None and missing_current_resolution:
        blocking_reasons.append("active_ticker_without_current_resolution_observation")
    if requires_fresh_vintage and active_audit.active_non_yahoo_rows:
        blocking_reasons.append("active_universe_contains_non_yahoo_rows")
    if transition_findings.height:
        blocking_reasons.append("adjustment_factor_transition_discontinuity")
    if removed_old_keys and not policy.allow_historical_price_key_removals:
        blocking_reasons.append("historical_price_keys_removed")
    if missing_seed_keys:
        blocking_reasons.append("eodhd_seed_coverage_lost")
    if (
        material_old_revisions.height or return_availability_changes.height
    ) and not policy.allow_historical_price_revisions:
        blocking_reasons.append("unreviewed_historical_return_revisions")

    report = {
        "gate_version": 1,
        "candidate_mode": candidate_mode.value,
        "policy": policy.to_manifest(),
        "candidate_rows": candidate.height,
        "candidate_tickers": candidate.select(pl.col("ticker").n_unique()).item(),
        "active_ticker_count": len(active),
        "missing_active_yahoo_tickers": list(active_audit.missing_active_tickers),
        "mixed_active_yahoo_vintage_tickers": list(active_audit.mixed_active_tickers),
        "active_global_yahoo_vintage_ids": list(active_audit.global_vintage_ids),
        "active_resolution_vintage_id": active_resolution_vintage_id,
        "active_tickers_without_current_resolution_observation": missing_current_resolution,
        "audited_carried_active_rows": active_audit.carried_active_rows,
        "audited_carried_active_tickers": active_audit.carried_active_tickers,
        "active_non_yahoo_rows": active_audit.active_non_yahoo_rows,
        "transition_factor_findings": transition_findings.height,
        "historical_daily_return_revisions_over_threshold": material_old_revisions.height,
        "daily_return_audit_storage_tolerance": policy.historical_return_revision_threshold,
        "historical_return_availability_changes": return_availability_changes.height,
        "historical_return_revision_tickers": (
            material_old_revisions.select(pl.col("ticker").n_unique()).item()
            if material_old_revisions.height
            else 0
        ),
        "historical_return_revision_examples": material_old_revisions.with_columns(
            pl.col("date").cast(pl.String)
        )
        .head(20)
        .to_dicts(),
        "historical_price_keys_removed": removed_old_keys,
        "historical_price_key_removal_tickers": (
            historical_key_removals.select(pl.col("ticker").n_unique()).item()
            if removed_old_keys
            else 0
        ),
        "historical_price_key_removal_examples": historical_key_removals.with_columns(
            pl.col("date").cast(pl.String)
        )
        .head(20)
        .to_dicts(),
        "missing_inactive_eodhd_seed_keys": missing_seed_keys,
        "historical_revision_override_enabled": policy.allow_historical_price_revisions,
        "historical_key_removal_override_enabled": policy.allow_historical_price_key_removals,
        "security_identity_policy": {
            "candidate_compliant": True,
            "previous_rows_rejected": (
                previous_identity.rejected.height if previous_identity is not None else 0
            ),
        },
        "blocking_reasons": blocking_reasons,
        "passed": not blocking_reasons,
    }
    return PriceGateResult(
        report,
        revisions,
        transition_findings,
        historical_key_removals,
    )


def validate_price_candidate(result: PriceGateResult) -> None:
    validate_price_gate_report(result.report)


def validate_price_gate_report(report: dict[str, object]) -> None:
    """Apply a recorded price gate only at the publication boundary."""

    blocking_reasons = report.get("blocking_reasons", [])
    if blocking_reasons:
        raise RuntimeError(
            "Canonical price candidate failed publication gates: "
            f"{blocking_reasons}. No package was published."
        )


def _audit_active_vintages(
    *,
    lineage: pl.DataFrame,
    active_tickers: Sequence[str],
    active_resolution_vintage_id: str | None,
    candidate_mode: PriceCandidateMode,
    observed_active_tickers: Sequence[str],
) -> ActiveVintageAudit:
    active = tuple(_normalize_ticker(ticker) for ticker in active_tickers)
    active_lineage = lineage.filter(pl.col("ticker").is_in(active))
    non_yahoo_rows = active_lineage.filter(pl.col("source") != "yfinance").height
    yahoo = active_lineage.filter(pl.col("source") == "yfinance")
    vintages = yahoo.group_by("ticker").agg(
        pl.col("source_vintage_id").n_unique().alias("vintage_count")
    )
    mixed = tuple(
        vintages.filter(pl.col("vintage_count") != 1).get_column("ticker").to_list()
    )
    global_vintages = tuple(
        yahoo.select(pl.col("source_vintage_id").drop_nulls().unique())
        .get_column("source_vintage_id")
        .to_list()
    )
    current = {_normalize_ticker(ticker) for ticker in observed_active_tickers}
    carried_rows = 0
    carried_tickers = 0
    if (
        active_resolution_vintage_id is not None
        and candidate_mode == PriceCandidateMode.FRESH_ACTIVE_VINTAGE
    ):
        current = set(
            yahoo.filter(pl.col("source_vintage_id") == active_resolution_vintage_id)
            .get_column("ticker")
            .unique()
            .to_list()
        )
        carried = yahoo.filter(pl.col("source_vintage_id") != active_resolution_vintage_id)
        carried_rows = carried.height
        carried_tickers = (
            carried.select(pl.col("ticker").n_unique()).item() if carried.height else 0
        )
    refreshed = current or set(vintages.get_column("ticker").to_list())
    return ActiveVintageAudit(
        active_tickers=active,
        active_non_yahoo_rows=non_yahoo_rows,
        mixed_active_tickers=mixed,
        global_vintage_ids=global_vintages,
        current_resolution_tickers=frozenset(current),
        carried_active_rows=carried_rows,
        carried_active_tickers=carried_tickers,
        missing_active_tickers=tuple(sorted(set(active) - refreshed)),
    )


def _transition_factor_findings(lineage: pl.DataFrame, *, policy: PriceGatePolicy) -> pl.DataFrame:
    frame = (
        lineage.filter(
            pl.col("close").is_not_null()
            & (pl.col("close") != 0)
            & pl.col("adjusted_close").is_not_null()
        )
        .sort(["ticker", "date"])
        .with_columns(
            (pl.col("adjusted_close") / pl.col("close")).alias("adjustment_factor"),
            pl.col("source_vintage_id").shift(1).over("ticker").alias("previous_source_vintage_id"),
            pl.col("source").shift(1).over("ticker").alias("previous_source"),
        )
        .with_columns(
            pl.col("adjustment_factor").shift(1).over("ticker").alias("previous_adjustment_factor")
        )
        .with_columns(
            (
                (pl.col("source_vintage_id") != pl.col("previous_source_vintage_id"))
                | (pl.col("source") != pl.col("previous_source"))
            )
            .fill_null(False)
            .alias("is_transition"),
            ((pl.col("adjustment_factor") / pl.col("previous_adjustment_factor")) - 1.0)
            .abs()
            .alias("absolute_factor_jump"),
        )
    )
    return frame.filter(
        pl.col("is_transition")
        & (pl.col("absolute_factor_jump") > policy.transition_factor_jump_threshold)
    ).select(
        "ticker",
        "date",
        "previous_source",
        "source",
        "previous_source_vintage_id",
        "source_vintage_id",
        "previous_adjustment_factor",
        "adjustment_factor",
        "absolute_factor_jump",
    )


def _daily_return_revisions(
    previous: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    storage_tolerance: float,
) -> pl.DataFrame:
    old = _daily_returns(previous, "previous")
    new = _daily_returns(candidate, "candidate")
    difference = pl.col("candidate_daily_return") - pl.col("previous_daily_return")
    return (
        old.join(new, on=["ticker", "date"], how="inner")
        .with_columns(
            difference.alias("daily_return_difference"),
            difference.abs().alias("absolute_daily_return_difference"),
        )
        .filter(
            (
                pl.col("previous_daily_return").is_null()
                != pl.col("candidate_daily_return").is_null()
            )
            | (
                pl.col("absolute_daily_return_difference")
                > storage_tolerance
            )
        )
        .sort("absolute_daily_return_difference", descending=True, nulls_last=True)
    )


def _daily_returns(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    return (
        frame.sort(["ticker", "date"])
        .with_columns(
            pl.col("adjusted_close").pct_change().over("ticker").alias(f"{prefix}_daily_return")
        )
        .select("ticker", "date", f"{prefix}_daily_return")
    )


def _normalize_prices(frame: pl.DataFrame | None) -> pl.DataFrame:
    if frame is None or frame.is_empty():
        return pl.DataFrame(
            schema={"ticker": pl.String, "date": pl.Date, "adjusted_close": pl.Float64}
        )
    date_expr = (
        pl.col("date").str.to_date(strict=False)
        if frame.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    return frame.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        date_expr.alias("date"),
        pl.col("adjusted_close").cast(pl.Float64, strict=False),
    )


def _normalize_lineage(frame: pl.DataFrame) -> pl.DataFrame:
    required = {"ticker", "date", "close", "adjusted_close", "source", "source_vintage_id"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Candidate price lineage is missing columns: {sorted(missing)}")
    date_expr = (
        pl.col("date").str.to_date(strict=False)
        if frame.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    return frame.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        date_expr.alias("date"),
    )


def _require_unique(frame: pl.DataFrame, *, label: str) -> None:
    duplicate_count = frame.height - frame.select(pl.struct(["ticker", "date"]).n_unique()).item()
    if duplicate_count:
        raise RuntimeError(f"{label} has {duplicate_count} duplicate ticker/date keys")


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).upper()
    return value if value.endswith(".US") else f"{value}.US"


def _empty_revision_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "date": pl.Date,
            "previous_daily_return": pl.Float64,
            "candidate_daily_return": pl.Float64,
            "daily_return_difference": pl.Float64,
            "absolute_daily_return_difference": pl.Float64,
        }
    )
