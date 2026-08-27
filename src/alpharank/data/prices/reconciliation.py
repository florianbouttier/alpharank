"""Preserve validated price history and append only newly observed returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import polars as pl

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
    PRICE_VALUE_COLUMNS,
)
from alpharank.data.security_identity import apply_security_identity_policy

PRICE_RECONCILIATION_CONTRACT = "validated_history_return_extension_v1"


@dataclass(frozen=True, slots=True)
class PriceReconciliationResult:
    prices: pl.DataFrame
    lineage: pl.DataFrame
    extension_audit: pl.DataFrame
    report: dict[str, object]
    observed_active_tickers: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PriceReconciliationContext:
    active_tickers: Sequence[str]
    preserved_terminal_tickers: Sequence[str]
    incomplete_provider_tickers: Sequence[str]
    run_id: str


@dataclass(frozen=True, slots=True)
class ReconciliationReportInputs:
    previous: pl.DataFrame
    lineage: pl.DataFrame
    extension_audit: pl.DataFrame
    observed_tickers: tuple[str, ...]
    refreshable_tickers: list[str]
    unresolved_tickers: list[dict[str, str]]
    retained_incomplete_tickers: list[dict[str, str]]
    new_ticker_rows: int
    run_id: str
    previous_identity_rejected_rows: int
    current_identity_rejected_rows: int


def reconcile_validated_price_history(
    *,
    previous_validated_lineage: pl.DataFrame,
    current_yahoo_observation: pl.DataFrame,
    context: PriceReconciliationContext,
) -> PriceReconciliationResult:
    """Keep every validated key and derive only dates after each ticker anchor."""

    previous, current, previous_rejected, current_rejected = _prepare_reconciliation_inputs(
        previous_validated_lineage, current_yahoo_observation
    )
    active = {_normalize_ticker(ticker) for ticker in context.active_tickers}
    terminal = {_normalize_ticker(ticker) for ticker in context.preserved_terminal_tickers}
    incomplete = {_normalize_ticker(ticker) for ticker in context.incomplete_provider_tickers}
    refreshable = sorted(active - terminal)
    observed = tuple(sorted(set(current.get_column("ticker").unique().to_list())))
    previous_by_ticker = _partition_by_ticker(previous.filter(pl.col("ticker").is_in(refreshable)))
    current_by_ticker = _partition_by_ticker(current.filter(pl.col("ticker").is_in(refreshable)))
    frames = [previous]
    audit_frames: list[pl.DataFrame] = []
    issues: list[dict[str, str]] = []
    retained_incomplete: list[dict[str, str]] = []
    new_ticker_rows = 0
    for ticker in refreshable:
        prior_ticker = previous_by_ticker.get(ticker)
        current_ticker = current_by_ticker.get(ticker)
        prior_ticker = previous.clear() if prior_ticker is None else prior_ticker
        current_ticker = current.clear() if current_ticker is None else current_ticker
        if current_ticker.is_empty():
            issues.append({"ticker": ticker, "reason": "current_provider_observation_missing"})
            continue
        if prior_ticker.is_empty():
            frames.append(current_ticker)
            new_ticker_rows += current_ticker.height
            continue
        extension, audit, issue = _build_ticker_extension(
            prior_ticker, current_ticker, context.run_id
        )
        if issue is not None:
            finding = {"ticker": ticker, "reason": issue}
            if ticker in incomplete:
                retained_incomplete.append(finding)
            else:
                issues.append(finding)
            continue
        if not extension.is_empty():
            frames.append(extension)
            audit_frames.append(audit)
    lineage = _combine_reconciled_frames(frames)
    _validate_preserved_prefix(previous=previous, reconciled=lineage)
    extension_audit = _combine_audits(audit_frames)
    report = _build_report(
        ReconciliationReportInputs(
            previous=previous,
            lineage=lineage,
            extension_audit=extension_audit,
            observed_tickers=observed,
            refreshable_tickers=refreshable,
            unresolved_tickers=issues,
            retained_incomplete_tickers=retained_incomplete,
            new_ticker_rows=new_ticker_rows,
            run_id=context.run_id,
            previous_identity_rejected_rows=previous_rejected,
            current_identity_rejected_rows=current_rejected,
        )
    )
    return PriceReconciliationResult(
        prices=lineage.select(PRICE_VALUE_COLUMNS),
        lineage=lineage,
        extension_audit=extension_audit,
        report=report,
        observed_active_tickers=observed,
    )


def _prepare_reconciliation_inputs(
    previous: pl.DataFrame,
    current: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, int, int]:
    previous_identity = apply_security_identity_policy(
        previous, ticker_column="ticker", date_column="date"
    )
    current_identity = apply_security_identity_policy(
        current, ticker_column="ticker", date_column="date"
    )
    normalized_previous = _normalize_lineage(previous_identity.frame)
    normalized_current = _normalize_lineage(current_identity.frame)
    return (
        normalized_previous,
        normalized_current,
        previous_identity.rejected.height,
        current_identity.rejected.height,
    )


def _build_ticker_extension(
    previous: pl.DataFrame,
    current: pl.DataFrame,
    run_id: str,
) -> tuple[pl.DataFrame, pl.DataFrame, str | None]:
    anchor = previous.sort("date").tail(1).row(0, named=True)
    anchor_date = anchor["date"]
    provider_anchor = current.filter(pl.col("date") == anchor_date)
    if provider_anchor.height != 1:
        return current.clear(), _empty_audit(), "provider_anchor_for_validated_tail_missing"
    with_returns = current.with_columns(
        pl.col("adjusted_close").pct_change().alias("provider_daily_return")
    )
    tail = with_returns.filter(pl.col("date") > anchor_date)
    if tail.is_empty():
        return current.clear(), _empty_audit(), None
    if tail.filter(
        pl.col("provider_daily_return").is_null()
        | ~pl.col("provider_daily_return").is_finite()
        | (pl.col("provider_daily_return") <= -1.0)
    ).height:
        return current.clear(), _empty_audit(), "provider_tail_return_unusable"
    extension = _scale_tail_from_validated_anchor(tail, anchor, run_id)
    audit = extension.select(
        "ticker",
        pl.lit(anchor_date).alias("validated_anchor_date"),
        "date",
        "provider_daily_return",
        pl.col("provider_adjusted_close"),
        pl.col("adjusted_close").alias("selected_adjusted_close"),
        "adjustment_bridge_factor",
        pl.lit("new_return_appended_to_validated_history").alias("selection_reason"),
    )
    return extension.select(PRICE_LINEAGE_COLUMNS), audit, None


def _scale_tail_from_validated_anchor(
    tail: pl.DataFrame,
    anchor: dict[str, object],
    run_id: str,
) -> pl.DataFrame:
    anchor_adjusted = float(anchor["adjusted_close"])
    anchor_close = float(anchor["close"])
    anchor_factor = anchor_adjusted / anchor_close if anchor_close else 1.0
    extension = tail.with_columns(
        pl.col("adjusted_close").alias("provider_adjusted_close"),
        (pl.lit(anchor_adjusted) * (pl.col("provider_daily_return") + 1.0).cum_prod()).alias(
            "selected_adjusted_close"
        ),
    ).with_columns(
        pl.col("selected_adjusted_close").alias("adjusted_close"),
        (pl.col("selected_adjusted_close") / anchor_factor).alias("selected_close"),
    )
    for column in ("open", "high", "low"):
        extension = extension.with_columns(
            pl.when(pl.col("close").is_not_null() & (pl.col("close") != 0.0))
            .then(pl.col(column) / pl.col("close") * pl.col("selected_close"))
            .otherwise(None)
            .alias(column)
        )
    return extension.with_columns(
        pl.col("selected_close").alias("close"),
        pl.lit("yfinance_return_ledger").alias("source"),
        pl.lit("prices_yfinance_return_extension").alias("dataset"),
        pl.lit(run_id).alias("source_vintage_id"),
        pl.lit(run_id).alias("return_source_vintage_id"),
        pl.lit(ADJUSTMENT_POLICY_VERSION).alias("adjustment_policy_version"),
        (pl.col("selected_adjusted_close") / pl.col("provider_adjusted_close")).alias(
            "adjustment_bridge_factor"
        ),
        pl.lit(anchor.get("eodhd_seed_sha256")).cast(pl.String).alias("eodhd_seed_sha256"),
        pl.lit(None).cast(pl.String).alias("correction_overlay_id"),
    )


def _normalize_lineage(frame: pl.DataFrame) -> pl.DataFrame:
    missing_values = set(PRICE_VALUE_COLUMNS) - set(frame.columns)
    if missing_values:
        raise ValueError(f"Price reconciliation input is missing: {sorted(missing_values)}")
    normalized = frame.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.String),
    )
    defaults = {
        "source": pl.lit("yfinance"),
        "dataset": pl.lit("prices_yfinance"),
        "ingestion_run_id": pl.lit("unknown"),
        "ingested_at": pl.lit("unknown"),
        "source_vintage_id": pl.col("ingestion_run_id"),
        "return_source_vintage_id": pl.col("ingestion_run_id"),
        "adjustment_policy_version": pl.lit(ADJUSTMENT_POLICY_VERSION),
        "adjustment_bridge_factor": pl.lit(1.0),
        "eodhd_seed_sha256": pl.lit(None).cast(pl.String),
        "correction_overlay_id": pl.lit(None).cast(pl.String),
    }
    for column, expression in defaults.items():
        if column not in normalized.columns:
            normalized = normalized.with_columns(expression.alias(column))
    return normalized.select(PRICE_LINEAGE_COLUMNS).sort(["ticker", "date"])


def _combine_reconciled_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .sort(["ticker", "date"])
        .unique(subset=["ticker", "date"], keep="first", maintain_order=True)
        .select(PRICE_LINEAGE_COLUMNS)
    )


def _partition_by_ticker(frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    return {
        key[0] if isinstance(key, tuple) else key: ticker_frame
        for key, ticker_frame in frame.partition_by("ticker", as_dict=True).items()
    }


def _validate_preserved_prefix(*, previous: pl.DataFrame, reconciled: pl.DataFrame) -> None:
    selected = reconciled.join(
        previous.select("ticker", "date"), on=["ticker", "date"], how="inner"
    ).select(PRICE_LINEAGE_COLUMNS)
    if previous.height != selected.height or not previous.equals(selected, null_equal=True):
        raise RuntimeError("Reconciled prices changed a previously validated ticker/date row")


def _build_report(inputs: ReconciliationReportInputs) -> dict[str, object]:
    return {
        "contract": PRICE_RECONCILIATION_CONTRACT,
        "run_id": inputs.run_id,
        "selection_rule": "retain_validated_keys_and_append_new_provider_daily_returns",
        "previous_validated_rows": inputs.previous.height,
        "previous_validated_rows_changed": 0,
        "previous_identity_rejected_rows": inputs.previous_identity_rejected_rows,
        "current_identity_rejected_rows": inputs.current_identity_rejected_rows,
        "candidate_rows": inputs.lineage.height,
        "new_ticker_history_rows": inputs.new_ticker_rows,
        "return_extension_rows": inputs.extension_audit.height,
        "return_extension_tickers": inputs.extension_audit.select(
            pl.col("ticker").n_unique()
        ).item()
        if inputs.extension_audit.height
        else 0,
        "refreshable_active_ticker_count": len(inputs.refreshable_tickers),
        "current_provider_observed_ticker_count": len(inputs.observed_tickers),
        "retained_incomplete_provider_tickers": inputs.retained_incomplete_tickers,
        "unresolved_tickers": inputs.unresolved_tickers,
        "blocking_reasons": ["unresolved_validated_return_extension"]
        if inputs.unresolved_tickers
        else [],
        "passed": not inputs.unresolved_tickers,
        "canonical_storage": "one validated history plus new return-derived rows",
        "provider_evidence": "immutable RAW delta archive and run-scoped revision audit",
    }


def _combine_audits(frames: list[pl.DataFrame]) -> pl.DataFrame:
    return (
        pl.concat(frames, how="vertical_relaxed").sort(["ticker", "date"])
        if frames
        else _empty_audit()
    )


def _empty_audit() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "validated_anchor_date": pl.String,
            "date": pl.String,
            "provider_daily_return": pl.Float64,
            "provider_adjusted_close": pl.Float64,
            "selected_adjusted_close": pl.Float64,
            "adjustment_bridge_factor": pl.Float64,
            "selection_reason": pl.String,
        }
    )


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).upper()
    return value if value.endswith(".US") else f"{value}.US"
