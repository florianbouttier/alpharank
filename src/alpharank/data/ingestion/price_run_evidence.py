"""Persist run-scoped price publication gates and reconciliation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.data.ingestion.price_publication_candidate import PricePublicationCandidate
from alpharank.data.ingestion.storage import write_json
from alpharank.data.open_source.price_quality import (
    audit_extreme_adjusted_price_moves,
    load_reviewed_extreme_price_moves,
)


@dataclass(frozen=True, slots=True)
class ExtremePriceMoveEvidenceContext:
    previous_prices: pl.DataFrame | None
    event_since: str
    active_tickers: Sequence[str]
    preserved_terminal_tickers: Sequence[str]
    reviewed_registry_path: Path | None


def persist_price_candidate_evidence(
    *,
    run_dir: Path,
    source_refresh_contract: dict[str, object],
    eodhd_seed_manifest: dict[str, object],
    candidate: PricePublicationCandidate,
    persistent_registry: pl.DataFrame,
) -> None:
    """Write the provider observation, canonical gate, and selected extension."""

    hybrid = candidate.hybrid
    gate = candidate.gate
    source_refresh_contract.update(
        {
            "eodhd_price_seed": eodhd_seed_manifest,
            "price_composition": hybrid.composition_report,
            "price_provider_revision_guard": candidate.provider_gate.report,
            "price_revision_diagnostic": candidate.revision_diagnostic,
            "price_revision_guard": gate.report,
            "price_ticker_transition": candidate.ticker_transition.report,
        }
    )
    write_json(run_dir / "price_composition.json", hybrid.composition_report)
    write_json(run_dir / "price_revision_guard.json", gate.report)
    write_json(run_dir / "price_provider_revision_guard.json", candidate.provider_gate.report)
    write_json(run_dir / "price_revision_diagnostic.json", candidate.revision_diagnostic)
    write_json(
        run_dir / "price_ticker_transition.json",
        candidate.ticker_transition.report,
    )
    candidate.ticker_transition.audit.write_parquet(
        run_dir / "price_ticker_transition_audit.parquet"
    )
    persistent_registry.write_parquet(run_dir / "persistent_price_history_registry.parquet")
    candidate.provider_gate.daily_return_revisions.write_parquet(
        run_dir / "price_daily_return_revisions.parquet"
    )
    gate.daily_return_revisions.write_parquet(
        run_dir / "price_canonical_daily_return_revisions.parquet"
    )
    gate.transition_factor_findings.write_parquet(
        run_dir / "price_transition_factor_findings.parquet"
    )
    gate.historical_key_removals.write_parquet(run_dir / "price_historical_key_removals.parquet")
    if candidate.reconciliation is None:
        return
    reconciliation = candidate.reconciliation
    source_refresh_contract["price_revision_reconciliation"] = reconciliation.report
    write_json(run_dir / "price_revision_reconciliation.json", reconciliation.report)
    reconciliation.extension_audit.write_parquet(run_dir / "price_return_extension_audit.parquet")


def persist_extreme_price_move_evidence(
    *,
    run_dir: Path,
    source_refresh_contract: dict[str, object],
    candidate_prices: pl.DataFrame,
    context: ExtremePriceMoveEvidenceContext,
) -> None:
    """Record a deferred gate over only the candidate's new canonical keys."""

    review_keys = _resolve_price_review_keys(
        candidate_prices=candidate_prices,
        previous_prices=context.previous_prices,
        event_since=context.event_since,
    )
    quality_tickers = _quality_tickers(context)
    reviewed_moves = None
    reviewed_manifest: dict[str, object] | None = None
    if context.reviewed_registry_path is not None:
        reviewed_moves, reviewed_manifest = load_reviewed_extreme_price_moves(
            context.reviewed_registry_path
        )
    gate = audit_extreme_adjusted_price_moves(
        candidate_prices,
        review_keys=review_keys,
        tickers=quality_tickers,
        reviewed_moves=reviewed_moves,
    )
    source_refresh_contract["price_extreme_move_guard"] = gate.report
    write_json(run_dir / "price_extreme_move_guard.json", gate.report)
    gate.findings.write_parquet(run_dir / "price_extreme_move_findings.parquet")
    gate.unreviewed.write_parquet(run_dir / "price_extreme_move_unreviewed.parquet")
    if reviewed_manifest is not None:
        _persist_reviewed_move_matches(
            run_dir=run_dir,
            source_refresh_contract=source_refresh_contract,
            manifest=reviewed_manifest,
            reviewed=gate.reviewed,
        )


def _resolve_price_review_keys(
    *,
    candidate_prices: pl.DataFrame,
    previous_prices: pl.DataFrame | None,
    event_since: str,
) -> pl.DataFrame:
    candidate_keys = candidate_prices.select("ticker", "date").with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
    )
    if previous_prices is None:
        return candidate_keys.filter(pl.col("date") >= pl.lit(event_since).cast(pl.Date))
    previous_keys = previous_prices.select("ticker", "date").with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
    )
    return candidate_keys.join(previous_keys, on=["ticker", "date"], how="anti")


def _quality_tickers(context: ExtremePriceMoveEvidenceContext) -> list[str]:
    terminal = {_normalize_ticker(ticker) for ticker in context.preserved_terminal_tickers}
    return [
        normalized
        for ticker in context.active_tickers
        if (normalized := _normalize_ticker(ticker)) not in terminal
    ]


def _persist_reviewed_move_matches(
    *,
    run_dir: Path,
    source_refresh_contract: dict[str, object],
    manifest: dict[str, object],
    reviewed: pl.DataFrame,
) -> None:
    report = {
        **manifest,
        "matched_count": reviewed.height,
        "matched_events": reviewed.with_columns(pl.col("date").cast(pl.String)).to_dicts(),
    }
    source_refresh_contract["reviewed_extreme_price_moves"] = report
    write_json(run_dir / "reviewed_extreme_price_moves.json", report)


def _normalize_ticker(ticker: str) -> str:
    return f"{str(ticker).upper().removesuffix('.US')}.US"
