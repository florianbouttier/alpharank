"""Persist run-scoped price publication gates and reconciliation evidence."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.ingestion.price_publication_candidate import PricePublicationCandidate
from alpharank.data.ingestion.storage import write_json


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
        }
    )
    write_json(run_dir / "price_composition.json", hybrid.composition_report)
    write_json(run_dir / "price_revision_guard.json", gate.report)
    write_json(run_dir / "price_provider_revision_guard.json", candidate.provider_gate.report)
    write_json(run_dir / "price_revision_diagnostic.json", candidate.revision_diagnostic)
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
