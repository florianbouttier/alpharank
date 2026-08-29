"""Write one immutable canonical price package and its publication evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import polars as pl

from alpharank.data.ingestion.acquisition_status import build_price_publication_guard
from alpharank.data.open_source.price_quality import ExtremePriceMoveGateResult
from alpharank.data.prices import (
    EodhdSeed,
    HybridPriceResult,
    PriceGateResult,
    PriceTickerTransitionResult,
)
from alpharank.data.security_identity import (
    SECURITY_IDENTITY_POLICY_ID,
    SecurityIdentityApplication,
)


@dataclass(frozen=True, slots=True)
class PricePackageRequest:
    """All paths and immutable context required to build a price package."""

    run_id: str
    source_refresh_contract: Mapping[str, object]
    previous_lineage_path: Path
    previous_resolution: str
    previous_composition_id: str | None
    fresh_yahoo_path: Path
    benchmark_path: Path
    constituents_path: Path
    eodhd_seed_path: Path
    output_dir: Path
    expected_through: str
    start_date: str
    preserve_terminal_tickers: tuple[str, ...]
    constituent_registry_path: Path
    reviewed_move_registry_path: Path
    base_package_dir: Path | None = None
    expected_benchmark_run_id: str | None = None
    data_freshness: Mapping[str, object] | None = None
    sec_package_dir: Path | None = None
    reassessment: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class PreparedPricePackage:
    """Validated in-memory payload and reports ready for immutable writing."""

    result: HybridPriceResult
    revision_gate: PriceGateResult
    extreme_gate: ExtremePriceMoveGateResult
    benchmark_prices: pl.DataFrame
    constituents: SecurityIdentityApplication
    history_registry: pl.DataFrame
    history_summary: Mapping[str, object]
    seed: EodhdSeed
    security_identities: pl.DataFrame
    reviewed_registry_manifest: Mapping[str, object]
    data_freshness: Mapping[str, object]
    ticker_transition: PriceTickerTransitionResult


def write_price_package(
    request: PricePackageRequest,
    prepared: PreparedPricePackage,
) -> dict[str, object]:
    """Write payloads, audit evidence and a hash-bound manifest."""

    output_dir = request.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    (output_dir / "lineage").mkdir(parents=True)
    (output_dir / "audit").mkdir()
    _write_payloads(output_dir, prepared)
    source_contract = _build_source_contract(request, prepared)
    manifest = _build_manifest(request, prepared, source_contract)
    write_json(output_dir / "lineage" / "manifest.json", manifest)
    return manifest


def _write_payloads(output_dir: Path, prepared: PreparedPricePackage) -> None:
    result = prepared.result
    revision = prepared.revision_gate
    extreme = prepared.extreme_gate
    result.prices.write_parquet(output_dir / "US_Finalprice.parquet")
    prepared.benchmark_prices.write_parquet(output_dir / "SP500Price.parquet")
    prepared.constituents.frame.write_csv(output_dir / "SP500_Constituents.csv")
    result.lineage.write_parquet(output_dir / "lineage" / "prices_open_source_lineage.parquet")
    prepared.ticker_transition.audit.write_parquet(
        output_dir / "lineage" / "price_ticker_transition_audit.parquet"
    )
    registry_path = prepared.ticker_transition.report.get("registry_path")
    if registry_path:
        shutil.copy2(
            Path(str(registry_path)),
            output_dir / "lineage" / "price_ticker_transition_policy.json",
        )
    prepared.history_registry.write_parquet(
        output_dir / "lineage" / "persistent_price_history_registry.parquet"
    )
    audit = output_dir / "audit"
    revision.daily_return_revisions.write_parquet(audit / "price_daily_return_revisions.parquet")
    revision.transition_factor_findings.write_parquet(
        audit / "price_transition_factor_findings.parquet"
    )
    revision.historical_key_removals.write_parquet(audit / "price_historical_key_removals.parquet")
    extreme.findings.write_parquet(audit / "price_extreme_move_findings.parquet")
    extreme.unreviewed.write_parquet(audit / "price_extreme_move_unreviewed.parquet")
    extreme.reviewed.write_parquet(audit / "price_extreme_move_reviewed.parquet")
    write_json(audit / "price_revision_guard.json", revision.report)
    write_json(audit / "price_extreme_move_guard.json", extreme.report)
    write_json(audit / "price_composition.json", result.composition_report)
    write_json(audit / "price_ticker_transition.json", prepared.ticker_transition.report)


def _build_source_contract(
    request: PricePackageRequest,
    prepared: PreparedPricePackage,
) -> dict[str, object]:
    source_contract = dict(request.source_refresh_contract)
    source_contract.update(
        {
            "contract_version": 2,
            "price_composition": prepared.result.composition_report,
            "price_revision_guard": prepared.revision_gate.report,
            "price_extreme_move_guard": prepared.extreme_gate.report,
            "previous_validated_price_lineage": {
                **file_record(request.previous_lineage_path),
                "resolution": request.previous_resolution,
                "composition_id": request.previous_composition_id,
            },
            "fresh_yahoo_vintage": file_record(request.fresh_yahoo_path),
            "eodhd_price_seed": prepared.seed.manifest(),
            "persistent_price_history": {
                **prepared.history_summary,
                "semantics": (
                    "Every ticker/date published by the preceding validated lineage is "
                    "retained when the ticker leaves the active refresh universe."
                ),
                "routine_deletion_allowed": False,
            },
            "security_identity": _security_identity_contract(prepared),
            "reviewed_extreme_price_moves": _reviewed_move_contract(prepared),
            "price_ticker_transition": _ticker_transition_contract(
                request.output_dir.resolve(), prepared
            ),
        }
    )
    source_contract["policy"] = {
        **dict(source_contract["policy"]),
        "allow_historical_price_revisions": False,
        "allow_historical_price_key_removals": False,
    }
    source_contract["price_publication_guard"] = build_price_publication_guard(source_contract)
    if request.reassessment is not None:
        source_contract["deferred_publication"] = dict(request.reassessment)
    return source_contract


def _build_manifest(
    request: PricePackageRequest,
    prepared: PreparedPricePackage,
    source_contract: Mapping[str, object],
) -> dict[str, object]:
    output_dir = request.output_dir.resolve()
    composition = prepared.result.composition_report
    publication_gate = source_contract["price_publication_guard"]
    return {
        "contract_version": 2,
        "scope": "canonical_price_package",
        "run_id": request.run_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "base_full_ingestion_package": (
            str(request.base_package_dir.resolve()) if request.base_package_dir else None
        ),
        "acquisition_run_reassessment": request.reassessment,
        "source_refresh_contract": source_contract,
        "data_freshness": request.data_freshness or prepared.data_freshness,
        "output_sha256": {
            name: sha256_file(output_dir / name)
            for name in ("US_Finalprice.parquet", "SP500Price.parquet", "SP500_Constituents.csv")
        },
        "validation": {
            "inactive_history_byte_preserved": True,
            "all_previous_validated_inactive_history_preserved": True,
            "open_source_only_inactive_history_persisted": True,
            "active_history_single_fresh_yahoo_vintage": (
                composition["audited_carried_active_rows"] == 0
            ),
            "active_history_audited_resolution_run": True,
            "active_history_audited_carried_rows": composition["audited_carried_active_rows"],
            "price_revision_guard_passed": prepared.revision_gate.report["passed"],
            "price_extreme_move_guard_passed": prepared.extreme_gate.report["passed"],
            "price_publication_guard_passed": publication_gate["passed"],
            "deferred_publication_without_network": request.reassessment is not None,
            "security_identity_policy_applied": True,
            "price_ticker_transition_policy_applied": True,
            "price_ticker_transition_added_rows": prepared.ticker_transition.audit.height,
        },
        "artifacts": {
            "price_lineage": file_record(
                output_dir / "lineage" / "prices_open_source_lineage.parquet"
            ),
            "persistent_price_history_registry": file_record(
                output_dir / "lineage" / "persistent_price_history_registry.parquet"
            ),
            "price_ticker_transition_audit": file_record(
                output_dir / "lineage" / "price_ticker_transition_audit.parquet"
            ),
            "price_ticker_transition_registry": file_record(
                output_dir / "lineage" / "price_ticker_transition_policy.json"
            ),
        },
    }


def _security_identity_contract(prepared: PreparedPricePackage) -> dict[str, object]:
    registry_path = Path(
        prepared.security_identities.get_column("registry_path").drop_nulls().unique().item()
    )
    return {
        "policy_id": SECURITY_IDENTITY_POLICY_ID,
        "registry": file_record(registry_path),
        "price_lineage": prepared.result.composition_report["security_identity"],
        "constituents": prepared.constituents.report,
    }


def _reviewed_move_contract(prepared: PreparedPricePackage) -> dict[str, object]:
    return {
        **prepared.reviewed_registry_manifest,
        "matched_count": prepared.extreme_gate.reviewed.height,
        "matched_events": prepared.extreme_gate.reviewed.with_columns(
            pl.col("date").cast(pl.String)
        ).to_dicts(),
    }


def _ticker_transition_contract(
    output_dir: Path,
    prepared: PreparedPricePackage,
) -> dict[str, object]:
    report = prepared.ticker_transition.report
    return {
        **report,
        "registry": file_record(output_dir / "lineage" / "price_ticker_transition_policy.json"),
        "audit": file_record(output_dir / "lineage" / "price_ticker_transition_audit.parquet"),
    }


def file_record(path: Path) -> dict[str, object]:
    resolved = path.resolve()
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
