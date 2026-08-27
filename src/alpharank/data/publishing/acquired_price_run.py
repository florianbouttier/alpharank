"""Reassess a completed acquisition run without contacting data providers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

from alpharank.data.ingestion.acquisition_status import ACQUISITION_STATUS_CONTRACT
from alpharank.data.publishing.price_package_output import PricePackageRequest, file_record

REQUIRED_ACQUISITION_SOURCES = frozenset(
    {
        "yahoo_prices",
        "yahoo_metadata",
        "yfinance_earnings",
        "sec_submissions",
        "sec_companyfacts",
        "sec_filing_documents",
        "simfin_fundamentals",
        "yfinance_fundamentals",
    }
)


def build_acquired_price_request(
    *,
    run_dir: Path,
    sec_package_dir: Path,
    constituents_path: Path,
    eodhd_seed_path: Path,
    output_dir: Path,
    expected_through: str,
    start_date: str,
    preserve_terminal_tickers: tuple[str, ...],
    constituent_registry_path: Path,
    reviewed_move_registry_path: Path,
    previous_lineage_path: Path | None = None,
) -> PricePackageRequest:
    """Bind a new candidate to the immutable evidence of one acquired run."""

    resolved_run_dir = run_dir.resolve()
    manifest = load_acquisition_run_manifest(resolved_run_dir)
    previous = resolve_acquired_previous_lineage(
        manifest,
        explicit=previous_lineage_path,
    )
    run_id = str(manifest["run_id"])
    contract = manifest["source_refresh_contract"]
    if not isinstance(contract, Mapping):
        raise RuntimeError("Acquisition source refresh contract is invalid")
    reassessment = {
        "contract": "acquired_run_deferred_publication_v1",
        "network_accessed": False,
        "acquisition_run_id": run_id,
        "acquisition_run_dir": str(resolved_run_dir),
        "acquisition_status": file_record(resolved_run_dir / "acquisition_status.json"),
        "original_source_refresh_contract": file_record(
            resolved_run_dir / "source_refresh_contract.json"
        ),
        "original_price_publication_guard": file_record(
            resolved_run_dir / "price_publication_guard.json"
        ),
        "review_registry": file_record(reviewed_move_registry_path.resolve()),
    }
    previous_contract = contract.get("previous_validated_price_lineage")
    composition_id = (
        str(previous_contract.get("composition_id"))
        if isinstance(previous_contract, Mapping) and previous_contract.get("composition_id")
        else None
    )
    return PricePackageRequest(
        run_id=run_id,
        source_refresh_contract=contract,
        previous_lineage_path=previous,
        previous_resolution="acquisition_run_bound_lineage",
        previous_composition_id=composition_id,
        fresh_yahoo_path=resolved_run_dir / "raw" / "prices_yfinance.parquet",
        benchmark_path=resolved_run_dir / "raw" / "prices_spy_yfinance.parquet",
        constituents_path=constituents_path.resolve(),
        eodhd_seed_path=eodhd_seed_path.resolve(),
        output_dir=output_dir.resolve(),
        expected_through=expected_through,
        start_date=start_date,
        preserve_terminal_tickers=preserve_terminal_tickers,
        constituent_registry_path=constituent_registry_path.resolve(),
        reviewed_move_registry_path=reviewed_move_registry_path.resolve(),
        expected_benchmark_run_id=run_id,
        sec_package_dir=sec_package_dir.resolve(),
        reassessment=reassessment,
    )


def load_acquisition_run_manifest(run_dir: Path) -> dict[str, object]:
    """Validate a completed acquisition before any gate is reassessed."""

    status = _read_json(run_dir / "acquisition_status.json")
    contract = _read_json(run_dir / "source_refresh_contract.json")
    run_id = str(status.get("run_id") or "").strip()
    if status.get("contract") != ACQUISITION_STATUS_CONTRACT:
        raise RuntimeError("Acquisition run has an unsupported status contract")
    expected_phase = "all_declared_sources_attempted_before_publication_decision"
    if status.get("phase") != expected_phase:
        raise RuntimeError("Acquisition run did not finish all declared source attempts")
    if not run_id or run_id != run_dir.name:
        raise RuntimeError("Acquisition run id does not match its immutable directory")
    source_rows = _source_inventory(status)
    missing_sources = sorted(REQUIRED_ACQUISITION_SOURCES - set(source_rows))
    if missing_sources:
        raise RuntimeError(f"Acquisition status omits declared sources: {missing_sources}")
    prices = source_rows["yahoo_prices"]
    if not str(prices.get("status", "")).startswith("downloaded"):
        raise RuntimeError("Acquisition run has no downloaded Yahoo price observation")
    if int(prices.get("downloaded_rows", 0)) <= 0:
        raise RuntimeError("Acquisition run downloaded zero Yahoo price rows")
    _validate_source_contract(contract)
    return {"run_id": run_id, "source_refresh_contract": contract}


def _source_inventory(status: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    sources = status.get("sources")
    if not isinstance(sources, list):
        raise RuntimeError("Acquisition status has no source inventory")
    return {
        str(item.get("source")): item
        for item in sources
        if isinstance(item, Mapping) and item.get("source")
    }


def _validate_source_contract(contract: Mapping[str, object]) -> None:
    if contract.get("snapshot_scope") != "full_ingestion":
        raise RuntimeError("Deferred publication requires a full-ingestion acquisition")
    policy = contract.get("policy")
    if not isinstance(policy, Mapping) or policy.get("require_eodhd_price_seed") is not True:
        raise RuntimeError("Deferred publication requires the immutable EODHD seed policy")
    semantics = contract.get("source_semantics")
    if not isinstance(semantics, Mapping):
        raise RuntimeError("Acquisition source semantics are missing")
    yahoo = semantics.get("yfinance_prices")
    if not isinstance(yahoo, Mapping):
        raise RuntimeError("Acquisition Yahoo source semantics are missing")
    if yahoo.get("network_missing_tickers") or yahoo.get(
        "benchmark_network_missing_tickers"
    ):
        raise RuntimeError("Acquisition Yahoo active or benchmark coverage is incomplete")
    for source in ("sec_companyfacts", "sec_submissions"):
        value = semantics.get(source)
        if not isinstance(value, Mapping) or value.get("active_network_complete") is not True:
            raise RuntimeError(f"Acquisition {source} active-universe refresh is incomplete")


def resolve_acquired_previous_lineage(
    manifest: Mapping[str, object],
    *,
    explicit: Path | None,
) -> Path:
    contract = manifest.get("source_refresh_contract")
    if not isinstance(contract, Mapping):
        raise RuntimeError("Acquisition manifest has no source refresh contract")
    previous = contract.get("previous_validated_price_lineage")
    if not isinstance(previous, Mapping) or not previous.get("path"):
        raise RuntimeError("Acquisition run does not bind its previous validated lineage")
    acquired_path = Path(str(previous["path"])).resolve()
    if explicit is not None and explicit.resolve() != acquired_path:
        raise RuntimeError("Explicit previous lineage differs from the acquisition baseline")
    if not acquired_path.is_file():
        raise FileNotFoundError(acquired_path)
    return acquired_path


def _read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
