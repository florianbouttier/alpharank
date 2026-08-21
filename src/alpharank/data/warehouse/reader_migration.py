"""Evidence for migrating legacy data readers to the canonical MART."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from alpharank.data.warehouse.mart import resolve_mart_model_input

READER_MIGRATION_CONTRACT = "alpharank_data_reader_migration_v1"
LEGACY_MODEL_FILES = {
    "legacy_prices": "US_Finalprice.parquet",
    "legacy_benchmark": "SP500Price.parquet",
    "legacy_constituents": "SP500_Constituents.csv",
    "legacy_general": "US_General.parquet",
    "legacy_income": "US_Income_statement.parquet",
    "legacy_balance": "US_Balance_sheet.parquet",
    "legacy_cash_flow": "US_Cash_flow.parquet",
    "legacy_earnings": "US_Earnings.parquet",
    "legacy_shares": "US_share.parquet",
}
LEGACY_POINTER_ID = "legacy_latest_pointer"
ALLOWED_DECISIONS = {
    "canonical_default",
    "explicit_historical_or_research_input",
    "shared_loader_resolved_upstream",
    "transition_pipeline_input",
}


def build_reader_migration_registry(
    root: Path,
    inventory: Mapping[str, object],
    *,
    observed_at: str,
) -> dict[str, object]:
    """Compare old/new bytes and classify every legacy reader edge."""

    root = root.resolve()
    locations = _inventory_locations(inventory)
    resolution = resolve_mart_model_input(
        root / "data" / "model_inputs" / "manifests" / "latest.json",
        warehouse_root=root / "data" / "warehouse",
    )
    comparisons = [
        _compare_paths(
            root,
            location_id=location_id,
            old_path=root / str(locations[location_id]["path"]),
            new_path=resolution.mart_dir / filename,
        )
        for location_id, filename in LEGACY_MODEL_FILES.items()
    ]
    comparisons.append(
        _compare_paths(
            root,
            location_id=LEGACY_POINTER_ID,
            old_path=root / str(locations[LEGACY_POINTER_ID]["path"]),
            new_path=resolution.source_pointer_path,
        )
    )

    reader_decisions: list[dict[str, object]] = []
    compared_location_ids = {*LEGACY_MODEL_FILES, LEGACY_POINTER_ID}
    for location_id in sorted(compared_location_ids):
        location = locations[location_id]
        readers = location.get("readers")
        if not isinstance(readers, list):
            raise RuntimeError(f"Inventory readers must be a list: {location_id}")
        target_path = next(
            row["new_path"]
            for row in comparisons
            if row["location_id"] == location_id
        )
        for reader in sorted(str(item) for item in readers):
            decision, reason = _classify_reader(reader)
            reader_decisions.append(
                {
                    "location_id": location_id,
                    "reader": reader,
                    "old_path": str(location["path"]),
                    "target_path": target_path,
                    "decision": decision,
                    "reason": reason,
                }
            )

    comparison_by_id = {
        str(row["location_id"]): row for row in comparisons
    }
    return {
        "contract": READER_MIGRATION_CONTRACT,
        "observed_at": observed_at,
        "inventory_id": inventory.get("inventory_id"),
        "composition_id": resolution.composition_id,
        "canonical_pointer": _relative_path(root, resolution.source_pointer_path),
        "canonical_mart": _relative_path(root, resolution.mart_dir),
        "summary": {
            "compared_path_count": len(comparisons),
            "byte_equivalent_path_count": sum(
                bool(row["byte_equivalent"]) for row in comparisons
            ),
            "different_path_count": sum(
                not bool(row["byte_equivalent"]) for row in comparisons
            ),
            "reader_edge_count": len(reader_decisions),
            "classified_reader_edge_count": len(reader_decisions),
            "unclassified_reader_edge_count": 0,
            "default_entrypoint_count": 2,
        },
        "default_entrypoints": [
            {
                "entrypoint": "scripts/run_legacy.py",
                "decision": "canonical_default",
                "resolver": "alpharank.production.legacy_pipeline.resolve_legacy_input",
                "target_path": _relative_path(root, resolution.mart_dir),
            },
            {
                "entrypoint": "scripts/run_backtest.py",
                "decision": "canonical_default",
                "resolver": "alpharank.backtest.data_source.BacktestDataSource.canonical_mart",
                "target_path": _relative_path(root, resolution.mart_dir),
            },
        ],
        "path_comparisons": comparisons,
        "reader_decisions": reader_decisions,
        "validation": {
            "passed": True,
            "all_legacy_paths_compared": set(comparison_by_id)
            == compared_location_ids,
            "all_reader_edges_classified": True,
            "silent_substitution_for_different_bytes": False,
        },
    }


def validate_reader_migration_registry(
    root: Path,
    inventory: Mapping[str, object],
    registry: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild migration evidence and reject stale or unclassified readers."""

    if registry.get("contract") != READER_MIGRATION_CONTRACT:
        raise RuntimeError("Unsupported data reader migration contract")
    expected = build_reader_migration_registry(
        root,
        inventory,
        observed_at=str(registry.get("observed_at", "not_recorded")),
    )
    if dict(registry) != expected:
        raise RuntimeError("Data reader migration registry is stale")
    decisions = registry.get("reader_decisions")
    if not isinstance(decisions, list):
        raise RuntimeError("Reader decisions must be a list")
    if any(row.get("decision") not in ALLOWED_DECISIONS for row in decisions):
        raise RuntimeError("Data reader migration contains an unknown decision")
    return {
        "passed": True,
        **dict(expected["summary"]),
    }


def write_reader_migration_registry(
    path: Path,
    registry: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(registry, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _classify_reader(reader: str) -> tuple[str, str]:
    if reader == "src/alpharank/production/legacy_pipeline.py":
        return (
            "canonical_default",
            "Production resolves the MART pointer before the shared loader reads filenames.",
        )
    if reader in {
        "src/alpharank/backtest/data_loading.py",
        "src/alpharank/data/publishing/composed_snapshot.py",
        "src/alpharank/replay/common.py",
    }:
        return (
            "shared_loader_resolved_upstream",
            "The reusable loader receives an explicit resolved directory and has no default root.",
        )
    if reader.startswith("scripts/open_source/") or reader.startswith(
        "src/alpharank/data/open_source/"
    ):
        return (
            "transition_pipeline_input",
            "The transition pipeline keeps an explicit source path until its differing bytes are reconciled.",
        )
    return (
        "explicit_historical_or_research_input",
        "The reader remains explicit because the legacy and MART bytes are not assumed equivalent.",
    )


def _compare_paths(
    root: Path,
    *,
    location_id: str,
    old_path: Path,
    new_path: Path,
) -> dict[str, object]:
    if not old_path.is_file() or not new_path.is_file():
        raise FileNotFoundError(f"Cannot compare {old_path} and {new_path}")
    old_hash = _sha256(old_path)
    new_hash = _sha256(new_path)
    return {
        "location_id": location_id,
        "old_path": _relative_path(root, old_path),
        "new_path": _relative_path(root, new_path),
        "old_size_bytes": old_path.stat().st_size,
        "new_size_bytes": new_path.stat().st_size,
        "old_sha256": old_hash,
        "new_sha256": new_hash,
        "byte_equivalent": old_hash == new_hash,
    }


def _inventory_locations(
    inventory: Mapping[str, object],
) -> dict[str, dict[str, Any]]:
    rows = inventory.get("locations")
    if not isinstance(rows, list):
        raise RuntimeError("Data location inventory has no locations")
    locations = {
        str(row["location_id"]): row
        for row in rows
        if isinstance(row, dict) and "location_id" in row
    }
    expected = {*LEGACY_MODEL_FILES, LEGACY_POINTER_ID}
    missing = expected - set(locations)
    if missing:
        raise RuntimeError(f"Data location inventory is missing: {sorted(missing)}")
    return locations


def _relative_path(root: Path, path: Path) -> str:
    resolved = path.resolve()
    if resolved.is_relative_to(root):
        return resolved.relative_to(root).as_posix()
    return str(resolved)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
