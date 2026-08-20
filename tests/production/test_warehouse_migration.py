from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.mart import resolve_mart_model_input
from alpharank.data.warehouse_migration import (
    catalog_existing_eodhd,
    migrate_validated_snapshot_to_warehouse,
    promote_mart_pointer,
    rollback_mart_pointer,
    validate_eodhd_catalog,
    validate_warehouse_mart,
)

MODEL_FILES = (
    "US_Finalprice.parquet",
    "SP500Price.parquet",
    "SP500_Constituents.csv",
    "US_General.parquet",
    "US_Income_statement.parquet",
    "US_Balance_sheet.parquet",
    "US_Cash_flow.parquet",
    "US_Earnings.parquet",
    "US_share.parquet",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_composed_snapshot(root: Path) -> tuple[Path, Path]:
    snapshot = root / "model_inputs" / "history" / "snapshot_old"
    snapshot.mkdir(parents=True)
    prices = pl.DataFrame(
        {
            "date": ["2026-07-31", "2026-07-31"],
            "open": [1.0, 2.0],
            "high": [1.1, 2.1],
            "low": [0.9, 1.9],
            "close": [1.0, 2.0],
            "volume": [100.0, 200.0],
            "adjusted_close": [1.0, 2.0],
            "ticker": ["AAA.US", "BBB.US"],
        }
    )
    prices.write_parquet(snapshot / "US_Finalprice.parquet")
    prices.select("ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume").write_parquet(
        snapshot / "SP500Price.parquet"
    )
    (snapshot / "SP500_Constituents.csv").write_text("date,ticker\n2026-07-01,AAA.US\n")
    for name in MODEL_FILES[3:]:
        pl.DataFrame({"ticker": ["AAA.US"], "value": [1.0]}).write_parquet(snapshot / name)
    lineage = prices.with_columns(
        pl.lit("eodhd_frozen_history").alias("source"),
        pl.lit("US_Finalprice").alias("dataset"),
        pl.lit("seed").alias("ingestion_run_id"),
        pl.lit("2026-08-16T00:00:00+00:00").alias("ingested_at"),
    )
    lineage_dir = snapshot / "lineage" / "prices"
    lineage_dir.mkdir(parents=True)
    lineage.write_parquet(lineage_dir / "prices_open_source_lineage.parquet")
    (lineage_dir / "manifest.json").write_text("{}")
    output_hashes = {name: _sha256(snapshot / name) for name in MODEL_FILES}
    manifest = {
        "contract_version": 1,
        "scope": "alpharank_composed_model_input",
        "composition_id": "composition-test",
        "snapshot_dir": str(snapshot),
        "output_sha256": output_hashes,
        "validation": {"passed": True},
    }
    (snapshot / "lineage" / "manifest.json").write_text(json.dumps(manifest))
    (snapshot / "snapshot_manifest.json").write_text(json.dumps(manifest))
    pointer = root / "model_inputs" / "manifests" / "latest.json"
    pointer.parent.mkdir(parents=True)
    pointer.write_text(
        json.dumps(
            {
                "composition_id": "composition-test",
                "snapshot_dir": str(snapshot),
                "manifest_path": str(snapshot / "lineage" / "manifest.json"),
                "generated_at": "2026-08-16T00:00:00+00:00",
            }
        )
    )
    return snapshot, pointer


def test_eodhd_catalog_deduplicates_bytes_and_reconstructs_every_source(tmp_path: Path) -> None:
    eodhd = tmp_path / "data" / "eodhd"
    first = eodhd / "history" / "run1" / "US_Finalprice.parquet"
    second = eodhd / "output" / "US_Finalprice.parquet"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    first.write_bytes(b"same immutable evidence")
    second.write_bytes(first.read_bytes())

    catalog_path = catalog_existing_eodhd(
        eodhd_root=eodhd,
        archive_dir=tmp_path / "warehouse" / "raw" / "eodhd",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    validation = validate_eodhd_catalog(catalog_path)
    repeated = catalog_existing_eodhd(
        eodhd_root=eodhd,
        archive_dir=tmp_path / "warehouse" / "raw" / "eodhd",
        generated_at="2026-08-19T20:00:00+00:00",
    )

    assert repeated == catalog_path
    assert validation == {"passed": True, "source_file_count": 2, "unique_object_count": 1}


def test_live008_builds_byte_identical_mart_and_preserves_source(tmp_path: Path) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "US_Finalprice.parquet").write_bytes(b"paid seed")
    source_inventory = {path.relative_to(snapshot): _sha256(path) for path in snapshot.rglob("*") if path.is_file()}

    result = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    validation = validate_warehouse_mart(
        mart_dir=result.mart_dir,
        def_manifest_path=result.def_manifest,
        source_snapshot_dir=snapshot,
    )

    assert result.model_file_count == 9
    assert validation["source_to_mart_hash_parity"] is True
    assert {name: _sha256(result.mart_dir / name) for name in MODEL_FILES} == {
        name: _sha256(snapshot / name) for name in MODEL_FILES
    }
    assert source_inventory == {
        path.relative_to(snapshot): _sha256(path) for path in snapshot.rglob("*") if path.is_file()
    }
    assert json.loads(pointer.read_text())["snapshot_dir"] == str(snapshot)


def test_mart_pointer_promotion_is_atomic_and_rollback_restores_exact_bytes(tmp_path: Path) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    result = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        generated_at="2026-08-19T20:00:00+00:00",
    )
    before = pointer.read_bytes()

    promotion = promote_mart_pointer(
        pointer_path=pointer,
        mart_dir=result.mart_dir,
        promotion_root=tmp_path / "data" / "warehouse" / "manifests" / "promotions",
        migration_id=result.migration_id,
        generated_at="2026-08-19T20:01:00+00:00",
    )
    assert json.loads(pointer.read_text())["snapshot_dir"] == str(result.mart_dir)

    rollback_mart_pointer(pointer_path=pointer, promotion_manifest_path=promotion)
    assert pointer.read_bytes() == before


def test_legacy_model_input_resolves_only_validated_mart_with_exact_parity(
    tmp_path: Path,
) -> None:
    snapshot, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    source_hashes = {name: _sha256(snapshot / name) for name in MODEL_FILES}
    migrated = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        promote=True,
        generated_at="2026-08-20T12:00:00+00:00",
    )

    resolved = resolve_mart_model_input(
        pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
    )

    assert resolved.mart_dir == migrated.mart_dir
    assert resolved.composition_id == "composition-test"
    assert resolved.model_file_sha256 == source_hashes
    assert resolved.to_manifest()["source_snapshot_dir"] == str(snapshot.resolve())


def test_mart_input_resolution_rejects_changed_model_bytes(tmp_path: Path) -> None:
    _, pointer = _seed_composed_snapshot(tmp_path / "data")
    eodhd = tmp_path / "data" / "eodhd" / "output"
    eodhd.mkdir(parents=True)
    (eodhd / "seed.json").write_text('{"source":"eodhd"}')
    migrated = migrate_validated_snapshot_to_warehouse(
        project_root=tmp_path,
        latest_pointer_path=pointer,
        warehouse_root=tmp_path / "data" / "warehouse",
        promote=True,
        generated_at="2026-08-20T12:00:00+00:00",
    )
    (migrated.mart_dir / "US_Finalprice.parquet").write_bytes(b"changed")

    with pytest.raises(RuntimeError, match="hash mismatch|bytes differ"):
        resolve_mart_model_input(
            pointer,
            warehouse_root=tmp_path / "data" / "warehouse",
        )
