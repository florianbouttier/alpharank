from __future__ import annotations

from pathlib import Path

import pytest

from alpharank.data.warehouse.historical_migration import (
    build_historical_root_catalog,
    validate_historical_root_catalog,
    write_historical_root_catalog,
)


def test_historical_roots_are_referenced_by_hash_without_copy(tmp_path: Path) -> None:
    archive = tmp_path / "data" / "archive"
    archive.mkdir(parents=True)
    (archive / "first.parquet").write_bytes(b"same bytes")
    (archive / "second.parquet").write_bytes(b"same bytes")
    legacy = tmp_path / "data" / "US_Finalprice.parquet"
    legacy.write_bytes(b"other bytes")
    catalog = build_historical_root_catalog(
        {"archive": archive, "legacy_price": legacy},
        generated_at="2026-08-20",
    )
    catalog_path = (
        tmp_path / "warehouse" / "manifests" / "historical" / "manifest.json"
    )
    write_historical_root_catalog(catalog_path, catalog)

    validation = validate_historical_root_catalog(catalog_path)

    assert validation["file_count"] == 3
    assert validation["unique_object_count"] == 2
    assert validation["duplicate_file_count"] == 1
    assert validation["copy_count"] == 0
    assert validation["download_count"] == 0
    assert not (tmp_path / "warehouse" / "objects").exists()
    assert {record["source_path"] for record in catalog["files"]} == {
        str(legacy.resolve()),
        str((archive / "first.parquet").resolve()),
        str((archive / "second.parquet").resolve()),
    }


def test_historical_catalog_detects_source_mutation(tmp_path: Path) -> None:
    source = tmp_path / "data" / "legacy.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"before")
    catalog = build_historical_root_catalog(
        {"legacy": source},
        generated_at="2026-08-20",
    )
    catalog_path = tmp_path / "warehouse" / "manifest.json"
    write_historical_root_catalog(catalog_path, catalog)
    source.write_bytes(b"after")

    with pytest.raises(RuntimeError, match="source size changed|source hash changed"):
        validate_historical_root_catalog(catalog_path)
