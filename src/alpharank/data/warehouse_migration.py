from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import polars as pl

from alpharank.data.composed_snapshot import validate_composed_model_snapshot
from alpharank.data.open_source.raw_archive import register_immutable_raw_file
from alpharank.data.snapshot_publication import publish_mart_snapshot
from alpharank.data.snapshot_storage import copy_snapshot_file
from alpharank.data.warehouse import WarehousePaths

EODHD_CATALOG_CONTRACT = "alpharank_eodhd_raw_catalog_v1"
STG_BOOTSTRAP_CONTRACT = "alpharank_stg_price_bootstrap_v1"
DEF_COMPOSITION_CONTRACT = "alpharank_def_composition_v1"
MART_CONTRACT = "alpharank_warehouse_mart_v1"
PROMOTION_CONTRACT = "alpharank_mart_pointer_promotion_v1"


@dataclass(frozen=True)
class WarehouseMigrationResult:
    migration_id: str
    composition_id: str
    source_snapshot_dir: Path
    eodhd_catalog_manifest: Path
    stg_manifest: Path
    def_manifest: Path
    mart_dir: Path
    mart_manifest: Path
    model_file_count: int
    promotion_manifest: Path | None


def migrate_validated_snapshot_to_warehouse(
    *,
    project_root: Path,
    latest_pointer_path: Path,
    warehouse_root: Path,
    promote: bool = False,
    generated_at: str | None = None,
) -> WarehouseMigrationResult:
    """Bootstrap RAW/STG/DEF/MART without changing the validated economics."""

    project_root = project_root.resolve()
    latest_pointer_path = latest_pointer_path.resolve()
    paths = WarehousePaths(warehouse_root.resolve())
    paths.ensure()
    timestamp = generated_at or datetime.now(timezone.utc).isoformat()

    source_pointer = _read_json(latest_pointer_path)
    source_snapshot_dir = _recorded_path(
        source_pointer.get("snapshot_dir"), latest_pointer_path
    )
    source_validation = validate_composed_model_snapshot(source_snapshot_dir)
    composition_id = str(source_validation["composition_id"])
    if source_pointer.get("composition_id") != composition_id:
        raise RuntimeError("Production pointer and validated snapshot composition differ")
    source_inventory_before = _tree_inventory(source_snapshot_dir)
    source_manifest = _read_json(source_snapshot_dir / "lineage" / "manifest.json")
    output_hashes = source_manifest.get("output_sha256")
    if not isinstance(output_hashes, Mapping) or len(output_hashes) != 9:
        raise RuntimeError("LIVE-008 requires exactly nine hashed model files")

    eodhd_catalog = catalog_existing_eodhd(
        eodhd_root=project_root / "data" / "eodhd",
        archive_dir=paths.raw / "eodhd",
        generated_at=timestamp,
    )
    stg_manifest = _bootstrap_stg_prices(
        source_snapshot_dir=source_snapshot_dir,
        destination_root=paths.stg / "prices" / composition_id,
        composition_id=composition_id,
        generated_at=timestamp,
    )
    def_manifest = _bootstrap_def_composition(
        source_snapshot_dir=source_snapshot_dir,
        stg_manifest_path=stg_manifest,
        destination_root=paths.definitive / "compositions" / composition_id,
        composition_id=composition_id,
        expected_hashes=dict(output_hashes),
        generated_at=timestamp,
    )
    mart_dir, mart_manifest = _build_mart(
        source_snapshot_dir=source_snapshot_dir,
        def_manifest_path=def_manifest,
        destination_root=paths.mart,
        composition_id=composition_id,
        expected_hashes=dict(output_hashes),
        generated_at=timestamp,
    )
    validation = validate_warehouse_mart(
        mart_dir=mart_dir,
        def_manifest_path=def_manifest,
        source_snapshot_dir=source_snapshot_dir,
    )
    migration_id = f"live008_{composition_id[:12]}"
    migration_manifest = paths.manifests / "migrations" / migration_id / "manifest.json"
    _write_json_atomic(
        migration_manifest,
        {
            "contract": MART_CONTRACT,
            "migration_id": migration_id,
            "generated_at": timestamp,
            "composition_id": composition_id,
            "source_pointer_path": str(latest_pointer_path),
            "source_pointer_sha256": _sha256(latest_pointer_path),
            "source_snapshot_dir": str(source_snapshot_dir),
            "source_snapshot_inventory_sha256": _inventory_sha256(source_inventory_before),
            "eodhd_catalog_manifest": _file_record(eodhd_catalog),
            "stg_manifest": _file_record(stg_manifest),
            "def_manifest": _file_record(def_manifest),
            "mart_manifest": _file_record(mart_manifest),
            "validation": validation,
        },
    )
    _write_json_atomic(
        paths.manifests / "latest_mart.json",
        {
            "contract": MART_CONTRACT,
            "migration_id": migration_id,
            "composition_id": composition_id,
            "snapshot_dir": str(mart_dir),
            "manifest_path": str(mart_dir / "lineage" / "manifest.json"),
            "migration_manifest_path": str(migration_manifest),
            "generated_at": timestamp,
        },
    )

    source_inventory_after = _tree_inventory(source_snapshot_dir)
    if source_inventory_after != source_inventory_before:
        raise RuntimeError("LIVE-008 changed the immutable source snapshot")

    promotion_manifest = None
    if promote:
        promotion_manifest = promote_mart_pointer(
            pointer_path=latest_pointer_path,
            mart_dir=mart_dir,
            promotion_root=paths.manifests / "promotions",
            migration_id=migration_id,
            generated_at=timestamp,
        )
    return WarehouseMigrationResult(
        migration_id=migration_id,
        composition_id=composition_id,
        source_snapshot_dir=source_snapshot_dir,
        eodhd_catalog_manifest=eodhd_catalog,
        stg_manifest=stg_manifest,
        def_manifest=def_manifest,
        mart_dir=mart_dir,
        mart_manifest=mart_manifest,
        model_file_count=len(output_hashes),
        promotion_manifest=promotion_manifest,
    )


def catalog_existing_eodhd(
    *,
    eodhd_root: Path,
    archive_dir: Path,
    generated_at: str,
) -> Path:
    """Catalog every retained EODHD file by hash without downloading or rewriting it."""

    eodhd_root = eodhd_root.resolve()
    archive_dir = archive_dir.resolve()
    source_files = sorted(path for path in eodhd_root.rglob("*") if path.is_file())
    if not source_files:
        raise RuntimeError(f"No retained EODHD files found under {eodhd_root}")
    records: list[dict[str, Any]] = []
    for source_path in source_files:
        relative_path = source_path.relative_to(eodhd_root).as_posix()
        source_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:24]
        expected_sha256 = _sha256(source_path)
        manifest_path = archive_dir / "sources" / source_id / "manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if (
                manifest.get("sha256") != expected_sha256
                or manifest.get("original_path") != str(source_path)
            ):
                raise RuntimeError(f"EODHD RAW source id collision or mutation: {source_path}")
        else:
            manifest_path = register_immutable_raw_file(
                archive_dir=archive_dir,
                source_id=source_id,
                source_path=source_path,
                source="eodhd",
                dataset=source_path.name,
                observed_at=generated_at,
            )
        manifest = _read_json(manifest_path)
        object_path = Path(str(manifest["object_path"]))
        if _sha256(object_path) != expected_sha256:
            raise RuntimeError(f"EODHD RAW object hash mismatch: {object_path}")
        records.append(
            {
                "relative_path": relative_path,
                "source_id": source_id,
                "source_manifest_path": str(manifest_path),
                "object_path": str(object_path),
                "size_bytes": source_path.stat().st_size,
                "sha256": expected_sha256,
            }
        )
    catalog_payload = {
        "contract": EODHD_CATALOG_CONTRACT,
        "generated_at": generated_at,
        "eodhd_root": str(eodhd_root),
        "source_file_count": len(records),
        "unique_object_count": len({record["sha256"] for record in records}),
        "total_source_bytes": sum(record["size_bytes"] for record in records),
        "unique_object_bytes": sum(
            next(record["size_bytes"] for record in records if record["sha256"] == digest)
            for digest in {record["sha256"] for record in records}
        ),
        "files": records,
    }
    catalog_id = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    catalog_payload["catalog_id"] = catalog_id
    catalog_path = archive_dir / "catalogs" / catalog_id / "manifest.json"
    if catalog_path.exists():
        existing = _read_json(catalog_path)
        if existing.get("files") != records:
            raise RuntimeError(f"Existing EODHD catalog differs: {catalog_path}")
    else:
        _write_json_atomic(catalog_path, catalog_payload)
    validate_eodhd_catalog(catalog_path)
    _write_json_atomic(
        archive_dir / "manifests" / "latest_catalog.json",
        {
            "contract": EODHD_CATALOG_CONTRACT,
            "catalog_id": catalog_id,
            "manifest_path": str(catalog_path),
            "source_file_count": len(records),
            "unique_object_count": catalog_payload["unique_object_count"],
        },
    )
    return catalog_path


def validate_eodhd_catalog(catalog_path: Path) -> dict[str, Any]:
    payload = _read_json(catalog_path)
    if payload.get("contract") != EODHD_CATALOG_CONTRACT:
        raise RuntimeError(f"Unsupported EODHD catalog: {catalog_path}")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) != payload.get("source_file_count"):
        raise RuntimeError("EODHD catalog file count mismatch")
    for record in files:
        source_path = Path(payload["eodhd_root"]) / record["relative_path"]
        object_path = Path(record["object_path"])
        if _sha256(source_path) != record["sha256"]:
            raise RuntimeError(f"EODHD source changed after cataloguing: {source_path}")
        if _sha256(object_path) != record["sha256"]:
            raise RuntimeError(f"EODHD RAW reconstruction failed: {object_path}")
    return {
        "passed": True,
        "source_file_count": len(files),
        "unique_object_count": len({record["sha256"] for record in files}),
    }


def validate_warehouse_mart(
    *,
    mart_dir: Path,
    def_manifest_path: Path,
    source_snapshot_dir: Path,
) -> dict[str, Any]:
    mart_validation = validate_composed_model_snapshot(mart_dir)
    def_manifest = _read_json(def_manifest_path)
    expected = def_manifest.get("output_sha256")
    if not isinstance(expected, Mapping) or len(expected) != 9:
        raise RuntimeError("DEF composition does not expose nine model files")
    source_manifest = _read_json(source_snapshot_dir / "lineage" / "manifest.json")
    if dict(expected) != source_manifest.get("output_sha256"):
        raise RuntimeError("DEF composition differs from the validated source snapshot")
    observed = {name: _sha256(mart_dir / name) for name in expected}
    if observed != dict(expected):
        raise RuntimeError("MART model file hashes differ from DEF")
    return {
        **mart_validation,
        "model_file_count": len(observed),
        "def_to_mart_hash_parity": True,
        "source_to_mart_hash_parity": True,
    }


def promote_mart_pointer(
    *,
    pointer_path: Path,
    mart_dir: Path,
    promotion_root: Path,
    migration_id: str,
    generated_at: str,
) -> Path:
    """Atomically promote one validated MART and retain an exact rollback payload."""

    pointer_path = pointer_path.resolve()
    mart_dir = mart_dir.resolve()
    validate_composed_model_snapshot(mart_dir)
    before_bytes = pointer_path.read_bytes()
    before = json.loads(before_bytes)
    promotion_id = f"{generated_at.replace(':', '').replace('-', '')}_{migration_id}"
    promotion_dir = promotion_root.resolve() / promotion_id
    if promotion_dir.exists():
        raise FileExistsError(f"Promotion already exists: {promotion_dir}")
    promotion_dir.mkdir(parents=True, exist_ok=False)
    before_path = promotion_dir / "before.json"
    before_path.write_bytes(before_bytes)
    try:
        publication = publish_mart_snapshot(
            mart_dir=mart_dir,
            warehouse_root=mart_dir.parents[1],
            pointer_path=pointer_path,
            publication_root=promotion_root.resolve().parent
            / "snapshot_publications",
            publication_id=promotion_id,
            migration_id=migration_id,
            generated_at=generated_at,
        )
        promoted = _read_json(pointer_path)
        _write_json_atomic(promotion_dir / "after.json", promoted)
        validate_composed_model_snapshot(_recorded_path(promoted["snapshot_dir"], pointer_path))
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _write_bytes_atomic(pointer_path, before_bytes)
        raise
    promotion_manifest = promotion_dir / "manifest.json"
    _write_json_atomic(
        promotion_manifest,
        {
            "contract": PROMOTION_CONTRACT,
            "promotion_id": promotion_id,
            "migration_id": migration_id,
            "generated_at": generated_at,
            "pointer_path": str(pointer_path),
            "before": {
                "path": str(before_path),
                "sha256": hashlib.sha256(before_bytes).hexdigest(),
                "composition_id": before.get("composition_id"),
                "snapshot_dir": before.get("snapshot_dir"),
            },
            "after": {
                "path": str(promotion_dir / "after.json"),
                "sha256": _sha256(promotion_dir / "after.json"),
                "composition_id": promoted["composition_id"],
                "snapshot_dir": promoted["snapshot_dir"],
                "publication_id": publication.publication_id,
                "publication_manifest": _file_record(
                    publication.publication_manifest_path
                ),
            },
            "validation": {"passed": True, "atomic_replace": True},
        },
    )
    return promotion_manifest


def rollback_mart_pointer(*, pointer_path: Path, promotion_manifest_path: Path) -> None:
    """Restore the exact pre-promotion pointer after verifying the retained bytes."""

    manifest = _read_json(promotion_manifest_path)
    if manifest.get("contract") != PROMOTION_CONTRACT:
        raise RuntimeError("Unsupported MART promotion manifest")
    before = manifest.get("before", {})
    before_path = Path(str(before.get("path")))
    before_bytes = before_path.read_bytes()
    if hashlib.sha256(before_bytes).hexdigest() != before.get("sha256"):
        raise RuntimeError("Rollback pointer payload hash mismatch")
    _write_bytes_atomic(pointer_path.resolve(), before_bytes)


def _bootstrap_stg_prices(
    *,
    source_snapshot_dir: Path,
    destination_root: Path,
    composition_id: str,
    generated_at: str,
) -> Path:
    source = source_snapshot_dir / "lineage" / "prices" / "prices_open_source_lineage.parquet"
    destination = destination_root / "prices_open_source_lineage.parquet"
    manifest_path = destination_root / "manifest.json"
    if not manifest_path.exists():
        destination_root.mkdir(parents=True, exist_ok=False)
        copy_snapshot_file(source, destination)
        counts = _price_key_counts(destination)
        _write_json_atomic(
            manifest_path,
            {
                "contract": STG_BOOTSTRAP_CONTRACT,
                "generated_at": generated_at,
                "composition_id": composition_id,
                "source_path": str(source),
                "source_sha256": _sha256(source),
                "output_path": str(destination),
                "output_sha256": _sha256(destination),
                **counts,
                "normalization": "validated lineage bootstrap; no value selection or fill",
                "validation": {"passed": True, "unique_ticker_date": True},
            },
        )
    payload = _read_json(manifest_path)
    if _sha256(destination) != payload.get("output_sha256"):
        raise RuntimeError("STG price bootstrap hash mismatch")
    _price_key_counts(destination)
    return manifest_path


def _bootstrap_def_composition(
    *,
    source_snapshot_dir: Path,
    stg_manifest_path: Path,
    destination_root: Path,
    composition_id: str,
    expected_hashes: dict[str, str],
    generated_at: str,
) -> Path:
    manifest_path = destination_root / "manifest.json"
    if not manifest_path.exists():
        staging = destination_root.parent / f".{destination_root.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        try:
            storage_modes = []
            for name in sorted(expected_hashes):
                storage_modes.append(copy_snapshot_file(source_snapshot_dir / name, staging / name))
            stg = _read_json(stg_manifest_path)
            lineage_source = Path(stg["output_path"])
            storage_modes.append(
                copy_snapshot_file(
                    lineage_source,
                    staging / "lineage" / "prices_open_source_lineage.parquet",
                )
            )
            observed = {name: _sha256(staging / name) for name in expected_hashes}
            if observed != expected_hashes:
                raise RuntimeError("DEF bootstrap changed model-file bytes")
            counts = _price_key_counts(staging / "lineage" / "prices_open_source_lineage.parquet")
            payload = {
                "contract": DEF_COMPOSITION_CONTRACT,
                "generated_at": generated_at,
                "composition_id": composition_id,
                "source_snapshot_dir": str(source_snapshot_dir),
                "stg_manifest": _file_record(stg_manifest_path),
                "output_sha256": observed,
                "price_lineage": {
                    "path": str(destination_root / "lineage" / "prices_open_source_lineage.parquet"),
                    "sha256": _sha256(staging / "lineage" / "prices_open_source_lineage.parquet"),
                    **counts,
                },
                "resolution": "bootstrap from the last validated composition; exact key provenance retained",
                "storage_mode_counts": {
                    mode: storage_modes.count(mode) for mode in sorted(set(storage_modes))
                },
                "validation": {
                    "passed": True,
                    "unique_ticker_date": True,
                    "source_model_hash_parity": True,
                },
            }
            _write_json_atomic(staging / "manifest.json", payload)
            os.replace(staging, destination_root)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            shutil.rmtree(staging, ignore_errors=True)
            raise
    payload = _read_json(manifest_path)
    observed = {name: _sha256(destination_root / name) for name in expected_hashes}
    if payload.get("output_sha256") != observed or observed != expected_hashes:
        raise RuntimeError("Existing DEF composition failed hash validation")
    return manifest_path


def _build_mart(
    *,
    source_snapshot_dir: Path,
    def_manifest_path: Path,
    destination_root: Path,
    composition_id: str,
    expected_hashes: dict[str, str],
    generated_at: str,
) -> tuple[Path, Path]:
    mart_dir = destination_root / f"alpharank_input_{composition_id}"
    warehouse_manifest_path = destination_root / "manifests" / composition_id / "manifest.json"
    if not mart_dir.exists():
        staging = destination_root / f".{mart_dir.name}.staging"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        def_dir = def_manifest_path.parent
        try:
            for source in sorted(path for path in source_snapshot_dir.rglob("*") if path.is_file()):
                relative = source.relative_to(source_snapshot_dir)
                selected_source = def_dir / relative.name if relative.as_posix() in expected_hashes else source
                copy_snapshot_file(selected_source, staging / relative)
            composed_manifest_path = staging / "lineage" / "manifest.json"
            composed = _read_json(composed_manifest_path)
            composed["snapshot_dir"] = str(mart_dir)
            composed["warehouse"] = {
                "contract": MART_CONTRACT,
                "migration_id": f"live008_{composition_id[:12]}",
                "generated_at": generated_at,
                "source_snapshot_dir": str(source_snapshot_dir),
                "def_manifest_path": str(def_manifest_path),
                "def_manifest_sha256": _sha256(def_manifest_path),
                "economic_change": False,
            }
            _write_json_atomic(composed_manifest_path, composed)
            _write_json_atomic(staging / "snapshot_manifest.json", composed)
            observed = {name: _sha256(staging / name) for name in expected_hashes}
            if observed != expected_hashes:
                raise RuntimeError("MART bootstrap changed model-file bytes")
            os.replace(staging, mart_dir)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError):
            shutil.rmtree(staging, ignore_errors=True)
            raise
    mart_validation = validate_composed_model_snapshot(mart_dir)
    if mart_validation["composition_id"] != composition_id:
        raise RuntimeError("Existing MART composition id mismatch")
    _write_json_atomic(
        warehouse_manifest_path,
        {
            "contract": MART_CONTRACT,
            "generated_at": generated_at,
            "composition_id": composition_id,
            "snapshot_dir": str(mart_dir),
            "composed_manifest_path": str(mart_dir / "lineage" / "manifest.json"),
            "def_manifest": _file_record(def_manifest_path),
            "output_sha256": expected_hashes,
            "validation": {
                "passed": True,
                "model_file_count": len(expected_hashes),
                "def_to_mart_hash_parity": True,
                "source_to_mart_hash_parity": True,
            },
        },
    )
    return mart_dir, warehouse_manifest_path


def _price_key_counts(path: Path) -> dict[str, int]:
    counts = (
        pl.scan_parquet(path)
        .select(
            pl.len().alias("row_count"),
            pl.struct(["ticker", "date"]).n_unique().alias("unique_key_count"),
        )
        .collect()
        .row(0, named=True)
    )
    if counts["row_count"] != counts["unique_key_count"]:
        raise RuntimeError(f"Duplicate ticker/date keys in {path}")
    return {key: int(value) for key, value in counts.items()}


def _tree_inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha256(path)
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    }


def _inventory_sha256(inventory: Mapping[str, str]) -> str:
    return hashlib.sha256(
        json.dumps(dict(inventory), sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _recorded_path(value: Any, pointer_path: Path) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = pointer_path.parent / path
    return path.resolve()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected a JSON object: {path}")
    return payload


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, sort_keys=False) + "\n").encode("utf-8"),
    )


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)
