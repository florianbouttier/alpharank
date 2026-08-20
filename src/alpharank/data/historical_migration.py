"""Reference-first catalogues for historical data-root migrations."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

HISTORICAL_MIGRATION_CONTRACT = "alpharank_historical_root_catalog_v1"
HISTORICAL_MIGRATION_SUMMARY_CONTRACT = (
    "alpharank_historical_root_catalog_summary_v1"
)


def build_historical_root_catalog(
    source_roots: Mapping[str, Path],
    *,
    generated_at: str,
) -> dict[str, object]:
    """Hash every source file without downloading, copying or moving it."""

    resolved_roots = _validate_roots(source_roots)
    records: list[dict[str, object]] = []
    root_summaries: list[dict[str, object]] = []
    for root_id, root_path in sorted(resolved_roots.items()):
        source_files = (
            [root_path]
            if root_path.is_file()
            else sorted(path for path in root_path.rglob("*") if path.is_file())
        )
        root_records: list[dict[str, object]] = []
        for source_path in source_files:
            before = source_path.stat()
            digest = _sha256(source_path)
            after = source_path.stat()
            if (before.st_size, before.st_mtime_ns) != (
                after.st_size,
                after.st_mtime_ns,
            ):
                raise RuntimeError(
                    f"Historical source changed while hashing: {source_path}"
                )
            relative_path = (
                source_path.name
                if root_path.is_file()
                else source_path.relative_to(root_path).as_posix()
            )
            record = {
                "root_id": root_id,
                "relative_path": relative_path,
                "source_path": str(source_path),
                "size_bytes": before.st_size,
                "sha256": digest,
                "migration_storage": "reference_existing_bytes",
            }
            records.append(record)
            root_records.append(record)
        root_summaries.append(
            {
                "root_id": root_id,
                "source_path": str(root_path),
                "source_kind": "file" if root_path.is_file() else "directory",
                "file_count": len(root_records),
                "size_bytes": sum(int(record["size_bytes"]) for record in root_records),
                "inventory_sha256": _inventory_sha256(root_records),
            }
        )

    digest_counts = Counter(str(record["sha256"]) for record in records)
    unique_bytes = {
        str(record["sha256"]): int(record["size_bytes"])
        for record in records
    }
    total_bytes = sum(int(record["size_bytes"]) for record in records)
    catalog_id = _inventory_sha256(records)
    return {
        "contract": HISTORICAL_MIGRATION_CONTRACT,
        "catalog_id": catalog_id,
        "generated_at": generated_at,
        "migration_mode": "reference_and_hash_before_any_copy",
        "root_count": len(root_summaries),
        "file_count": len(records),
        "total_source_bytes": total_bytes,
        "unique_object_count": len(digest_counts),
        "unique_object_bytes": sum(unique_bytes.values()),
        "duplicate_file_count": sum(count - 1 for count in digest_counts.values()),
        "duplicate_bytes": total_bytes - sum(unique_bytes.values()),
        "download_count": 0,
        "copy_count": 0,
        "roots": root_summaries,
        "files": records,
        "validation": {
            "passed": True,
            "all_source_files_referenced": True,
            "exact_content_hash_before_copy": True,
            "source_mutation_detected": False,
            "physical_migration_started": False,
        },
    }


def write_historical_root_catalog(path: Path, catalog: Mapping[str, object]) -> None:
    """Write one immutable catalogue, or accept an identical existing one."""

    path = path.resolve()
    payload = dict(catalog)
    if path.exists():
        if _read_json(path) != payload:
            raise FileExistsError(f"Historical migration catalog already differs: {path}")
        return
    _write_json_atomic(path, payload)


def validate_historical_root_catalog(path: Path) -> dict[str, object]:
    """Rehash every referenced source and prove that no copy was declared."""

    path = path.resolve()
    catalog = _read_json(path)
    if catalog.get("contract") != HISTORICAL_MIGRATION_CONTRACT:
        raise RuntimeError("Unsupported historical migration catalog")
    records = catalog.get("files")
    if not isinstance(records, list) or len(records) != catalog.get("file_count"):
        raise RuntimeError("Historical migration file count mismatch")
    for raw_record in records:
        if not isinstance(raw_record, dict):
            raise RuntimeError("Historical migration record must be an object")
        source_path = Path(str(raw_record.get("source_path")))
        if not source_path.is_file():
            raise RuntimeError(f"Historical source is missing: {source_path}")
        if source_path.stat().st_size != raw_record.get("size_bytes"):
            raise RuntimeError(f"Historical source size changed: {source_path}")
        if _sha256(source_path) != raw_record.get("sha256"):
            raise RuntimeError(f"Historical source hash changed: {source_path}")
        if raw_record.get("migration_storage") != "reference_existing_bytes":
            raise RuntimeError("Historical migration copied data before validation")
    if _inventory_sha256(records) != catalog.get("catalog_id"):
        raise RuntimeError("Historical migration catalog id mismatch")
    if catalog.get("download_count") != 0 or catalog.get("copy_count") != 0:
        raise RuntimeError("Historical migration performed forbidden acquisition or copy")
    return {
        "passed": True,
        "catalog_id": catalog["catalog_id"],
        "root_count": catalog["root_count"],
        "file_count": len(records),
        "total_source_bytes": catalog["total_source_bytes"],
        "unique_object_count": catalog["unique_object_count"],
        "duplicate_file_count": catalog["duplicate_file_count"],
        "copy_count": 0,
        "download_count": 0,
    }


def build_historical_catalog_summary(
    catalog_path: Path,
) -> dict[str, object]:
    """Return the tracked review summary for a full ignored data catalogue."""

    catalog_path = catalog_path.resolve()
    catalog = _read_json(catalog_path)
    return {
        "contract": HISTORICAL_MIGRATION_SUMMARY_CONTRACT,
        "generated_at": catalog["generated_at"],
        "catalog_id": catalog["catalog_id"],
        "catalog_manifest_path": str(catalog_path),
        "catalog_manifest_sha256": _sha256(catalog_path),
        "migration_mode": catalog["migration_mode"],
        "root_count": catalog["root_count"],
        "file_count": catalog["file_count"],
        "total_source_bytes": catalog["total_source_bytes"],
        "unique_object_count": catalog["unique_object_count"],
        "unique_object_bytes": catalog["unique_object_bytes"],
        "duplicate_file_count": catalog["duplicate_file_count"],
        "duplicate_bytes": catalog["duplicate_bytes"],
        "download_count": catalog["download_count"],
        "copy_count": catalog["copy_count"],
        "roots": catalog["roots"],
        "validation": catalog["validation"],
    }


def write_historical_catalog_summary(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    _write_json_atomic(path.resolve(), dict(summary))


def _validate_roots(source_roots: Mapping[str, Path]) -> dict[str, Path]:
    if not source_roots:
        raise ValueError("Historical migration requires source roots")
    resolved: dict[str, Path] = {}
    for root_id, path in source_roots.items():
        if not root_id or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in root_id):
            raise ValueError(f"Invalid historical root id: {root_id!r}")
        resolved_path = path.resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(resolved_path)
        resolved[root_id] = resolved_path
    items = list(resolved.items())
    for index, (root_id, root_path) in enumerate(items):
        for other_id, other_path in items[index + 1 :]:
            if root_path == other_path:
                raise ValueError(f"Historical roots overlap: {root_id}, {other_id}")
            if root_path.is_dir() and other_path.is_relative_to(root_path):
                raise ValueError(f"Historical roots overlap: {root_id}, {other_id}")
            if other_path.is_dir() and root_path.is_relative_to(other_path):
                raise ValueError(f"Historical roots overlap: {root_id}, {other_id}")
    return resolved


def _inventory_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
