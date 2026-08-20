"""Immutable publication manifests that reference validated MART contents."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from alpharank.data.mart import validate_mart_model_input

SNAPSHOT_PUBLICATION_CONTRACT = "alpharank_mart_snapshot_publication_v1"
SNAPSHOT_POINTER_CONTRACT = "alpharank_mart_snapshot_pointer_v1"


@dataclass(frozen=True)
class SnapshotPublicationResult:
    publication_id: str
    publication_manifest_path: Path
    pointer_path: Path
    composition_id: str
    mart_dir: Path
    file_count: int
    inventory_sha256: str


def publish_mart_snapshot(
    *,
    mart_dir: Path,
    warehouse_root: Path,
    pointer_path: Path,
    publication_root: Path,
    publication_id: str,
    migration_id: str,
    generated_at: str,
) -> SnapshotPublicationResult:
    """Publish a MART by reference, with a complete immutable file inventory."""

    if re.fullmatch(r"[A-Za-z0-9_.+-]+", publication_id) is None:
        raise ValueError(f"Unsafe snapshot publication id: {publication_id!r}")
    validated = validate_mart_model_input(
        mart_dir,
        warehouse_root=warehouse_root,
    )
    inventory = _tree_inventory(validated.mart_dir)
    inventory_sha256 = _inventory_sha256(inventory)
    publication_dir = publication_root.resolve() / publication_id
    if publication_dir.exists():
        raise FileExistsError(f"Snapshot publication already exists: {publication_dir}")
    publication_dir.mkdir(parents=True, exist_ok=False)
    publication_manifest_path = publication_dir / "manifest.json"
    publication = {
        "contract": SNAPSHOT_PUBLICATION_CONTRACT,
        "publication_id": publication_id,
        "migration_id": migration_id,
        "generated_at": generated_at,
        "composition_id": validated.composition_id,
        "snapshot_dir": str(validated.mart_dir),
        "storage": "reference_immutable_mart_without_payload_copy",
        "composed_manifest_path": str(validated.composed_manifest_path),
        "warehouse_manifest_path": str(validated.warehouse_manifest_path),
        "source_snapshot_dir": str(validated.source_snapshot_dir),
        "model_file_sha256": validated.model_file_sha256,
        "file_count": len(inventory),
        "inventory_sha256": inventory_sha256,
        "files": inventory,
        "validation": {
            "passed": True,
            "complete_tree_inventory": True,
            "def_to_mart_hash_parity": True,
            "source_to_mart_hash_parity": True,
            "payload_copy_count": 0,
        },
    }
    _write_json_atomic(publication_manifest_path, publication)
    pointer_path = pointer_path.resolve()
    pointer = {
        "contract": SNAPSHOT_POINTER_CONTRACT,
        "publication_id": publication_id,
        "composition_id": validated.composition_id,
        "snapshot_dir": str(validated.mart_dir),
        "manifest_path": str(validated.composed_manifest_path),
        "publication_manifest_path": str(publication_manifest_path),
        "publication_manifest_sha256": _sha256(publication_manifest_path),
        "snapshot_inventory_sha256": inventory_sha256,
        "generated_at": generated_at,
        "warehouse_migration_id": migration_id,
    }
    _write_json_atomic(pointer_path, pointer)
    validate_snapshot_publication(pointer_path)
    return SnapshotPublicationResult(
        publication_id=publication_id,
        publication_manifest_path=publication_manifest_path,
        pointer_path=pointer_path,
        composition_id=validated.composition_id,
        mart_dir=validated.mart_dir,
        file_count=len(inventory),
        inventory_sha256=inventory_sha256,
    )


def validate_snapshot_publication(pointer_path: Path) -> dict[str, object]:
    """Revalidate the publication manifest, complete MART tree and pointer."""

    pointer_path = pointer_path.resolve()
    pointer = _read_json(pointer_path)
    if pointer.get("contract") != SNAPSHOT_POINTER_CONTRACT:
        raise RuntimeError("Unsupported snapshot pointer contract")
    publication_manifest_path = Path(
        _non_empty_string(
            pointer.get("publication_manifest_path"),
            "publication_manifest_path",
        )
    ).resolve()
    if _sha256(publication_manifest_path) != pointer.get(
        "publication_manifest_sha256"
    ):
        raise RuntimeError("Snapshot publication manifest hash mismatch")
    publication = _read_json(publication_manifest_path)
    if publication.get("contract") != SNAPSHOT_PUBLICATION_CONTRACT:
        raise RuntimeError("Unsupported snapshot publication contract")
    for key in ("publication_id", "composition_id", "snapshot_dir"):
        if publication.get(key) != pointer.get(key):
            raise RuntimeError(f"Snapshot pointer differs from publication: {key}")
    mart_dir = Path(_non_empty_string(publication.get("snapshot_dir"), "snapshot_dir"))
    warehouse_manifest_path = Path(
        _non_empty_string(
            publication.get("warehouse_manifest_path"),
            "warehouse_manifest_path",
        )
    ).resolve()
    warehouse_root = warehouse_manifest_path.parents[3]
    validated = validate_mart_model_input(mart_dir, warehouse_root=warehouse_root)
    if validated.composition_id != publication.get("composition_id"):
        raise RuntimeError("Snapshot publication MART identity mismatch")
    inventory = _tree_inventory(validated.mart_dir)
    inventory_sha256 = _inventory_sha256(inventory)
    if inventory != publication.get("files"):
        raise RuntimeError("Snapshot publication file inventory differs")
    if inventory_sha256 != publication.get("inventory_sha256"):
        raise RuntimeError("Snapshot publication inventory hash mismatch")
    if inventory_sha256 != pointer.get("snapshot_inventory_sha256"):
        raise RuntimeError("Snapshot pointer inventory hash mismatch")
    if len(inventory) != publication.get("file_count"):
        raise RuntimeError("Snapshot publication file count mismatch")
    validation = publication.get("validation")
    if not isinstance(validation, dict) or validation.get("payload_copy_count") != 0:
        raise RuntimeError("Snapshot publication duplicated MART payloads")
    return {
        "passed": True,
        "publication_id": publication["publication_id"],
        "composition_id": publication["composition_id"],
        "snapshot_dir": str(validated.mart_dir),
        "file_count": len(inventory),
        "inventory_sha256": inventory_sha256,
        "payload_copy_count": 0,
    }


def _tree_inventory(root: Path) -> list[dict[str, object]]:
    return [
        {
            "path": path.relative_to(root).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(item for item in root.rglob("*") if item.is_file())
    ]


def _inventory_sha256(inventory: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _non_empty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"Missing {label}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
