"""Resolution and validation of canonical model-ready MART inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from alpharank.data.composed_snapshot import validate_composed_model_snapshot

MART_CONTRACT_ID = "alpharank_warehouse_mart_v1"


@dataclass(frozen=True)
class MartInputResolution:
    composition_id: str
    mart_dir: Path
    composed_manifest_path: Path
    warehouse_manifest_path: Path
    source_pointer_path: Path
    source_pointer_sha256: str
    source_snapshot_dir: Path
    model_file_sha256: dict[str, str]

    def to_manifest(self) -> dict[str, object]:
        return {
            "contract": MART_CONTRACT_ID,
            "composition_id": self.composition_id,
            "mart_dir": str(self.mart_dir),
            "composed_manifest_path": str(self.composed_manifest_path),
            "warehouse_manifest_path": str(self.warehouse_manifest_path),
            "source_pointer_path": str(self.source_pointer_path),
            "source_pointer_sha256": self.source_pointer_sha256,
            "source_snapshot_dir": str(self.source_snapshot_dir),
            "model_file_sha256": self.model_file_sha256,
        }


@dataclass(frozen=True)
class ValidatedMart:
    composition_id: str
    mart_dir: Path
    composed_manifest_path: Path
    warehouse_manifest_path: Path
    source_snapshot_dir: Path
    model_file_sha256: dict[str, str]


def resolve_mart_model_input(
    pointer_path: Path,
    *,
    warehouse_root: Path | None = None,
) -> MartInputResolution:
    """Resolve one immutable MART and require exact DEF/source hash parity."""

    pointer_path = pointer_path.resolve()
    pointer = _read_json(pointer_path)
    composition_id = _non_empty_string(pointer.get("composition_id"), "composition_id")
    mart_dir = _recorded_path(pointer.get("snapshot_dir"), pointer_path)
    data_root = pointer_path.parents[2]
    resolved_warehouse_root = (
        warehouse_root.resolve() if warehouse_root is not None else data_root / "warehouse"
    )
    canonical_mart_root = (resolved_warehouse_root / "mart").resolve()
    if not mart_dir.is_relative_to(canonical_mart_root):
        raise RuntimeError(f"Model input is outside canonical MART: {mart_dir}")

    validated = validate_mart_model_input(
        mart_dir,
        warehouse_root=resolved_warehouse_root,
    )
    if validated.composition_id != composition_id:
        raise RuntimeError("MART pointer and composed manifest identities differ")

    return MartInputResolution(
        composition_id=composition_id,
        mart_dir=mart_dir,
        composed_manifest_path=validated.composed_manifest_path,
        warehouse_manifest_path=validated.warehouse_manifest_path,
        source_pointer_path=pointer_path,
        source_pointer_sha256=_sha256(pointer_path),
        source_snapshot_dir=validated.source_snapshot_dir,
        model_file_sha256=validated.model_file_sha256,
    )


def validate_mart_model_input(
    mart_dir: Path,
    *,
    warehouse_root: Path,
) -> ValidatedMart:
    """Validate one MART without relying on a mutable publication pointer."""

    mart_dir = mart_dir.resolve()
    warehouse_root = warehouse_root.resolve()
    canonical_mart_root = (warehouse_root / "mart").resolve()
    if not mart_dir.is_relative_to(canonical_mart_root):
        raise RuntimeError(f"Model input is outside canonical MART: {mart_dir}")
    validation = validate_composed_model_snapshot(mart_dir)
    composition_id = _non_empty_string(
        validation.get("composition_id"), "composition_id"
    )
    composed_manifest_path = mart_dir / "lineage" / "manifest.json"
    composed = _read_json(composed_manifest_path)
    warehouse_lineage = composed.get("warehouse")
    if not isinstance(warehouse_lineage, Mapping):
        raise RuntimeError("MART composed manifest has no warehouse lineage")
    if warehouse_lineage.get("contract") != MART_CONTRACT_ID:
        raise RuntimeError("Unsupported MART lineage contract")
    source_snapshot_dir = Path(
        _non_empty_string(
            warehouse_lineage.get("source_snapshot_dir"),
            "warehouse.source_snapshot_dir",
        )
    ).resolve()

    warehouse_manifest_path = (
        canonical_mart_root / "manifests" / composition_id / "manifest.json"
    )
    warehouse_manifest = _read_json(warehouse_manifest_path)
    if warehouse_manifest.get("contract") != MART_CONTRACT_ID:
        raise RuntimeError("Unsupported MART model-input contract")
    if warehouse_manifest.get("composition_id") != composition_id:
        raise RuntimeError("MART warehouse manifest identity mismatch")
    if _recorded_path(
        warehouse_manifest.get("snapshot_dir"), warehouse_manifest_path
    ) != mart_dir:
        raise RuntimeError("MART warehouse manifest points to another directory")
    parity = warehouse_manifest.get("validation")
    if not isinstance(parity, Mapping) or not all(
        parity.get(flag) is True
        for flag in (
            "passed",
            "def_to_mart_hash_parity",
            "source_to_mart_hash_parity",
        )
    ):
        raise RuntimeError("MART has no validated DEF/source hash parity")
    raw_hashes = warehouse_manifest.get("output_sha256")
    if not isinstance(raw_hashes, Mapping) or not raw_hashes:
        raise RuntimeError("MART manifest has no model-file hashes")
    expected_hashes = {str(name): str(digest) for name, digest in raw_hashes.items()}
    observed_hashes = {name: _sha256(mart_dir / name) for name in expected_hashes}
    if observed_hashes != expected_hashes:
        raise RuntimeError("MART model-file bytes differ from its validated manifest")
    return ValidatedMart(
        composition_id=composition_id,
        mart_dir=mart_dir,
        composed_manifest_path=composed_manifest_path,
        warehouse_manifest_path=warehouse_manifest_path,
        source_snapshot_dir=source_snapshot_dir,
        model_file_sha256=observed_hashes,
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return payload


def _recorded_path(value: object, pointer_path: Path) -> Path:
    path = Path(_non_empty_string(value, "recorded path"))
    if not path.is_absolute():
        path = pointer_path.parent / path
    return path.resolve()


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
