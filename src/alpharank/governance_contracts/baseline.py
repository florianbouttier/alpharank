"""Immutable methodology baseline sealing and validation contract."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from alpharank.data.snapshot_storage import copy_snapshot_file
from alpharank.governance_contracts.common import (
    files_under as _files_under,
)
from alpharank.governance_contracts.common import (
    sha256_path as _sha256_path,
)
from alpharank.governance_contracts.contracts import (
    BASELINE_CONTRACT_VERSION,
    BASELINE_MANIFEST_NAME,
    BASELINE_SEAL_NAME,
    BaselineValidationError,
)


@dataclass(frozen=True)
class _InventoryEntry:
    relative_path: str
    size_bytes: int
    sha256: str
    storage_mode: str


def seal_baseline_package(
    *,
    package_dir: Path,
    baseline_id: str,
    sources: Mapping[str, Path],
    approved_by: str,
    implementation_commit: str,
    methodology_status: str = "audited_biased_not_causal",
    source_snapshot_id: str | None = None,
    known_limitations: tuple[str, ...] = (),
    sealed_at: datetime | None = None,
) -> dict[str, Any]:
    """Copy audited artifacts into a new write-once baseline package.

    The destination must not exist. Every source file is copied to an
    independent path, preferring APFS copy-on-write clones, and every payload
    file is inventoried. The completed directory is atomically renamed and all
    write bits are removed only after the manifest and its detached seal exist.
    """

    destination = package_dir.resolve()
    if destination.exists():
        raise FileExistsError(
            f"Baseline package already exists and cannot be overwritten: {destination}"
        )
    identifier = str(baseline_id).strip()
    if not identifier:
        raise ValueError("baseline_id must be non-empty.")
    if not sources:
        raise ValueError("At least one baseline source is required.")
    normalized_sources = _validate_sources(sources)
    timestamp = sealed_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        raise ValueError("sealed_at must include an explicit timezone.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.tmp-{uuid4().hex}"
    temporary.mkdir(parents=False, exist_ok=False)
    inventory: list[_InventoryEntry] = []
    try:
        payload_dir = temporary / "payload"
        payload_dir.mkdir()
        for label, source in normalized_sources.items():
            target_root = payload_dir / label
            if source.is_dir():
                target_root.mkdir()
                for source_file in _files_under(source):
                    relative = source_file.relative_to(source)
                    target = target_root / relative
                    storage_mode = copy_snapshot_file(source_file, target)
                    inventory.append(
                        _inventory_entry(
                            target,
                            package_root=temporary,
                            storage_mode=storage_mode,
                        )
                    )
            else:
                target_root.mkdir()
                target = target_root / source.name
                storage_mode = copy_snapshot_file(source, target)
                inventory.append(
                    _inventory_entry(
                        target,
                        package_root=temporary,
                        storage_mode=storage_mode,
                    )
                )

        inventory.sort(key=lambda entry: entry.relative_path)
        root_sha256 = _inventory_sha256(inventory)
        manifest = {
            "baseline_contract_version": BASELINE_CONTRACT_VERSION,
            "baseline_id": identifier,
            "methodology_status": methodology_status,
            "causal_validation": False,
            "sealed_at_utc": timestamp.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "approved_by": str(approved_by),
            "implementation_commit": str(implementation_commit),
            "source_snapshot_id": source_snapshot_id,
            "known_limitations": list(known_limitations),
            "storage_contract": {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "semantics": "independent byte-identical immutable payload paths",
                "storage_mode_counts": _storage_mode_counts(inventory),
            },
            "source_roots": {label: str(path) for label, path in normalized_sources.items()},
            "payload_file_count": len(inventory),
            "payload_size_bytes": sum(entry.size_bytes for entry in inventory),
            "payload_inventory_sha256": root_sha256,
            "inventory": [
                {
                    "relative_path": entry.relative_path,
                    "size_bytes": entry.size_bytes,
                    "sha256": entry.sha256,
                    "storage_mode": entry.storage_mode,
                }
                for entry in inventory
            ],
        }
        manifest_path = temporary / BASELINE_MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest_sha256 = _sha256_path(manifest_path)
        (temporary / BASELINE_SEAL_NAME).write_text(
            manifest_sha256 + "  " + BASELINE_MANIFEST_NAME + "\n",
            encoding="utf-8",
        )
        _remove_write_bits(temporary)
        temporary.rename(destination)
    except (KeyError, OSError, RuntimeError, TypeError, ValueError):
        _make_tree_owner_writable(temporary)
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    validate_baseline_package(destination)
    return manifest


def validate_baseline_package(package_dir: Path) -> dict[str, Any]:
    """Fail closed if a sealed baseline differs from its detached inventory."""

    root = package_dir.resolve()
    manifest_path = root / BASELINE_MANIFEST_NAME
    seal_path = root / BASELINE_SEAL_NAME
    errors: list[str] = []
    if not root.is_dir():
        raise BaselineValidationError(f"Baseline package does not exist: {root}")
    if not manifest_path.is_file():
        errors.append(f"missing {BASELINE_MANIFEST_NAME}")
    if not seal_path.is_file():
        errors.append(f"missing {BASELINE_SEAL_NAME}")
    if errors:
        raise BaselineValidationError("; ".join(errors))

    expected_manifest_sha = seal_path.read_text(encoding="utf-8").split()[0]
    actual_manifest_sha = _sha256_path(manifest_path)
    if expected_manifest_sha != actual_manifest_sha:
        errors.append("baseline manifest SHA-256 mismatch")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise BaselineValidationError("baseline manifest is not valid JSON") from exc

    if manifest.get("baseline_contract_version") != BASELINE_CONTRACT_VERSION:
        errors.append("unsupported baseline contract version")
    if manifest.get("causal_validation") is not False:
        errors.append("audited biased baseline must not claim causal validation")
    inventory_rows = manifest.get("inventory")
    if not isinstance(inventory_rows, list) or not inventory_rows:
        errors.append("baseline payload inventory is missing or empty")
        inventory_rows = []

    expected_by_path: dict[str, dict[str, Any]] = {}
    for row in inventory_rows:
        if not isinstance(row, dict) or not row.get("relative_path"):
            errors.append("invalid baseline inventory row")
            continue
        relative_path = str(row["relative_path"])
        if relative_path in expected_by_path:
            errors.append(f"duplicate baseline inventory path: {relative_path}")
        expected_by_path[relative_path] = row

    payload_dir = root / "payload"
    actual_paths = {path.relative_to(root).as_posix() for path in _files_under(payload_dir)}
    all_package_files = {path.relative_to(root).as_posix() for path in _files_under(root)}
    allowed_package_files = actual_paths | {
        BASELINE_MANIFEST_NAME,
        BASELINE_SEAL_NAME,
    }
    for unexpected in sorted(all_package_files - allowed_package_files):
        errors.append(f"unexpected sealed package file: {unexpected}")
    expected_paths = set(expected_by_path)
    for missing in sorted(expected_paths - actual_paths):
        errors.append(f"missing sealed payload file: {missing}")
    for unexpected in sorted(actual_paths - expected_paths):
        errors.append(f"unexpected sealed payload file: {unexpected}")

    actual_entries: list[_InventoryEntry] = []
    for relative_path in sorted(expected_paths & actual_paths):
        path = root / relative_path
        row = expected_by_path[relative_path]
        actual_sha = _sha256_path(path)
        actual_size = path.stat().st_size
        if actual_sha != row.get("sha256"):
            errors.append(f"sealed payload SHA-256 mismatch: {relative_path}")
        if actual_size != row.get("size_bytes"):
            errors.append(f"sealed payload size mismatch: {relative_path}")
        actual_entries.append(
            _InventoryEntry(
                relative_path=relative_path,
                size_bytes=actual_size,
                sha256=actual_sha,
                storage_mode=str(row.get("storage_mode", "unknown")),
            )
        )
    if len(inventory_rows) != manifest.get("payload_file_count"):
        errors.append("payload_file_count does not match inventory")
    if sum(entry.size_bytes for entry in actual_entries) != manifest.get("payload_size_bytes"):
        errors.append("payload_size_bytes does not match inventory")
    if actual_entries and _inventory_sha256(actual_entries) != manifest.get(
        "payload_inventory_sha256"
    ):
        errors.append("payload inventory SHA-256 mismatch")

    for path in [root, manifest_path, seal_path, payload_dir, *root.rglob("*")]:
        if path.exists() and path.stat().st_mode & 0o222:
            errors.append(f"sealed baseline path remains writable: {path.relative_to(root)}")

    if errors:
        raise BaselineValidationError("; ".join(errors))
    return {
        "baseline_id": manifest["baseline_id"],
        "manifest_sha256": actual_manifest_sha,
        "payload_inventory_sha256": manifest["payload_inventory_sha256"],
        "payload_file_count": manifest["payload_file_count"],
        "payload_size_bytes": manifest["payload_size_bytes"],
        "passed": True,
    }


def _validate_sources(sources: Mapping[str, Path]) -> dict[str, Path]:
    normalized: dict[str, Path] = {}
    for raw_label, raw_path in sources.items():
        label = str(raw_label).strip()
        if not label or label in {".", ".."} or "/" in label or "\\" in label:
            raise ValueError(f"Invalid baseline source label: {raw_label!r}")
        source = Path(raw_path).resolve()
        if not source.exists():
            raise FileNotFoundError(f"Baseline source does not exist: {source}")
        if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
            raise ValueError(f"Baseline sources must not contain symlinks: {source}")
        if label in normalized:
            raise ValueError(f"Duplicate baseline source label: {label}")
        normalized[label] = source
    return dict(sorted(normalized.items()))


def _inventory_entry(path: Path, *, package_root: Path, storage_mode: str) -> _InventoryEntry:
    return _InventoryEntry(
        relative_path=path.relative_to(package_root).as_posix(),
        size_bytes=path.stat().st_size,
        sha256=_sha256_path(path),
        storage_mode=storage_mode,
    )


def _inventory_sha256(entries: list[_InventoryEntry]) -> str:
    digest = hashlib.sha256()
    for entry in sorted(entries, key=lambda item: item.relative_path):
        digest.update(entry.relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(entry.size_bytes).encode("ascii"))
        digest.update(b"\0")
        digest.update(entry.sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _storage_mode_counts(entries: list[_InventoryEntry]) -> dict[str, int]:
    return {
        mode: sum(entry.storage_mode == mode for entry in entries)
        for mode in sorted({entry.storage_mode for entry in entries})
    }


def _remove_write_bits(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(path.stat().st_mode & ~0o222)
    root.chmod(root.stat().st_mode & ~0o222)


def _make_tree_owner_writable(root: Path) -> None:
    if not root.exists():
        return
    for path in [root, *root.rglob("*")]:
        try:
            path.chmod(path.stat().st_mode | 0o700)
        except OSError:
            continue
