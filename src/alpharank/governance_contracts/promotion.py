"""Atomic run reservation and methodology promotion."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from alpharank.governance_contracts.common import (
    atomic_replace_json as _atomic_replace_json,
)
from alpharank.governance_contracts.common import (
    directory_hashes as _directory_hashes,
)
from alpharank.governance_contracts.common import (
    promotion_timestamp as _promotion_timestamp,
)


def reserve_run_directory(run_dir: Path) -> Path:
    """Atomically reserve a never-before-used run directory."""

    destination = run_dir.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise FileExistsError(
            f"Run directory already exists and cannot be reused: {destination}"
        ) from error
    return destination


def promote_methodology_version(
    *,
    pointer_path: Path,
    version_dir: Path,
    version_id: str,
    approved_by: str,
    reason: str,
    changed_at: datetime | None = None,
) -> dict[str, Any]:
    """Atomically promote a version while preserving every prior record."""

    identifier = str(version_id).strip()
    if not identifier:
        raise ValueError("version_id must be non-empty.")
    version = version_dir.resolve()
    if not version.is_dir():
        raise FileNotFoundError(f"Methodology version directory not found: {version}")
    timestamp = _promotion_timestamp(changed_at)
    pointer = _read_promotion_pointer(pointer_path)
    records = dict(pointer.get("version_records") or {})
    current = pointer.get("active_version")
    if current == identifier:
        raise ValueError(f"Methodology version is already active: {identifier}")
    if current in records:
        records[current] = {**records[current], "status": "superseded"}
    records[identifier] = {
        "version_id": identifier,
        "version_dir": str(version),
        "artifact_hashes": _directory_hashes(version),
        "status": "active",
    }
    actions = list(pointer.get("actions") or [])
    actions.append(
        {
            "action": "promote",
            "from_version": current,
            "to_version": identifier,
            "approved_by": str(approved_by),
            "reason": str(reason),
            "changed_at_utc": timestamp,
        }
    )
    payload = {
        "promotion_contract_version": 1,
        "active_version": identifier,
        "active_record": records[identifier],
        "version_records": records,
        "actions": actions,
    }
    _atomic_replace_json(pointer_path, payload)
    return payload


def rollback_methodology_version(
    *,
    pointer_path: Path,
    target_version_id: str,
    approved_by: str,
    reason: str,
    changed_at: datetime | None = None,
) -> dict[str, Any]:
    """Atomically reactivate an intact prior version by its sealed hashes."""

    pointer = _read_promotion_pointer(pointer_path, required=True)
    records = dict(pointer.get("version_records") or {})
    target_id = str(target_version_id).strip()
    if target_id not in records:
        raise KeyError(f"Unknown methodology version: {target_id}")
    target = dict(records[target_id])
    version_dir = Path(str(target["version_dir"]))
    actual_hashes = _directory_hashes(version_dir)
    if actual_hashes != target.get("artifact_hashes"):
        raise RuntimeError(f"Cannot rollback to modified version: {target_id}")
    current = pointer.get("active_version")
    if current in records:
        records[current] = {**records[current], "status": "superseded"}
    target["status"] = "active"
    records[target_id] = target
    actions = list(pointer.get("actions") or [])
    actions.append(
        {
            "action": "rollback",
            "from_version": current,
            "to_version": target_id,
            "approved_by": str(approved_by),
            "reason": str(reason),
            "changed_at_utc": _promotion_timestamp(changed_at),
        }
    )
    payload = {
        **pointer,
        "active_version": target_id,
        "active_record": target,
        "version_records": records,
        "actions": actions,
    }
    _atomic_replace_json(pointer_path, payload)
    return payload


def _read_promotion_pointer(path: Path, *, required: bool = False) -> dict[str, Any]:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Promotion pointer not found: {path}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
