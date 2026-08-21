"""Read-only observation and rollback policy for historical data roots."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

ARCHIVE_POLICY_CONTRACT = "alpharank_legacy_data_archive_policy_v1"
CATALOG_SUMMARY_CONTRACT = "alpharank_historical_root_catalog_summary_v1"
READER_MIGRATION_CONTRACT = "alpharank_data_reader_migration_v1"


def build_legacy_archive_policy(
    root: Path,
    catalog_summary: Mapping[str, object],
    reader_registry: Mapping[str, object],
    *,
    observation_started_at: str,
    minimum_observation_days: int = 30,
) -> dict[str, object]:
    """Freeze catalogued roots by contract and define reversible archival."""

    root = root.resolve()
    if catalog_summary.get("contract") != CATALOG_SUMMARY_CONTRACT:
        raise RuntimeError("Unsupported historical root catalog summary")
    if reader_registry.get("contract") != READER_MIGRATION_CONTRACT:
        raise RuntimeError("Unsupported reader migration registry")
    if minimum_observation_days < 1:
        raise ValueError("The read-only observation period must be positive")
    started_at = date.fromisoformat(observation_started_at)
    archive_not_before = started_at + timedelta(days=minimum_observation_days)

    raw_roots = catalog_summary.get("roots")
    if not isinstance(raw_roots, list) or not raw_roots:
        raise RuntimeError("Historical root catalog has no roots")
    roots = []
    for raw_root in raw_roots:
        if not isinstance(raw_root, Mapping):
            raise RuntimeError("Historical root declaration must be an object")
        source_path = Path(str(raw_root["source_path"])).resolve()
        roots.append(
            {
                "root_id": str(raw_root["root_id"]),
                "source_path": _relative_path(root, source_path),
                "source_kind": str(raw_root["source_kind"]),
                "file_count": int(raw_root["file_count"]),
                "size_bytes": int(raw_root["size_bytes"]),
                "inventory_sha256": str(raw_root["inventory_sha256"]),
                "write_policy": "deny_for_governed_repository_code",
                "archive_mode": "catalog_reference_without_payload_move",
                "archive_state": "read_only_observation",
            }
        )

    return {
        "contract": ARCHIVE_POLICY_CONTRACT,
        "catalog_id": catalog_summary["catalog_id"],
        "reader_registry_composition_id": reader_registry["composition_id"],
        "root_count": len(roots),
        "file_count": int(catalog_summary["file_count"]),
        "total_source_bytes": int(catalog_summary["total_source_bytes"]),
        "observation": {
            "started_at": started_at.isoformat(),
            "minimum_days": minimum_observation_days,
            "archive_not_before": archive_not_before.isoformat(),
            "state": "open",
            "promotion_condition": (
                "No governed write and no undeclared reader during the full window."
            ),
        },
        "physical_actions": {
            "permissions_changed": False,
            "payload_moved": False,
            "payload_deleted": False,
            "copy_count": 0,
            "download_count": 0,
        },
        "rollback": {
            "available": True,
            "trigger": (
                "A declared replay or transition cannot resolve its explicit legacy input."
            ),
            "steps": [
                "Stop the affected run before it publishes results.",
                "Resolve the root by root_id from this policy.",
                "Verify its inventory_sha256 against the immutable historical catalog.",
                "Restore only the affected reader's explicit path; do not change a canonical default.",
                "Record the exception and restart the observation window.",
            ],
        },
        "roots": sorted(roots, key=lambda item: str(item["root_id"])),
        "validation": {
            "passed": True,
            "all_catalogued_roots_frozen_by_contract": True,
            "archive_is_reference_only": True,
            "observation_window_defined": True,
            "rollback_defined": True,
            "automatic_deletion": False,
        },
    }


def validate_legacy_archive_policy(
    root: Path,
    catalog_summary: Mapping[str, object],
    reader_registry: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Reject policy drift against catalogued roots and reader evidence."""

    if policy.get("contract") != ARCHIVE_POLICY_CONTRACT:
        raise RuntimeError("Unsupported legacy archive policy")
    observation = policy.get("observation")
    if not isinstance(observation, Mapping):
        raise RuntimeError("Legacy archive policy has no observation window")
    expected = build_legacy_archive_policy(
        root,
        catalog_summary,
        reader_registry,
        observation_started_at=str(observation["started_at"]),
        minimum_observation_days=int(observation["minimum_days"]),
    )
    if dict(policy) != expected:
        raise RuntimeError("Legacy archive policy is stale")
    return {
        "passed": True,
        "root_count": expected["root_count"],
        "file_count": expected["file_count"],
        "total_source_bytes": expected["total_source_bytes"],
        "archive_not_before": observation["archive_not_before"],
        "payload_moved": False,
        "payload_deleted": False,
    }


def assert_legacy_path_not_writable(
    root: Path,
    candidate: Path,
    policy: Mapping[str, object],
) -> None:
    """Fail before governed code writes within a frozen historical root."""

    root = root.resolve()
    candidate = candidate.resolve()
    raw_roots = policy.get("roots")
    if not isinstance(raw_roots, list):
        raise RuntimeError("Legacy archive policy has no roots")
    for raw_root in raw_roots:
        if not isinstance(raw_root, Mapping):
            raise RuntimeError("Legacy archive root must be an object")
        frozen_root = (root / str(raw_root["source_path"])).resolve()
        if candidate == frozen_root or candidate.is_relative_to(frozen_root):
            raise PermissionError(
                f"Governed write targets frozen legacy root {raw_root['root_id']}: "
                f"{candidate}"
            )


def write_legacy_archive_policy(
    path: Path,
    policy: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(policy, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _relative_path(root: Path, path: Path) -> str:
    if path.is_relative_to(root):
        return path.relative_to(root).as_posix()
    if "data" in path.parts:
        data_index = path.parts.index("data")
        return Path(*path.parts[data_index:]).as_posix()
    return str(path)
