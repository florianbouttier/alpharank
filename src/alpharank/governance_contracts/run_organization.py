"""Inventory and governance contracts for run result directories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RUN_ROOT_INVENTORY_CONTRACT = "alpharank_run_root_inventory_v1"
RUN_STATUSES = {"candidate", "validated", "published", "failed"}
_COMPACT_DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
_DASHED_DATE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")


def build_run_root_inventory(
    outputs_root: Path,
    *,
    observed_at: str,
) -> dict[str, object]:
    """Inventory each immediate output directory from metadata and manifests."""

    outputs_root = outputs_root.resolve()
    if not outputs_root.is_dir():
        raise FileNotFoundError(outputs_root)
    records = [
        _inventory_run_root(outputs_root, path)
        for path in sorted(
            (item for item in outputs_root.iterdir() if item.is_dir()),
            key=lambda item: item.name,
        )
    ]
    families = Counter(str(record["family"]) for record in records)
    statuses = Counter(str(record["status"]) for record in records)
    inventory_sha256 = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "contract": RUN_ROOT_INVENTORY_CONTRACT,
        "observed_at": observed_at,
        "scope": "Immediate outputs directories; metadata and root manifest only.",
        "inventory_sha256": inventory_sha256,
        "summary": {
            "run_root_count": len(records),
            "file_count": sum(int(record["file_count"]) for record in records),
            "size_bytes": sum(int(record["size_bytes"]) for record in records),
            "family_counts": dict(sorted(families.items())),
            "status_counts": dict(sorted(statuses.items())),
        },
        "run_roots": records,
        "validation": {
            "passed": True,
            "all_immediate_directories_registered": True,
            "payload_content_read": False,
            "undeclared_status_inferred_from_name": False,
        },
    }


def validate_run_root_inventory(
    outputs_root: Path,
    inventory: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the inventory and reject directory, metadata or manifest drift."""

    if inventory.get("contract") != RUN_ROOT_INVENTORY_CONTRACT:
        raise RuntimeError("Unsupported run root inventory contract")
    expected = build_run_root_inventory(
        outputs_root,
        observed_at=str(inventory.get("observed_at", "not_recorded")),
    )
    if dict(inventory) != expected:
        raise RuntimeError("Run root inventory is stale")
    return {"passed": True, **dict(expected["summary"])}


def write_run_root_inventory(
    path: Path,
    inventory: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(inventory, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _inventory_run_root(outputs_root: Path, run_root: Path) -> dict[str, object]:
    file_count = 0
    size_bytes = 0
    latest_mtime_ns = run_root.stat().st_mtime_ns
    for path in run_root.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        file_count += 1
        size_bytes += stat.st_size
        latest_mtime_ns = max(latest_mtime_ns, stat.st_mtime_ns)
    run_date, date_source = _run_date(run_root.name, latest_mtime_ns)
    status, status_source = _manifest_status(run_root / "manifest.json")
    return {
        "path": run_root.relative_to(outputs_root.parent).as_posix(),
        "root_name": run_root.name,
        "family": _run_family(run_root.name),
        "run_date": run_date,
        "date_source": date_source,
        "status": status,
        "status_source": status_source,
        "file_count": file_count,
        "size_bytes": size_bytes,
        "latest_mtime_utc": datetime.fromtimestamp(
            latest_mtime_ns / 1_000_000_000,
            tz=timezone.utc,
        ).isoformat(),
    }


def _run_date(name: str, latest_mtime_ns: int) -> tuple[str, str]:
    dashed = _DASHED_DATE.search(name)
    if dashed:
        return dashed.group(1), "root_name"
    compact = _COMPACT_DATE.search(name)
    if compact:
        value = compact.group(1)
        return f"{value[:4]}-{value[4:6]}-{value[6:]}", "root_name"
    observed = datetime.fromtimestamp(
        latest_mtime_ns / 1_000_000_000,
        tz=timezone.utc,
    )
    return observed.date().isoformat(), "latest_file_mtime"


def _manifest_status(manifest_path: Path) -> tuple[str, str]:
    if not manifest_path.is_file():
        return "legacy_unclassified", "no_root_manifest"
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "legacy_unclassified", "unreadable_root_manifest"
    if not isinstance(payload, Mapping):
        return "legacy_unclassified", "root_manifest_without_explicit_status"
    status = payload.get("status")
    if status in RUN_STATUSES:
        return str(status), "root_manifest"
    return "legacy_unclassified", "root_manifest_without_explicit_status"


def _run_family(name: str) -> str:
    if name.startswith("."):
        return "tooling"
    if _DASHED_DATE.fullmatch(name):
        return "dated_legacy_runs"
    prefixes = (
        ("_staging", "staging"),
        ("checkpoints", "checkpoints"),
        ("common_", "replay"),
        ("data_", "data_quality"),
        ("ema_", "ema_research"),
        ("generalized_ema_", "ema_research"),
        ("legacy_", "legacy"),
        ("live", "live"),
        ("methodology", "methodology"),
        ("multihorizon", "boosting"),
        ("open_source", "open_source"),
        ("portfolio_", "portfolio_research"),
        ("production_refresh", "production_refresh"),
        ("research_dashboard", "reporting"),
        ("sec_", "sec_research"),
    )
    for prefix, family in prefixes:
        if name.startswith(prefix):
            return family
    return "other_legacy"
