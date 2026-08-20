"""Inventory and governance contracts for run result directories."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

RUN_ROOT_INVENTORY_CONTRACT = "alpharank_run_root_inventory_v1"
RUN_PATH_CONTRACT = "alpharank_run_path_v1"
RUN_MANIFEST_CONTRACT = "alpharank_run_manifest_v1"
RUN_LOG_LINK_CONTRACT = "alpharank_run_log_link_v1"
RUN_STATUSES = {"candidate", "validated", "published", "failed"}
RUN_STATUS_TRANSITIONS = {
    "candidate": {"validated", "failed"},
    "validated": {"published", "failed"},
    "published": set(),
    "failed": set(),
}
_COMPACT_DATE = re.compile(r"(?<!\d)(20\d{6})(?!\d)")
_DASHED_DATE = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")
_FAMILY = re.compile(r"[a-z][a-z0-9]*(?:_[a-z0-9]+)*")
_RUN_ID = re.compile(r"20\d{6}T\d{6}Z_[a-z0-9][a-z0-9_]*")


def canonical_run_dir(outputs_root: Path, *, family: str, run_id: str) -> Path:
    """Return the only valid directory shape for a newly created run."""

    if _FAMILY.fullmatch(family) is None:
        raise ValueError(f"Invalid run family: {family!r}")
    if _RUN_ID.fullmatch(run_id) is None:
        raise ValueError(f"Invalid run id: {run_id!r}")
    name_tokens = {family, *run_id.split("_")}
    reserved = name_tokens & RUN_STATUSES
    if reserved:
        raise ValueError(f"Run path embeds manifest status: {sorted(reserved)}")
    return outputs_root.resolve() / family / run_id


def validate_canonical_run_dir(outputs_root: Path, run_dir: Path) -> dict[str, str]:
    """Validate one path as exactly outputs/<family>/<run_id>."""

    outputs_root = outputs_root.resolve()
    run_dir = run_dir.resolve()
    if not run_dir.is_relative_to(outputs_root):
        raise ValueError(f"Run directory is outside outputs: {run_dir}")
    relative = run_dir.relative_to(outputs_root)
    if len(relative.parts) != 2:
        raise ValueError(f"Run directory must have exactly two parts: {relative}")
    family, run_id = relative.parts
    expected = canonical_run_dir(outputs_root, family=family, run_id=run_id)
    if expected != run_dir:
        raise ValueError(f"Run directory is not canonical: {run_dir}")
    return {
        "contract": RUN_PATH_CONTRACT,
        "family": family,
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
    }


def initialize_run_manifest(
    outputs_root: Path,
    *,
    family: str,
    run_id: str,
    created_at: str,
) -> Path:
    """Create one canonical run directory with an explicit candidate manifest."""

    run_dir = canonical_run_dir(outputs_root, family=family, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    manifest = {
        "contract": RUN_MANIFEST_CONTRACT,
        "family": family,
        "run_id": run_id,
        "run_dir": run_dir.relative_to(outputs_root.resolve().parent).as_posix(),
        "created_at": created_at,
        "status": "candidate",
        "status_history": [
            {
                "status": "candidate",
                "changed_at": created_at,
                "reason": "run_initialized",
            }
        ],
        "artifacts": [],
        "logs": [],
    }
    manifest_path = run_dir / "manifest.json"
    write_run_manifest(manifest_path, manifest)
    validate_run_manifest(outputs_root, manifest_path)
    return manifest_path


def transition_run_status(
    manifest: Mapping[str, object],
    *,
    new_status: str,
    changed_at: str,
    reason: str,
) -> dict[str, object]:
    """Return a manifest with one allowed, auditable status transition."""

    current = str(manifest.get("status"))
    if new_status not in RUN_STATUSES:
        raise ValueError(f"Unknown run status: {new_status}")
    if new_status not in RUN_STATUS_TRANSITIONS.get(current, set()):
        raise ValueError(f"Invalid run status transition: {current} -> {new_status}")
    if not reason.strip():
        raise ValueError("Run status transition requires a reason")
    updated = deepcopy(dict(manifest))
    raw_history = updated.get("status_history")
    if not isinstance(raw_history, list):
        raise RuntimeError("Run manifest status_history must be a list")
    updated["status"] = new_status
    raw_history.append(
        {
            "status": new_status,
            "changed_at": changed_at,
            "reason": reason,
        }
    )
    return updated


def validate_run_manifest(
    outputs_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Validate path identity, current status and complete status history."""

    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("contract") != RUN_MANIFEST_CONTRACT:
        raise RuntimeError("Unsupported run manifest contract")
    family = str(payload.get("family"))
    run_id = str(payload.get("run_id"))
    run_dir = canonical_run_dir(outputs_root, family=family, run_id=run_id)
    if manifest_path != run_dir / "manifest.json":
        raise RuntimeError("Run manifest path differs from its identity")
    recorded_run_dir = run_dir.relative_to(outputs_root.resolve().parent).as_posix()
    if payload.get("run_dir") != recorded_run_dir:
        raise RuntimeError("Run manifest records another directory")
    status = payload.get("status")
    history = payload.get("status_history")
    if status not in RUN_STATUSES or not isinstance(history, list) or not history:
        raise RuntimeError("Run manifest has no explicit valid status history")
    if any(not isinstance(entry, Mapping) for entry in history):
        raise RuntimeError("Run manifest status history entries must be objects")
    if history[-1].get("status") != status:
        raise RuntimeError("Run manifest status differs from status history")
    observed = [str(entry.get("status")) for entry in history]
    if observed[0] != "candidate":
        raise RuntimeError("Run status history must start as candidate")
    for previous, current in zip(observed, observed[1:], strict=False):
        if current not in RUN_STATUS_TRANSITIONS.get(previous, set()):
            raise RuntimeError(f"Invalid recorded status transition: {previous} -> {current}")
    return {
        "passed": True,
        "family": family,
        "run_id": run_id,
        "status": status,
        "transition_count": len(history) - 1,
    }


def write_run_manifest(path: Path, manifest: Mapping[str, object]) -> None:
    """Atomically write a small run manifest without moving result payloads."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def canonical_log_path(
    project_root: Path,
    *,
    family: str,
    run_id: str,
    filename: str = "run.log",
) -> Path:
    """Return logs/<family>/<run_id>/<filename> for one canonical run."""

    canonical_run_dir(project_root.resolve() / "outputs", family=family, run_id=run_id)
    if Path(filename).name != filename or not filename.endswith(".log"):
        raise ValueError(f"Invalid run log filename: {filename!r}")
    return project_root.resolve() / "logs" / family / run_id / filename


def register_run_log(
    project_root: Path,
    *,
    manifest_path: Path,
    log_path: Path,
    role: str,
) -> dict[str, object]:
    """Write bidirectional log links and return the updated run manifest."""

    project_root = project_root.resolve()
    outputs_root = project_root / "outputs"
    report = validate_run_manifest(outputs_root, manifest_path)
    if not role.strip():
        raise ValueError("Run log role must be explicit")
    expected_log = canonical_log_path(
        project_root,
        family=str(report["family"]),
        run_id=str(report["run_id"]),
        filename=log_path.name,
    )
    log_path = log_path.resolve()
    if log_path != expected_log or not log_path.is_file():
        raise RuntimeError(f"Run log is missing or outside its canonical directory: {log_path}")
    manifest_path = manifest_path.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Run manifest must be an object")
    raw_logs = payload.get("logs")
    if not isinstance(raw_logs, list):
        raise RuntimeError("Run manifest logs must be a list")
    sidecar_path = log_path.with_suffix(f"{log_path.suffix}.run.json")
    relative_log = log_path.relative_to(project_root).as_posix()
    relative_manifest = manifest_path.relative_to(project_root).as_posix()
    relative_sidecar = sidecar_path.relative_to(project_root).as_posix()
    log_record = {
        "log_id": hashlib.sha256(relative_log.encode("utf-8")).hexdigest(),
        "path": relative_log,
        "sidecar_path": relative_sidecar,
        "role": role,
        "size_bytes": log_path.stat().st_size,
        "sha256": _sha256_file(log_path),
    }
    if any(not isinstance(entry, Mapping) for entry in raw_logs):
        raise RuntimeError("Run manifest log entries must be objects")
    if any(entry.get("path") == relative_log for entry in raw_logs):
        raise RuntimeError(f"Run log is already registered: {relative_log}")
    raw_logs.append(log_record)
    sidecar = {
        "contract": RUN_LOG_LINK_CONTRACT,
        "family": report["family"],
        "run_id": report["run_id"],
        "log": log_record,
        "run_manifest_path": relative_manifest,
    }
    _write_json_atomic(sidecar_path, sidecar)
    write_run_manifest(manifest_path, payload)
    validate_run_log_links(project_root, manifest_path)
    return payload


def validate_run_log_links(
    project_root: Path,
    manifest_path: Path,
) -> dict[str, object]:
    """Follow every manifest-to-log link and each log sidecar back to the run."""

    project_root = project_root.resolve()
    manifest_path = manifest_path.resolve()
    validate_run_manifest(project_root / "outputs", manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_logs = payload.get("logs")
    if not isinstance(raw_logs, list):
        raise RuntimeError("Run manifest logs must be a list")
    relative_manifest = manifest_path.relative_to(project_root).as_posix()
    for record in raw_logs:
        if not isinstance(record, Mapping):
            raise RuntimeError("Run log record must be an object")
        log_path = (project_root / str(record["path"])).resolve()
        sidecar_path = (project_root / str(record["sidecar_path"])).resolve()
        if not log_path.is_file() or _sha256_file(log_path) != record.get("sha256"):
            raise RuntimeError(f"Run log bytes differ from manifest: {log_path}")
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if (
            not isinstance(sidecar, Mapping)
            or sidecar.get("contract") != RUN_LOG_LINK_CONTRACT
            or sidecar.get("run_manifest_path") != relative_manifest
            or sidecar.get("log") != record
        ):
            raise RuntimeError(f"Run log sidecar does not link back: {sidecar_path}")
    return {"passed": True, "log_count": len(raw_logs), "bidirectional": True}


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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
