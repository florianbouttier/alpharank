"""Exact duplicate measurement and reversible retention for run results."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

RUN_RETENTION_REPORT_CONTRACT = "alpharank_run_retention_report_v1"
RUN_RETENTION_SUMMARY_CONTRACT = "alpharank_run_retention_summary_v1"


def build_run_retention_report(
    outputs_root: Path,
    *,
    generated_at: str,
) -> dict[str, object]:
    """Measure exact duplicates without moving or deleting any result."""

    outputs_root = outputs_root.resolve()
    if not outputs_root.is_dir():
        raise FileNotFoundError(outputs_root)
    files = sorted(path for path in outputs_root.rglob("*") if path.is_file())
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_size[path.stat().st_size].append(path)

    duplicate_groups: list[dict[str, object]] = []
    hashed_file_count = 0
    content_object_count = 0
    for size_bytes, candidates in sorted(by_size.items()):
        if len(candidates) == 1:
            content_object_count += 1
            continue
        by_hash: dict[str, list[Path]] = defaultdict(list)
        for path in candidates:
            by_hash[_sha256(path)].append(path)
            hashed_file_count += 1
        content_object_count += len(by_hash)
        for digest, matches in sorted(by_hash.items()):
            if len(matches) < 2:
                continue
            relative_paths = [
                path.relative_to(outputs_root).as_posix() for path in sorted(matches)
            ]
            reclaimable_bytes = size_bytes * (len(relative_paths) - 1)
            duplicate_groups.append(
                {
                    "sha256": digest,
                    "size_bytes": size_bytes,
                    "copy_count": len(relative_paths),
                    "reclaimable_bytes": reclaimable_bytes,
                    "recovery_source": relative_paths[0],
                    "retention_candidates": relative_paths[1:],
                    "paths": relative_paths,
                }
            )

    duplicate_groups.sort(
        key=lambda group: (
            -int(group["reclaimable_bytes"]),
            str(group["sha256"]),
        )
    )
    total_bytes = sum(path.stat().st_size for path in files)
    report_id = _records_sha256(duplicate_groups)
    return {
        "contract": RUN_RETENTION_REPORT_CONTRACT,
        "report_id": report_id,
        "generated_at": generated_at,
        "outputs_root": str(outputs_root),
        "measurement": "exact_sha256_after_size_grouping",
        "file_count": len(files),
        "total_bytes": total_bytes,
        "size_collision_file_count": hashed_file_count,
        "unique_content_count": content_object_count,
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": sum(
            int(group["copy_count"]) - 1 for group in duplicate_groups
        ),
        "reclaimable_bytes": sum(
            int(group["reclaimable_bytes"]) for group in duplicate_groups
        ),
        "duplicate_groups": duplicate_groups,
        "retention_proposal": {
            "state": "proposal_only",
            "automatic_deletion": False,
            "deletion_count": 0,
            "rule": (
                "Keep every published target and one verified recovery source per exact SHA-256 group."
            ),
            "preconditions": [
                "Close the observation window and obtain a separate owner decision.",
                "Rehash each candidate and its recovery source immediately before action.",
                "Exclude every path referenced by a run manifest, latest pointer or rollback record.",
                "Move candidates to a reversible quarantine before any later deletion decision.",
            ],
            "rollback": (
                "Restore the quarantined path from recovery_source and verify the recorded SHA-256."
            ),
        },
        "validation": {
            "passed": True,
            "exact_content_hash": True,
            "payload_copied": False,
            "payload_moved": False,
            "payload_deleted": False,
        },
    }


def write_run_retention_report(
    path: Path,
    report: Mapping[str, object],
) -> None:
    """Write one immutable full report, accepting only identical replay."""

    path = path.resolve()
    payload = dict(report)
    if path.exists():
        existing = _read_json(path)
        if existing != payload:
            raise FileExistsError(f"Run retention report already differs: {path}")
        return
    _write_json_atomic(path, payload)


def validate_run_retention_report(
    outputs_root: Path,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Rebuild the exact report and reject any measurement drift."""

    if report.get("contract") != RUN_RETENTION_REPORT_CONTRACT:
        raise RuntimeError("Unsupported run retention report contract")
    expected = build_run_retention_report(
        outputs_root,
        generated_at=str(report.get("generated_at", "not_recorded")),
    )
    if dict(report) != expected:
        raise RuntimeError("Run retention report is stale")
    return {
        "passed": True,
        "report_id": expected["report_id"],
        "file_count": expected["file_count"],
        "duplicate_group_count": expected["duplicate_group_count"],
        "duplicate_file_count": expected["duplicate_file_count"],
        "reclaimable_bytes": expected["reclaimable_bytes"],
        "deletion_count": 0,
    }


def build_run_retention_summary(
    report_path: Path,
    *,
    top_group_count: int = 50,
) -> dict[str, object]:
    """Build the tracked review summary for the full ignored report."""

    report_path = report_path.resolve()
    report = _read_json(report_path)
    raw_groups = report.get("duplicate_groups")
    if not isinstance(raw_groups, list):
        raise RuntimeError("Run retention report has no duplicate groups")
    return {
        "contract": RUN_RETENTION_SUMMARY_CONTRACT,
        "report_id": report["report_id"],
        "generated_at": report["generated_at"],
        "report_path": str(report_path),
        "report_sha256": _sha256(report_path),
        "measurement": report["measurement"],
        "file_count": report["file_count"],
        "total_bytes": report["total_bytes"],
        "size_collision_file_count": report["size_collision_file_count"],
        "unique_content_count": report["unique_content_count"],
        "duplicate_group_count": report["duplicate_group_count"],
        "duplicate_file_count": report["duplicate_file_count"],
        "reclaimable_bytes": report["reclaimable_bytes"],
        "top_duplicate_groups": raw_groups[:top_group_count],
        "retention_proposal": report["retention_proposal"],
        "validation": report["validation"],
    }


def write_run_retention_summary(
    path: Path,
    summary: Mapping[str, object],
) -> None:
    _write_json_atomic(path.resolve(), dict(summary))


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


def _records_sha256(records: list[dict[str, object]]) -> str:
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
