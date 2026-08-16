#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Iterable

from alpharank.data.snapshot_storage import copy_snapshot_file


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HISTORY_ROOT = PROJECT_ROOT / "data" / "open_source" / "history" / "output"


def _sha256(path: Path) -> tuple[Path, str]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return path, digest.hexdigest()


def _candidate_files(history_root: Path) -> list[Path]:
    return sorted(
        path
        for path in history_root.rglob("*")
        if path.is_file() and not path.is_symlink() and ".compaction-tmp" not in path.name
    )


def compact_history(history_root: Path, *, dry_run: bool = False, workers: int = 4) -> dict[str, object]:
    files = _candidate_files(history_root)
    by_size: dict[int, list[Path]] = defaultdict(list)
    for path in files:
        by_size[path.stat().st_size].append(path)

    hash_candidates = [path for paths in by_size.values() if len(paths) > 1 for path in paths]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        hashed = list(executor.map(_sha256, hash_candidates))

    by_content: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path, digest in hashed:
        by_content[(path.stat().st_size, digest)].append(path)

    duplicate_groups = [paths for paths in by_content.values() if len(paths) > 1]
    replaced = 0
    logical_duplicate_bytes = 0
    storage_modes: dict[str, int] = defaultdict(int)
    for paths in duplicate_groups:
        canonical = paths[0]
        size = canonical.stat().st_size
        for duplicate in paths[1:]:
            logical_duplicate_bytes += size
            if dry_run:
                continue
            temporary = duplicate.with_name(f".{duplicate.name}.compaction-tmp")
            temporary.unlink(missing_ok=True)
            mode = copy_snapshot_file(canonical, temporary)
            if _sha256(temporary)[1] != _sha256(canonical)[1]:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(f"Compaction verification failed for {duplicate}")
            os.replace(temporary, duplicate)
            storage_modes[mode] += 1
            replaced += 1

    return {
        "history_root": str(history_root.resolve()),
        "dry_run": dry_run,
        "file_count": len(files),
        "hashed_file_count": len(hash_candidates),
        "duplicate_group_count": len(duplicate_groups),
        "duplicate_file_count": sum(len(paths) - 1 for paths in duplicate_groups),
        "logical_duplicate_bytes": logical_duplicate_bytes,
        "replaced_file_count": replaced,
        "storage_mode_counts": dict(sorted(storage_modes.items())),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deduplicate immutable output snapshots with verified copy-on-write clones."
    )
    parser.add_argument("--history-root", type=Path, default=DEFAULT_HISTORY_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = compact_history(args.history_root.resolve(), dry_run=args.dry_run, workers=args.workers)
    report["completed_at"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    report_path = args.history_root.resolve().parent / (
        f"compaction_report_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    )
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
