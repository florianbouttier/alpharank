from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any

from alpharank.data.publishing.snapshot_storage import copy_snapshot_file


def snapshot_output_directory(
    source_dir: Path,
    *,
    history_root: Path,
    snapshot_prefix: str,
    metadata: dict[str, Any] | None = None,
) -> Path | None:
    if not source_dir.exists() or not any(source_dir.iterdir()):
        return None

    history_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    snapshot_dir = history_root / f"{snapshot_prefix}_{timestamp}"
    suffix = 1
    while snapshot_dir.exists():
        suffix += 1
        snapshot_dir = history_root / f"{snapshot_prefix}_{timestamp}_{suffix}"

    storage_modes: list[str] = []

    def copy_file(source: str, destination: str) -> str:
        storage_modes.append(copy_snapshot_file(source, destination))
        return destination

    shutil.copytree(source_dir, snapshot_dir, copy_function=copy_file)
    if metadata is not None:
        (snapshot_dir / "snapshot_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (snapshot_dir / "storage_manifest.json").write_text(
        json.dumps(
            {
                "strategy": "copy_on_write_with_physical_copy_fallback",
                "semantics": "independent path with byte-identical content; APFS clones are copy-on-write",
                "source_dir": str(source_dir.resolve()),
                "file_count": len(storage_modes),
                "storage_mode_counts": {
                    mode: storage_modes.count(mode) for mode in sorted(set(storage_modes))
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return snapshot_dir
