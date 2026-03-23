from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from typing import Any


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

    shutil.copytree(source_dir, snapshot_dir)
    if metadata is not None:
        (snapshot_dir / "snapshot_manifest.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return snapshot_dir
