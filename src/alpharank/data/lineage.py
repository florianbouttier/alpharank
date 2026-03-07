from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None


TEMPORAL_COLUMN_CANDIDATES = (
    "date",
    "year_month",
    "filing_date",
    "reportDate",
    "report_date",
)


def _now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _snapshot_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _frame_shape(df: Any) -> tuple[int, int]:
    if pl is not None and isinstance(df, pl.DataFrame):
        return df.height, df.width
    if isinstance(df, pd.DataFrame):
        return int(df.shape[0]), int(df.shape[1])
    return 0, 0


def _max_temporal_values(df: Any) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for col in TEMPORAL_COLUMN_CANDIDATES:
        if pl is not None and isinstance(df, pl.DataFrame) and col in df.columns and not df.is_empty():
            try:
                value = df.select(pl.col(col).max().alias("max_value")).item()
                if value is not None:
                    result[col] = str(value)
            except Exception:
                continue
        elif isinstance(df, pd.DataFrame) and col in df.columns and not df.empty:
            try:
                series = pd.to_datetime(df[col], errors="coerce")
                value = series.max()
                if pd.notna(value):
                    result[col] = str(value)
            except Exception:
                try:
                    value = df[col].max()
                    if pd.notna(value):
                        result[col] = str(value)
                except Exception:
                    continue
    return result


def summarize_frame(df: Any) -> Dict[str, Any]:
    rows, cols = _frame_shape(df)
    summary: Dict[str, Any] = {
        "rows": rows,
        "cols": cols,
        "max_temporal_values": _max_temporal_values(df),
    }
    if pl is not None and isinstance(df, pl.DataFrame):
        summary["columns"] = list(df.columns)
    elif isinstance(df, pd.DataFrame):
        summary["columns"] = list(df.columns)
    else:
        summary["columns"] = []
    return summary


def _dataset_manifest_entry(
    *,
    name: str,
    path: Path,
    frame: Any | None = None,
    snapshot_copy_path: Path | None = None,
) -> Dict[str, Any]:
    stat = path.stat()
    entry: Dict[str, Any] = {
        "name": name,
        "canonical_path": str(path.resolve()),
        "size_bytes": int(stat.st_size),
        "modified_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "sha256": _sha256(path),
    }
    if snapshot_copy_path is not None:
        entry["snapshot_copy_path"] = str(snapshot_copy_path.resolve())
    if frame is not None:
        entry["summary"] = summarize_frame(frame)
    return entry


def load_latest_manifest(data_dir: Path) -> Dict[str, Any] | None:
    latest_path = data_dir / "latest_snapshot.json"
    if not latest_path.exists():
        return None
    return json.loads(latest_path.read_text(encoding="utf-8"))


def write_manifest(
    *,
    manifest_path: Path,
    files: Mapping[str, Path],
    frames: Mapping[str, Any] | None = None,
    snapshot_id: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    datasets: Dict[str, Any] = {}
    for name, path in files.items():
        frame = frames.get(name) if frames is not None else None
        datasets[name] = _dataset_manifest_entry(name=name, path=path, frame=frame)

    manifest: Dict[str, Any] = {
        "snapshot_id": snapshot_id or _snapshot_id(),
        "generated_at": _now_str(),
        "manifest_path": str(manifest_path.resolve()),
        "datasets": datasets,
    }
    if extra:
        manifest.update(dict(extra))

    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def create_snapshot(
    *,
    data_dir: Path,
    files: Mapping[str, Path],
    frames: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    snapshot_id = _snapshot_id()
    snapshot_dir = data_dir / "_snapshots" / snapshot_id
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    copied_paths: Dict[str, Path] = {}
    for name, path in files.items():
        copied = snapshot_dir / path.name
        shutil.copy2(path, copied)
        copied_paths[name] = copied

    datasets: Dict[str, Any] = {}
    for name, path in files.items():
        frame = frames.get(name) if frames is not None else None
        datasets[name] = _dataset_manifest_entry(
            name=name,
            path=path,
            frame=frame,
            snapshot_copy_path=copied_paths[name],
        )

    manifest = {
        "snapshot_id": snapshot_id,
        "generated_at": _now_str(),
        "data_dir": str(data_dir.resolve()),
        "snapshot_dir": str(snapshot_dir.resolve()),
        "manifest_path": str((snapshot_dir / "manifest.json").resolve()),
        "datasets": datasets,
    }

    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (data_dir / "latest_snapshot.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
