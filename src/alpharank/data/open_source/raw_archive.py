from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from alpharank.data.snapshot_storage import copy_snapshot_file


RAW_DELTA_CONTRACT = "alpharank_raw_delta_archive_v1"
IMMUTABLE_FILE_CONTRACT = "alpharank_immutable_raw_file_v1"


@dataclass(frozen=True)
class RawArchiveResult:
    run_id: str
    run_dir: Path
    manifest_path: Path
    parent_run_id: str | None
    input_row_count: int
    stored_content_row_count: int
    unchanged_row_count: int
    inserted_row_count: int
    updated_row_count: int
    restored_row_count: int
    missing_row_count: int
    snapshot_sha256: str


def archive_raw_frame_delta(
    *,
    archive_dir: Path,
    run_id: str,
    frame: pl.DataFrame,
    key_columns: Sequence[str],
    source: str,
    dataset: str,
    observed_at: str,
    request: dict[str, Any] | None = None,
) -> RawArchiveResult:
    """Archive one provider observation without storing unchanged rows again.

    The raw archive is an immutable event chain. Each run records inserts,
    changes, restorations and keys missing from the provider response. An
    identical observation therefore writes an empty event parquet plus a new
    manifest, while remaining exactly reconstructible through its parent.
    """

    archive_dir = archive_dir.resolve()
    runs_dir = archive_dir / "runs"
    manifests_dir = archive_dir / "manifests"
    run_dir = runs_dir / run_id
    if run_dir.exists():
        raise FileExistsError(f"Raw archive run already exists: {run_dir}")
    if not key_columns:
        raise ValueError("Raw archive key_columns cannot be empty")
    missing_columns = [column for column in key_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(f"Raw archive key columns are missing: {missing_columns}")
    if frame.select(pl.struct(list(key_columns)).is_duplicated().any()).item():
        raise ValueError(f"Raw archive input contains duplicate keys: {list(key_columns)}")

    data_columns = list(frame.columns)
    ordered = frame.sort(list(key_columns))
    parent_run_id = _read_latest_run_id(manifests_dir / "latest.json")
    previous_state, content_by_hash, parent_manifest = _reconstruct_internal(
        archive_dir=archive_dir,
        run_id=parent_run_id,
    )
    if parent_manifest is not None:
        _require_compatible_contract(
            parent_manifest=parent_manifest,
            source=source,
            dataset=dataset,
            key_columns=key_columns,
            data_columns=data_columns,
        )

    current_state: dict[tuple[Any, ...], str] = {}
    current_rows: dict[tuple[Any, ...], tuple[Any, ...]] = {}
    for row in ordered.iter_rows(named=False):
        key = tuple(row[data_columns.index(column)] for column in key_columns)
        row_sha256 = _row_sha256(data_columns, row)
        current_state[key] = row_sha256
        current_rows[key] = row

    event_rows: list[dict[str, Any]] = []
    counts = {"inserted": 0, "updated": 0, "restored": 0, "missing": 0}
    unchanged_count = 0
    for key, row_sha256 in current_state.items():
        previous_sha256 = previous_state.get(key)
        if previous_sha256 == row_sha256:
            unchanged_count += 1
            continue
        seen_before = row_sha256 in content_by_hash
        if previous_sha256 is None and seen_before:
            event_type = "restored"
        elif previous_sha256 is None:
            event_type = "inserted"
        else:
            event_type = "updated"
        counts[event_type] += 1
        row = current_rows[key]
        event = {
            "event_type": event_type,
            "row_sha256": row_sha256,
            "previous_row_sha256": previous_sha256,
            "stores_content": not seen_before,
        }
        for index, column in enumerate(data_columns):
            event[column] = row[index] if (not seen_before or column in key_columns) else None
        event_rows.append(event)

    for key, previous_sha256 in previous_state.items():
        if key in current_state:
            continue
        counts["missing"] += 1
        event = {
            "event_type": "missing",
            "row_sha256": None,
            "previous_row_sha256": previous_sha256,
            "stores_content": False,
        }
        for column, value in zip(key_columns, key):
            event[column] = value
        for column in data_columns:
            event.setdefault(column, None)
        event_rows.append(event)

    event_schema = {
        "event_type": pl.String,
        "row_sha256": pl.String,
        "previous_row_sha256": pl.String,
        "stores_content": pl.Boolean,
        **dict(ordered.schema),
    }
    events = pl.DataFrame(event_rows, schema=event_schema, strict=False)
    if not events.is_empty():
        events = events.sort([*key_columns, "event_type"])

    snapshot_sha256 = _snapshot_sha256(current_state)
    run_dir.mkdir(parents=True, exist_ok=False)
    events_path = run_dir / "events.parquet"
    events.write_parquet(events_path)
    event_file_sha256 = _file_sha256(events_path)
    manifest = {
        "contract": RAW_DELTA_CONTRACT,
        "run_id": run_id,
        "parent_run_id": parent_run_id,
        "source": source,
        "dataset": dataset,
        "observed_at": observed_at,
        "request": request or {},
        "key_columns": list(key_columns),
        "data_columns": data_columns,
        "schema": {column: str(dtype) for column, dtype in ordered.schema.items()},
        "input_row_count": ordered.height,
        "event_row_count": events.height,
        "stored_content_row_count": events.filter(pl.col("stores_content")).height,
        "unchanged_row_count": unchanged_count,
        "inserted_row_count": counts["inserted"],
        "updated_row_count": counts["updated"],
        "restored_row_count": counts["restored"],
        "missing_row_count": counts["missing"],
        "snapshot_sha256": snapshot_sha256,
        "events_path": "events.parquet",
        "events_sha256": event_file_sha256,
    }
    manifest_path = run_dir / "manifest.json"
    _write_json_atomic(manifest_path, manifest)
    manifests_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(
        manifests_dir / "latest.json",
        {
            "contract": RAW_DELTA_CONTRACT,
            "run_id": run_id,
            "manifest_path": str(manifest_path),
            "snapshot_sha256": snapshot_sha256,
        },
    )
    return RawArchiveResult(
        run_id=run_id,
        run_dir=run_dir,
        manifest_path=manifest_path,
        parent_run_id=parent_run_id,
        input_row_count=ordered.height,
        stored_content_row_count=manifest["stored_content_row_count"],
        unchanged_row_count=unchanged_count,
        inserted_row_count=counts["inserted"],
        updated_row_count=counts["updated"],
        restored_row_count=counts["restored"],
        missing_row_count=counts["missing"],
        snapshot_sha256=snapshot_sha256,
    )


def reconstruct_raw_frame(*, archive_dir: Path, run_id: str | None = None) -> pl.DataFrame:
    """Reconstruct the exact provider frame observed by a raw archive run."""

    archive_dir = archive_dir.resolve()
    selected_run_id = run_id or _read_latest_run_id(archive_dir / "manifests" / "latest.json")
    state, content_by_hash, manifest = _reconstruct_internal(
        archive_dir=archive_dir,
        run_id=selected_run_id,
    )
    if manifest is None:
        return pl.DataFrame()
    rows = [content_by_hash[row_sha256] for _, row_sha256 in sorted(state.items(), key=lambda item: item[0])]
    if not rows:
        return pl.DataFrame(schema={column: _dtype_from_name(dtype) for column, dtype in manifest["schema"].items()})
    return pl.DataFrame(rows, schema=manifest["data_columns"], orient="row").cast(
        {column: _dtype_from_name(dtype) for column, dtype in manifest["schema"].items()},
        strict=False,
    ).sort(manifest["key_columns"])


def register_immutable_raw_file(
    *,
    archive_dir: Path,
    source_id: str,
    source_path: Path,
    source: str,
    dataset: str,
    observed_at: str,
) -> Path:
    """Register a local source file once and reuse its content-addressed object."""

    archive_dir = archive_dir.resolve()
    source_path = source_path.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    digest = _file_sha256(source_path)
    suffix = "".join(source_path.suffixes)
    object_path = archive_dir / "objects" / digest[:2] / f"{digest}{suffix}"
    if not object_path.exists():
        object_path.parent.mkdir(parents=True, exist_ok=True)
        copy_snapshot_file(source_path, object_path)
    manifest_path = archive_dir / "sources" / source_id / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Immutable raw source id already exists: {source_id}")
    _write_json_atomic(
        manifest_path,
        {
            "contract": IMMUTABLE_FILE_CONTRACT,
            "source_id": source_id,
            "source": source,
            "dataset": dataset,
            "observed_at": observed_at,
            "original_path": str(source_path),
            "object_path": str(object_path),
            "size_bytes": source_path.stat().st_size,
            "sha256": digest,
        },
    )
    return manifest_path


def _reconstruct_internal(
    *,
    archive_dir: Path,
    run_id: str | None,
) -> tuple[dict[tuple[Any, ...], str], dict[str, tuple[Any, ...]], dict[str, Any] | None]:
    if run_id is None:
        return {}, {}, None
    manifests: list[dict[str, Any]] = []
    seen_runs: set[str] = set()
    cursor = run_id
    while cursor is not None:
        if cursor in seen_runs:
            raise RuntimeError(f"Raw archive parent cycle detected at run {cursor}")
        seen_runs.add(cursor)
        manifest_path = archive_dir / "runs" / cursor / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("contract") != RAW_DELTA_CONTRACT:
            raise RuntimeError(f"Unsupported raw archive contract: {manifest_path}")
        manifests.append(manifest)
        cursor = manifest.get("parent_run_id")
    manifests.reverse()

    state: dict[tuple[Any, ...], str] = {}
    content_by_hash: dict[str, tuple[Any, ...]] = {}
    for manifest in manifests:
        events_path = archive_dir / "runs" / manifest["run_id"] / manifest["events_path"]
        if _file_sha256(events_path) != manifest["events_sha256"]:
            raise RuntimeError(f"Raw archive event hash mismatch: {events_path}")
        events = pl.read_parquet(events_path)
        key_columns = manifest["key_columns"]
        data_columns = manifest["data_columns"]
        for event in events.iter_rows(named=True):
            key = tuple(event[column] for column in key_columns)
            if event["event_type"] == "missing":
                state.pop(key, None)
                continue
            row_sha256 = str(event["row_sha256"])
            if event["stores_content"]:
                content_by_hash[row_sha256] = tuple(event[column] for column in data_columns)
            if row_sha256 not in content_by_hash:
                raise RuntimeError(f"Raw archive content reference is unresolved: {row_sha256}")
            state[key] = row_sha256
        if _snapshot_sha256(state) != manifest["snapshot_sha256"]:
            raise RuntimeError(f"Raw archive snapshot hash mismatch for run {manifest['run_id']}")
    return state, content_by_hash, manifests[-1]


def _require_compatible_contract(
    *,
    parent_manifest: dict[str, Any],
    source: str,
    dataset: str,
    key_columns: Sequence[str],
    data_columns: Sequence[str],
) -> None:
    expected = {
        "source": source,
        "dataset": dataset,
        "key_columns": list(key_columns),
        "data_columns": list(data_columns),
    }
    actual = {key: parent_manifest.get(key) for key in expected}
    if actual != expected:
        raise ValueError(f"Raw archive contract changed: expected={expected}, previous={actual}")


def _row_sha256(columns: Sequence[str], row: Sequence[Any]) -> str:
    payload = [[column, _canonical_value(value)] for column, value in zip(columns, row)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _snapshot_sha256(state: dict[tuple[Any, ...], str]) -> str:
    digest = hashlib.sha256()
    for key, row_sha256 in sorted(state.items(), key=lambda item: item[0]):
        digest.update(json.dumps([_canonical_value(value) for value in key], separators=(",", ":")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(row_sha256.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        return value
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_latest_run_id(path: Path) -> str | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload.get("run_id")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _dtype_from_name(name: str) -> pl.DataType:
    mapping = {
        "String": pl.String,
        "Utf8": pl.String,
        "Float64": pl.Float64,
        "Float32": pl.Float32,
        "Int64": pl.Int64,
        "Int32": pl.Int32,
        "UInt64": pl.UInt64,
        "UInt32": pl.UInt32,
        "Boolean": pl.Boolean,
        "Date": pl.Date,
        "Datetime(time_unit='us', time_zone=None)": pl.Datetime("us"),
    }
    if name not in mapping:
        raise ValueError(f"Unsupported raw archive dtype: {name}")
    return mapping[name]
