"""Columnar implementation of the immutable RAW v1 event-chain contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Sequence

import polars as pl

RAW_DELTA_CONTRACT = "alpharank_raw_delta_archive_v1"


@dataclass(frozen=True)
class RawFrameDelta:
    """One observed frame and its changes, without Python row dictionaries."""

    ordered: pl.DataFrame
    events: pl.DataFrame
    parent_manifest: dict[str, Any] | None
    snapshot_sha256: str
    unchanged_count: int


def build_frame_delta(
    *,
    archive_dir: Path,
    run_id: str | None,
    frame: pl.DataFrame,
    key_columns: Sequence[str],
) -> RawFrameDelta:
    ordered = frame.sort(list(key_columns))
    current = _with_row_hashes(ordered)
    previous, content, manifest = reconstruct_archive_index(archive_dir=archive_dir, run_id=run_id)
    if manifest is None:
        previous = current.head(0).select([*key_columns, "row_sha256"])
        content = _content_hashes(current.head(0))
    changed = current.join(
        previous.rename({"row_sha256": "previous_row_sha256"}),
        on=list(key_columns),
        how="left",
        nulls_equal=True,
    ).filter(~pl.col("row_sha256").eq_missing(pl.col("previous_row_sha256")))
    unchanged = ordered.height - changed.height
    changed = _changed_events(changed, content, ordered.schema, key_columns)
    missing = _missing_events(previous, current, ordered.schema, key_columns)
    events = pl.concat([changed, missing]).sort([*key_columns, "event_type"])
    return RawFrameDelta(
        ordered=ordered,
        events=events,
        parent_manifest=manifest,
        snapshot_sha256=frame_snapshot_sha256(current, key_columns),
        unchanged_count=unchanged,
    )


def _changed_events(
    changed: pl.DataFrame,
    content: pl.DataFrame,
    schema: pl.Schema,
    key_columns: Sequence[str],
) -> pl.DataFrame:
    changed = changed.with_columns(_content_hashes(changed).to_series())
    seen = _known_references(content, changed.select("_raw_content_hash")).with_columns(
        pl.lit(True).alias("_raw_seen")
    )
    changed = changed.join(seen, on="_raw_content_hash", how="left").with_columns(
        pl.col("_raw_seen").is_null().alias("stores_content"),
        pl.when(pl.col("previous_row_sha256").is_not_null())
        .then(pl.lit("updated"))
        .when(pl.col("_raw_seen").fill_null(False))
        .then(pl.lit("restored"))
        .otherwise(pl.lit("inserted"))
        .alias("event_type"),
    )
    values = [
        pl.col(column)
        if column in key_columns
        else pl.when(pl.col("stores_content"))
        .then(pl.col(column))
        .otherwise(pl.lit(None, dtype=dtype))
        .alias(column)
        for column, dtype in schema.items()
    ]
    return changed.select(
        "event_type",
        "row_sha256",
        "previous_row_sha256",
        "stores_content",
        *values,
    )


def _missing_events(
    previous: pl.DataFrame,
    current: pl.DataFrame,
    schema: pl.Schema,
    key_columns: Sequence[str],
) -> pl.DataFrame:
    missing = previous.join(
        current.select(list(key_columns)),
        on=list(key_columns),
        how="anti",
        nulls_equal=True,
    )
    values = [
        pl.col(column) if column in key_columns else pl.lit(None, dtype=dtype).alias(column)
        for column, dtype in schema.items()
    ]
    return missing.select(
        pl.lit("missing").alias("event_type"),
        pl.lit(None, dtype=pl.String).alias("row_sha256"),
        pl.col("row_sha256").alias("previous_row_sha256"),
        pl.lit(False).alias("stores_content"),
        *values,
    )


def reconstruct_archive_index(
    *,
    archive_dir: Path,
    run_id: str | None,
) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, Any] | None]:
    """Validate ancestors, retaining keys and hashes but no historical payloads."""
    manifests = _manifest_chain(archive_dir, run_id)
    if not manifests:
        return pl.DataFrame(), pl.DataFrame(), None
    state: pl.DataFrame | None = None
    content: pl.DataFrame | None = None
    for manifest in manifests:
        path = archive_dir / "runs" / manifest["run_id"] / manifest["events_path"]
        if file_sha256(path) != manifest["events_sha256"]:
            raise RuntimeError(f"Raw archive event hash mismatch: {path}")
        keys = manifest["key_columns"]
        events = pl.read_parquet(
            path, columns=[*keys, "row_sha256", "stores_content", "event_type"]
        )
        references = events.filter(pl.col("event_type") != "missing")
        new_content = _content_hashes(references.filter(pl.col("stores_content")))
        content = (
            new_content if content is None else pl.concat([content, new_content], rechunk=False)
        )
        updates = events.select(
            *keys,
            pl.when(pl.col("event_type") == "missing")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("row_sha256"))
            .alias("row_sha256"),
        )
        state = updates if state is None else pl.concat([state, updates])
        state = compact_columns(
            state.unique(subset=keys, keep="last").filter(pl.col("row_sha256").is_not_null())
        )
        requested = _content_hashes(references)
        unresolved = requested.join(
            _known_references(content, requested),
            on="_raw_content_hash",
            how="anti",
        )
        if unresolved.height:
            raise RuntimeError("Raw archive content reference is unresolved")
        if frame_snapshot_sha256(state, keys) != manifest["snapshot_sha256"]:
            raise RuntimeError(f"Raw archive snapshot hash mismatch for run {manifest['run_id']}")
    assert state is not None and content is not None
    return state, content, manifests[-1]


def _content_hashes(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.select(
        pl.when(pl.col("row_sha256").str.contains("^[0-9a-f]{64}$"))
        .then(pl.col("row_sha256").str.decode("hex", strict=False))
        .otherwise(pl.lit(None, dtype=pl.Binary))
        .alias("_raw_content_hash")
    )


def _known_references(content: pl.DataFrame, requested: pl.DataFrame) -> pl.DataFrame:
    """Build lookup tables on this vintage's references, not all historical hashes."""
    return content.join(requested, on="_raw_content_hash", how="semi").unique(
        subset="_raw_content_hash"
    )


def resolve_state_content(
    *,
    archive_dir: Path,
    run_id: str,
    state: pl.DataFrame,
) -> pl.DataFrame:
    """Retain only referenced values, reading each visited file separately."""
    manifests = _manifest_chain(archive_dir, run_id)
    remaining = state.select("row_sha256")
    selected: list[pl.DataFrame] = []
    for manifest in reversed(manifests):
        path = archive_dir / "runs" / manifest["run_id"] / manifest["events_path"]
        stored = (
            pl.read_parquet(path)
            .filter(pl.col("stores_content") & (pl.col("event_type") != "missing"))
            .select([*manifest["data_columns"], "row_sha256"])
        )
        matches = compact_columns(stored.join(remaining, on="row_sha256", how="semi"))
        selected.append(matches)
        remaining = remaining.join(matches.select("row_sha256"), on="row_sha256", how="anti")
        if remaining.is_empty():
            break
    if remaining.height:
        raise RuntimeError("Raw archive content reference is unresolved")
    return pl.concat(selected, how="vertical_relaxed").unique(subset="row_sha256", keep="first")


def compact_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Detach filtered string views from obsolete full-vintage backing buffers."""
    return pl.from_arrow(frame.to_arrow(compat_level=pl.CompatLevel.oldest()))


def _manifest_chain(archive_dir: Path, run_id: str | None) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    seen: set[str] = set()
    cursor = run_id
    while cursor is not None:
        if cursor in seen:
            raise RuntimeError(f"Raw archive parent cycle detected at run {cursor}")
        seen.add(cursor)
        path = archive_dir / "runs" / cursor / "manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("contract") != RAW_DELTA_CONTRACT:
            raise RuntimeError(f"Unsupported raw archive contract: {path}")
        manifests.append(manifest)
        cursor = manifest.get("parent_run_id")
    return list(reversed(manifests))


def _with_row_hashes(frame: pl.DataFrame) -> pl.DataFrame:
    chunks = [
        pl.Series("row_sha256", [row_sha256(frame.columns, row) for row in chunk.iter_rows()])
        for chunk in frame.iter_slices(65_536)
    ]
    hashes = pl.concat(chunks) if chunks else pl.Series("row_sha256", [], dtype=pl.String)
    return frame.with_columns(hashes)


def row_sha256(columns: Sequence[str], row: Sequence[Any]) -> str:
    payload = [[column, _canonical_value(value)] for column, value in zip(columns, row)]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def frame_snapshot_sha256(frame: pl.DataFrame, key_columns: Sequence[str]) -> str:
    """Preserve the exact v1 digest while iterating bounded row batches."""
    digest = hashlib.sha256()
    ordered = frame.select([*key_columns, "row_sha256"]).sort(list(key_columns))
    for row in ordered.iter_rows():
        key = json.dumps([_canonical_value(value) for value in row[:-1]], separators=(",", ":"))
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(row[-1].encode("ascii"))
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


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
