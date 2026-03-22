from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl


@dataclass(frozen=True)
class OpenSourceLivePaths:
    base_dir: Path
    audit_root_dir: Path | None = None

    @property
    def root_dir(self) -> Path:
        return self.base_dir.parent

    @property
    def raw_dir(self) -> Path:
        return self.base_dir / "raw"

    @property
    def target_dir(self) -> Path:
        return self.base_dir / "target"

    @property
    def clean_dir(self) -> Path:
        return self.target_dir

    @property
    def legacy_dir(self) -> Path:
        return self.target_dir / "legacy_compatible"

    @property
    def audit_dir(self) -> Path:
        return self.audit_root_dir or (self.root_dir / "audit")

    @property
    def manifests_dir(self) -> Path:
        return self.base_dir / "manifests"

    @property
    def runs_dir(self) -> Path:
        return self.base_dir / "runs"

    @property
    def latest_manifest_path(self) -> Path:
        return self.manifests_dir / "latest_run.json"

    def ensure(self) -> None:
        for directory in (
            self.root_dir,
            self.raw_dir,
            self.target_dir,
            self.legacy_dir,
            self.audit_dir,
            self.manifests_dir,
            self.runs_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / run_id


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_run_manifest(paths: OpenSourceLivePaths, run_id: str, manifest: dict[str, Any]) -> Path:
    run_manifest_path = paths.run_dir(run_id) / "manifest.json"
    write_json(run_manifest_path, manifest)
    write_json(paths.latest_manifest_path, manifest)
    return run_manifest_path


def append_run_delta(path: Path, frame: pl.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(path)


def upsert_parquet(
    path: Path,
    frame: pl.DataFrame,
    *,
    key_cols: Iterable[str],
    order_cols: Iterable[str] = (),
) -> pl.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    key_list = list(key_cols)
    order_list = list(order_cols)

    if path.exists():
        existing = pl.read_parquet(path)
        merged = pl.concat([existing, frame], how="diagonal_relaxed")
    else:
        merged = frame

    if merged.is_empty():
        merged.write_parquet(path)
        return merged

    sort_cols = [column for column in [*key_list, *order_list] if column in merged.columns]
    if sort_cols:
        merged = merged.sort(sort_cols)
    if key_list:
        merged = merged.unique(subset=key_list, keep="last", maintain_order=True)
        merged = merged.sort([column for column in key_list if column in merged.columns])
    merged.write_parquet(path)
    return merged


def coerce_schema(frame: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    expressions: list[pl.Expr] = []
    for column, dtype in schema.items():
        if column in frame.columns:
            expressions.append(pl.col(column).cast(dtype, strict=False))
        else:
            expressions.append(pl.lit(None).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(schema))
