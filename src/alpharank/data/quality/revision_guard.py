from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Mapping

import polars as pl


_DATASETS: dict[str, tuple[str, tuple[str, ...]]] = {
    "income_statement": (
        "US_Income_statement.parquet",
        ("ticker", "date", "filing_date"),
    ),
    "balance_sheet": (
        "US_Balance_sheet.parquet",
        ("ticker", "date", "filing_date"),
    ),
    "cash_flow": (
        "US_Cash_flow.parquet",
        ("ticker", "date", "filing_date"),
    ),
    "shares": ("US_share.parquet", ("ticker", "date")),
    "earnings": ("US_Earnings.parquet", ("ticker", "date", "reportDate")),
}


def audit_historical_revisions(
    *,
    previous_output_dir: Path,
    candidate_paths: Mapping[str, Path],
    expected_through: str,
    guard_days: int,
) -> dict[str, object]:
    cutoff = date.fromisoformat(expected_through) - timedelta(days=guard_days)
    datasets: dict[str, dict[str, object]] = {}
    for name, (file_name, keys) in _DATASETS.items():
        previous_path = previous_output_dir / file_name
        candidate_path = candidate_paths.get(file_name)
        if candidate_path is None or not previous_path.exists():
            continue
        previous = _historical_rows(pl.read_parquet(previous_path), cutoff=cutoff)
        candidate = _historical_rows(pl.read_parquet(candidate_path), cutoff=cutoff)
        datasets[name] = _compare_keyed_rows(previous, candidate, keys=keys)

    blocked_datasets = [
        name
        for name, result in datasets.items()
        if any(int(result[field]) > 0 for field in ("added_rows", "removed_rows", "changed_common_rows"))
    ]
    return {
        "guard_version": 1,
        "cutoff_date": cutoff.isoformat(),
        "guard_days": guard_days,
        "datasets": datasets,
        "blocked_datasets": blocked_datasets,
        "historical_revisions_detected": bool(blocked_datasets),
    }


def _historical_rows(frame: pl.DataFrame, *, cutoff: date) -> pl.DataFrame:
    if frame.is_empty() or "date" not in frame.columns:
        return frame
    return frame.filter(
        pl.col("date").cast(pl.String, strict=False).str.slice(0, 10)
        <= cutoff.isoformat()
    )


def _compare_keyed_rows(
    previous: pl.DataFrame,
    candidate: pl.DataFrame,
    *,
    keys: tuple[str, ...],
) -> dict[str, object]:
    if previous.columns != candidate.columns:
        return {
            "previous_rows": previous.height,
            "candidate_rows": candidate.height,
            "added_rows": candidate.height,
            "removed_rows": previous.height,
            "changed_common_rows": 0,
            "schema_changed": True,
            "previous_columns": previous.columns,
            "candidate_columns": candidate.columns,
        }
    _require_unique(previous, keys=keys, label="previous")
    _require_unique(candidate, keys=keys, label="candidate")
    previous_hashes = previous.select(
        *keys,
        pl.struct(previous.columns).hash(seed=42).alias("previous_hash"),
    )
    candidate_hashes = candidate.select(
        *keys,
        pl.struct(candidate.columns).hash(seed=42).alias("candidate_hash"),
    )
    added = candidate_hashes.join(previous_hashes, on=list(keys), how="anti")
    removed = previous_hashes.join(candidate_hashes, on=list(keys), how="anti")
    changed = (
        previous_hashes.join(candidate_hashes, on=list(keys), how="inner")
        .filter(pl.col("previous_hash") != pl.col("candidate_hash"))
        .select(*keys)
    )
    return {
        "previous_rows": previous.height,
        "candidate_rows": candidate.height,
        "added_rows": added.height,
        "removed_rows": removed.height,
        "changed_common_rows": changed.height,
        "schema_changed": False,
        "added_key_examples": added.select(keys).sort(list(keys)).head(20).to_dicts(),
        "removed_key_examples": removed.select(keys).sort(list(keys)).head(20).to_dicts(),
        "changed_key_examples": changed.sort(list(keys)).head(20).to_dicts(),
    }


def _require_unique(frame: pl.DataFrame, *, keys: tuple[str, ...], label: str) -> None:
    if frame.is_empty():
        return
    unique_count = frame.select(pl.struct(keys).n_unique()).item()
    if unique_count != frame.height:
        raise RuntimeError(
            f"Historical revision guard found {frame.height - unique_count} duplicate "
            f"{label} natural keys: {keys}"
        )
