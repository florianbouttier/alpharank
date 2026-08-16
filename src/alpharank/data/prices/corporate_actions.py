"""Versioned, reviewed corporate-action evidence for price reconciliation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl


SPLIT_SCHEMA = {
    "ticker": pl.String,
    "date": pl.String,
    "split_ratio": pl.Float64,
    "source": pl.String,
    "source_url": pl.String,
}


def load_confirmed_stock_splits(
    registry_path: Path,
    *,
    tickers: Iterable[str] | None = None,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    selected = (
        {str(ticker).upper() for ticker in tickers} if tickers is not None else None
    )
    rows = []
    for event in payload.get("events", []):
        ticker = str(event.get("ticker", "")).upper()
        if event.get("action") != "stock_split":
            continue
        if selected is not None and ticker not in selected:
            continue
        rows.append(
            {
                "ticker": ticker,
                "date": str(event["effective_date"]),
                "split_ratio": float(event["split_ratio"]),
                "source": str(event["source"]),
                "source_url": str(event["source_url"]),
            }
        )
    frame = pl.DataFrame(rows, schema=SPLIT_SCHEMA) if rows else pl.DataFrame(schema=SPLIT_SCHEMA)
    digest = hashlib.sha256(registry_path.read_bytes()).hexdigest()
    return frame.sort(["ticker", "date"]), {
        "registry_id": payload.get("registry_id"),
        "path": str(registry_path.resolve()),
        "sha256": digest,
        "selected_event_count": frame.height,
    }


def combine_stock_split_evidence(
    *frames: pl.DataFrame,
) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return pl.DataFrame(schema=SPLIT_SCHEMA)
    normalized = []
    for frame in non_empty:
        normalized.append(
            frame.with_columns(
                pl.col("ticker").cast(pl.String).str.to_uppercase(),
                pl.col("date").cast(pl.String),
                pl.col("split_ratio").cast(pl.Float64),
                (
                    pl.col("source_url").cast(pl.String)
                    if "source_url" in frame.columns
                    else pl.lit(None).cast(pl.String)
                ).alias("source_url"),
            ).select(*SPLIT_SCHEMA)
        )
    return (
        pl.concat(normalized, how="vertical_relaxed")
        .with_columns(pl.col("source_url").is_not_null().alias("_reviewed"))
        .sort(["ticker", "date", "_reviewed"])
        .unique(subset=["ticker", "date", "split_ratio"], keep="last")
        .drop("_reviewed")
        .sort(["ticker", "date"])
    )
