from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from alpharank.data.prices.contracts import EODHD_SOURCE


PERSISTENT_PRICE_HISTORY_POLICY_ID = "published_price_history_v1"


@dataclass(frozen=True)
class PersistentPriceHistorySource:
    lineage_path: Path
    price_manifest_path: Path
    snapshot_dir: Path
    composition_id: str


def resolve_previous_validated_price_lineage(
    latest_composed_manifest_path: Path,
) -> PersistentPriceHistorySource:
    """Resolve the price lineage retained by the current composed snapshot."""

    pointer_path = latest_composed_manifest_path.expanduser().resolve()
    pointer = _read_json(pointer_path)
    snapshot_dir = _resolve_recorded_path(pointer.get("snapshot_dir"), pointer_path)
    composition_id = str(pointer.get("composition_id") or "")
    if not composition_id:
        raise RuntimeError(f"Composed snapshot pointer has no composition id: {pointer_path}")

    snapshot_manifest_path = snapshot_dir / "lineage" / "manifest.json"
    snapshot_manifest = _read_json(snapshot_manifest_path)
    if snapshot_manifest.get("composition_id") != composition_id:
        raise RuntimeError("Latest pointer and composed snapshot composition ids differ")
    if snapshot_manifest.get("validation", {}).get("passed") is not True:
        raise RuntimeError("Latest composed snapshot is not marked valid")

    price_lineage_dir = snapshot_dir / "lineage" / "prices"
    price_manifest_path = price_lineage_dir / "manifest.json"
    price_manifest = _read_json(price_manifest_path)
    gate = price_manifest.get("source_refresh_contract", {}).get(
        "price_revision_guard", {}
    )
    if gate.get("passed") is not True:
        raise RuntimeError("Latest composed snapshot price gate is not marked valid")

    lineage_path = price_lineage_dir / "prices_open_source_lineage.parquet"
    if not lineage_path.is_file():
        raise FileNotFoundError(
            f"Latest composed snapshot has no retained price lineage: {lineage_path}"
        )
    expected = price_manifest.get("artifacts", {}).get("price_lineage", {}).get(
        "sha256"
    )
    if expected and _sha256(lineage_path) != expected:
        raise RuntimeError("Retained price lineage hash does not match its manifest")
    return PersistentPriceHistorySource(
        lineage_path=lineage_path,
        price_manifest_path=price_manifest_path,
        snapshot_dir=snapshot_dir,
        composition_id=composition_id,
    )


def build_persistent_price_history_registry(
    lineage: pl.DataFrame,
    *,
    active_tickers: Sequence[str],
    preserved_terminal_tickers: Sequence[str] = (),
) -> pl.DataFrame:
    """Describe how every published ticker will persist into the next refresh."""

    required = {
        "ticker",
        "date",
        "source",
        "source_vintage_id",
        "ingestion_run_id",
        "ingested_at",
    }
    missing = required - set(lineage.columns)
    if missing:
        raise ValueError(f"Price lineage is missing registry columns: {sorted(missing)}")
    active = sorted({_normalize_ticker(ticker) for ticker in active_tickers})
    terminal = sorted(
        {_normalize_ticker(ticker) for ticker in preserved_terminal_tickers}
    )
    normalized = lineage.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.String),
    )
    return (
        normalized.group_by("ticker")
        .agg(
            pl.len().alias("row_count"),
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            (pl.col("source") == EODHD_SOURCE).any().alias("has_eodhd_seed"),
            (pl.col("source") != EODHD_SOURCE)
            .any()
            .alias("has_open_source_history"),
            pl.col("source")
            .drop_nulls()
            .unique()
            .sort()
            .str.join("|")
            .alias("sources"),
            pl.col("ingestion_run_id")
            .sort_by("date")
            .last()
            .alias("latest_ingestion_run_id"),
            pl.col("source_vintage_id")
            .sort_by("date")
            .last()
            .alias("latest_source_vintage_id"),
            pl.col("ingested_at").max().alias("latest_ingested_at"),
        )
        .with_columns(
            pl.col("ticker").is_in(active).alias("current_active"),
            pl.col("ticker").is_in(terminal).alias("terminal_carry_forward"),
            pl.lit(PERSISTENT_PRICE_HISTORY_POLICY_ID).alias("persistence_policy_id"),
        )
        .with_columns(
            pl.when(pl.col("terminal_carry_forward"))
            .then(pl.lit("terminal_carry_forward"))
            .when(pl.col("current_active"))
            .then(pl.lit("active_refreshed"))
            .when(pl.col("has_eodhd_seed"))
            .then(pl.lit("inactive_eodhd_seeded"))
            .otherwise(pl.lit("inactive_open_source_only"))
            .alias("persistence_class")
        )
        .sort("ticker")
    )


def persistent_history_summary(registry: pl.DataFrame) -> dict[str, Any]:
    """Return stable manifest counters for a persistent-history registry."""

    counts = {
        row["persistence_class"]: int(row["len"])
        for row in registry.group_by("persistence_class").len().to_dicts()
    }
    open_only = registry.filter(
        pl.col("persistence_class") == "inactive_open_source_only"
    )
    non_eodhd_persisted = registry.filter(
        (~pl.col("current_active") | pl.col("terminal_carry_forward"))
        & ~pl.col("has_eodhd_seed")
    )
    return {
        "policy_id": PERSISTENT_PRICE_HISTORY_POLICY_ID,
        "ticker_count": registry.height,
        "row_count": int(registry.get_column("row_count").sum()),
        "persistence_class_counts": counts,
        "inactive_open_source_only_tickers": open_only.get_column("ticker").to_list(),
        "non_eodhd_persisted_tickers": non_eodhd_persisted.get_column(
            "ticker"
        ).to_list(),
    }


def _resolve_recorded_path(value: object, pointer_path: Path) -> Path:
    if not value:
        raise RuntimeError(f"Composed snapshot pointer has no snapshot_dir: {pointer_path}")
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = pointer_path.parent / path
    return path.resolve()


def _normalize_ticker(ticker: str) -> str:
    value = str(ticker).strip().upper()
    return value if value.endswith(".US") else f"{value}.US"


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Manifest must be a JSON object: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
