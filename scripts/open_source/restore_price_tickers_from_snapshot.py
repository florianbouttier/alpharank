#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from alpharank.data.open_source.ingestion_frames import _with_price_ingestion_metadata
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    acquire_process_json_lock,
    append_run_delta,
    new_run_id,
    read_json,
    utc_now_iso,
    write_json,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_DIR = PROJECT_ROOT / "data" / "open_source" / "official"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Restore active Yahoo ticker rows from an immutable published snapshot."
    )
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--tickers", nargs="+", required=True)
    parser.add_argument("--official-dir", type=Path, default=DEFAULT_OFFICIAL_DIR)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()

    snapshot_dir = args.snapshot_dir.resolve()
    snapshot_prices_path = snapshot_dir / "lineage" / "prices_open_source.parquet"
    snapshot_manifest_path = snapshot_dir / "snapshot_manifest.json"
    if not snapshot_prices_path.exists() or not snapshot_manifest_path.exists():
        raise FileNotFoundError(
            "The recovery source must be a retained output snapshot with prices and manifest."
        )

    paths = OpenSourceLivePaths(args.official_dir.resolve())
    paths.ensure()
    acquire_process_json_lock(
        paths.manifests_dir / "nightly.lock.json",
        operation="restore_price_tickers_from_snapshot",
    )
    run_id = new_run_id()
    recovered_at = utc_now_iso()
    requested = tuple(sorted({f"{ticker.upper().removesuffix('.US')}.US" for ticker in args.tickers}))
    snapshot_prices = (
        pl.read_parquet(snapshot_prices_path)
        .filter(pl.col("ticker").cast(pl.String).is_in(requested))
        .select(["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"])
        .sort(["ticker", "date"])
    )
    recovered = {
        ticker: snapshot_prices.filter(pl.col("ticker") == ticker)
        for ticker in requested
    }
    missing = [ticker for ticker, frame in recovered.items() if frame.is_empty()]
    if missing:
        raise RuntimeError(f"Snapshot has no recoverable rows for: {missing}")

    raw_path = paths.raw_dir / "prices_yfinance.parquet"
    active = pl.read_parquet(raw_path)
    quarantined = active.filter(pl.col("ticker").cast(pl.String).is_in(requested))
    retained = active.filter(~pl.col("ticker").cast(pl.String).is_in(requested))
    restored = _with_price_ingestion_metadata(
        snapshot_prices,
        dataset="prices_yfinance_restored_from_published_snapshot",
        source="yfinance",
        run_id=run_id,
        ingested_at=recovered_at,
    )
    updated = pl.concat([retained, restored], how="diagonal_relaxed").sort(
        ["ticker", "date", "source"]
    )

    run_dir = paths.run_dir(run_id)
    append_run_delta(run_dir / "quarantine" / "prices_yfinance_replaced.parquet", quarantined)
    append_run_delta(run_dir / "raw" / "prices_yfinance_restored.parquet", restored)
    updated.write_parquet(raw_path)
    prior_manifest = read_json(paths.latest_manifest_path)
    source_manifest = read_json(snapshot_manifest_path)
    write_json(
        run_dir / "manifest.json",
        {
            "run_id": run_id,
            "mode": "restore_price_tickers_from_snapshot",
            "generated_at": recovered_at,
            "reason": args.reason,
            "tickers": requested,
            "source_snapshot": str(snapshot_dir),
            "source_snapshot_run_id": (
                source_manifest.get("run_id") if isinstance(source_manifest, dict) else None
            ),
            "prior_published_run_id": (
                prior_manifest.get("run_id") if isinstance(prior_manifest, dict) else None
            ),
            "quarantined_rows": quarantined.height,
            "restored_rows": restored.height,
            "publication_status": "active_raw_repaired_not_published",
        },
    )
    print(f"Recovery run id: {run_id}")
    print(f"Restored tickers: {', '.join(requested)}")
    print(f"Quarantined active rows: {quarantined.height}")
    print(f"Restored snapshot rows: {restored.height}")
    print("No output package was published; run a validated refresh next.")


if __name__ == "__main__":
    main()
