from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from alpharank.data.open_source.storage import release_json_lock, try_acquire_json_lock, upsert_parquet


def test_upsert_parquet_replaces_overlapping_keys_with_latest_row(tmp_path: Path) -> None:
    path = tmp_path / "prices.parquet"
    first = pl.DataFrame(
        {
            "ticker": ["AAPL.US", "MSFT.US"],
            "date": ["2025-01-01", "2025-01-01"],
            "adjusted_close": [1.0, 2.0],
            "source": ["yfinance", "yfinance"],
            "ingested_at": ["2026-03-22T10:00:00+00:00", "2026-03-22T10:00:00+00:00"],
        }
    )
    second = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [3.0],
            "source": ["yfinance"],
            "ingested_at": ["2026-03-22T11:00:00+00:00"],
        }
    )

    upsert_parquet(path, first, key_cols=["ticker", "date", "source"], order_cols=["ingested_at"])
    merged = upsert_parquet(path, second, key_cols=["ticker", "date", "source"], order_cols=["ingested_at"])

    assert merged.height == 2
    assert merged.filter(pl.col("ticker") == "AAPL.US")["adjusted_close"].to_list() == [3.0]


def test_json_lock_blocks_live_pid_and_can_be_released(tmp_path: Path) -> None:
    lock_path = tmp_path / "nightly.lock.json"
    acquired, existing = try_acquire_json_lock(lock_path, {"pid": os.getpid(), "started_at": "2026-03-23T23:00:00"})
    assert acquired is True
    assert existing is None

    acquired_again, existing_again = try_acquire_json_lock(lock_path, {"pid": os.getpid(), "started_at": "2026-03-23T23:05:00"})
    assert acquired_again is False
    assert existing_again is not None
    assert existing_again["pid"] == os.getpid()

    release_json_lock(lock_path)
    acquired_after_release, existing_after_release = try_acquire_json_lock(
        lock_path,
        {"pid": os.getpid(), "started_at": "2026-03-23T23:10:00"},
    )
    assert acquired_after_release is True
    assert existing_after_release is None


def test_json_lock_reclaims_stale_pid(tmp_path: Path) -> None:
    lock_path = tmp_path / "nightly.lock.json"
    lock_path.write_text('{"pid": 999999, "started_at": "2026-03-23T22:00:00"}', encoding="utf-8")

    acquired, existing = try_acquire_json_lock(lock_path, {"pid": os.getpid(), "started_at": "2026-03-23T23:00:00"})

    assert acquired is True
    assert existing is None
