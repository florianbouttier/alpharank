from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.storage import upsert_parquet


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
