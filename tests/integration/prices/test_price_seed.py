from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from alpharank.data.prices.seed import load_eodhd_seed


def test_load_eodhd_seed_is_hashed_normalized_and_immutable(tmp_path: Path) -> None:
    path = tmp_path / "US_Finalprice.parquet"
    pl.DataFrame(
        {
            "date": [datetime(2020, 1, 2), datetime(2020, 1, 3)],
            "open": [10.0, 11.0],
            "high": [11.0, 12.0],
            "low": [9.0, 10.0],
            "close": [10.5, 11.5],
            "volume": [100.0, 200.0],
            "adjusted_close": [10.0, 11.0],
            "ticker": ["BF-B.US", "BF-B.US"],
        }
    ).write_parquet(path)

    seed = load_eodhd_seed(path, start_date="2020-01-03")

    assert len(seed.sha256) == 64
    assert seed.row_count == 1
    assert seed.ticker_count == 1
    assert seed.frame["ticker"].to_list() == ["BF.B.US"]
    assert seed.frame["source"].to_list() == ["eodhd_frozen_history"]
    assert seed.frame["eodhd_seed_sha256"].to_list() == [seed.sha256]
    assert seed.manifest()["immutable"] is True
