from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl

from alpharank.data.ingestion.prices import (
    _load_reference_tickers as load_ingestion_reference_tickers,
)
from alpharank.data.ingestion.transition import (
    _load_reference_tickers as load_transition_reference_tickers,
)


def test_load_reference_tickers_accepts_date_typed_price_column(tmp_path: Path) -> None:
    reference_dir = tmp_path / "reference"
    reference_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "ticker": ["AAPL.US", "MSFT.US", "OLD.US"],
            "date": [date(2026, 4, 25), date(2026, 4, 24), date(2024, 12, 31)],
        }
    ).write_parquet(reference_dir / "US_Finalprice.parquet")

    expected = ("AAPL", "MSFT")
    assert load_ingestion_reference_tickers(reference_dir, start_date="2025-01-01") == expected
    assert load_transition_reference_tickers(reference_dir, start_date="2025-01-01") == expected
