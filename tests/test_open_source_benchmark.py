from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.data.open_source.benchmark import build_price_alignment


def test_build_price_alignment_accepts_mixed_date_types() -> None:
    eodhd_prices = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": [date(2026, 5, 6)],
            "adjusted_close": [100.0],
            "close": [100.0],
            "open": [99.0],
            "high": [101.0],
            "low": [98.0],
            "volume": [1_000],
        }
    )
    open_prices = pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": ["2026-05-06"],
            "adjusted_close": [100.0],
            "close": [100.0],
            "open": [99.0],
            "high": [101.0],
            "low": [98.0],
            "volume": [1_000],
        }
    )

    aligned = build_price_alignment(eodhd_prices, open_prices)

    assert aligned["match_status"].to_list() == ["matched"]
    assert aligned["date"].to_list() == [date(2026, 5, 6)]
