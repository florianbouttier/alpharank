from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.data.open_source.price_quality import (
    assert_no_extreme_adjusted_price_moves,
    find_extreme_adjusted_price_moves,
)
from alpharank.data.open_source.storage import merge_upsert_frames


def test_price_quality_flags_partial_split_scale() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["MNST.US"] * 4,
            "date": [
                date(2026, 7, 30),
                date(2026, 7, 31),
                date(2026, 8, 7),
                date(2026, 8, 10),
            ],
            "adjusted_close": [97.65, 48.19, 45.18, 91.43],
        }
    )

    findings = find_extreme_adjusted_price_moves(
        prices,
        event_since="2026-07-31",
        tickers=["MNST.US"],
    )

    assert findings["date"].to_list() == [date(2026, 7, 31), date(2026, 8, 10)]
    with pytest.raises(RuntimeError, match="discontinuities"):
        assert_no_extreme_adjusted_price_moves(
            prices,
            event_since="2026-07-31",
            tickers=["MNST.US"],
        )


def test_price_quality_ignores_moves_before_refresh_window() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US", "A.US"],
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2026, 8, 10)],
            "adjusted_close": [100.0, 50.0, 52.0],
        }
    )
    findings = find_extreme_adjusted_price_moves(prices, event_since="2026-08-01")
    assert findings.is_empty()


def test_merge_upsert_frames_has_no_persistence_side_effect() -> None:
    existing = pl.DataFrame(
        {"ticker": ["A.US"], "date": [date(2026, 8, 7)], "value": [1.0], "seq": [1]}
    )
    delta = pl.DataFrame(
        {"ticker": ["A.US"], "date": [date(2026, 8, 7)], "value": [2.0], "seq": [2]}
    )
    merged = merge_upsert_frames(
        existing,
        delta,
        key_cols=["ticker", "date"],
        order_cols=["seq"],
    )
    assert existing["value"].to_list() == [1.0]
    assert merged["value"].to_list() == [2.0]
