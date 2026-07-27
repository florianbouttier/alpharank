from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.multihorizon.live import (
    _filter_holding_month_universe,
    previous_completed_month,
)


def test_previous_completed_month_handles_year_boundary() -> None:
    assert previous_completed_month(date(2026, 7, 24)) == date(2026, 6, 1)
    assert previous_completed_month(date(2026, 1, 15)) == date(2025, 12, 1)


def test_filter_holding_month_universe_removes_known_exit(tmp_path) -> None:
    constituents_path = tmp_path / "SP500_Constituents.csv"
    pl.DataFrame(
        {
            "Date": [
                date(2026, 6, 1),
                date(2026, 7, 1),
                date(2026, 7, 1),
            ],
            "Ticker": ["EPAM", "WDC", "BRK.B"],
            "Name": ["EPAM", "Western Digital", "Berkshire Hathaway"],
        }
    ).write_csv(constituents_path)
    frame = pl.DataFrame(
        {
            "ticker": ["EPAM.US", "WDC.US", "BRK.B.US"],
            "score": [2.0, 1.0, 0.5],
        }
    )

    filtered, audit = _filter_holding_month_universe(
        frame,
        constituents_path=constituents_path,
        holding_month=date(2026, 7, 1),
    )

    assert filtered["ticker"].to_list() == ["WDC.US", "BRK.B.US"]
    assert audit["applied"] is True
    assert audit["removed_tickers"] == ["EPAM.US"]
