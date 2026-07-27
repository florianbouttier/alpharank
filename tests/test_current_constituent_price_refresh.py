from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.data.open_source.constituents import (
    current_constituent_price_coverage,
)


def test_current_constituent_coverage_preserves_class_share_dots(tmp_path) -> None:
    constituents_path = tmp_path / "SP500_Constituents.csv"
    pl.DataFrame(
        {
            "Date": [date(2026, 7, 1), date(2026, 7, 1)],
            "Ticker": ["BRK.B", "BF.B"],
            "Name": ["Berkshire Hathaway", "Brown-Forman"],
        }
    ).write_csv(constituents_path)
    prices = pl.DataFrame(
        {
            "ticker": ["BRK.B.US", "BF.B.US"],
            "date": [date(2026, 7, 23), date(2026, 7, 24)],
            "adjusted_close": [500.0, 30.0],
        }
    )

    summary, coverage = current_constituent_price_coverage(
        prices,
        constituents_path=constituents_path,
    )

    assert summary["member_count"] == 2
    assert summary["missing_price_count"] == 0
    assert summary["latest_common_price_date"] == "2026-07-23"
    assert set(coverage["ticker"]) == {"BRK.B.US", "BF.B.US"}
