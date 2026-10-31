from __future__ import annotations

import polars as pl
import pytest

from alpharank.utils.returns import calculate_daily_returns, calculate_monthly_returns


def test_return_calculations_accept_iso_dates_under_polars_two() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["ABC.US", "ABC.US", "ABC.US"],
            "date": ["2026-08-31", "2026-09-30", "2026-10-30"],
            "close": [100.0, 110.0, 121.0],
        }
    )

    daily = calculate_daily_returns(prices)
    monthly = calculate_monthly_returns(prices)

    expected_dates = ["2026-08-31", "2026-09-30", "2026-10-30"]
    assert daily["date"].dt.strftime("%Y-%m-%d").tolist() == expected_dates
    assert daily["daily_return"].iloc[1:].tolist() == pytest.approx([0.1, 0.1])
    assert monthly["date"].dt.strftime("%Y-%m-%d").tolist() == expected_dates
    assert monthly["monthly_return"].iloc[1:].tolist() == pytest.approx([0.1, 0.1])
