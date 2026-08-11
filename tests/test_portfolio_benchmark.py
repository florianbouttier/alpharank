from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.benchmark import (
    SPY_PRICE_RETURN,
    SPY_TOTAL_RETURN,
    monthly_benchmark_returns,
)


def test_benchmark_convention_separates_price_and_total_returns() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["SPY.US"] * 4,
            "date": [
                date(2020, 1, 30),
                date(2020, 1, 31),
                date(2020, 2, 27),
                date(2020, 2, 28),
            ],
            "close": [99.0, 100.0, 101.0, 102.0],
            "adjusted_close": [98.0, 100.0, 102.0, 103.0],
        }
    )

    total = monthly_benchmark_returns(prices, convention=SPY_TOTAL_RETURN)
    price = monthly_benchmark_returns(prices, convention=SPY_PRICE_RETURN)

    assert total["monthly_return"][1] == pytest.approx(0.03)
    assert price["monthly_return"][1] == pytest.approx(0.02)
    assert total["benchmark_price_column"].unique().to_list() == ["adjusted_close"]
    assert total["benchmark_includes_distributions"].unique().to_list() == [True]


def test_benchmark_rejects_multiple_tickers() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["SPY.US", "IVV.US"],
            "date": [date(2020, 1, 31)] * 2,
            "adjusted_close": [100.0, 100.0],
        }
    )

    with pytest.raises(ValueError, match="one ticker"):
        monthly_benchmark_returns(prices)
