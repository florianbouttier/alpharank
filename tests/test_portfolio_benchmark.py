from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.portfolio.benchmark import (
    SPY_PRICE_RETURN,
    SPY_TOTAL_RETURN,
    monthly_benchmark_returns,
)
from alpharank.data.processing import PricesDataPreprocessor
from alpharank.utils.frame_backend import to_polars


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


def test_asset_and_benchmark_return_conventions_match() -> None:
    assets = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "AAA.US"],
            "date": [
                date(2020, 1, 31),
                date(2020, 2, 27),
                date(2020, 3, 31),
            ],
            "close": [100.0, 110.0, 121.0],
            "adjusted_close": [100.0, 110.0, 120.0],
        }
    )
    benchmark = pl.DataFrame(
        {
            "ticker": ["SPY.US", "SPY.US", "SPY.US"],
            "date": [
                date(2020, 1, 31),
                date(2020, 2, 28),
                date(2020, 3, 31),
            ],
            "close": [100.0, 103.0, 106.0],
            "adjusted_close": [100.0, 105.0, 110.0],
        }
    )

    aligned = to_polars(
        PricesDataPreprocessor.prices_vs_index(
            index=benchmark,
            prices=assets,
            column_close_index="adjusted_close",
            column_close_prices="adjusted_close",
            backend="polars",
        )
    ).with_columns(pl.col("date").cast(pl.Date)).sort("date")

    assert aligned["date"].to_list() == [date(2020, 1, 31), date(2020, 3, 31)]
    assert aligned["adjusted_close_index"].to_list() == [100.0, 110.0]
    assert aligned["dr_vs_index"][1] == pytest.approx((120.0 / 110.0) - 1.0)
    assert date(2020, 2, 27) not in aligned["date"].to_list()
    assert date(2020, 2, 28) not in aligned["date"].to_list()
