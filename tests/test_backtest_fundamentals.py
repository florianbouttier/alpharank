from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl
import pytest

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.fundamentals import build_monthly_fundamental_features


def test_backtest_fundamentals_keep_ratios_and_growth_not_raw_dollars() -> None:
    quarter_dates = [
        date(2017, 3, 31),
        date(2017, 6, 30),
        date(2017, 9, 30),
        date(2017, 12, 31),
        date(2018, 3, 31),
        date(2018, 6, 30),
        date(2018, 9, 30),
        date(2018, 12, 31),
        date(2019, 3, 31),
        date(2019, 6, 30),
        date(2019, 9, 30),
        date(2019, 12, 31),
        date(2020, 3, 31),
        date(2020, 6, 30),
        date(2020, 9, 30),
        date(2020, 12, 31),
    ]

    def _quarter_df(value_name: str, values: list[float]) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "ticker": ["AAA.US"] * len(quarter_dates),
                "date": quarter_dates,
                "filing_date": quarter_dates,
                value_name: values,
            }
        )

    income = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "filing_date": quarter_dates,
            "totalRevenue": [100 + 5 * i for i in range(len(quarter_dates))],
            "netIncome": [10 + i for i in range(len(quarter_dates))],
            "ebitda": [15 + i for i in range(len(quarter_dates))],
            "ebit": [12 + i for i in range(len(quarter_dates))],
            "grossProfit": [30 + 2 * i for i in range(len(quarter_dates))],
        }
    )
    balance = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "filing_date": quarter_dates,
            "commonStockSharesOutstanding": [100.0] * len(quarter_dates),
            "totalStockholderEquity": [200 + 3 * i for i in range(len(quarter_dates))],
            "netDebt": [50 + i for i in range(len(quarter_dates))],
            "totalAssets": [400 + 5 * i for i in range(len(quarter_dates))],
            "cashAndShortTermInvestments": [20 + i for i in range(len(quarter_dates))],
        }
    )
    cash = _quarter_df("freeCashFlow", [8 + i for i in range(len(quarter_dates))])
    earnings = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "reportDate": quarter_dates,
            "epsActual": [0.5 + 0.05 * i for i in range(len(quarter_dates))],
        }
    )
    monthly_prices = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "date": [date(2021, 1, 29), date(2021, 2, 26)],
            "year_month": [date(2021, 1, 1), date(2021, 2, 1)],
            "last_close": [50.0, 55.0],
        }
    )

    features = build_monthly_fundamental_features(
        monthly_prices=monthly_prices,
        balance_sheet=balance,
        income_statement=income,
        cash_flow=cash,
        earnings=earnings,
    )

    assert "market_cap" not in features.columns
    assert "enterprise_value" not in features.columns
    assert "total_revenue_ttm" not in features.columns
    assert "net_income_ttm" not in features.columns
    assert "ebitda_ttm" not in features.columns
    assert "free_cashflow_ttm" not in features.columns
    assert "shares_outstanding_avg4q" not in features.columns

    expected_growth_cols = [
        "total_revenue_ttm_growth_1q",
        "total_revenue_ttm_growth_4q",
        "total_revenue_ttm_growth_12q",
        "net_income_ttm_growth_1q",
        "ebitda_ttm_growth_4q",
        "free_cashflow_ttm_growth_12q",
        "eps_actual_ttm_growth_1q",
    ]
    for col in expected_growth_cols:
        assert col in features.columns

    assert "net_margin_ttm" in features.columns
    assert "sales_yield" in features.columns
    assert "ebitda_to_ev" in features.columns
    assert "earnings_yield" in features.columns
    assert "share_dilution_1q" in features.columns


def test_backtest_fundamentals_use_only_reported_data_as_of_month_end() -> None:
    quarter_dates = [
        date(2019, 6, 30),
        date(2019, 9, 30),
        date(2019, 12, 31),
        date(2020, 3, 31),
        date(2020, 6, 30),
    ]
    filing_dates = [
        date(2019, 8, 1),
        date(2019, 11, 1),
        date(2020, 2, 1),
        date(2020, 5, 1),
        date(2020, 8, 15),
    ]

    income = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "filing_date": filing_dates,
            "totalRevenue": [100.0, 110.0, 120.0, 130.0, 200.0],
            "netIncome": [10.0, 11.0, 12.0, 13.0, 40.0],
            "ebitda": [15.0, 16.0, 17.0, 18.0, 45.0],
            "ebit": [12.0, 13.0, 14.0, 15.0, 38.0],
            "grossProfit": [30.0, 32.0, 34.0, 36.0, 70.0],
        }
    )
    balance = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "filing_date": filing_dates,
            "commonStockSharesOutstanding": [100.0] * len(quarter_dates),
            "totalStockholderEquity": [200.0, 205.0, 210.0, 215.0, 260.0],
            "netDebt": [50.0, 50.0, 49.0, 48.0, 45.0],
            "totalAssets": [400.0, 410.0, 420.0, 430.0, 500.0],
            "cashAndShortTermInvestments": [20.0, 20.0, 21.0, 22.0, 30.0],
        }
    )
    cash = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "filing_date": filing_dates,
            "freeCashFlow": [8.0, 9.0, 10.0, 11.0, 28.0],
        }
    )
    earnings = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(quarter_dates),
            "date": quarter_dates,
            "reportDate": filing_dates,
            "epsActual": [0.5, 0.6, 0.7, 0.8, 1.5],
        }
    )
    monthly_prices = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "date": [date(2020, 7, 31), date(2020, 8, 31)],
            "year_month": [date(2020, 7, 1), date(2020, 8, 1)],
            "last_close": [50.0, 52.0],
        }
    )

    features = build_monthly_fundamental_features(
        monthly_prices=monthly_prices,
        balance_sheet=balance,
        income_statement=income,
        cash_flow=cash,
        earnings=earnings,
    ).sort("year_month")

    july_row, august_row = features.to_dicts()

    assert july_row["net_margin_ttm"] == pytest.approx(46.0 / 460.0)
    assert august_row["net_margin_ttm"] == pytest.approx(76.0 / 560.0)
    assert july_row["earnings_yield"] == pytest.approx(2.6 / 50.0)
    assert august_row["earnings_yield"] == pytest.approx(3.6 / 52.0)
    assert july_row["net_margin_ttm"] < august_row["net_margin_ttm"]
