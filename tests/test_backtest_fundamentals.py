from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

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
    assert "price_to_sales" in features.columns
    assert "ev_to_ebitda" in features.columns
