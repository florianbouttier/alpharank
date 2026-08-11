from __future__ import annotations

import pandas as pd
import pytest

from alpharank.data.processing import FundamentalProcessor


def _statement_frames(reverse: bool = False) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = [
        {
            "ticker": "AAA.US",
            "date": "2020-03-31",
            "filing_date": "2020-05-01",
            "commonStockSharesOutstanding": 10.0,
            "totalStockholderEquity": 100.0,
            "netDebt": 0.0,
            "totalAssets": 100.0,
            "cashAndShortTermInvestments": 0.0,
        },
        {
            "ticker": "AAA.US",
            "date": "2020-06-30",
            "filing_date": "2020-05-01",
            "commonStockSharesOutstanding": 20.0,
            "totalStockholderEquity": 200.0,
            "netDebt": 0.0,
            "totalAssets": 200.0,
            "cashAndShortTermInvestments": 0.0,
        },
        {
            "ticker": "NUL.US",
            "date": "2020-03-31",
            "filing_date": None,
            "commonStockSharesOutstanding": 10.0,
            "totalStockholderEquity": 100.0,
            "netDebt": 0.0,
            "totalAssets": 100.0,
            "cashAndShortTermInvestments": 0.0,
        },
    ]
    if reverse:
        rows = list(reversed(rows))

    balance = pd.DataFrame(rows)
    revenue_by_key = {
        ("AAA.US", "2020-03-31"): 100.0,
        ("AAA.US", "2020-06-30"): 500.0,
        ("NUL.US", "2020-03-31"): 100.0,
    }
    net_income_by_key = {
        ("AAA.US", "2020-03-31"): 10.0,
        ("AAA.US", "2020-06-30"): 20.0,
        ("NUL.US", "2020-03-31"): 10.0,
    }
    income = pd.DataFrame(
        [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "filing_date": row["filing_date"],
                "totalRevenue": revenue_by_key[(row["ticker"], row["date"])],
                "grossProfit": revenue_by_key[(row["ticker"], row["date"])] / 2,
                "operatingIncome": net_income_by_key[(row["ticker"], row["date"])],
                "incomeBeforeTax": net_income_by_key[(row["ticker"], row["date"])],
                "netIncome": net_income_by_key[(row["ticker"], row["date"])],
                "ebit": net_income_by_key[(row["ticker"], row["date"])],
                "ebitda": net_income_by_key[(row["ticker"], row["date"])],
            }
            for row in rows
        ]
    )
    free_cash_flow_by_key = {
        ("AAA.US", "2020-03-31"): 10.0,
        ("AAA.US", "2020-06-30"): 20.0,
        ("NUL.US", "2020-03-31"): 10.0,
    }
    cashflow = pd.DataFrame(
        [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "filing_date": row["filing_date"],
                "freeCashFlow": free_cash_flow_by_key[(row["ticker"], row["date"])],
            }
            for row in rows
        ]
    )
    eps_by_key = {
        ("AAA.US", "2020-03-31"): 1.0,
        ("AAA.US", "2020-06-30"): 2.0,
        ("NUL.US", "2020-03-31"): 1.0,
    }
    earnings = pd.DataFrame(
        [
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "reportDate": row["filing_date"],
                "epsActual": eps_by_key[(row["ticker"], row["date"])],
            }
            for row in rows
        ]
    )
    return balance, income, cashflow, earnings


def _ratios(reverse: bool = False) -> pd.DataFrame:
    balance, income, cashflow, earnings = _statement_frames(reverse=reverse)
    monthly_return = pd.DataFrame(
        [
            {"ticker": "AAA.US", "date": "2020-05-31", "last_close": 10.0},
            {"ticker": "NUL.US", "date": "2020-05-31", "last_close": 10.0},
        ]
    )
    out = FundamentalProcessor.calculate_pe_ratios(
        balance=balance,
        earnings=earnings,
        cashflow=cashflow,
        income=income,
        earning_choice="netincome_rolling",
        monthly_return=monthly_return,
        list_date_to_maximise=["filing_date_income", "filing_date_balance"],
        backend="polars",
    )
    out["year_month"] = out["year_month"].astype(str)
    return out.sort_values(["ticker", "year_month"]).reset_index(drop=True)


def test_pe_ratios_are_deterministic_for_same_day_fundamental_filings() -> None:
    forward = _ratios(reverse=False)
    reversed_input = _ratios(reverse=True)

    pd.testing.assert_frame_equal(forward, reversed_input)

    assert forward["ticker"].to_list() == ["AAA.US"]
    row = forward.iloc[0]
    assert row["year_month"] == "2020-05"
    assert row["market_cap"] == pytest.approx(150.0)
    assert row["pe"] == pytest.approx(2.5)
