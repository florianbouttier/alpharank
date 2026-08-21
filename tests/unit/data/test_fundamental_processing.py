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

    assert forward.empty


def test_ttm_requires_four_distinct_quarters() -> None:
    dates = ["2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31"]
    filing_dates = ["2020-05-01", "2020-08-01", "2020-11-01", "2021-02-01"]
    base = {
        "ticker": ["AAA.US"] * 4,
        "date": dates,
        "filing_date": filing_dates,
    }
    balance = pd.DataFrame(
        base
        | {
            "commonStockSharesOutstanding": [10.0] * 4,
            "totalStockholderEquity": [100.0] * 4,
            "netDebt": [0.0] * 4,
            "totalAssets": [100.0] * 4,
            "cashAndShortTermInvestments": [0.0] * 4,
        }
    )
    income = pd.DataFrame(
        base
        | {
            "totalRevenue": [10.0, 20.0, 30.0, 40.0],
            "grossProfit": [5.0, 10.0, 15.0, 20.0],
            "operatingIncome": [1.0, 2.0, 3.0, 4.0],
            "incomeBeforeTax": [1.0, 2.0, 3.0, 4.0],
            "netIncome": [1.0, 2.0, 3.0, 4.0],
            "ebit": [1.0, 2.0, 3.0, 4.0],
            "ebitda": [1.0, 2.0, 3.0, 4.0],
        }
    )
    cashflow = pd.DataFrame(base | {"freeCashFlow": [1.0, 2.0, 3.0, 4.0]})
    earnings = pd.DataFrame(
        {
            "ticker": ["AAA.US"] * 4,
            "date": dates,
            "reportDate": filing_dates,
            "epsActual": [0.1, 0.2, 0.3, 0.4],
        }
    )

    result = FundamentalProcessor.calculate_fundamental_ratios(
        balance=balance,
        cashflow=cashflow,
        income=income,
        earnings=earnings,
        list_kpi_toincrease=[],
        list_ratios_toincrease=[],
        list_kpi_toaccelerate=[],
        list_lag_increase=[],
        list_ratios_to_augment=[],
        list_date_to_maximise=["filing_date_income", "filing_date_balance"],
        backend="polars",
    ).sort_values("quarter_end")

    assert result["totalrevenue_rolling"].iloc[:3].isna().all()
    assert result["totalrevenue_rolling"].iloc[3] == pytest.approx(100.0)
    assert result["freecashflow_rolling"].iloc[3] == pytest.approx(10.0)
    assert result["epsactual_rolling"].iloc[3] == pytest.approx(1.0)

    missing_quarter = income[income["date"] != "2020-09-30"]
    incomplete = FundamentalProcessor.calculate_fundamental_ratios(
        balance=balance[balance["date"] != "2020-09-30"],
        cashflow=cashflow[cashflow["date"] != "2020-09-30"],
        income=missing_quarter,
        earnings=earnings[earnings["date"] != "2020-09-30"],
        list_kpi_toincrease=[],
        list_ratios_toincrease=[],
        list_kpi_toaccelerate=[],
        list_lag_increase=[],
        list_ratios_to_augment=[],
        list_date_to_maximise=["filing_date_income", "filing_date_balance"],
        backend="polars",
    )
    assert incomplete["totalrevenue_rolling"].isna().all()
