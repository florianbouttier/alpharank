import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.data.processing import FundamentalProcessor, PricesDataPreprocessor

pl = pytest.importorskip("polars")


def _prices():
    dates = pd.date_range("2023-01-01", periods=180, freq="D")
    frames = []
    for ticker in ["AAA.US", "BBB.US"]:
        vals = 100 + np.linspace(0, 10, len(dates)) + (0.5 if ticker == "BBB.US" else 0.0)
        frames.append(pd.DataFrame({"ticker": ticker, "date": dates, "adjusted_close": vals}))
    return pd.concat(frames, ignore_index=True)


def _fundamentals():
    q = pd.date_range("2022-03-31", periods=8, freq="Q")
    base = []
    for ticker in ["AAA.US", "BBB.US"]:
        for d in q:
            base.append(
                {
                    "ticker": ticker,
                    "date": d,
                    "filing_date": d + pd.Timedelta(days=20),
                    "commonStockSharesOutstanding": 1_000_000,
                    "totalStockholderEquity": 2_000_000,
                    "netDebt": 300_000,
                    "totalAssets": 4_500_000,
                    "cashAndShortTermInvestments": 200_000,
                }
            )
    balance = pd.DataFrame(base)

    income = balance[["ticker", "date", "filing_date"]].copy()
    income = income.assign(
        totalRevenue=500_000,
        grossProfit=200_000,
        operatingIncome=120_000,
        incomeBeforeTax=100_000,
        netIncome=80_000,
        ebit=115_000,
        ebitda=130_000,
    )

    cash = balance[["ticker", "date", "filing_date"]].copy()
    cash["freeCashFlow"] = 60_000

    earnings = balance[["ticker", "date"]].copy()
    earnings = earnings.rename(columns={"date": "date"})
    earnings["reportDate"] = earnings["date"] + pd.Timedelta(days=22)
    earnings["epsActual"] = 2.4
    return balance, income, cash, earnings


def test_price_preprocessor_polars():
    prices = _prices()
    idx = prices[prices["ticker"] == "AAA.US"][["date", "adjusted_close"]].rename(columns={"adjusted_close": "sp500_close"})
    pl_out = PricesDataPreprocessor.prices_vs_index(idx, prices, "sp500_close", "adjusted_close", backend="polars")
    pl_out = pl_out.sort_values(["ticker", "date"]).reset_index(drop=True)
    assert {"close_vs_index", "dr_vs_index"}.issubset(set(pl_out.columns))
    assert len(pl_out) == len(prices)


def test_fundamental_processor_polars():
    prices = _prices()
    monthly = PricesDataPreprocessor.calculate_monthly_returns(
        prices, column_date="date", column_close="adjusted_close", backend="polars"
    )
    balance, income, cash, earnings = _fundamentals()
    pe_pl = FundamentalProcessor.calculate_pe_ratios(
        balance=balance,
        earnings=earnings,
        cashflow=cash,
        income=income,
        earning_choice="epsactual_rolling",
        monthly_return=monthly.copy(),
        list_date_to_maximise=[],
        backend="polars",
    )
    pe_pl = pe_pl.sort_values(["ticker", "year_month"]).reset_index(drop=True)
    assert len(pe_pl) > 0
    for col in ["pe", "ps_ratio", "pb_ratio", "ev_ebitda_ratio", "market_cap"]:
        assert col in pe_pl.columns


def test_processing_pandas_backend_disabled():
    prices = _prices()
    idx = prices[prices["ticker"] == "AAA.US"][["date", "adjusted_close"]].rename(columns={"adjusted_close": "sp500_close"})
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        PricesDataPreprocessor.prices_vs_index(idx, prices, "sp500_close", "adjusted_close", backend="pandas")
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        PricesDataPreprocessor.calculate_monthly_returns(prices, column_date="date", column_close="adjusted_close", backend="pandas")
