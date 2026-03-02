import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))
from alpharank.utils.returns import calculate_daily_returns, calculate_monthly_returns

pl = pytest.importorskip("polars")


def _sample_prices() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=120, freq="D")
    rows = []
    for ticker in ["AAA.US", "BBB.US", "CCC.US"]:
        level = 100 + rng.normal(0, 1, size=len(dates)).cumsum()
        rows.append(pd.DataFrame({"date": dates, "ticker": ticker, "close": level}))
    return pd.concat(rows, ignore_index=True)


def test_daily_returns_polars():
    df = _sample_prices()
    out_pl = calculate_daily_returns(df, price_column="close", date_column="date", ticker_column="ticker", return_column="dr", backend="polars")
    out_pl = out_pl.sort_values(["ticker", "date"]).reset_index(drop=True)
    assert "dr" in out_pl.columns
    assert len(out_pl) == len(df)
    assert out_pl["dr"].isna().sum() >= 3


def test_monthly_returns_polars():
    df = _sample_prices()
    out_pl = calculate_monthly_returns(df, price_column="close", date_column="date", ticker_column="ticker", return_column="monthly_return", backend="polars")
    out_pl = out_pl.sort_values(["ticker", "year_month"]).reset_index(drop=True)
    assert {"ticker", "year_month", "monthly_return", "last_close"}.issubset(set(out_pl.columns))
    assert len(out_pl) > 0


def test_returns_pandas_backend_disabled():
    df = _sample_prices()
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        calculate_daily_returns(df, price_column="close", date_column="date", ticker_column="ticker", return_column="dr", backend="pandas")
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        calculate_monthly_returns(df, price_column="close", date_column="date", ticker_column="ticker", return_column="monthly_return", backend="pandas")
