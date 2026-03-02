import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.features.indicators import TechnicalIndicators
from alpharank.strategy.legacy import StrategyLearner

pl = pytest.importorskip("polars")


def _mock_prices() -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2021-01-01", periods=220, freq="D")
    frames = []
    for i, ticker in enumerate(["AAA.US", "BBB.US", "CCC.US"]):
        base = 50 + 10 * i
        close_vs_index = base + rng.normal(0, 0.2, size=len(dates)).cumsum()
        dr = pd.Series(close_vs_index).pct_change()
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "ticker": ticker,
                    "close_vs_index": close_vs_index,
                    "dr": dr,
                }
            )
        )
    out = pd.concat(frames, ignore_index=True)
    out["year_month"] = out["date"].dt.to_period("M")
    return out


def _mock_historical(prices: pd.DataFrame) -> pd.DataFrame:
    ym = sorted(prices["year_month"].unique())
    rows = []
    for ticker in prices["ticker"].unique():
        for month in ym:
            rows.append({"ticker": ticker, "year_month": month})
    return pd.DataFrame(rows)


def test_fiting_polars():
    prices = _mock_prices()
    historical = _mock_historical(prices)
    out_pl = StrategyLearner.fiting(
        df=prices.copy(),
        column_price="close_vs_index",
        historical_company=historical.copy(),
        n_long=30,
        n_short=8,
        func_movingaverage=TechnicalIndicators.ema,
        n_asset=2,
        backend="polars",
    )
    out_pl = out_pl.sort_values(["year_month", "ticker", "date"]).reset_index(drop=True)
    assert len(out_pl) > 0
    assert {"mtr", "quantile_mtr", "dr"}.issubset(set(out_pl.columns))


def test_fiting_pandas_backend_disabled():
    prices = _mock_prices()
    historical = _mock_historical(prices)
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        StrategyLearner.fiting(
            df=prices.copy(),
            column_price="close_vs_index",
            historical_company=historical.copy(),
            n_long=30,
            n_short=8,
            func_movingaverage=TechnicalIndicators.ema,
            n_asset=2,
            backend="pandas",
        )
