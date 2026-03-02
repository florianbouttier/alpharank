import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.strategy.legacy import StrategyLearner

pl = pytest.importorskip("polars")


def _mock_outputs():
    dates = [pd.Period("2024-01", freq="M"), pd.Period("2024-02", freq="M")]
    df1 = pd.DataFrame(
        {
            "year_month": dates * 3,
            "ticker": ["A.US"] * 2 + ["B.US"] * 2 + ["C.US"] * 2,
            "dr": [1.05, 1.02, 1.03, 1.01, np.nan, np.nan],
            "Sector": ["Tech"] * 2 + ["Fin"] * 2 + ["Energy"] * 2,
        }
    )
    df2 = pd.DataFrame(
        {
            "year_month": dates * 3,
            "ticker": ["B.US"] * 2 + ["C.US"] * 2 + ["D.US"] * 2,
            "dr": [1.03, 1.01, np.nan, np.nan, 1.10, 1.08],
            "Sector": ["Fin"] * 2 + ["Energy"] * 2 + ["Auto"] * 2,
        }
    )
    return [{"detailed": df1, "aggregated": pd.DataFrame()}, {"detailed": df2, "aggregated": pd.DataFrame()}]


def test_aggregate_portfolios_polars():
    outputs = _mock_outputs()
    pl_out = StrategyLearner.aggregate_portfolios(outputs, mode="frequency", union_mode=True, backend="polars")

    d_pl = pl_out["detailed"].sort_values(["year_month", "ticker"]).reset_index(drop=True)
    a_pl = pl_out["aggregated"].sort_values(["year_month"]).reset_index(drop=True)

    assert len(d_pl) > 0
    assert len(a_pl) > 0
    assert {"weight", "weight_normalized"}.issubset(set(d_pl.columns))
    assert "monthly_return" in a_pl.columns


def test_aggregate_portfolios_pandas_backend_disabled():
    outputs = _mock_outputs()
    with pytest.raises(ValueError, match="Pandas backend is disabled"):
        StrategyLearner.aggregate_portfolios(outputs, mode="frequency", union_mode=True, backend="pandas")
