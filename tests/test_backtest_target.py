from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.datasets import build_model_frame
from alpharank.backtest.pipeline import _binary_target


def test_backtest_target_uses_future_excess_return() -> None:
    monthly_prices = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "BBB.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1), date(2020, 2, 1)],
            "monthly_return": [0.01, 0.10, 0.01, 0.03],
        }
    )
    technical_features = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "BBB.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1), date(2020, 2, 1)],
            "momentum_1m": [1.0, 2.0, 3.0, 4.0],
        }
    )
    fundamental_features = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US", "BBB.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 1, 1), date(2020, 2, 1)],
            "quality_score": [10.0, 11.0, 12.0, 13.0],
        }
    )
    index_monthly = pl.DataFrame(
        {
            "year_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "index_monthly_return": [0.00, 0.05],
        }
    )
    constituents = pl.DataFrame(
        {
            "Ticker": ["AAA", "BBB"],
            "Date": [date(2020, 1, 1), date(2020, 1, 1)],
        }
    )

    frame, features_used, dropped_features = build_model_frame(
        monthly_prices=monthly_prices,
        technical_features=technical_features,
        fundamental_features=fundamental_features,
        index_monthly=index_monthly,
        constituents=constituents,
        start_month="2020-01",
        missing_feature_threshold=0.5,
    )

    jan_rows = frame.filter(pl.col("year_month") == date(2020, 1, 1)).sort("ticker")

    assert features_used == ["momentum_1m", "quality_score"]
    assert dropped_features == []
    assert jan_rows.get_column("future_return").to_list() == [0.10, 0.03]
    assert jan_rows.get_column("benchmark_future_return").to_list() == [0.05, 0.05]
    assert np.allclose(jan_rows.get_column("future_excess_return").to_numpy(), np.array([0.05, -0.02]))

    target = _binary_target(jan_rows, threshold=0.0)
    assert target.tolist() == [1, 0]
