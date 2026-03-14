from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import pytest
import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.config import TechnicalFeatureConfig
from alpharank.backtest.features import compute_technical_features


def test_backtest_technical_features_are_indicator_based_and_configurable() -> None:
    closes = [100.0, 102.0, 104.0, 103.0, 106.0, 109.0, 111.0, 114.0]
    months = [
        date(2020, 1, 1),
        date(2020, 2, 1),
        date(2020, 3, 1),
        date(2020, 4, 1),
        date(2020, 5, 1),
        date(2020, 6, 1),
        date(2020, 7, 1),
        date(2020, 8, 1),
    ]
    monthly_returns = [None]
    monthly_returns.extend((current / previous) - 1.0 for previous, current in zip(closes, closes[1:]))

    monthly_prices = pl.DataFrame(
        {
            "ticker": ["AAA.US"] * len(months),
            "year_month": months,
            "last_close": closes,
            "monthly_return": monthly_returns,
        }
    )

    config = TechnicalFeatureConfig(
        roc_windows=(1, 3),
        ema_pairs=((2, 4),),
        price_to_ema_spans=(2, 4),
        rsi_windows=(3, 6),
        rsi_ratio_pairs=((3, 6),),
        bollinger_windows=(3,),
        stochastic_windows=((3, 2),),
        range_windows=(3,),
        volatility_windows=(2, 4),
        volatility_ratio_pairs=((2, 4),),
    )

    features = compute_technical_features(monthly_prices, config=config).sort("year_month")

    assert "ret_lag_1" not in features.columns
    assert "mom_3m" not in features.columns
    assert set(features.columns) == {
        "ticker",
        "year_month",
        "price_roc_1m",
        "price_roc_3m",
        "ema_ratio_2_4",
        "price_to_ema_2",
        "price_to_ema_4",
        "rsi_3m",
        "rsi_6m",
        "rsi_ratio_3_6",
        "bollinger_percent_b_3m",
        "bollinger_bandwidth_3m",
        "stoch_d_3_2",
        "dist_to_3m_high",
        "dist_to_3m_low",
        "range_position_3m",
        "volatility_2m",
        "volatility_4m",
        "volatility_ratio_2_4",
    }

    last_row = features.to_dicts()[-1]
    assert last_row["price_roc_3m"] == pytest.approx((114.0 / 106.0) - 1.0)
    assert last_row["ema_ratio_2_4"] > 1.0
    assert last_row["price_to_ema_4"] > 0.0
    assert 0.0 <= last_row["range_position_3m"] <= 1.0
    assert last_row["bollinger_percent_b_3m"] is not None
    assert last_row["stoch_d_3_2"] is not None
    assert last_row["volatility_ratio_2_4"] is not None
