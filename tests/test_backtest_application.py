from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.application import (
    ApplicationBacktestConfig,
    filter_predictions_by_price_staleness,
    run_application_backtest,
)


def _predictions_fixture() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US", "AAA.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 2, 1), date(2020, 2, 1)],
            "decision_month": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 2, 1), date(2020, 2, 1)],
            "holding_month": [date(2020, 2, 1), date(2020, 2, 1), date(2020, 3, 1), date(2020, 3, 1)],
            "decision_asof_date": [date(2020, 1, 31), date(2019, 11, 30), date(2020, 2, 29), date(2020, 2, 29)],
            "future_return": [0.05, 0.01, -0.02, 0.03],
            "benchmark_future_return": [0.01, 0.01, -0.01, -0.01],
            "future_excess_return": [0.04, 0.0, -0.01, 0.04],
            "target_label": [1, 0, 0, 1],
            "prediction": [0.80, 0.55, 0.45, 0.90],
        }
    )


def test_filter_predictions_by_price_staleness_drops_stale_names() -> None:
    predictions = _predictions_fixture()

    filtered = filter_predictions_by_price_staleness(predictions, max_price_staleness_months=0)

    assert filtered.select("ticker").to_series().to_list() == ["AAA.US", "AAA.US", "BBB.US"]


def test_run_application_backtest_supports_top_n_and_prediction_threshold() -> None:
    predictions = _predictions_fixture()

    top_n_result = run_application_backtest(
        predictions,
        ApplicationBacktestConfig(name="top_n_1", selection_mode="top_n", top_n=1),
        risk_free_rate=0.02,
    )
    threshold_result = run_application_backtest(
        predictions,
        ApplicationBacktestConfig(
            name="pred_gt_0_60",
            selection_mode="prediction_threshold",
            prediction_threshold=0.60,
            max_price_staleness_months=0,
        ),
        risk_free_rate=0.02,
    )

    assert top_n_result.selections.height == 2
    assert top_n_result.selections.get_column("ticker").to_list() == ["AAA.US", "BBB.US"]
    assert threshold_result.selections.height == 2
    assert threshold_result.selections.get_column("ticker").to_list() == ["AAA.US", "BBB.US"]
    assert threshold_result.kpis.filter(pl.col("strategy") == "Portfolio").get_column("months").item() == 2.0
