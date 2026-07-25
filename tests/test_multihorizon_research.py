from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.multihorizon.data import _add_multihorizon_targets
from alpharank.multihorizon.data import _append_legacy_labels
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.metrics import score_predictions
from alpharank.multihorizon.splits import (
    PurgedCombinatorialMonthSplit,
    horizon_walk_forward_windows,
)
from alpharank.multihorizon.trading import build_monthly_top_n_returns


def test_future_target_requires_an_exact_calendar_gap() -> None:
    stock = pl.DataFrame(
        {
            "ticker": ["A", "A", "A"],
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1), date(2020, 4, 1)],
            "last_close": [100.0, 110.0, 130.0],
            "monthly_return": [0.0, 0.1, 130 / 110 - 1],
        }
    )
    index = pl.DataFrame(
        {
            "year_month": [
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 3, 1),
                date(2020, 4, 1),
            ],
            "index_close": [100.0, 101.0, 102.0, 103.0],
            "index_monthly_return": [0.0, 0.01, 0.01, 0.01],
        }
    )
    result = _add_multihorizon_targets(stock, index, [1])
    assert result["future_return_1m"].to_list()[0] == pytest.approx(0.1)
    assert result["future_return_1m"].to_list()[1:] == [None, None]


def test_preprocessor_uses_train_only_median_after_monthly_fill() -> None:
    train = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "x": [1.0, 3.0],
            "too_sparse": [None, 1.0],
        }
    )
    future = pl.DataFrame(
        {"decision_month": [date(2020, 3, 1)], "x": [None], "too_sparse": [999.0]}
    )
    preprocessor = fit_fold_preprocessor(train, ["x", "too_sparse"], max_missing_ratio=0.4)
    assert preprocessor.features == ("x",)
    _, matrix = preprocessor.transform(future)
    np.testing.assert_allclose(matrix, [[2.0]])


def test_outer_window_respects_label_maturity_and_purge() -> None:
    windows = horizon_walk_forward_windows(
        list(range(240)),
        horizon=36,
        min_train_months=120,
        validation_months=24,
        test_months=12,
        step_months=12,
    )
    first = windows[0]
    assert len(first.train_months) == 120
    assert first.validation_months[0] - first.train_months[-1] == 36
    assert first.test_months[0] - first.validation_months[-1] == 36


def test_inner_cpcv_removes_overlapping_label_intervals() -> None:
    months = [month for month in range(24) for _ in range(2)]
    splitter = PurgedCombinatorialMonthSplit(months, horizon=6, n_groups=4)
    for train_idx, test_idx in splitter.split():
        train_months = np.asarray(months)[train_idx]
        test_months = np.asarray(months)[test_idx]
        for train_month in train_months:
            assert not np.any(
                (train_month <= test_months + 6) & (train_month + 6 >= test_months)
            )


def test_legacy_teacher_join_does_not_leave_raw_weight_as_a_feature(tmp_path) -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["A.US"],
            "decision_month": [date(2020, 1, 1)],
        }
    )
    legacy_path = tmp_path / "legacy.parquet"
    pl.DataFrame(
        {
            "portfolio_model": ["Combined_Frequency"],
            "year_month": [date(2020, 2, 1)],
            "ticker": ["A.US"],
            "n_models": [3],
            "weight_normalized": [0.25],
        }
    ).write_parquet(legacy_path)

    result = _append_legacy_labels(frame, legacy_path)

    assert "weight_normalized" not in result.columns
    assert result["legacy_weight_normalized"].to_list() == [0.25]


def test_regression_report_contains_prediction_error_metrics() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 4,
            "ticker": ["A", "B", "C", "D"],
            "score": [0.3, 0.2, 0.1, 0.0],
            "future_excess_return_1m": [0.4, 0.1, -0.1, 0.0],
            "future_excess_rank_1m": [1.0, 0.75, 0.25, 0.5],
            "future_return_1m": [0.4, 0.1, -0.1, 0.0],
            "legacy_selected": [1, 0, 0, 0],
        }
    )
    metrics, _ = score_predictions(
        predictions,
        method="regression",
        horizon=1,
        top_n_values=(2,),
    )
    assert metrics["rmse"] == pytest.approx(
        np.sqrt(np.mean((np.array([0.4, 0.1, -0.1, 0.0]) - np.array([0.3, 0.2, 0.1, 0.0])) ** 2))
    )
    assert {"mae", "r2", "ndcg_at_10", "spearman_ic"} <= metrics.keys()


def test_monthly_trading_backtest_applies_turnover_cost() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 1, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
                date(2020, 2, 1),
            ],
            "ticker": ["A", "B", "C", "A", "B", "C"],
            "score": [3.0, 2.0, 1.0, 3.0, 1.0, 2.0],
            "future_return_1m": [0.10, 0.00, -0.10, 0.02, 0.00, 0.04],
            "benchmark_future_return_1m": [0.01] * 6,
        }
    )
    monthly = build_monthly_top_n_returns(
        predictions,
        top_n=2,
        transaction_cost_bps=10.0,
    )
    assert monthly["turnover"].to_list() == pytest.approx([1.0, 0.5])
    assert monthly["net_return"].to_list() == pytest.approx([0.049, 0.0295])
