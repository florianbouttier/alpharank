from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.multihorizon.data import _add_multihorizon_targets
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.splits import (
    PurgedCombinatorialMonthSplit,
    horizon_walk_forward_windows,
)


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
