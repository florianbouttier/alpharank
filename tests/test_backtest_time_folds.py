from __future__ import annotations

from datetime import date

import numpy as np

from alpharank.backtest.time_folds import (
    CombinatorialPurgedGroupTimeSeriesSplit,
    cpcv_fold_windows,
    walk_forward_windows,
)


def _months(count: int) -> list[date]:
    return [date(2020 + index // 12, index % 12 + 1, 1) for index in range(count)]


def test_cpcv_fold_windows_build_combinatorial_test_groups() -> None:
    windows = cpcv_fold_windows(_months(12), n_groups=4, test_group_count=2)

    assert len(windows) == 6
    assert {window.split_strategy for window in windows} == {"cpcv"}
    assert {window.test_group_indexes for window in windows} == {
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    }
    assert all(window.train_months for window in windows)
    assert all(window.val_months for window in windows)
    assert all(window.test_months for window in windows)


def test_cpcv_splitter_removes_validation_groups_from_training() -> None:
    groups = ["2020-01", "2020-01", "2020-02", "2020-02", "2020-03", "2020-03"]
    splitter = CombinatorialPurgedGroupTimeSeriesSplit(groups, n_groups=3, test_group_count=1)

    splits = list(splitter.split(np.zeros((len(groups), 2))))

    assert len(splits) == 3
    for train_idx, val_idx in splits:
        train_groups = {groups[index] for index in train_idx}
        val_groups = {groups[index] for index in val_idx}
        assert train_groups
        assert val_groups
        assert train_groups.isdisjoint(val_groups)


def test_walk_forward_windows_use_only_past_train_months() -> None:
    windows = walk_forward_windows(
        _months(18),
        min_train_months=6,
        val_months=3,
        test_months=1,
        max_windows=4,
    )

    assert len(windows) == 4
    for window in windows:
        assert window.split_strategy == "walk_forward"
        assert max(window.train_months) < min(window.val_months)
        assert max(window.val_months) < min(window.test_months)
