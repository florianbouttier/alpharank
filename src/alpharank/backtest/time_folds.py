from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import List, Sequence

import numpy as np
import polars as pl


@dataclass
class FoldWindow:
    fold_index: int
    train_months: List
    val_months: List
    test_months: List
    split_strategy: str = "rolling"
    test_group_indexes: tuple[int, ...] = ()


def split_months_into_folds(months: Sequence, n_folds: int) -> List[List]:
    months_list = list(months)
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2.")
    if len(months_list) < n_folds:
        raise ValueError(
            f"Not enough months ({len(months_list)}) for n_folds={n_folds}. "
            "Reduce n_folds or extend history."
        )

    base = len(months_list) // n_folds
    remainder = len(months_list) % n_folds

    folds: List[List] = []
    cursor = 0
    for i in range(n_folds):
        fold_size = base + (1 if i < remainder else 0)
        folds.append(months_list[cursor : cursor + fold_size])
        cursor += fold_size

    return folds


def rolling_fold_windows(months: Sequence, n_folds: int) -> List[FoldWindow]:
    if n_folds < 3:
        raise ValueError("n_folds must be >= 3 for train/val/test rolling windows.")
    folds = split_months_into_folds(months, n_folds=n_folds)

    windows: List[FoldWindow] = []
    for idx in range(n_folds - 2):
        train_months = [month for fold in folds[: idx + 1] for month in fold]
        val_months = folds[idx + 1]
        test_months = folds[idx + 2]
        windows.append(
            FoldWindow(
                fold_index=idx + 1,
                train_months=train_months,
                val_months=val_months,
                test_months=test_months,
                split_strategy="rolling",
            )
        )

    return windows


def _group_index_map(groups: Sequence[Sequence]) -> dict:
    return {month: group_idx for group_idx, group in enumerate(groups) for month in group}


def _embargoed_group_indexes(
    *,
    test_group_indexes: set[int],
    n_groups: int,
    embargo_groups: int,
) -> set[int]:
    if embargo_groups <= 0:
        return set()

    embargoed: set[int] = set()
    for group_idx in test_group_indexes:
        start = max(0, group_idx - embargo_groups)
        end = min(n_groups - 1, group_idx + embargo_groups)
        embargoed.update(range(start, end + 1))
    return embargoed - test_group_indexes


def cpcv_fold_windows(
    months: Sequence,
    *,
    n_groups: int,
    test_group_count: int = 2,
    embargo_groups: int = 0,
) -> List[FoldWindow]:
    """Build combinatorial purged CV windows over chronological month groups.

    Each window tests one combination of month groups, removes adjacent
    embargo groups from the training candidate set, and reserves one remaining
    group for model-selection diagnostics.
    """

    groups = split_months_into_folds(months, n_folds=n_groups)
    if test_group_count < 1 or test_group_count >= n_groups:
        raise ValueError("test_group_count must be between 1 and n_groups - 1.")

    windows: List[FoldWindow] = []
    all_group_indexes = set(range(n_groups))

    for fold_index, combo in enumerate(combinations(range(n_groups), test_group_count), start=1):
        test_group_indexes = set(combo)
        embargoed = _embargoed_group_indexes(
            test_group_indexes=test_group_indexes,
            n_groups=n_groups,
            embargo_groups=embargo_groups,
        )
        available = sorted(all_group_indexes - test_group_indexes - embargoed)
        if len(available) < 2:
            continue

        first_test_group = min(test_group_indexes)
        before_test = [group_idx for group_idx in available if group_idx < first_test_group]
        val_group = before_test[-1] if before_test else available[0]
        train_group_indexes = [group_idx for group_idx in available if group_idx != val_group]
        if not train_group_indexes:
            continue

        train_months = [month for group_idx in train_group_indexes for month in groups[group_idx]]
        val_months = list(groups[val_group])
        test_months = [month for group_idx in sorted(test_group_indexes) for month in groups[group_idx]]

        windows.append(
            FoldWindow(
                fold_index=fold_index,
                train_months=sorted(train_months),
                val_months=sorted(val_months),
                test_months=sorted(test_months),
                split_strategy="cpcv",
                test_group_indexes=tuple(sorted(test_group_indexes)),
            )
        )

    if not windows:
        raise ValueError(
            "CPCV configuration produced no usable windows. "
            "Reduce test_group_count or embargo_groups, or increase n_groups."
        )
    return windows


class CombinatorialPurgedGroupTimeSeriesSplit:
    """mlcraft-compatible CPCV splitter over ordered group labels."""

    def __init__(
        self,
        groups: Sequence,
        *,
        n_groups: int = 5,
        test_group_count: int = 1,
        embargo_groups: int = 0,
    ) -> None:
        self.groups = list(groups)
        self.n_groups = int(n_groups)
        self.test_group_count = int(test_group_count)
        self.embargo_groups = int(embargo_groups)

    def split(self, X, y=None):
        if not self.groups:
            return
        unique_groups = sorted(dict.fromkeys(self.groups))
        month_groups = split_months_into_folds(unique_groups, n_folds=min(self.n_groups, len(unique_groups)))
        group_index_by_value = _group_index_map(month_groups)
        row_group_indexes = np.asarray([group_index_by_value[group] for group in self.groups], dtype=int)
        n_groups = len(month_groups)
        all_indexes = np.arange(len(self.groups), dtype=int)

        for combo in combinations(range(n_groups), min(self.test_group_count, n_groups - 1)):
            test_group_indexes = set(combo)
            embargoed = _embargoed_group_indexes(
                test_group_indexes=test_group_indexes,
                n_groups=n_groups,
                embargo_groups=self.embargo_groups,
            )
            val_mask = np.isin(row_group_indexes, list(test_group_indexes))
            train_mask = ~np.isin(row_group_indexes, list(test_group_indexes | embargoed))
            train_idx = all_indexes[train_mask]
            val_idx = all_indexes[val_mask]
            if train_idx.size == 0 or val_idx.size == 0:
                continue
            yield np.sort(train_idx), np.sort(val_idx)


def filter_by_months(df: pl.DataFrame, months: Sequence) -> pl.DataFrame:
    if not months:
        return df.head(0)
    return df.filter(pl.col("year_month").is_in(list(months)))
