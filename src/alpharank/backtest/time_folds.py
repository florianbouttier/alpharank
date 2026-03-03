from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import polars as pl


@dataclass
class FoldWindow:
    fold_index: int
    train_months: List
    val_months: List
    test_months: List


def split_months_into_folds(months: Sequence, n_folds: int) -> List[List]:
    months_list = list(months)
    if n_folds < 3:
        raise ValueError("n_folds must be >= 3 for train/val/test rolling windows.")
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
            )
        )

    return windows


def filter_by_months(df: pl.DataFrame, months: Sequence) -> pl.DataFrame:
    if not months:
        return df.head(0)
    return df.filter(pl.col("year_month").is_in(list(months)))
