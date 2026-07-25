from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class HorizonWindow:
    fold: int
    train_months: tuple
    validation_months: tuple
    test_months: tuple
    horizon: int


def horizon_walk_forward_windows(
    months: Sequence,
    *,
    horizon: int,
    min_train_months: int,
    validation_months: int,
    test_months: int,
    step_months: int,
    max_windows: int | None = None,
) -> list[HorizonWindow]:
    """Expanding outer walk-forward with label maturity and overlap purge.

    At the first test month ``t``, a decision made at ``t-horizon`` is the
    latest row whose target is observable. A further ``horizon-1`` month purge
    separates training labels from the validation interval.
    """

    ordered = list(dict.fromkeys(sorted(months)))
    horizon = int(horizon)
    first_test = min_train_months + validation_months + 2 * (horizon - 1)
    windows: list[HorizonWindow] = []
    cursor = first_test
    while cursor + test_months <= len(ordered):
        validation_end = cursor - horizon + 1
        validation_start = validation_end - validation_months
        train_end = validation_start - (horizon - 1)
        if train_end >= min_train_months:
            windows.append(
                HorizonWindow(
                    fold=len(windows) + 1,
                    train_months=tuple(ordered[:train_end]),
                    validation_months=tuple(ordered[validation_start:validation_end]),
                    test_months=tuple(ordered[cursor : cursor + test_months]),
                    horizon=horizon,
                )
            )
        cursor += step_months
    if max_windows and len(windows) > max_windows:
        windows = windows[-int(max_windows) :]
        windows = [
            HorizonWindow(
                fold=index,
                train_months=item.train_months,
                validation_months=item.validation_months,
                test_months=item.test_months,
                horizon=item.horizon,
            )
            for index, item in enumerate(windows, start=1)
        ]
    if not windows:
        raise ValueError("No mature horizon-aware outer window could be constructed.")
    return windows


class PurgedCombinatorialMonthSplit:
    """CPCV splitter that purges every training label overlapping a test label."""

    def __init__(
        self,
        months: Sequence,
        *,
        horizon: int,
        n_groups: int = 4,
        test_group_count: int = 1,
    ) -> None:
        self.months = list(months)
        self.horizon = int(horizon)
        self.n_groups = int(n_groups)
        self.test_group_count = int(test_group_count)

    def split(self, X=None, y=None):
        unique = list(dict.fromkeys(sorted(self.months)))
        if len(unique) < self.n_groups:
            raise ValueError("Not enough months for the requested CPCV groups.")
        chunks = [list(chunk) for chunk in np.array_split(np.asarray(unique, dtype=object), self.n_groups)]
        position = {month: index for index, month in enumerate(unique)}
        row_positions = np.asarray([position[month] for month in self.months])
        all_rows = np.arange(len(self.months))
        for selected in combinations(range(self.n_groups), self.test_group_count):
            test_months = [month for group in selected for month in chunks[group]]
            test_positions = np.asarray([position[month] for month in test_months])
            test_mask = np.isin(row_positions, test_positions)
            # A training observation covers [decision, decision+horizon].
            overlap = np.zeros(len(self.months), dtype=bool)
            for test_position in test_positions:
                overlap |= (row_positions <= test_position + self.horizon) & (
                    row_positions + self.horizon >= test_position
                )
            train_idx = all_rows[~overlap]
            test_idx = all_rows[test_mask]
            if train_idx.size and test_idx.size:
                yield np.sort(train_idx), np.sort(test_idx)
