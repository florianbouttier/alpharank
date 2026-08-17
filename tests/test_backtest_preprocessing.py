from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.backtest.pipeline import _fit_outer_fold_preprocessor


def test_preprocessing_is_fit_inside_each_outer_fold() -> None:
    train = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "stable": [1.0, 3.0],
            "train_sparse": [None, 5.0],
        }
    )
    future = pl.DataFrame(
        {
            "decision_month": [date(2020, 3, 1)],
            "stable": [9999.0],
            "train_sparse": [7.0],
        }
    )
    mutated_future = future.with_columns(
        pl.lit(-9999.0).alias("stable"),
        pl.lit(1234.0).alias("train_sparse"),
    )

    reference = _fit_outer_fold_preprocessor(
        train, ["stable", "train_sparse"], max_missing_ratio=0.4
    )
    candidate = _fit_outer_fold_preprocessor(
        train, ["stable", "train_sparse"], max_missing_ratio=0.4
    )

    assert reference.features == candidate.features == ("stable",)
    assert reference.global_medians == candidate.global_medians == {"stable": 2.0}
    transformed_reference, _ = reference.transform(future)
    transformed_candidate, _ = candidate.transform(mutated_future)
    assert transformed_reference["stable"][0] == 9999.0
    assert transformed_candidate["stable"][0] == -9999.0
    assert reference.transform(train)[0].equals(candidate.transform(train)[0])
