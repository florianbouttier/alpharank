from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import polars as pl


@dataclass(frozen=True)
class FoldPreprocessor:
    """Train-only feature selection and fallback medians."""

    features: tuple[str, ...]
    global_medians: dict[str, float]

    def transform(self, frame: pl.DataFrame) -> tuple[pl.DataFrame, np.ndarray]:
        transformed = frame.with_columns(
            [
                pl.when(pl.col(column).cast(pl.Float64, strict=False).is_finite())
                .then(pl.col(column).cast(pl.Float64, strict=False))
                .otherwise(None)
                .alias(column)
                for column in self.features
            ]
        )
        transformed = transformed.with_columns(
            [
                pl.col(column)
                .fill_null(pl.col(column).median().over("decision_month"))
                .fill_null(float(self.global_medians[column]))
                .fill_null(0.0)
                .alias(column)
                for column in self.features
            ]
        )
        matrix = transformed.select(self.features).to_numpy().astype(np.float32)
        return transformed, matrix


def fit_fold_preprocessor(
    train_frame: pl.DataFrame,
    candidate_features: Sequence[str],
    *,
    max_missing_ratio: float,
) -> FoldPreprocessor:
    """Fit sparse-column filtering and medians on the training rows only."""

    if train_frame.is_empty():
        raise ValueError("Cannot fit preprocessing on an empty frame.")
    features = list(candidate_features)
    finite_train = train_frame.with_columns(
        [
            pl.when(pl.col(column).cast(pl.Float64, strict=False).is_finite())
            .then(pl.col(column).cast(pl.Float64, strict=False))
            .otherwise(None)
            .alias(column)
            for column in features
        ]
    )
    missing = finite_train.select(
        [pl.col(column).is_null().mean().alias(column) for column in features]
    ).to_dicts()[0]
    kept = [
        column
        for column in features
        if float(missing.get(column, 1.0) or 0.0) <= float(max_missing_ratio)
    ]
    if not kept:
        raise ValueError("All candidate features were removed by train-only sparse filtering.")
    medians_row = finite_train.select([pl.col(column).median().alias(column) for column in kept]).to_dicts()[0]
    medians = {
        column: float(value) if value is not None and np.isfinite(float(value)) else 0.0
        for column, value in medians_row.items()
    }
    return FoldPreprocessor(features=tuple(kept), global_medians=medians)
