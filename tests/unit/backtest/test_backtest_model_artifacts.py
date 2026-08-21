from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import xgboost as xgb

from alpharank.backtest.model_artifacts import (
    load_serialized_fold_predictor,
    serialize_fold_model,
)
from alpharank.multihorizon.preprocessing import FoldPreprocessor


def test_serialized_model_reproduces_oos_predictions(tmp_path: Path) -> None:
    matrix = np.asarray(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=np.float32
    )
    labels = np.asarray([0, 0, 1, 1], dtype=np.float32)
    booster = xgb.train(
        {
            "objective": "binary:logistic",
            "max_depth": 2,
            "eta": 0.3,
            "seed": 42,
            "nthread": 1,
        },
        xgb.DMatrix(matrix, label=labels),
        num_boost_round=5,
    )
    wrapper = SimpleNamespace(model_=booster, best_num_iterations_=None)
    preprocessor = FoldPreprocessor(
        features=("x1", "x2"), global_medians={"x1": 0.5, "x2": 0.5}
    )
    frame = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 4,
            "ticker": ["A", "B", "C", "D"],
            "x1": matrix[:, 0],
            "x2": matrix[:, 1],
        }
    )
    expected = booster.predict(xgb.DMatrix(matrix))

    manifest = serialize_fold_model(
        fold_dir=tmp_path,
        model=wrapper,
        preprocessor=preprocessor,
        seed=42,
        fold_metadata={"fold": 1, "test_month_start": "2020-01-01"},
    )
    replayed = load_serialized_fold_predictor(tmp_path).predict(frame)

    np.testing.assert_array_equal(replayed, expected)
    expected_rank = np.argsort(-expected, kind="stable")
    replayed_rank = np.argsort(-replayed, kind="stable")
    np.testing.assert_array_equal(replayed_rank, expected_rank)
    assert manifest["features"] == ["x1", "x2"]
    assert manifest["seed"] == 42
    assert manifest["fold_metadata"]["fold"] == 1
