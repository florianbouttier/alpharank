from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.pipeline import _build_prediction_debug_frames


def test_build_prediction_debug_frames_marks_scored_rows() -> None:
    model_frame = pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US", "CCC.US"],
            "year_month": [date(2020, 1, 1), date(2020, 1, 1), date(2020, 2, 1)],
            "feature_a": [1.0, 2.0, 3.0],
            "monthly_return": [0.01, 0.02, 0.03],
            "future_return": [0.05, 0.01, 0.04],
            "benchmark_future_return": [0.02, 0.02, 0.03],
            "future_excess_return": [0.03, -0.01, 0.01],
            "future_relative_return": [0.0294, -0.0098, 0.0097],
        }
    )
    predictions = pl.DataFrame(
        {
            "ticker": ["AAA.US", "BBB.US"],
            "year_month": [date(2020, 1, 1), date(2020, 1, 1)],
            "monthly_return": [0.01, 0.02],
            "future_return": [0.05, 0.01],
            "benchmark_future_return": [0.02, 0.02],
            "future_excess_return": [0.03, -0.01],
            "future_relative_return": [0.0294, -0.0098],
            "prediction": [0.8, 0.2],
            "target_label": [1, 0],
            "fold": [1, 1],
            "objective_score": [0.7, 0.7],
            "objective_score_val": [0.65, 0.65],
        }
    )
    fold_index = pl.DataFrame(
        {
            "fold": [1, 2],
            "status": ["completed", "skipped"],
            "skip_reason": [None, "fold_min_val_rows"],
            "train_month_start": ["2019-01-01", "2020-01-01"],
            "train_month_end": ["2019-12-01", "2020-03-01"],
            "val_month_start": ["2020-01-01", "2020-04-01"],
            "val_month_end": ["2020-01-01", "2020-04-01"],
            "test_month_start": ["2020-02-01", "2020-05-01"],
            "test_month_end": ["2020-02-01", "2020-05-01"],
            "train_positive_rate": [0.4, None],
            "val_positive_rate": [0.5, None],
            "test_positive_rate": [0.6, None],
        }
    )

    debug_long, debug_full = _build_prediction_debug_frames(
        model_frame=model_frame,
        predictions=predictions,
        fold_index=fold_index,
        top_n=1,
    )

    assert debug_long.height == 2
    assert debug_long.get_column("selected_top_n").to_list() == [True, False]
    assert debug_long.get_column("status").to_list() == ["completed", "completed"]

    full_rows = debug_full.sort(["year_month", "ticker"])
    assert full_rows.get_column("is_scored").to_list() == [True, True, False]
    assert full_rows.get_column("has_target").to_list() == [True, True, True]
