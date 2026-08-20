from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
import xgboost as xgb

from alpharank.backtest.model_artifacts import serialize_fold_model
from alpharank.multihorizon.config import LATEST_COMMON_COMPARISON_PROFILE
from alpharank.multihorizon.metrics import build_prediction_portfolios
from alpharank.multihorizon.preprocessing import FoldPreprocessor
from alpharank.replay import validate_boosting_v2_replay


def test_boosting_v2_replay_is_serialized_and_causal(tmp_path: Path) -> None:
    run_dir = tmp_path / "boosting"
    combination = run_dir / "classification_h06"
    fold_dir = combination / "fold_01"
    fold_dir.mkdir(parents=True)
    matrix = np.asarray([[0.0], [0.5], [1.0]], dtype=np.float32)
    booster = xgb.train(
        {"objective": "binary:logistic", "seed": 43, "nthread": 1},
        xgb.DMatrix(matrix, label=np.asarray([0, 0, 1])),
        num_boost_round=3,
    )
    wrapper = SimpleNamespace(model_=booster, best_num_iterations_=None)
    preprocessor = FoldPreprocessor(features=("x",), global_medians={"x": 0.5})
    serialize_fold_model(
        fold_dir=fold_dir,
        model=wrapper,
        preprocessor=preprocessor,
        seed=43,
        fold_metadata={"fold": 1},
    )
    scores = booster.predict(xgb.DMatrix(matrix))
    replay = pl.DataFrame(
        {
            "decision_month": [date(2024, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "x": matrix[:, 0],
            "expected_raw_score": scores,
        }
    )
    replay_path = fold_dir / "oos_replay.parquet"
    replay.write_parquet(replay_path)
    (fold_dir / "oos_replay_manifest.json").write_text(
        json.dumps(
            {
                "oos_replay_file": replay_path.name,
                "oos_replay_sha256": _sha256(replay_path),
            }
        ),
        encoding="utf-8",
    )
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2024, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "legacy_selected": [0, 1, 0],
            "future_excess_return_1m": [None, 0.02, 0.03],
            "future_excess_return_6m": [0.01, 0.02, 0.03],
            "score": scores,
            "fold": [1, 1, 1],
        }
    )
    predictions.write_parquet(combination / "predictions.parquet")
    build_prediction_portfolios(
        predictions, horizon=6, top_n_values=(5, 10, 20)
    ).with_columns(
        pl.lit(1).alias("fold"),
        pl.lit("classification").alias("method"),
        pl.lit(6).alias("horizon"),
    ).write_csv(combination / "portfolio_monthly.csv")
    pl.DataFrame(
        {
            "fold": [1, 1, 1],
            "split": ["train", "validation", "test"],
            "population_rows": [3, 3, 3],
            "trainable_rows": [3, 3, 3],
            "evaluable_rows": [3, 3, 3],
            "terminal_event_resolved_rows": [0, 0, 0],
            "provisional_last_observation_rows": [0, 0, 0],
            "horizon_pending_rows": [0, 0, 0],
            "benchmark_target_unavailable_rows": [0, 0, 0],
            "ticker_target_unavailable_rows": [0, 0, 0],
            "terminal_event_unresolved_rows": [0, 0, 0],
        }
    ).write_csv(combination / "fold_target_censoring.csv")
    composition_id = "b" * 64
    config = {
        **LATEST_COMMON_COMPARISON_PROFILE,
        "top_n_values": [5, 10, 20],
        "score_only_end_month": "2024-01",
    }
    journal = pl.DataFrame(
        schema={"ticker": pl.String, "decision_month": pl.Date}
    )
    journal_parquet = run_dir / "provisional_target_journal.parquet"
    journal_csv = run_dir / "provisional_target_journal.csv"
    journal.write_parquet(journal_parquet)
    journal.write_csv(journal_csv)
    manifest = {
        "config": config,
        "methodology_identity": {
            "methodology_version": "v2-causal",
            "composition_id": composition_id,
        },
        "runtime_provenance": {"git": {"dirty": False}},
        "provisional_target_policy": {
            "journal_rows": 0,
            "journal_parquet": {
                "path": str(journal_parquet),
                "sha256": _sha256(journal_parquet),
            },
            "journal_csv": {
                "path": str(journal_csv),
                "sha256": _sha256(journal_csv),
            },
        },
        "results": {"combinations": [{"method": "classification", "horizon": 6}]},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    report = validate_boosting_v2_replay(
        run_dir, expected_composition_id=composition_id
    )

    assert report["passed"] is True
    assert report["fold_count"] == 1
    assert report["maximum_absolute_score_replay_error"] == 0.0
    replay.with_columns((pl.col("x") + 1.0).alias("x")).write_parquet(replay_path)
    with pytest.raises(RuntimeError, match="hash mismatch"):
        validate_boosting_v2_replay(run_dir, expected_composition_id=composition_id)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
