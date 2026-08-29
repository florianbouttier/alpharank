from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alpharank.replay.prediction_universes import build_boosting_holdings
from alpharank.replay.trend_eligibility import (
    build_causal_trend_eligibility_registry,
    filter_predictions_to_causal_trend_universe,
)


def test_strict_majority_respects_normal_and_inverted_ema_pairs(tmp_path: Path) -> None:
    pairs = ((10, 30), (50, 20), (5, 100))
    replay = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1)] * 4,
            "ticker": ["ALL", "TWO", "ONE", "MISSING"],
            "relative_ema_ratio_10_30": [1.1, 1.1, 1.1, 1.1],
            "relative_ema_ratio_50_20": [0.9, 1.1, 1.1, 0.9],
            "relative_ema_ratio_5_100": [1.2, 1.2, 0.8, None],
        }
    )
    run_dir, candidates = _write_boosting_fixture(tmp_path, pairs, replay)

    registry = build_causal_trend_eligibility_registry(run_dir, candidates)
    rows = registry.frame.sort("ticker").select(
        "ticker",
        "trend_observed_pair_count",
        "trend_positive_pair_count",
        "trend_eligible",
        "trend_eligibility_reason",
    )

    assert rows.to_dicts() == [
        {
            "ticker": "ALL",
            "trend_observed_pair_count": 3,
            "trend_positive_pair_count": 3,
            "trend_eligible": True,
            "trend_eligibility_reason": "eligible_strict_majority",
        },
        {
            "ticker": "MISSING",
            "trend_observed_pair_count": 2,
            "trend_positive_pair_count": 2,
            "trend_eligible": False,
            "trend_eligibility_reason": "incomplete_pair_coverage",
        },
        {
            "ticker": "ONE",
            "trend_observed_pair_count": 3,
            "trend_positive_pair_count": 1,
            "trend_eligible": False,
            "trend_eligibility_reason": "non_positive_majority",
        },
        {
            "ticker": "TWO",
            "trend_observed_pair_count": 3,
            "trend_positive_pair_count": 2,
            "trend_eligible": True,
            "trend_eligibility_reason": "eligible_strict_majority",
        },
    ]


def test_eligibility_does_not_depend_on_score_or_future_returns(tmp_path: Path) -> None:
    replay = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1), date(2025, 1, 1)],
            "ticker": ["UP", "DOWN"],
            "relative_ema_ratio_10_30": [1.1, 0.9],
        }
    )
    run_dir, candidates = _write_boosting_fixture(tmp_path, ((10, 30),), replay)
    changed = candidates.with_columns(
        (-pl.col("score")).alias("score"),
        (-pl.col("future_return_1m")).alias("future_return_1m"),
    )

    baseline = build_causal_trend_eligibility_registry(run_dir, candidates)
    mutated = build_causal_trend_eligibility_registry(run_dir, changed)

    assert baseline.frame.equals(mutated.frame)
    assert filter_predictions_to_causal_trend_universe(candidates, baseline)[
        "ticker"
    ].to_list() == ["UP"]


def test_missing_pair_column_fails_closed_per_candidate(tmp_path: Path) -> None:
    replay = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1)],
            "ticker": ["ONLY"],
            "relative_ema_ratio_10_30": [1.1],
        }
    )
    run_dir, candidates = _write_boosting_fixture(
        tmp_path,
        ((10, 30), (20, 50)),
        replay,
    )

    registry = build_causal_trend_eligibility_registry(run_dir, candidates)

    assert registry.frame["trend_observed_pair_count"].item() == 1
    assert registry.frame["trend_required_pair_count"].item() == 2
    assert registry.frame["trend_eligible"].item() is False


def test_registry_rejects_prediction_key_drift_and_tampered_replay(tmp_path: Path) -> None:
    replay = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1)],
            "ticker": ["UP"],
            "relative_ema_ratio_10_30": [1.1],
        }
    )
    run_dir, candidates = _write_boosting_fixture(tmp_path, ((10, 30),), replay)

    with pytest.raises(ValueError, match="exactly match predictions"):
        build_causal_trend_eligibility_registry(
            run_dir,
            candidates.with_columns(pl.lit("OTHER").alias("ticker")),
        )

    replay_path = run_dir / "classification_h06/fold_01/oos_replay.parquet"
    replay.with_columns(pl.lit(0.8).alias("relative_ema_ratio_10_30")).write_parquet(replay_path)
    with pytest.raises(ValueError, match="hash does not match"):
        build_causal_trend_eligibility_registry(run_dir, candidates)


def test_native_and_trend_universes_are_ranked_separately() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2025, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "score": [3.0, 2.0, 1.0],
            "future_return_1m": [0.1, 0.2, 0.3],
            "benchmark_future_return_1m": [0.0, 0.0, 0.0],
        }
    )

    holdings = build_boosting_holdings(
        (("native", predictions), ("causal_trend", predictions.filter(pl.col("ticker") != "A"))),
        top_n_values=(1,),
    )

    assert holdings.select("strategy", "ticker").sort("strategy").to_dicts() == [
        {"strategy": "Boosting Top 1", "ticker": "A"},
        {"strategy": "Boosting Top 1 | Causal trend", "ticker": "B"},
    ]


def _write_boosting_fixture(
    root: Path,
    pairs: tuple[tuple[int, int], ...],
    replay: pl.DataFrame,
) -> tuple[Path, pl.DataFrame]:
    run_dir = root / "boosting"
    classification_dir = run_dir / "classification_h06"
    fold_dir = classification_dir / "fold_01"
    fold_dir.mkdir(parents=True)
    replay_path = fold_dir / "oos_replay.parquet"
    replay.write_parquet(replay_path)
    (fold_dir / "oos_replay_manifest.json").write_text(
        json.dumps(
            {
                "fold": 1,
                "oos_replay_file": replay_path.name,
                "oos_replay_sha256": _hash(replay_path),
            }
        ),
        encoding="utf-8",
    )
    pl.DataFrame(
        {
            "fold": [1],
            "train_cutoff": ["2024-12-01"],
            "winner_pair_count": [len(pairs)],
            "winner_pairs": [json.dumps(pairs)],
        }
    ).write_csv(classification_dir / "fold_feature_manifest.csv")
    candidates = replay.select("decision_month", "ticker").with_columns(
        pl.lit(1).cast(pl.Int32).alias("fold"),
        pl.lit(1.0).alias("score"),
        pl.lit(0.1).alias("future_return_1m"),
    )
    return run_dir, candidates


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
