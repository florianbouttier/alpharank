from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl
from run_ema_rich_future_target_models import (  # noqa: E402
    _recomposition_by_month,
    _recomposition_summary,
)
from run_signal_copy_models import (  # noqa: E402
    _binary_target,
    _classifier_params,
    _group_sizes,
    _matrix,
    _month_rank_target,
    _predict_xgb,
    _rank_params,
    _regression_params,
    _train_xgb,
)

from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows  # noqa: E402
from alpharank.utils.xgboost_runtime import load_xgboost  # noqa: E402


@dataclass(frozen=True)
class AtomicFutureTargetConfig:
    feature_frame_path: Path | None = None
    output_dir: Path = Path("outputs")
    max_windows: int = 999
    min_train_months: int = 24
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    threshold: float = 0.05
    seed: int = 42


def _latest_feature_frame(output_dir: Path) -> Path:
    candidates = sorted(output_dir.glob("legacy_atomic_feature_frame_*/legacy_atomic_feature_frame.parquet"))
    if not candidates:
        raise FileNotFoundError("No legacy atomic feature frame found. Run build_legacy_atomic_feature_frame.py first.")
    return candidates[-1]


def _feature_columns(frame: pl.DataFrame) -> tuple[list[str], list[str], list[str]]:
    atomic = [
        column
        for column in frame.columns
        if column.startswith("legacy_atomic_") or column.startswith("legacy_optuna_")
    ]
    ema = [
        column
        for column in frame.columns
        if column.startswith("ema_ratio_") or column.startswith("price_to_ema_")
    ]
    return atomic, ema, atomic + ema


def _train_predict_classifier(
    *,
    xgb,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    features: Sequence[str],
    threshold: float,
    seed: int,
) -> np.ndarray:
    y_train = _binary_target(train_df, threshold)
    if np.unique(y_train).size < 2:
        return np.zeros(test_df.height, dtype=float)
    model = _train_xgb(
        xgb,
        _matrix(train_df, features),
        y_train,
        params=_classifier_params(seed),
        rounds=250,
    )
    return _predict_xgb(xgb, model, _matrix(test_df, features))


def _train_predict_regression(*, xgb, train_df: pl.DataFrame, test_df: pl.DataFrame, features: Sequence[str], seed: int) -> np.ndarray:
    y_train = np.clip(train_df.get_column("future_excess_return").to_numpy(), -0.30, 0.30).astype(np.float32)
    model = _train_xgb(
        xgb,
        _matrix(train_df, features),
        y_train,
        params=_regression_params(seed),
        rounds=250,
    )
    return _predict_xgb(xgb, model, _matrix(test_df, features))


def _train_predict_ranker(*, xgb, train_df: pl.DataFrame, test_df: pl.DataFrame, features: Sequence[str], seed: int) -> np.ndarray:
    train_rank_df = train_df.sort("year_month")
    model = _train_xgb(
        xgb,
        _matrix(train_rank_df, features),
        _month_rank_target(train_rank_df, "future_excess_return"),
        params=_rank_params(seed),
        rounds=250,
        group=_group_sizes(train_rank_df),
    )
    return _predict_xgb(xgb, model, _matrix(test_df, features))


def run(config: AtomicFutureTargetConfig) -> Path:
    xgb = load_xgboost()
    feature_frame_path = config.feature_frame_path or _latest_feature_frame(config.output_dir)
    frame = pl.read_parquet(feature_frame_path).with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    atomic_features, ema_features, atomic_plus_ema_features = _feature_columns(frame)
    if not atomic_features:
        raise ValueError("No atomic feature columns found in feature frame.")

    months = frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )
    run_dir = config.output_dir / f"atomic_feature_future_target_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pl.DataFrame] = []
    for position, window in enumerate(windows, start=1):
        train_df = filter_by_months(frame, window.train_months).filter(pl.col("future_excess_return").is_not_null())
        test_df = filter_by_months(frame, window.test_months).filter(pl.col("future_excess_return").is_not_null())
        if train_df.height < 250 or test_df.height < 80:
            continue

        seed = config.seed + int(window.fold_index)
        predictions = test_df.select(
            [
                "ticker",
                "year_month",
                "holding_month",
                "future_excess_return",
                "legacy_selected",
                "legacy_atomic_vote_count",
                "legacy_atomic_max_quantile_mtr",
            ]
        ).with_columns(
            pl.Series(
                "atomic_classifier_gt5",
                _train_predict_classifier(
                    xgb=xgb,
                    train_df=train_df,
                    test_df=test_df,
                    features=atomic_features,
                    threshold=config.threshold,
                    seed=seed,
                ),
                dtype=pl.Float64,
            ),
            pl.Series(
                "atomic_classifier_gt0",
                _train_predict_classifier(
                    xgb=xgb,
                    train_df=train_df,
                    test_df=test_df,
                    features=atomic_features,
                    threshold=0.0,
                    seed=seed,
                ),
                dtype=pl.Float64,
            ),
            pl.Series(
                "atomic_regression",
                _train_predict_regression(xgb=xgb, train_df=train_df, test_df=test_df, features=atomic_features, seed=seed),
                dtype=pl.Float64,
            ),
            pl.Series(
                "atomic_rank_pairwise",
                _train_predict_ranker(xgb=xgb, train_df=train_df, test_df=test_df, features=atomic_features, seed=seed),
                dtype=pl.Float64,
            ),
            pl.Series(
                "atomic_plus_ema_rank_pairwise",
                _train_predict_ranker(
                    xgb=xgb,
                    train_df=train_df,
                    test_df=test_df,
                    features=atomic_plus_ema_features,
                    seed=seed,
                ),
                dtype=pl.Float64,
            ),
            pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
        )
        prediction_frames.append(predictions)
        print(f"[{position}/{len(windows)}] {window.test_months[0]} rows={test_df.height}", flush=True)

    predictions = pl.concat(prediction_frames, how="vertical")
    model_cols = [
        "legacy_atomic_vote_count",
        "legacy_atomic_max_quantile_mtr",
        "atomic_classifier_gt5",
        "atomic_classifier_gt0",
        "atomic_regression",
        "atomic_rank_pairwise",
        "atomic_plus_ema_rank_pairwise",
    ]
    recomposition = _recomposition_by_month(predictions, model_cols)
    summary = _recomposition_summary(recomposition)

    predictions.write_parquet(run_dir / "predictions.parquet")
    recomposition.write_csv(run_dir / "recomposition_by_month.csv")
    summary.write_csv(run_dir / "recomposition_summary.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "feature_frame_path": str(feature_frame_path),
                "atomic_feature_count": len(atomic_features),
                "ema_feature_count": len(ema_features),
                "models": model_cols,
                "primary_metric": "recomposition_pct",
                "primary_metric_formula": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "target_note": "Les modeles apprennent seulement future_excess_return ; legacy_selected sert uniquement au diagnostic de recomposition.",
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(summary)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future-return models on atomic Legacy signal features.")
    parser.add_argument("--feature-frame-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        AtomicFutureTargetConfig(
            feature_frame_path=args.feature_frame_path,
            output_dir=args.output_dir,
            max_windows=args.max_windows,
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
