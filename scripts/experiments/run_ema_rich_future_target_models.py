from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from run_signal_copy_models import (  # noqa: E402
    DEFAULT_LEGACY_PATH,
    DEFAULT_SOURCE_RUN,
    IDENTITY_COLS,
    _append_legacy,
    _binary_target,
    _classifier_params,
    _group_sizes,
    _kpis,
    _load_legacy_labels,
    _matrix,
    _month_rank_target,
    _monthly_returns,
    _rank_params,
    _regression_params,
    _top_n,
    _train_xgb,
    _predict_xgb,
)
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows  # noqa: E402
from alpharank.utils.xgboost_runtime import load_xgboost  # noqa: E402


@dataclass(frozen=True)
class EmaRichFutureConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    max_windows: int = 12
    min_train_months: int = 24
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    top_n: int = 10
    threshold: float = 0.05
    seed: int = 42


def _ema_features(features: Sequence[str]) -> list[str]:
    return [feature for feature in features if feature.startswith("ema_ratio_") or feature.startswith("price_to_ema_")]


def _add_ema_rich_features(frame: pl.DataFrame, ema_features: Sequence[str]) -> tuple[pl.DataFrame, list[str]]:
    rank_cols = [f"{feature}_rank_month" for feature in ema_features]
    z_cols = [f"{feature}_z_month" for feature in ema_features]
    top_cols = [f"{feature}_top25_flag" for feature in ema_features]

    ranked = frame.with_columns(
        [
            (pl.col(feature).rank(method="average").over("year_month") / pl.len().over("year_month")).alias(rank_col)
            for feature, rank_col in zip(ema_features, rank_cols, strict=True)
        ]
    )
    zscored = ranked.with_columns(
        [
            pl.when(pl.col(feature).std().over("year_month") > 1e-12)
            .then(
                (pl.col(feature) - pl.col(feature).mean().over("year_month"))
                / pl.col(feature).std().over("year_month")
            )
            .otherwise(0.0)
            .alias(z_col)
            for feature, z_col in zip(ema_features, z_cols, strict=True)
        ]
    )
    flagged = zscored.with_columns(
        [(pl.col(rank_col) >= 0.75).cast(pl.Int8).alias(top_col) for rank_col, top_col in zip(rank_cols, top_cols, strict=True)]
    )
    enriched = flagged.with_columns(
        pl.mean_horizontal(rank_cols).alias("ema_rank_mean"),
        pl.max_horizontal(rank_cols).alias("ema_rank_max"),
        pl.mean_horizontal(z_cols).alias("ema_z_mean"),
        pl.max_horizontal(z_cols).alias("ema_z_max"),
        pl.sum_horizontal(top_cols).alias("ema_top25_vote_count"),
    )
    rich_features = (
        list(ema_features)
        + rank_cols
        + z_cols
        + top_cols
        + ["ema_rank_mean", "ema_rank_max", "ema_z_mean", "ema_z_max", "ema_top25_vote_count"]
    )
    return enriched, rich_features


def _clone_rows(predictions: pl.DataFrame, model_cols: Sequence[str]) -> list[dict]:
    rows: list[dict] = []
    for model_name in model_cols:
        total_overlap = 0
        total_legacy = 0
        for month_df in predictions.partition_by("holding_month", maintain_order=True):
            legacy_count = int(month_df.get_column("legacy_selected").sum())
            if legacy_count <= 0:
                continue
            top = _top_n(month_df, model_name, legacy_count)
            total_overlap += int(top.get_column("legacy_selected").sum())
            total_legacy += legacy_count
        rows.append(
            {
                "model": model_name,
                "recall_at_legacy_k": total_overlap / total_legacy if total_legacy else 0.0,
                "total_overlap": total_overlap,
                "total_legacy": total_legacy,
            }
        )
    return rows


def run_experiment(config: EmaRichFutureConfig) -> Path:
    xgb = load_xgboost()
    meta = json.loads((config.source_run / "metadata.json").read_text())
    base_features = list(meta["features_used"])
    ema_base_features = _ema_features(base_features)
    model_frame, rich_features = _add_ema_rich_features(
        pl.read_parquet(config.source_run / "model_frame.parquet"),
        ema_base_features,
    )
    legacy = _load_legacy_labels(config.legacy_path)

    months = model_frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )
    run_dir = config.output_dir / f"ema_rich_future_target_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pl.DataFrame] = []
    for position, window in enumerate(windows, start=1):
        train_df = _append_legacy(
            filter_by_months(model_frame, window.train_months).filter(pl.col("future_excess_return").is_not_null()),
            legacy,
        )
        test_df = _append_legacy(
            filter_by_months(model_frame, window.test_months).filter(pl.col("future_excess_return").is_not_null()),
            legacy,
        )
        if train_df.height < 250 or test_df.height < 80:
            continue

        seed = config.seed + int(window.fold_index)
        y_train = _binary_target(train_df, config.threshold)
        y_test = _binary_target(test_df, config.threshold)
        if np.unique(y_train).size < 2:
            continue

        X_train = _matrix(train_df, rich_features)
        X_test = _matrix(test_df, rich_features)

        classifier = _train_xgb(
            xgb,
            X_train,
            y_train,
            params=_classifier_params(seed),
            rounds=250,
        )
        classifier_score = _predict_xgb(xgb, classifier, X_test)

        y_reg = np.clip(train_df.get_column("future_excess_return").to_numpy(), -0.30, 0.30).astype(np.float32)
        regression = _train_xgb(
            xgb,
            X_train,
            y_reg,
            params=_regression_params(seed),
            rounds=250,
        )
        regression_score = _predict_xgb(xgb, regression, X_test)

        train_rank_df = train_df.sort("year_month")
        ranker = _train_xgb(
            xgb,
            _matrix(train_rank_df, rich_features),
            _month_rank_target(train_rank_df, "future_excess_return"),
            params=_rank_params(seed),
            rounds=250,
            group=_group_sizes(train_rank_df),
        )
        rank_score = _predict_xgb(xgb, ranker, X_test)

        prediction_frames.append(
            test_df.select(IDENTITY_COLS + ["legacy_selected"]).with_columns(
                pl.Series("target_label", y_test, dtype=pl.Int8),
                pl.Series("ema_rich_classifier", classifier_score, dtype=pl.Float64),
                pl.Series("ema_rich_regression", regression_score, dtype=pl.Float64),
                pl.Series("ema_rich_rank_pairwise", rank_score, dtype=pl.Float64),
                pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
            )
        )
        print(f"[{position}/{len(windows)}] {window.test_months[0]} rows={test_df.height}")

    predictions = pl.concat(prediction_frames, how="vertical")
    model_cols = ["ema_rich_classifier", "ema_rich_regression", "ema_rich_rank_pairwise"]

    monthly_frames: list[pl.DataFrame] = []
    kpi_rows: list[dict] = []
    for model_name in model_cols:
        selections = _top_n(predictions, model_name, config.top_n)
        monthly = _monthly_returns(selections).with_columns(pl.lit(model_name).alias("model"))
        monthly_frames.append(monthly)
        kpi_rows.append(_kpis(monthly, model_name))

    kpis = pl.DataFrame(kpi_rows).sort("active_compounded", descending=True)
    clone = pl.DataFrame(_clone_rows(predictions, model_cols)).sort("recall_at_legacy_k", descending=True)

    predictions.write_parquet(run_dir / "predictions.parquet")
    pl.concat(monthly_frames, how="vertical").write_parquet(run_dir / "monthly_returns.parquet")
    kpis.write_csv(run_dir / "kpis.csv")
    clone.write_csv(run_dir / "clone.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": config.__dict__ | {
                    "source_run": str(config.source_run),
                    "legacy_path": str(config.legacy_path),
                    "output_dir": str(config.output_dir),
                },
                "ema_base_features": ema_base_features,
                "features": rich_features,
                "models": model_cols,
                "target_note": "All models train on future_excess_return labels only; legacy labels are used only for diagnostics.",
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(kpis)
    print(clone)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future-return models with month-normalized EMA features.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_experiment(
        EmaRichFutureConfig(
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            output_dir=args.output_dir,
            max_windows=args.max_windows,
            top_n=args.top_n,
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
