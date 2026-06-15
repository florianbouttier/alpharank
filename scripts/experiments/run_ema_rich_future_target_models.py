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
    _load_legacy_labels,
    _matrix,
    _month_rank_target,
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


def _ticker_list(values: Sequence[str]) -> str:
    return ",".join(sorted(str(value) for value in values))


def _recomposition_by_month(predictions: pl.DataFrame, model_cols: Sequence[str]) -> pl.DataFrame:
    rows: list[dict] = []
    for month_df in predictions.partition_by("holding_month", maintain_order=True):
        month = month_df.get_column("holding_month")[0]
        legacy_tickers = set(
            month_df.filter(pl.col("legacy_selected") == 1).get_column("ticker").cast(pl.Utf8).to_list()
        )
        legacy_count = len(legacy_tickers)
        if legacy_count <= 0:
            continue
        for model_name in model_cols:
            selected_tickers = set(_top_n(month_df, model_name, legacy_count).get_column("ticker").cast(pl.Utf8).to_list())
            common_tickers = selected_tickers & legacy_tickers
            rows.append(
                {
                    "model": model_name,
                    "holding_month": month,
                    "common_count": len(common_tickers),
                    "legacy_count": legacy_count,
                    "recomposition_pct": len(common_tickers) / legacy_count,
                    "legacy_tickers": _ticker_list(legacy_tickers),
                    "model_tickers": _ticker_list(selected_tickers),
                    "common_tickers": _ticker_list(common_tickers),
                    "missed_legacy_tickers": _ticker_list(legacy_tickers - selected_tickers),
                    "added_model_tickers": _ticker_list(selected_tickers - legacy_tickers),
                }
            )
    return pl.DataFrame(rows).sort(["model", "holding_month"])


def _recomposition_summary(recomposition: pl.DataFrame) -> pl.DataFrame:
    return (
        recomposition.group_by("model")
        .agg(
            pl.sum("common_count").alias("common_count"),
            pl.sum("legacy_count").alias("legacy_count"),
            (pl.sum("common_count") / pl.sum("legacy_count")).alias("recomposition_pct"),
            pl.mean("recomposition_pct").alias("mean_monthly_recomposition_pct"),
            pl.median("recomposition_pct").alias("median_monthly_recomposition_pct"),
            pl.len().alias("months"),
        )
        .sort("recomposition_pct", descending=True)
    )


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
        y_train_gt5 = _binary_target(train_df, config.threshold)
        y_test_gt5 = _binary_target(test_df, config.threshold)
        y_train_gt0 = _binary_target(train_df, 0.0)
        if np.unique(y_train_gt5).size < 2 or np.unique(y_train_gt0).size < 2:
            continue

        X_train = _matrix(train_df, rich_features)
        X_test = _matrix(test_df, rich_features)

        classifier_gt5 = _train_xgb(
            xgb,
            X_train,
            y_train_gt5,
            params=_classifier_params(seed),
            rounds=250,
        )
        classifier_gt5_score = _predict_xgb(xgb, classifier_gt5, X_test)

        classifier_gt0 = _train_xgb(
            xgb,
            X_train,
            y_train_gt0,
            params=_classifier_params(seed),
            rounds=250,
        )
        classifier_gt0_score = _predict_xgb(xgb, classifier_gt0, X_test)

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
            test_df.select(IDENTITY_COLS + ["legacy_selected", "ema_top25_vote_count"]).with_columns(
                pl.Series("target_gt5", y_test_gt5, dtype=pl.Int8),
                pl.col("ema_top25_vote_count").cast(pl.Float64).alias("ema_signal_count"),
                pl.Series("future_excess_classifier_gt5", classifier_gt5_score, dtype=pl.Float64),
                pl.Series("future_excess_classifier_gt0", classifier_gt0_score, dtype=pl.Float64),
                pl.Series("future_excess_regression", regression_score, dtype=pl.Float64),
                pl.Series("future_excess_rank_pairwise", rank_score, dtype=pl.Float64),
                pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
            ).drop("ema_top25_vote_count")
        )
        print(f"[{position}/{len(windows)}] {window.test_months[0]} rows={test_df.height}")

    predictions = pl.concat(prediction_frames, how="vertical")
    model_cols = [
        "ema_signal_count",
        "future_excess_regression",
        "future_excess_classifier_gt0",
        "future_excess_classifier_gt5",
        "future_excess_rank_pairwise",
    ]

    recomposition = _recomposition_by_month(predictions, model_cols)
    summary = _recomposition_summary(recomposition)

    predictions.write_parquet(run_dir / "predictions.parquet")
    recomposition.write_csv(run_dir / "recomposition_by_month.csv")
    summary.write_csv(run_dir / "recomposition_summary.csv")
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
                "primary_metric": "recomposition_pct",
                "primary_metric_formula": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "selection_note": "Chaque mois, le modele choisit exactement le meme nombre de tickers que Legacy.",
                "target_note": "Les modeles apprennent seulement des labels future_excess_return ; Legacy sert uniquement au diagnostic de recomposition.",
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(summary)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train future-return models with month-normalized EMA features.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--max-windows", type=int, default=999)
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
            threshold=args.threshold,
        )
    )


if __name__ == "__main__":
    main()
