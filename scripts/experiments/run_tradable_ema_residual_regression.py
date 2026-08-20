from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import polars as pl
from run_ema_rich_future_target_models import (  # noqa: E402
    _recomposition_by_month,
    _recomposition_summary,
)
from run_signal_copy_models import (  # noqa: E402
    DEFAULT_LEGACY_PATH,
    DEFAULT_SOURCE_RUN,
    _append_legacy,
    _load_legacy_labels,
)
from run_tradable_ema_regression_optuna import (  # noqa: E402
    _add_cross_sectional_features,
    _ema_base_features,
    _fit_mlcraft_regressor,
    _matrix,
    _predict,
    _target,
    _technical_base_features,
)

from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows

DEFAULT_BASE_WARM_START_JSON = Path("outputs/tradable_ema_regression_optuna_20260621_003954/warm_start_candidates.json")
BASE_FALLBACK_PARAMS: dict[str, Any] = {
    "num_boost_round": 925,
    "learning_rate": 0.01882630757879268,
    "max_depth": 3,
    "subsample": 0.9384254969886616,
    "colsample_bytree": 0.768734334773562,
    "min_child_weight": 16.583657065507783,
    "gamma": 0.09688026169284028,
    "alpha": 0.3112792901752597,
    "lambda": 3.2971804082507203,
}
RESIDUAL_DEFAULT_PARAMS: dict[str, Any] = {
    "num_boost_round": 650,
    "learning_rate": 0.025,
    "max_depth": 3,
    "subsample": 0.86,
    "colsample_bytree": 0.86,
    "min_child_weight": 4.0,
    "gamma": 0.0,
    "alpha": 0.0,
    "lambda": 2.0,
}


@dataclass(frozen=True)
class TradableEmaResidualRegressionConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    base_warm_start_json: Path | None = DEFAULT_BASE_WARM_START_JSON
    max_windows: int = 24
    min_train_months: int = 60
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    target_clip: float = 0.30
    residual_shrinkages: tuple[float, ...] = (0.25, 0.50, 1.0)
    seed: int = 42


def _score_name(shrinkage: float) -> str:
    token = f"{shrinkage:.2f}".replace(".", "_")
    return f"ema_plus_residual_{token}_regression"


def _load_best_params(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return dict(BASE_FALLBACK_PARAMS)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("warm_start_params", [])
    if not rows:
        return dict(BASE_FALLBACK_PARAMS)
    return dict(rows[0].get("params", BASE_FALLBACK_PARAMS))


def _non_ema_technical_features(columns: Iterable[str]) -> list[str]:
    ema_features = set(_ema_base_features(columns))
    return [feature for feature in _technical_base_features(columns) if feature not in ema_features]


def _load_frame(config: TradableEmaResidualRegressionConfig) -> tuple[pl.DataFrame, list[str], list[str]]:
    meta = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(meta["features_used"])
    ema_base = _ema_base_features(source_features)
    residual_base = _non_ema_technical_features(source_features)
    if not ema_base:
        raise ValueError("No EMA base features found in source metadata.")
    if not residual_base:
        raise ValueError("No non-EMA technical residual features found in source metadata.")

    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    frame, ema_features = _add_cross_sectional_features(frame, ema_base, prefix="ema")
    frame, residual_features = _add_cross_sectional_features(frame, residual_base, prefix="technical_residual")
    legacy = _load_legacy_labels(config.legacy_path)
    return _append_legacy(frame, legacy), ema_features, residual_features


def _score_predictions(
    frame: pl.DataFrame,
    *,
    base_scores: np.ndarray,
    residual_scores: np.ndarray,
    shrinkages: Sequence[float],
) -> pl.DataFrame:
    columns = [
        pl.Series("ema_base_regression", base_scores, dtype=pl.Float64),
        pl.Series("ema_residual_component", residual_scores, dtype=pl.Float64),
    ]
    for shrinkage in shrinkages:
        columns.append(
            pl.Series(_score_name(shrinkage), base_scores + float(shrinkage) * residual_scores, dtype=pl.Float64)
        )
    return frame.select(["ticker", "year_month", "holding_month", "future_excess_return", "legacy_selected"]).with_columns(
        columns
    )


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 2:
        return float("nan")
    left = a[mask]
    right = b[mask]
    if float(np.std(left)) < 1e-12 or float(np.std(right)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def _rank(values: np.ndarray) -> np.ndarray:
    return (
        pl.DataFrame({"value": values.astype(float)})
        .with_columns(pl.col("value").rank(method="average").alias("rank"))
        .get_column("rank")
        .to_numpy()
    )


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _corr(_rank(a), _rank(b))


def _prediction_metric_rows(predictions: pl.DataFrame, score_cols: Sequence[str], target_clip: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_raw = predictions.get_column("future_excess_return").to_numpy().astype(float)
    y_clip = np.clip(y_raw, -target_clip, target_clip)
    for score_col in score_cols:
        scores = predictions.get_column(score_col).to_numpy().astype(float)
        monthly_rows: list[dict[str, float]] = []
        for _month, month_df in predictions.partition_by("year_month", as_dict=True).items():
            ranked = month_df.sort(score_col, descending=True)
            month_y = month_df.get_column("future_excess_return").to_numpy().astype(float)
            month_scores = month_df.get_column(score_col).to_numpy().astype(float)
            monthly_rows.append(
                {
                    "spearman": _spearman(month_scores, month_y),
                    "top5_excess_mean": float(ranked.head(5).get_column("future_excess_return").mean()),
                    "top7_excess_mean": float(ranked.head(7).get_column("future_excess_return").mean()),
                    "top10_excess_mean": float(ranked.head(10).get_column("future_excess_return").mean()),
                    "top10_hit_rate_gt0": float((ranked.head(10).get_column("future_excess_return") > 0).mean()),
                }
            )
        monthly = pl.DataFrame(monthly_rows)
        rows.append(
            {
                "model": score_col,
                "rows": predictions.height,
                "months": predictions.select("year_month").n_unique(),
                "rmse_clipped": float(np.sqrt(np.mean((scores - y_clip) ** 2))),
                "mae_clipped": float(np.mean(np.abs(scores - y_clip))),
                "pearson_clipped": _corr(scores, y_clip),
                "spearman_global": _spearman(scores, y_raw),
                "monthly_spearman_mean": float(monthly.get_column("spearman").mean()),
                "monthly_spearman_median": float(monthly.get_column("spearman").median()),
                "monthly_top5_excess_mean": float(monthly.get_column("top5_excess_mean").mean()),
                "monthly_top7_excess_mean": float(monthly.get_column("top7_excess_mean").mean()),
                "monthly_top10_excess_mean": float(monthly.get_column("top10_excess_mean").mean()),
                "monthly_top10_hit_rate_gt0": float(monthly.get_column("top10_hit_rate_gt0").mean()),
            }
        )
    return rows


def run(config: TradableEmaResidualRegressionConfig) -> Path:
    run_dir = config.output_dir / f"tradable_ema_residual_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, ema_features, residual_features = _load_frame(config)
    base_params = _load_best_params(config.base_warm_start_json)
    residual_params = dict(RESIDUAL_DEFAULT_PARAMS)
    months = frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )

    prediction_frames: list[pl.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    score_cols = ["ema_base_regression"] + [_score_name(shrinkage) for shrinkage in config.residual_shrinkages]

    for position, window in enumerate(windows, start=1):
        fold_label = f"fold_{window.fold_index:03d}"
        train_df = filter_by_months(frame, window.train_months).filter(pl.col("future_excess_return").is_not_null())
        val_df = filter_by_months(frame, window.val_months).filter(pl.col("future_excess_return").is_not_null())
        test_df = filter_by_months(frame, window.test_months).filter(pl.col("future_excess_return").is_not_null())
        if train_df.height < 250 or val_df.height < 80 or test_df.height < 80:
            continue

        print(f"[{position}/{len(windows)}] {fold_label} train={train_df.height} val={val_df.height} test={test_df.height}", flush=True)
        seed = config.seed + int(window.fold_index)
        fit_df = pl.concat([train_df, val_df], how="vertical")

        base_model = _fit_mlcraft_regressor(
            params=base_params,
            X=_matrix(fit_df, ema_features),
            y=_target(fit_df, config.target_clip),
            seed=seed,
        )
        fit_base_scores = _predict(base_model, _matrix(fit_df, ema_features))
        residual_target = _target(fit_df, config.target_clip) - fit_base_scores.astype(np.float32)
        residual_model = _fit_mlcraft_regressor(
            params=residual_params,
            X=_matrix(fit_df, residual_features),
            y=residual_target.astype(np.float32),
            seed=seed,
        )

        test_base_scores = _predict(base_model, _matrix(test_df, ema_features))
        test_residual_scores = _predict(residual_model, _matrix(test_df, residual_features))
        fold_predictions = _score_predictions(
            test_df,
            base_scores=test_base_scores,
            residual_scores=test_residual_scores,
            shrinkages=config.residual_shrinkages,
        )
        prediction_frames.append(fold_predictions.with_columns(pl.lit(fold_label).alias("fold")))
        y_test = _target(test_df, config.target_clip)
        fold_row: dict[str, Any] = {
            "fold": fold_label,
            "train_month_start": str(window.train_months[0]),
            "train_month_end": str(window.train_months[-1]),
            "val_month_start": str(window.val_months[0]),
            "val_month_end": str(window.val_months[-1]),
            "test_month_start": str(window.test_months[0]),
            "test_month_end": str(window.test_months[-1]),
            "base_rmse_clipped": float(np.sqrt(np.mean((test_base_scores - y_test) ** 2))),
            "fit_residual_target_std": float(np.std(residual_target)),
            "test_residual_prediction_std": float(np.std(test_residual_scores)),
        }
        for shrinkage in config.residual_shrinkages:
            score_col = _score_name(shrinkage)
            score = fold_predictions.get_column(score_col).to_numpy()
            fold_row[f"{score_col}_rmse_clipped"] = float(np.sqrt(np.mean((score - y_test) ** 2)))
        fold_rows.append(fold_row)

    predictions = pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame()
    recomposition = _recomposition_by_month(predictions, score_cols) if not predictions.is_empty() else pl.DataFrame()
    summary = _recomposition_summary(recomposition) if not recomposition.is_empty() else pl.DataFrame()
    prediction_metrics = (
        pl.DataFrame(_prediction_metric_rows(predictions, score_cols, config.target_clip))
        if not predictions.is_empty()
        else pl.DataFrame()
    )
    folds = pl.DataFrame(fold_rows) if fold_rows else pl.DataFrame()

    predictions.write_parquet(run_dir / "predictions.parquet")
    recomposition.write_csv(run_dir / "recomposition_by_month.csv")
    summary.write_csv(run_dir / "recomposition_summary.csv")
    prediction_metrics.write_csv(run_dir / "prediction_metrics.csv")
    folds.write_csv(run_dir / "fold_metrics.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "base_warm_start_json": str(config.base_warm_start_json) if config.base_warm_start_json else None,
                "max_windows": config.max_windows,
                "min_train_months": config.min_train_months,
                "val_months": config.val_months,
                "test_months": config.test_months,
                "step_months": config.step_months,
                "target": f"future_excess_return clipped to +/-{config.target_clip}",
                "method": "Base mlcraft XGBoost regression on EMA-only features, then residual mlcraft XGBoost regression on non-EMA technical features.",
                "base_score_col": "ema_base_regression",
                "residual_component_col": "ema_residual_component",
                "score_cols": score_cols,
                "base_params": base_params,
                "residual_params": residual_params,
                "ema_features": ema_features,
                "residual_features": residual_features,
                "residual_shrinkages": list(config.residual_shrinkages),
                "primary_metric": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "prediction_metrics": "prediction_metrics.csv",
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(summary)
    print(prediction_metrics)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test EMA base regression plus non-EMA technical residual regression.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--base-warm-start-json", type=Path, default=DEFAULT_BASE_WARM_START_JSON)
    parser.add_argument("--max-windows", type=int, default=24)
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--target-clip", type=float, default=0.30)
    parser.add_argument("--residual-shrinkages", type=float, nargs="*", default=[0.25, 0.50, 1.0])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        TradableEmaResidualRegressionConfig(
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            output_dir=args.output_dir,
            base_warm_start_json=args.base_warm_start_json,
            max_windows=args.max_windows,
            min_train_months=args.min_train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
            target_clip=args.target_clip,
            residual_shrinkages=tuple(args.residual_shrinkages),
        )
    )


if __name__ == "__main__":
    main()
