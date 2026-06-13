from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import polars as pl

from alpharank.backtest.time_folds import (
    CombinatorialPurgedGroupTimeSeriesSplit,
    filter_by_months,
    walk_forward_windows,
)
from alpharank.backtest.tuning import safe_auc
from alpharank.utils.xgboost_runtime import load_xgboost


DEFAULT_SOURCE_RUN = Path("outputs/xgboost_timefold_backtest_20260612_175250")
DEFAULT_SHAP_PATH = Path("outputs/xgboost_timefold_backtest_20260611_013248/shap_feature_importance_exploratory.csv")

IDENTITY_COLS = [
    "ticker",
    "year_month",
    "decision_month",
    "holding_month",
    "decision_asof_date",
    "holding_asof_date",
    "benchmark_holding_asof_date",
    "holding_period_complete",
    "monthly_return",
    "future_return",
    "benchmark_future_return",
    "future_excess_return",
    "future_relative_return",
]


@dataclass(frozen=True)
class ExperimentConfig:
    source_run: Path
    shap_path: Path
    output_dir: Path
    top_features: int = 12
    max_windows: int = 12
    min_train_months: int = 24
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    top_n: int = 10
    threshold: float = 0.05
    inner_groups: int = 5
    seed: int = 42


def _sigmoid(raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -40.0, 40.0)))


def _logit(prob: np.ndarray) -> np.ndarray:
    prob = np.clip(np.asarray(prob, dtype=float), 1e-5, 1.0 - 1e-5)
    return np.log(prob / (1.0 - prob))


def _binary_target(frame: pl.DataFrame, threshold: float) -> np.ndarray:
    return (frame.get_column("future_excess_return").to_numpy() > float(threshold)).astype(np.int8)


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).to_numpy().astype(np.float32, copy=False)


def _load_features(config: ExperimentConfig, model_frame: pl.DataFrame) -> list[str]:
    if config.shap_path.exists():
        shap = pl.read_csv(config.shap_path)
        features = [
            row["feature"]
            for row in shap.sort("mean_abs_shap", descending=True).head(config.top_features).to_dicts()
            if row["feature"] in model_frame.columns
        ]
    else:
        features = []

    if len(features) < min(config.top_features, 6):
        fallback = [
            "volatility_12m",
            "volatility_24m",
            "volatility_36m",
            "asset_turnover_ttm",
            "volatility_48m",
            "total_revenue_ttm_growth_4q",
            "dist_to_21m_high",
            "bollinger_percent_b_6m",
            "dist_to_12m_high",
            "net_margin_ttm",
            "bollinger_bandwidth_12m",
            "total_revenue_ttm_growth_1q",
        ]
        features = [feature for feature in fallback if feature in model_frame.columns][: config.top_features]

    if not features:
        raise ValueError("No selected feature is available in the model frame.")
    return features


def _base_params(seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.012,
        "max_depth": 3,
        "subsample": 0.75,
        "colsample_bytree": 0.85,
        "min_child_weight": 3.0,
        "gamma": 2.0,
        "alpha": 1.0,
        "lambda": 2.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }


def _residual_params(seed: int) -> dict:
    return {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.02,
        "max_depth": 2,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 8.0,
        "gamma": 3.0,
        "alpha": 2.0,
        "lambda": 5.0,
        "seed": seed,
        "verbosity": 0,
        "nthread": -1,
    }


def _train_xgb(xgb, X: np.ndarray, y: np.ndarray, params: dict, rounds: int, base_margin: np.ndarray | None = None):
    dtrain = xgb.DMatrix(X, label=y)
    if base_margin is not None:
        dtrain.set_base_margin(base_margin)
    return xgb.train(params=params, dtrain=dtrain, num_boost_round=int(rounds), verbose_eval=False)


def _predict_xgb(xgb, model, X: np.ndarray, base_margin: np.ndarray | None = None) -> np.ndarray:
    dmatrix = xgb.DMatrix(X)
    if base_margin is not None:
        dmatrix.set_base_margin(base_margin)
    return np.asarray(model.predict(dmatrix), dtype=float)


def _oof_base_predictions(
    *,
    xgb,
    X_train: np.ndarray,
    y_train: np.ndarray,
    train_groups: Sequence,
    params: dict,
    rounds: int,
    inner_groups: int,
) -> np.ndarray:
    groups = list(train_groups)
    splitter = CombinatorialPurgedGroupTimeSeriesSplit(
        groups,
        n_groups=min(int(inner_groups), len(set(groups))),
        test_group_count=1,
        embargo_groups=0,
    )
    oof = np.full(y_train.shape[0], np.nan, dtype=float)
    for inner_train_idx, inner_val_idx in splitter.split(X_train, y_train):
        if np.unique(y_train[inner_train_idx]).size < 2:
            continue
        model = _train_xgb(
            xgb,
            X_train[inner_train_idx],
            y_train[inner_train_idx],
            params=params,
            rounds=rounds,
        )
        oof[inner_val_idx] = _predict_xgb(xgb, model, X_train[inner_val_idx])

    if np.isnan(oof).any():
        fallback = _train_xgb(xgb, X_train, y_train, params=params, rounds=rounds)
        fallback_pred = _predict_xgb(xgb, fallback, X_train)
        oof = np.where(np.isnan(oof), fallback_pred, oof)
    return oof


def _top_n_predictions(frame: pl.DataFrame, score_col: str, top_n: int) -> pl.DataFrame:
    return (
        frame.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("rank"))
        .filter(pl.col("rank") <= int(top_n))
        .sort(["year_month", "rank"])
    )


def _monthly_returns(selections: pl.DataFrame) -> pl.DataFrame:
    return (
        selections.group_by("holding_month")
        .agg(
            pl.mean("future_return").alias("portfolio_return"),
            pl.first("benchmark_future_return").alias("benchmark_return"),
            pl.mean("target_label").alias("hit_rate"),
            pl.len().alias("n_positions"),
        )
        .rename({"holding_month": "year_month"})
        .with_columns((pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return"))
        .sort("year_month")
    )


def _kpis(monthly: pl.DataFrame, strategy: str) -> dict:
    if monthly.is_empty():
        return {"strategy": strategy}
    total_return = float((1.0 + monthly.get_column("portfolio_return")).product() - 1.0)
    benchmark_return = float((1.0 + monthly.get_column("benchmark_return")).product() - 1.0)
    active_return = float((1.0 + monthly.get_column("active_return")).product() - 1.0)
    return {
        "strategy": strategy,
        "months": monthly.height,
        "total_return": total_return,
        "benchmark_return": benchmark_return,
        "active_return": active_return,
        "avg_active_return": float(monthly.get_column("active_return").mean()),
        "active_win_months": int((monthly.get_column("active_return") > 0).sum()),
        "avg_hit_rate": float(monthly.get_column("hit_rate").mean()),
        "worst_active_month": float(monthly.get_column("active_return").min()),
        "best_active_month": float(monthly.get_column("active_return").max()),
    }


def run_experiment(config: ExperimentConfig) -> Path:
    xgb = load_xgboost()
    model_frame = pl.read_parquet(config.source_run / "model_frame.parquet")
    features = _load_features(config, model_frame)

    months = model_frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )

    run_dir = config.output_dir / f"residual_init_score_experiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    prediction_frames: list[pl.DataFrame] = []
    fold_rows: list[dict] = []
    base_params = _base_params(config.seed)
    residual_params = _residual_params(config.seed)

    for position, window in enumerate(windows, start=1):
        train_df = filter_by_months(model_frame, window.train_months).filter(pl.col("future_excess_return").is_not_null())
        val_df = filter_by_months(model_frame, window.val_months).filter(pl.col("future_excess_return").is_not_null())
        test_df = filter_by_months(model_frame, window.test_months).filter(pl.col("future_excess_return").is_not_null())
        if train_df.height < 250 or val_df.height < 80 or test_df.height < 80:
            continue

        X_train = _matrix(train_df, features)
        X_val = _matrix(val_df, features)
        X_test = _matrix(test_df, features)
        y_train = _binary_target(train_df, config.threshold)
        y_val = _binary_target(val_df, config.threshold)
        y_test = _binary_target(test_df, config.threshold)
        if np.unique(y_train).size < 2:
            continue

        seed = config.seed + int(window.fold_index)
        fold_base_params = {**base_params, "seed": seed}
        fold_residual_params = {**residual_params, "seed": seed}

        base_oof = _oof_base_predictions(
            xgb=xgb,
            X_train=X_train,
            y_train=y_train,
            train_groups=train_df.get_column("year_month").to_list(),
            params=fold_base_params,
            rounds=300,
            inner_groups=config.inner_groups,
        )
        base_model = _train_xgb(xgb, X_train, y_train, params=fold_base_params, rounds=300)
        base_train = _predict_xgb(xgb, base_model, X_train)
        base_val = _predict_xgb(xgb, base_model, X_val)
        base_test = _predict_xgb(xgb, base_model, X_test)

        residual_model = _train_xgb(
            xgb,
            X_train,
            y_train,
            params=fold_residual_params,
            rounds=150,
            base_margin=_logit(base_oof),
        )
        residual_train = _predict_xgb(xgb, residual_model, X_train, base_margin=_logit(base_train))
        residual_val = _predict_xgb(xgb, residual_model, X_val, base_margin=_logit(base_val))
        residual_test = _predict_xgb(xgb, residual_model, X_test, base_margin=_logit(base_test))

        prediction_frames.append(
            test_df.select(IDENTITY_COLS).with_columns(
                pl.Series("base_prediction", base_test, dtype=pl.Float64),
                pl.Series("residual_prediction", residual_test, dtype=pl.Float64),
                pl.Series("target_label", y_test, dtype=pl.Int8),
                pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
            )
        )
        fold_rows.append(
            {
                "fold": window.fold_index,
                "position": position,
                "test_month": str(window.test_months[0]),
                "train_rows": train_df.height,
                "val_rows": val_df.height,
                "test_rows": test_df.height,
                "base_train_auc": safe_auc(y_train, base_train),
                "base_val_auc": safe_auc(y_val, base_val),
                "base_test_auc": safe_auc(y_test, base_test),
                "residual_train_auc": safe_auc(y_train, residual_train),
                "residual_val_auc": safe_auc(y_val, residual_val),
                "residual_test_auc": safe_auc(y_test, residual_test),
                "test_positive_rate": float(np.mean(y_test)),
            }
        )
        print(
            f"[{position}/{len(windows)}] {window.test_months[0]} "
            f"base_auc={fold_rows[-1]['base_test_auc']:.4f} "
            f"resid_auc={fold_rows[-1]['residual_test_auc']:.4f}"
        )

    predictions = pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame()
    base_sel = _top_n_predictions(predictions, "base_prediction", config.top_n)
    residual_sel = _top_n_predictions(predictions, "residual_prediction", config.top_n)
    base_monthly = _monthly_returns(base_sel)
    residual_monthly = _monthly_returns(residual_sel)
    fold_metrics = pl.DataFrame(fold_rows)
    kpis = pl.DataFrame([_kpis(base_monthly, "base_selected"), _kpis(residual_monthly, "residual_init_score")])

    predictions.write_parquet(run_dir / "predictions.parquet")
    base_sel.write_parquet(run_dir / "base_selections.parquet")
    residual_sel.write_parquet(run_dir / "residual_selections.parquet")
    base_monthly.write_parquet(run_dir / "base_monthly_returns.parquet")
    residual_monthly.write_parquet(run_dir / "residual_monthly_returns.parquet")
    fold_metrics.write_parquet(run_dir / "fold_metrics.parquet")
    kpis.write_parquet(run_dir / "kpis.parquet")
    kpis.write_csv(run_dir / "kpis.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(config.source_run),
                "shap_path": str(config.shap_path),
                "features": features,
                "config": config.__dict__ | {
                    "source_run": str(config.source_run),
                    "shap_path": str(config.shap_path),
                    "output_dir": str(config.output_dir),
                },
                "base_params": base_params,
                "residual_params": residual_params,
            },
            indent=2,
            default=str,
        )
    )
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test residual boosting with XGBoost base_margin on selected features.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--shap-path", type=Path, default=DEFAULT_SHAP_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top-features", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=12)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = run_experiment(
        ExperimentConfig(
            source_run=args.source_run,
            shap_path=args.shap_path,
            output_dir=args.output_dir,
            top_features=args.top_features,
            max_windows=args.max_windows,
            top_n=args.top_n,
            threshold=args.threshold,
        )
    )
    print(f"RUN_DIR={run_dir}")


if __name__ == "__main__":
    main()
