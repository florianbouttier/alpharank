from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import polars as pl

from alpharank.backtest.config import BacktestConfig
from alpharank.backtest.data_loading import load_raw_data
from alpharank.backtest.datasets import build_model_frame
from alpharank.backtest.explainability import (
    ShapFoldExplanation,
    collect_shap_explanation,
    generate_global_shap_report_pdf,
)
from alpharank.backtest.features import (
    compute_monthly_index_returns,
    compute_monthly_stock_prices,
    compute_technical_features,
)
from alpharank.backtest.fundamentals import build_monthly_fundamental_features
from alpharank.backtest.kpis import assert_no_numeric_na, compute_backtest_kpis, sanitize_numeric_frame
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n
from alpharank.backtest.reporting import (
    save_auc_score_overview,
    save_backtest_vs_sp500_plots,
    save_best_params_bar,
    save_bucket_frequency_curve,
    save_hyperparams_overview,
    save_learning_curve,
    save_lift_curve,
    save_optuna_trials_curve,
    save_optuna_visualizations,
    write_backtest_audit_report,
    write_html_report,
)
from alpharank.backtest.time_folds import filter_by_months, rolling_fold_windows
from alpharank.backtest.tuning import tune_and_fit_fold


@dataclass
class LearningArtifacts:
    run_dir: Path
    figures_dir: Path
    model_frame: pl.DataFrame
    predictions: pl.DataFrame
    fold_metrics: pl.DataFrame
    fold_index: pl.DataFrame
    best_params: pl.DataFrame
    features_used: List[str]
    dropped_features: List[str]
    fold_assets: List[Dict[str, Any]]
    shap_explanations: List[ShapFoldExplanation]
    total_windows: int


@dataclass
class BacktestPhaseArtifacts:
    selections: pl.DataFrame
    monthly_returns: pl.DataFrame
    backtest_kpis: pl.DataFrame
    split_kpis: pl.DataFrame
    fold_backtest_metrics: pl.DataFrame
    global_assets: Dict[str, Path]


@dataclass
class BacktestArtifacts:
    model_frame: pl.DataFrame
    predictions: pl.DataFrame
    selections: pl.DataFrame
    monthly_returns: pl.DataFrame
    kpis: pl.DataFrame
    split_kpis: pl.DataFrame
    fold_metrics: pl.DataFrame
    fold_index: pl.DataFrame
    best_params: pl.DataFrame
    features_used: List[str]
    dropped_features: List[str]
    output_paths: Dict[str, Path]


def _create_run_dir(output_dir: Path) -> Path:
    run_dir = output_dir / f"xgboost_timefold_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _format_eta(end_in_seconds: float) -> str:
    return (datetime.now() + timedelta(seconds=max(0.0, end_in_seconds))).strftime("%Y-%m-%d %H:%M:%S")


def _binary_target(df: pl.DataFrame, threshold: float) -> np.ndarray:
    return (df.get_column("future_excess_return").to_numpy() > threshold).astype(np.int8)


def _feature_matrix(df: pl.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return df.select(feature_cols).to_numpy()


def _positive_rate(y: np.ndarray) -> float:
    if y.size == 0:
        return 0.0
    return float(np.mean(y.astype(float)))


def _flatten_metrics(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items()}


def _save_fold_learning_outputs(
    fold_dir: Path,
    fold_predictions: pl.DataFrame,
    trials_rows: List[Dict[str, Any]],
    best_params: Dict[str, Any],
) -> None:
    fold_dir.mkdir(parents=True, exist_ok=True)
    fold_predictions.write_parquet(fold_dir / "predictions.parquet")

    if trials_rows:
        pl.DataFrame(trials_rows).write_csv(fold_dir / "optuna_trials.csv")
    else:
        pl.DataFrame({"trial_number": [], "objective": []}).write_csv(fold_dir / "optuna_trials.csv")

    (fold_dir / "best_params.json").write_text(json.dumps(best_params, indent=2), encoding="utf-8")


def _empty_predictions() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.Utf8,
            "year_month": pl.Date,
            "decision_month": pl.Date,
            "holding_month": pl.Date,
            "decision_asof_date": pl.Date,
            "holding_asof_date": pl.Date,
            "benchmark_holding_asof_date": pl.Date,
            "holding_period_complete": pl.Boolean,
            "monthly_return": pl.Float64,
            "future_return": pl.Float64,
            "benchmark_future_return": pl.Float64,
            "future_excess_return": pl.Float64,
            "future_relative_return": pl.Float64,
            "prediction": pl.Float64,
            "target_label": pl.Int8,
            "fold": pl.Int64,
            "objective_score": pl.Float64,
            "objective_score_val": pl.Float64,
        }
    )


def _empty_fold_metrics() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "fold": pl.Int64,
            "train_month_start": pl.Utf8,
            "train_month_end": pl.Utf8,
            "val_month_start": pl.Utf8,
            "val_month_end": pl.Utf8,
            "test_month_start": pl.Utf8,
            "test_month_end": pl.Utf8,
            "train_rows": pl.Int64,
            "val_rows": pl.Int64,
            "test_rows": pl.Int64,
            "objective_score": pl.Float64,
            "objective_score_val": pl.Float64,
            "train_auc": pl.Float64,
            "val_auc": pl.Float64,
            "test_auc": pl.Float64,
            "train_average_precision": pl.Float64,
            "val_average_precision": pl.Float64,
            "test_average_precision": pl.Float64,
            "train_logloss": pl.Float64,
            "val_logloss": pl.Float64,
            "test_logloss": pl.Float64,
            "train_brier": pl.Float64,
            "val_brier": pl.Float64,
            "test_brier": pl.Float64,
            "train_precision": pl.Float64,
            "val_precision": pl.Float64,
            "test_precision": pl.Float64,
            "train_recall": pl.Float64,
            "val_recall": pl.Float64,
            "test_recall": pl.Float64,
            "train_f1": pl.Float64,
            "val_f1": pl.Float64,
            "test_f1": pl.Float64,
            "train_accuracy": pl.Float64,
            "val_accuracy": pl.Float64,
            "test_accuracy": pl.Float64,
            "train_pred_positive_rate": pl.Float64,
            "val_pred_positive_rate": pl.Float64,
            "test_pred_positive_rate": pl.Float64,
            "train_realized_positive_rate": pl.Float64,
            "val_realized_positive_rate": pl.Float64,
            "test_realized_positive_rate": pl.Float64,
        }
    )


def _empty_fold_index() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "fold": pl.Int64,
            "status": pl.Utf8,
            "skip_reason": pl.Utf8,
            "train_month_start": pl.Utf8,
            "train_month_end": pl.Utf8,
            "val_month_start": pl.Utf8,
            "val_month_end": pl.Utf8,
            "test_month_start": pl.Utf8,
            "test_month_end": pl.Utf8,
            "train_rows": pl.Int64,
            "val_rows": pl.Int64,
            "test_rows": pl.Int64,
            "train_positive_rate": pl.Float64,
            "val_positive_rate": pl.Float64,
            "test_positive_rate": pl.Float64,
        }
    )


def _build_split_kpis(fold_metrics: pl.DataFrame) -> pl.DataFrame:
    base_metrics = [
        "auc",
        "average_precision",
        "logloss",
        "brier",
        "precision",
        "recall",
        "f1",
        "accuracy",
        "pred_positive_rate",
        "realized_positive_rate",
    ]

    if fold_metrics.is_empty():
        rows = []
        for split in ["train", "validation", "test"]:
            row: Dict[str, float | str] = {"split": split, "n_folds": 0.0}
            for metric in base_metrics:
                row[metric] = 0.0
            rows.append(row)
        return pl.DataFrame(rows)

    mapping = {"train": "train", "validation": "val", "test": "test"}
    rows = []
    for split_label, prefix in mapping.items():
        row: Dict[str, float | str] = {"split": split_label, "n_folds": float(fold_metrics.height)}
        for metric in base_metrics:
            col = f"{prefix}_{metric}"
            value = float(fold_metrics.get_column(col).mean() or 0.0) if col in fold_metrics.columns else 0.0
            row[metric] = value
        rows.append(row)

    return pl.DataFrame(rows)


def _prepare_modeling_frame(config: BacktestConfig) -> tuple[pl.DataFrame, List[str], List[str]]:
    raw = load_raw_data(config.data_dir)

    monthly_prices = compute_monthly_stock_prices(raw.final_price)
    index_monthly = compute_monthly_index_returns(raw.sp500_price)

    technical_features = compute_technical_features(monthly_prices)
    fundamental_features = build_monthly_fundamental_features(
        monthly_prices=monthly_prices,
        balance_sheet=raw.balance_sheet,
        income_statement=raw.income_statement,
        cash_flow=raw.cash_flow,
        earnings=raw.earnings,
    )

    model_frame, features_used, dropped_features = build_model_frame(
        monthly_prices=monthly_prices,
        technical_features=technical_features,
        fundamental_features=fundamental_features,
        index_monthly=index_monthly,
        constituents=raw.constituents,
        start_month=config.start_month,
        missing_feature_threshold=config.missing_feature_threshold,
    )

    return model_frame, features_used, dropped_features


def _build_fold_index_row(
    *,
    fold: int,
    status: str,
    skip_reason: str | None,
    train_months: List[Any],
    val_months: List[Any],
    test_months: List[Any],
    train_rows: int,
    val_rows: int,
    test_rows: int,
    train_positive_rate: float | None = None,
    val_positive_rate: float | None = None,
    test_positive_rate: float | None = None,
) -> Dict[str, Any]:
    return {
        "fold": int(fold),
        "status": status,
        "skip_reason": skip_reason,
        "train_month_start": str(train_months[0]) if train_months else None,
        "train_month_end": str(train_months[-1]) if train_months else None,
        "val_month_start": str(val_months[0]) if val_months else None,
        "val_month_end": str(val_months[-1]) if val_months else None,
        "test_month_start": str(test_months[0]) if test_months else None,
        "test_month_end": str(test_months[-1]) if test_months else None,
        "train_rows": int(train_rows),
        "val_rows": int(val_rows),
        "test_rows": int(test_rows),
        "train_positive_rate": train_positive_rate,
        "val_positive_rate": val_positive_rate,
        "test_positive_rate": test_positive_rate,
    }


def _build_prediction_debug_frames(
    *,
    model_frame: pl.DataFrame,
    predictions: pl.DataFrame,
    fold_index: pl.DataFrame,
    top_n: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    fold_meta_cols = [
        "fold",
        "status",
        "skip_reason",
        "train_month_start",
        "train_month_end",
        "val_month_start",
        "val_month_end",
        "test_month_start",
        "test_month_end",
        "train_positive_rate",
        "val_positive_rate",
        "test_positive_rate",
    ]

    if predictions.is_empty():
        scored_long = predictions.with_columns(
            pl.lit(None, dtype=pl.Int64).alias("prediction_rank_in_month"),
            pl.lit(False).alias("selected_top_n"),
            pl.lit(False).alias("is_scored"),
        )
    else:
        scored_long = (
            predictions.sort(["year_month", "prediction"], descending=[False, True])
            .with_columns(
                pl.col("prediction").rank(method="ordinal", descending=True).over("year_month").cast(pl.Int64).alias(
                    "prediction_rank_in_month"
                )
            )
            .with_columns(
                (pl.col("prediction_rank_in_month") <= pl.lit(top_n)).alias("selected_top_n"),
                pl.lit(True).alias("is_scored"),
            )
            .join(fold_index.select(fold_meta_cols), on="fold", how="left")
        )

    debug_core = [
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
    debug_pred = [
        "fold",
        "prediction",
        "target_label",
        "prediction_rank_in_month",
        "selected_top_n",
        "objective_score",
        "objective_score_val",
        "status",
        "skip_reason",
        "train_month_start",
        "train_month_end",
        "val_month_start",
        "val_month_end",
        "test_month_start",
        "test_month_end",
        "train_positive_rate",
        "val_positive_rate",
        "test_positive_rate",
        "is_scored",
    ]

    debug_long = (
        scored_long.select(debug_core + debug_pred)
        if not scored_long.is_empty()
        else pl.DataFrame(schema={**{col: model_frame.schema.get(col, pl.Float64) for col in debug_core}, **{
            "fold": pl.Int64,
            "prediction": pl.Float64,
            "target_label": pl.Int8,
            "prediction_rank_in_month": pl.Int64,
            "selected_top_n": pl.Boolean,
            "objective_score": pl.Float64,
            "objective_score_val": pl.Float64,
            "status": pl.Utf8,
            "skip_reason": pl.Utf8,
            "train_month_start": pl.Utf8,
            "train_month_end": pl.Utf8,
            "val_month_start": pl.Utf8,
            "val_month_end": pl.Utf8,
            "test_month_start": pl.Utf8,
            "test_month_end": pl.Utf8,
            "train_positive_rate": pl.Float64,
            "val_positive_rate": pl.Float64,
            "test_positive_rate": pl.Float64,
            "is_scored": pl.Boolean,
        }})
    )

    debug_full = (
        model_frame.join(
            scored_long.select(
                [
                    "ticker",
                    "year_month",
                    "fold",
                    "prediction",
                    "target_label",
                    "prediction_rank_in_month",
                    "selected_top_n",
                    "objective_score",
                    "objective_score_val",
                    "status",
                    "skip_reason",
                    "train_month_start",
                    "train_month_end",
                    "val_month_start",
                    "val_month_end",
                    "test_month_start",
                    "test_month_end",
                    "train_positive_rate",
                    "val_positive_rate",
                    "test_positive_rate",
                    "is_scored",
                ]
            ),
            on=["ticker", "year_month"],
            how="left",
        )
        .with_columns(
            pl.col("prediction").is_not_null().fill_null(False).alias("is_scored"),
            pl.col("future_excess_return").is_not_null().fill_null(False).alias("has_target"),
        )
        .sort(["year_month", "ticker"])
    )

    return debug_long.sort(["year_month", "prediction_rank_in_month", "ticker"]), debug_full


def run_learning_phase(config: BacktestConfig) -> LearningArtifacts:
    run_dir = _create_run_dir(config.output_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_frame, features_used, dropped_features = _prepare_modeling_frame(config)

    months = (
        model_frame.select("year_month")
        .unique()
        .sort("year_month")
        .get_column("year_month")
        .to_list()
    )

    windows = rolling_fold_windows(months, n_folds=config.n_folds)
    total_windows = len(windows)
    fold_durations: List[float] = []

    if config.verbose:
        print(
            "[Learning] start "
            f"months={len(months)} windows={total_windows} "
            f"trials_per_fold={config.n_optuna_trials}"
        )

    all_predictions: List[pl.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    fold_index_rows: List[Dict[str, Any]] = []
    best_param_rows: List[Dict[str, Any]] = []
    fold_assets: List[Dict[str, Any]] = []
    shap_explanations: List[ShapFoldExplanation] = []

    for window_position, window in enumerate(windows, start=1):
        fold_start = time.perf_counter()
        fold_label = f"fold_{window.fold_index:02d}"
        fold_dir = run_dir / fold_label
        fold_prefix = f"[Fold {window_position}/{total_windows} | {fold_label}]"

        train_df = filter_by_months(model_frame, window.train_months).filter(pl.col("future_excess_return").is_not_null())
        val_df = filter_by_months(model_frame, window.val_months).filter(pl.col("future_excess_return").is_not_null())
        test_df = filter_by_months(model_frame, window.test_months).filter(pl.col("future_excess_return").is_not_null())

        train_month_count = train_df.select(pl.col("year_month").n_unique()).item() if not train_df.is_empty() else 0

        if config.verbose:
            print(
                f"{fold_prefix} train={window.train_months[0]}->{window.train_months[-1]} "
                f"val={window.val_months[0]}->{window.val_months[-1]} "
                f"test={window.test_months[0]}->{window.test_months[-1]}"
            )
            print(
                f"{fold_prefix} rows train={train_df.height} val={val_df.height} test={test_df.height} "
                f"(train_months={train_month_count})"
            )

        if train_month_count < config.min_train_months:
            fold_index_rows.append(
                _build_fold_index_row(
                    fold=window.fold_index,
                    status="skipped",
                    skip_reason="min_train_months",
                    train_months=window.train_months,
                    val_months=window.val_months,
                    test_months=window.test_months,
                    train_rows=train_df.height,
                    val_rows=val_df.height,
                    test_rows=test_df.height,
                )
            )
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: train_months={train_month_count} < min_train_months={config.min_train_months}"
                )
            continue
        if train_df.height < config.fold_min_train_rows:
            fold_index_rows.append(
                _build_fold_index_row(
                    fold=window.fold_index,
                    status="skipped",
                    skip_reason="fold_min_train_rows",
                    train_months=window.train_months,
                    val_months=window.val_months,
                    test_months=window.test_months,
                    train_rows=train_df.height,
                    val_rows=val_df.height,
                    test_rows=test_df.height,
                )
            )
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: train_rows={train_df.height} < fold_min_train_rows={config.fold_min_train_rows}"
                )
            continue
        if val_df.height < config.fold_min_val_rows:
            fold_index_rows.append(
                _build_fold_index_row(
                    fold=window.fold_index,
                    status="skipped",
                    skip_reason="fold_min_val_rows",
                    train_months=window.train_months,
                    val_months=window.val_months,
                    test_months=window.test_months,
                    train_rows=train_df.height,
                    val_rows=val_df.height,
                    test_rows=test_df.height,
                )
            )
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: val_rows={val_df.height} < fold_min_val_rows={config.fold_min_val_rows}"
                )
            continue
        if test_df.height < config.fold_min_test_rows:
            fold_index_rows.append(
                _build_fold_index_row(
                    fold=window.fold_index,
                    status="skipped",
                    skip_reason="fold_min_test_rows",
                    train_months=window.train_months,
                    val_months=window.val_months,
                    test_months=window.test_months,
                    train_rows=train_df.height,
                    val_rows=val_df.height,
                    test_rows=test_df.height,
                )
            )
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: test_rows={test_df.height} < fold_min_test_rows={config.fold_min_test_rows}"
                )
            continue

        X_train = _feature_matrix(train_df, features_used)
        X_val = _feature_matrix(val_df, features_used)
        X_test = _feature_matrix(test_df, features_used)

        y_train = _binary_target(train_df, config.outperformance_threshold)
        y_val = _binary_target(val_df, config.outperformance_threshold)
        y_test = _binary_target(test_df, config.outperformance_threshold)
        train_positive_rate = _positive_rate(y_train)
        val_positive_rate = _positive_rate(y_val)
        test_positive_rate = _positive_rate(y_test)

        if config.verbose:
            print(
                f"{fold_prefix} target positive rate "
                f"train={train_positive_rate:.1%} "
                f"val={val_positive_rate:.1%} "
                f"test={test_positive_rate:.1%} "
                f"(outperformance_threshold={config.outperformance_threshold:.4f})"
            )

        fold_index_rows.append(
            _build_fold_index_row(
                fold=window.fold_index,
                status="completed",
                skip_reason=None,
                train_months=window.train_months,
                val_months=window.val_months,
                test_months=window.test_months,
                train_rows=train_df.height,
                val_rows=val_df.height,
                test_rows=test_df.height,
                train_positive_rate=train_positive_rate,
                val_positive_rate=val_positive_rate,
                test_positive_rate=test_positive_rate,
            )
        )

        tuned = tune_and_fit_fold(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            base_params=config.xgb_params,
            search_space=config.optuna_space,
            n_trials=config.n_optuna_trials,
            startup_trials=config.optuna_startup_trials,
            lambda_gap=config.optuna_lambda_gap,
            seed=config.random_seed + window.fold_index,
            progress_label=fold_prefix,
            show_progress=config.show_optuna_progress and config.verbose,
            progress_every=config.optuna_progress_every,
        )

        fold_score_train_test = float(
            tuned.test_auc - config.optuna_lambda_gap * abs(tuned.train_auc - tuned.test_auc)
        )

        fold_predictions = test_df.select(
            [
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
        ).with_columns(
            pl.Series("prediction", tuned.y_test_proba, dtype=pl.Float64),
            pl.Series("target_label", y_test, dtype=pl.Int8),
            pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
            pl.lit(fold_score_train_test).cast(pl.Float64).alias("objective_score"),
            pl.lit(tuned.objective_score).cast(pl.Float64).alias("objective_score_val"),
        )

        _save_fold_learning_outputs(
            fold_dir=fold_dir,
            fold_predictions=fold_predictions,
            trials_rows=tuned.trials_df,
            best_params=tuned.best_params,
        )

        fold_plot_assets: Dict[str, Any] = {"__label__": fold_label}

        learning_path = save_learning_curve(
            evals_result=tuned.evals_result,
            path=fold_dir / f"{fold_label}_learning_curve.png",
            fold_label=fold_label,
        )
        if learning_path is not None:
            fold_plot_assets["learning_curve"] = learning_path

        bucket_val = save_bucket_frequency_curve(
            y_true=y_val,
            y_score=tuned.y_val_proba,
            n_buckets=config.calibration_buckets,
            path=fold_dir / f"{fold_label}_validation_bucket_frequency.png",
            fold_label=fold_label,
            split_label="Validation",
        )
        if bucket_val is not None:
            fold_plot_assets["bucket_validation"] = bucket_val

        lift_val = save_lift_curve(
            y_true=y_val,
            y_score=tuned.y_val_proba,
            n_buckets=config.calibration_buckets,
            path=fold_dir / f"{fold_label}_validation_lift_curve.png",
            fold_label=fold_label,
            split_label="Validation",
        )
        if lift_val is not None:
            fold_plot_assets["lift_validation"] = lift_val

        bucket_test = save_bucket_frequency_curve(
            y_true=y_test,
            y_score=tuned.y_test_proba,
            n_buckets=config.calibration_buckets,
            path=fold_dir / f"{fold_label}_test_bucket_frequency.png",
            fold_label=fold_label,
            split_label="Test",
        )
        if bucket_test is not None:
            fold_plot_assets["bucket_test"] = bucket_test

        lift_test = save_lift_curve(
            y_true=y_test,
            y_score=tuned.y_test_proba,
            n_buckets=config.calibration_buckets,
            path=fold_dir / f"{fold_label}_test_lift_curve.png",
            fold_label=fold_label,
            split_label="Test",
        )
        if lift_test is not None:
            fold_plot_assets["lift_test"] = lift_test

        trial_plot = save_optuna_trials_curve(
            trials_rows=tuned.trials_df,
            path=fold_dir / f"{fold_label}_optuna_trials.png",
            fold_label=fold_label,
        )
        if trial_plot is not None:
            fold_plot_assets["optuna_trials"] = trial_plot

        hp_bar = save_best_params_bar(
            best_params=tuned.best_params,
            path=fold_dir / f"{fold_label}_best_hyperparams.png",
            fold_label=fold_label,
        )
        if hp_bar is not None:
            fold_plot_assets["best_hyperparams"] = hp_bar

        if config.save_optuna_all_plots:
            optuna_assets = save_optuna_visualizations(
                study=tuned.study,
                out_dir=fold_dir,
                fold_label=fold_label,
            )
            for key, value in optuna_assets.items():
                fold_plot_assets[key] = value

        shap_artifacts = collect_shap_explanation(
            model=tuned.model,
            X_test=X_test,
            feature_names=features_used,
            out_dir=fold_dir,
            fold_label=fold_label,
            max_samples=config.shap_sample_size,
            interaction_max_samples=min(config.shap_sample_size, 250),
        )
        for key, value in shap_artifacts.paths.items():
            fold_plot_assets[f"shap_{key}"] = value
        if shap_artifacts.explanation is not None:
            shap_explanations.append(shap_artifacts.explanation)

        fold_assets.append(fold_plot_assets)

        row: Dict[str, Any] = {
            "fold": window.fold_index,
            "train_month_start": str(window.train_months[0]),
            "train_month_end": str(window.train_months[-1]),
            "val_month_start": str(window.val_months[0]),
            "val_month_end": str(window.val_months[-1]),
            "test_month_start": str(window.test_months[0]),
            "test_month_end": str(window.test_months[-1]),
            "train_rows": tuned.train_size,
            "val_rows": tuned.val_size,
            "test_rows": tuned.test_size,
            "objective_score": fold_score_train_test,
            "objective_score_val": tuned.objective_score,
        }
        row.update(_flatten_metrics("train", tuned.train_metrics))
        row.update(_flatten_metrics("val", tuned.val_metrics))
        row.update(_flatten_metrics("test", tuned.test_metrics))

        fold_rows.append(row)

        numeric_params = {
            k: float(v)
            for k, v in tuned.best_params.items()
            if isinstance(v, (int, float))
        }
        best_param_rows.append({"fold": window.fold_index, **numeric_params})

        all_predictions.append(fold_predictions)

        fold_elapsed = time.perf_counter() - fold_start
        fold_durations.append(fold_elapsed)
        avg_fold_duration = float(np.mean(fold_durations)) if fold_durations else 0.0
        remaining_folds = max(total_windows - window_position, 0)
        eta_pipeline_seconds = avg_fold_duration * remaining_folds

        if config.verbose:
            print(
                f"{fold_prefix} completed "
                f"score_test_pen={fold_score_train_test:.4f} "
                f"(train_auc={tuned.train_auc:.4f}, val_auc={tuned.val_auc:.4f}, test_auc={tuned.test_auc:.4f}) "
                f"elapsed={_format_seconds(fold_elapsed)} "
                f"eta_learning={_format_seconds(eta_pipeline_seconds)} "
                f"finish_at~{_format_eta(eta_pipeline_seconds)}"
            )

    predictions = pl.concat(all_predictions, how="vertical") if all_predictions else _empty_predictions()
    fold_metrics = pl.DataFrame(fold_rows) if fold_rows else _empty_fold_metrics()
    fold_index = pl.DataFrame(fold_index_rows) if fold_index_rows else _empty_fold_index()
    best_params = pl.DataFrame(best_param_rows) if best_param_rows else pl.DataFrame(schema={"fold": pl.Int64})

    return LearningArtifacts(
        run_dir=run_dir,
        figures_dir=figures_dir,
        model_frame=model_frame,
        predictions=predictions,
        fold_metrics=fold_metrics,
        fold_index=fold_index,
        best_params=best_params,
        features_used=features_used,
        dropped_features=dropped_features,
        fold_assets=fold_assets,
        shap_explanations=shap_explanations,
        total_windows=total_windows,
    )


def run_backtest_phase(config: BacktestConfig, learning: LearningArtifacts) -> BacktestPhaseArtifacts:
    predictions = learning.predictions

    selections = select_top_n(predictions, top_n=config.top_n)

    monthly_returns = compute_monthly_portfolio_returns(
        selections.select(
            [
                "year_month",
                "decision_month",
                "holding_month",
                "future_return",
                "benchmark_future_return",
                "future_excess_return",
                "target_label",
                "prediction",
                "ticker",
            ]
        )
    )

    if selections.is_empty():
        fold_monthly = pl.DataFrame(
            schema={
                "fold": pl.Int64,
                "decision_month": pl.Date,
                "holding_month": pl.Date,
                "year_month": pl.Date,
                "portfolio_return": pl.Float64,
                "benchmark_return": pl.Float64,
                "hit_rate": pl.Float64,
                "n_positions": pl.Int64,
                "active_return": pl.Float64,
            }
        )
    else:
        fold_monthly = (
            selections.group_by(["fold", "holding_month"])
            .agg(
                pl.col("decision_month").min().alias("decision_month"),
                pl.mean("future_return").alias("portfolio_return"),
                pl.mean("benchmark_future_return").alias("benchmark_return"),
                pl.mean("target_label").alias("hit_rate"),
                pl.len().alias("n_positions"),
            )
            .with_columns(
                pl.col("benchmark_return").fill_null(0.0).alias("benchmark_return"),
                pl.col("hit_rate").fill_null(0.0).alias("hit_rate"),
            )
            .with_columns(
                (pl.col("portfolio_return") - pl.col("benchmark_return")).alias("active_return"),
                pl.col("holding_month").alias("year_month"),
            )
            .sort(["fold", "holding_month"])
        )

    fold_backtest_metrics = (
        fold_monthly.group_by("fold")
        .agg(
            ((pl.col("portfolio_return") + 1.0).product() - 1.0).alias("fold_portfolio_total_return"),
            ((pl.col("benchmark_return") + 1.0).product() - 1.0).alias("fold_benchmark_total_return"),
            pl.mean("hit_rate").alias("fold_avg_hit_rate"),
            pl.mean("n_positions").alias("fold_avg_positions"),
        )
        .with_columns(
            (pl.col("fold_portfolio_total_return") - pl.col("fold_benchmark_total_return")).alias(
                "fold_active_total_return"
            )
        )
        .sort("fold")
        if not fold_monthly.is_empty()
        else pl.DataFrame(
            schema={
                "fold": pl.Int64,
                "fold_portfolio_total_return": pl.Float64,
                "fold_benchmark_total_return": pl.Float64,
                "fold_avg_hit_rate": pl.Float64,
                "fold_avg_positions": pl.Float64,
                "fold_active_total_return": pl.Float64,
            }
        )
    )

    for row in fold_monthly.partition_by("fold", as_dict=True).items() if not fold_monthly.is_empty() else []:
        fold_key, frame = row
        fold_id = int(fold_key[0]) if isinstance(fold_key, tuple) else int(fold_key)
        fold_dir = learning.run_dir / f"fold_{fold_id:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(fold_dir / "monthly_returns.parquet")

    backtest_kpis = compute_backtest_kpis(
        monthly_returns=monthly_returns,
        risk_free_rate=config.risk_free_rate,
    )
    split_kpis = _build_split_kpis(learning.fold_metrics)

    global_assets: Dict[str, Path] = {}
    global_assets.update(save_auc_score_overview(learning.fold_metrics, learning.figures_dir))
    hp_global = save_hyperparams_overview(learning.best_params, learning.figures_dir)
    for key, value in hp_global.items():
        global_assets[f"hyperparam_{key}"] = value
    backtest_assets = save_backtest_vs_sp500_plots(monthly_returns, learning.figures_dir)
    for key, value in backtest_assets.items():
        global_assets[key] = value
    shap_report = generate_global_shap_report_pdf(
        explanations=learning.shap_explanations,
        out_path=learning.run_dir / "shap_global_report.pdf",
        max_features=config.shap_top_features,
    )
    if shap_report is not None:
        global_assets["shap_global_report"] = shap_report

    return BacktestPhaseArtifacts(
        selections=selections,
        monthly_returns=monthly_returns,
        backtest_kpis=backtest_kpis,
        split_kpis=split_kpis,
        fold_backtest_metrics=fold_backtest_metrics,
        global_assets=global_assets,
    )


def run_boosting_backtest(config: BacktestConfig) -> BacktestArtifacts:
    pipeline_start = time.perf_counter()

    if config.verbose:
        print("[Pipeline] phase 1/2: learning started")
    learning_start = time.perf_counter()
    learning = run_learning_phase(config)
    learning_elapsed = time.perf_counter() - learning_start
    if config.verbose:
        print(
            "[Pipeline] phase 1/2: learning completed "
            f"elapsed={_format_seconds(learning_elapsed)} "
            f"completed_folds={learning.fold_metrics.height}/{learning.total_windows}"
        )

    if config.verbose:
        print("[Pipeline] phase 2/2: backtest started")
    backtest_start = time.perf_counter()
    backtest_phase = run_backtest_phase(config, learning)
    backtest_elapsed = time.perf_counter() - backtest_start
    if config.verbose:
        print(
            "[Pipeline] phase 2/2: backtest completed "
            f"elapsed={_format_seconds(backtest_elapsed)} "
            f"months={backtest_phase.monthly_returns.height}"
        )

    fold_metrics = learning.fold_metrics.join(
        backtest_phase.fold_backtest_metrics,
        on="fold",
        how="left",
    )

    monthly_returns = sanitize_numeric_frame(backtest_phase.monthly_returns)
    fold_metrics = sanitize_numeric_frame(fold_metrics)
    fold_index = sanitize_numeric_frame(learning.fold_index)
    best_params = sanitize_numeric_frame(learning.best_params)
    backtest_kpis = sanitize_numeric_frame(backtest_phase.backtest_kpis)
    split_kpis = sanitize_numeric_frame(backtest_phase.split_kpis)
    debug_predictions_long, debug_predictions_full = _build_prediction_debug_frames(
        model_frame=learning.model_frame,
        predictions=learning.predictions,
        fold_index=fold_index,
        top_n=config.top_n,
    )

    assert_no_numeric_na(fold_metrics, context="fold_metrics")
    assert_no_numeric_na(backtest_kpis, context="backtest_kpis")
    assert_no_numeric_na(split_kpis, context="split_kpis")

    report_path = learning.run_dir / "training_report.html"
    write_html_report(
        title=config.report_title,
        output_path=report_path,
        backtest_kpis=backtest_kpis,
        split_kpis=split_kpis,
        fold_metrics=fold_metrics,
        best_params=best_params,
        global_assets=backtest_phase.global_assets,
        fold_assets=learning.fold_assets,
    )
    audit_report_path = learning.run_dir / "backtest_audit_report.html"
    write_backtest_audit_report(
        output_path=audit_report_path,
        monthly_returns=monthly_returns,
        selections=backtest_phase.selections,
        debug_predictions_long=debug_predictions_long,
        fold_index=fold_index,
        linked_artifacts={
            "selections": learning.run_dir / "selections.parquet",
            "debug_predictions_long": learning.run_dir / "debug_predictions_long.parquet",
            "debug_predictions_full": learning.run_dir / "debug_predictions_full.parquet",
            "fold_index": learning.run_dir / "fold_index.parquet",
            "monthly_returns": learning.run_dir / "monthly_returns.parquet",
        },
    )

    paths = {
        "run_dir": learning.run_dir,
        "model_frame": learning.run_dir / "model_frame.parquet",
        "predictions": learning.run_dir / "predictions.parquet",
        "selections": learning.run_dir / "selections.parquet",
        "monthly_returns": learning.run_dir / "monthly_returns.parquet",
        "fold_metrics": learning.run_dir / "fold_metrics.parquet",
        "fold_index": learning.run_dir / "fold_index.parquet",
        "best_params": learning.run_dir / "best_params.parquet",
        "split_kpis": learning.run_dir / "split_kpis.parquet",
        "kpis_parquet": learning.run_dir / "kpis.parquet",
        "kpis_csv": learning.run_dir / "kpis.csv",
        "debug_predictions_long": learning.run_dir / "debug_predictions_long.parquet",
        "debug_predictions_full": learning.run_dir / "debug_predictions_full.parquet",
        "report_html": report_path,
        "backtest_audit_report": audit_report_path,
        "metadata": learning.run_dir / "metadata.json",
    }
    shap_report_path = backtest_phase.global_assets.get("shap_global_report")
    if shap_report_path is not None:
        paths["shap_global_report"] = shap_report_path

    learning.model_frame.write_parquet(paths["model_frame"])
    learning.predictions.write_parquet(paths["predictions"])
    backtest_phase.selections.write_parquet(paths["selections"])
    monthly_returns.write_parquet(paths["monthly_returns"])
    fold_metrics.write_parquet(paths["fold_metrics"])
    fold_index.write_parquet(paths["fold_index"])
    best_params.write_parquet(paths["best_params"])
    split_kpis.write_parquet(paths["split_kpis"])
    backtest_kpis.write_parquet(paths["kpis_parquet"])
    backtest_kpis.write_csv(paths["kpis_csv"])
    debug_predictions_long.write_parquet(paths["debug_predictions_long"])
    debug_predictions_full.write_parquet(paths["debug_predictions_full"])

    metadata = {
        "created_at": datetime.now().isoformat(),
        "features_used": learning.features_used,
        "dropped_features": learning.dropped_features,
        "n_completed_folds": int(fold_metrics.height),
        "target_definition": "future_excess_return > outperformance_threshold",
        "config": {
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "start_month": config.start_month,
            "n_folds": config.n_folds,
            "top_n": config.top_n,
            "outperformance_threshold": config.outperformance_threshold,
            "min_train_months": config.min_train_months,
            "missing_feature_threshold": config.missing_feature_threshold,
            "n_optuna_trials": config.n_optuna_trials,
            "optuna_lambda_gap": config.optuna_lambda_gap,
            "optuna_startup_trials": config.optuna_startup_trials,
            "calibration_buckets": config.calibration_buckets,
            "save_optuna_all_plots": config.save_optuna_all_plots,
            "verbose": config.verbose,
            "show_optuna_progress": config.show_optuna_progress,
            "optuna_progress_every": config.optuna_progress_every,
            "risk_free_rate": config.risk_free_rate,
            "xgb_params": config.xgb_params,
            "optuna_space": config.optuna_space,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    total_elapsed = time.perf_counter() - pipeline_start
    if config.verbose:
        print(
            "[Pipeline] completed "
            f"elapsed={_format_seconds(total_elapsed)} "
            f"completed_folds={fold_metrics.height}/{learning.total_windows} "
            f"report={report_path}"
        )

    return BacktestArtifacts(
        model_frame=learning.model_frame,
        predictions=learning.predictions,
        selections=backtest_phase.selections,
        monthly_returns=monthly_returns,
        kpis=backtest_kpis,
        split_kpis=split_kpis,
        fold_metrics=fold_metrics,
        fold_index=fold_index,
        best_params=best_params,
        features_used=learning.features_used,
        dropped_features=learning.dropped_features,
        output_paths=paths,
    )
