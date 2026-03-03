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
from alpharank.backtest.explainability import generate_shap_plots
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
    save_best_params_bar,
    save_hyperparams_overview,
    save_learning_curve,
    save_lift_curve,
    save_optuna_trials_curve,
    write_html_report,
)
from alpharank.backtest.time_folds import filter_by_months, rolling_fold_windows
from alpharank.backtest.tuning import tune_and_fit_fold


@dataclass
class BacktestArtifacts:
    model_frame: pl.DataFrame
    predictions: pl.DataFrame
    selections: pl.DataFrame
    monthly_returns: pl.DataFrame
    kpis: pl.DataFrame
    fold_metrics: pl.DataFrame
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
    return (df.get_column("future_return").to_numpy() > threshold).astype(np.int8)


def _feature_matrix(df: pl.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return df.select(feature_cols).to_numpy()


def _fold_summary_returns(fold_returns: pl.DataFrame) -> Dict[str, float]:
    if fold_returns.is_empty():
        return {
            "fold_portfolio_total_return": 0.0,
            "fold_benchmark_total_return": 0.0,
            "fold_active_total_return": 0.0,
            "fold_avg_hit_rate": 0.0,
            "fold_avg_positions": 0.0,
        }

    p = fold_returns.get_column("portfolio_return").to_numpy()
    b = fold_returns.get_column("benchmark_return").to_numpy()

    return {
        "fold_portfolio_total_return": float(np.prod(1.0 + p) - 1.0),
        "fold_benchmark_total_return": float(np.prod(1.0 + b) - 1.0),
        "fold_active_total_return": float((np.prod(1.0 + p) - 1.0) - (np.prod(1.0 + b) - 1.0)),
        "fold_avg_hit_rate": float(fold_returns.get_column("hit_rate").mean() or 0.0),
        "fold_avg_positions": float(fold_returns.get_column("n_positions").mean() or 0.0),
    }


def _save_fold_outputs(
    fold_dir: Path,
    fold_predictions: pl.DataFrame,
    fold_selections: pl.DataFrame,
    fold_returns: pl.DataFrame,
    trials_rows: List[Dict[str, Any]],
    best_params: Dict[str, Any],
) -> Dict[str, Path]:
    fold_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "predictions": fold_dir / "predictions.parquet",
        "selections": fold_dir / "selections.parquet",
        "returns": fold_dir / "monthly_returns.parquet",
        "trials_csv": fold_dir / "optuna_trials.csv",
        "best_params_json": fold_dir / "best_params.json",
    }

    fold_predictions.write_parquet(paths["predictions"])
    fold_selections.write_parquet(paths["selections"])
    fold_returns.write_parquet(paths["returns"])

    if trials_rows:
        pl.DataFrame(trials_rows).write_csv(paths["trials_csv"])
    else:
        pl.DataFrame({"trial_number": [], "objective": []}).write_csv(paths["trials_csv"])

    paths["best_params_json"].write_text(json.dumps(best_params, indent=2), encoding="utf-8")
    return paths


def run_boosting_backtest(config: BacktestConfig) -> BacktestArtifacts:
    pipeline_start = time.perf_counter()
    raw = load_raw_data(config.data_dir)

    run_dir = _create_run_dir(config.output_dir)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

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
            "[Pipeline] start "
            f"months={len(months)} windows={total_windows} "
            f"trials_per_fold={config.n_optuna_trials} top_n={config.top_n}"
        )

    all_predictions: List[pl.DataFrame] = []
    all_selections: List[pl.DataFrame] = []
    all_returns: List[pl.DataFrame] = []
    fold_rows: List[Dict[str, Any]] = []
    best_param_rows: List[Dict[str, Any]] = []
    fold_assets: List[Dict[str, Any]] = []

    for window_position, window in enumerate(windows, start=1):
        fold_start = time.perf_counter()
        fold_label = f"fold_{window.fold_index:02d}"
        fold_dir = run_dir / fold_label
        fold_prefix = f"[Fold {window_position}/{total_windows} | {fold_label}]"

        train_df = filter_by_months(model_frame, window.train_months).filter(pl.col("future_return").is_not_null())
        val_df = filter_by_months(model_frame, window.val_months).filter(pl.col("future_return").is_not_null())
        test_df = filter_by_months(model_frame, window.test_months).filter(pl.col("future_return").is_not_null())

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
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: train_months={train_month_count} < min_train_months={config.min_train_months}"
                )
            continue
        if train_df.height < config.fold_min_train_rows:
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: train_rows={train_df.height} < fold_min_train_rows={config.fold_min_train_rows}"
                )
            continue
        if val_df.height < config.fold_min_val_rows:
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: val_rows={val_df.height} < fold_min_val_rows={config.fold_min_val_rows}"
                )
            continue
        if test_df.height < config.fold_min_test_rows:
            if config.verbose:
                print(
                    f"{fold_prefix} skipped: test_rows={test_df.height} < fold_min_test_rows={config.fold_min_test_rows}"
                )
            continue

        X_train = _feature_matrix(train_df, features_used)
        X_val = _feature_matrix(val_df, features_used)
        X_test = _feature_matrix(test_df, features_used)

        y_train = _binary_target(train_df, config.prediction_threshold)
        y_val = _binary_target(val_df, config.prediction_threshold)
        y_test = _binary_target(test_df, config.prediction_threshold)

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
            ["ticker", "year_month", "monthly_return", "future_return", "benchmark_future_return"]
        ).with_columns(
            pl.Series("prediction", tuned.y_test_proba, dtype=pl.Float64),
            pl.Series("target_label", y_test, dtype=pl.Int8),
            pl.lit(window.fold_index).cast(pl.Int64).alias("fold"),
            pl.lit(tuned.train_auc).cast(pl.Float64).alias("train_auc"),
            pl.lit(tuned.val_auc).cast(pl.Float64).alias("val_auc"),
            pl.lit(tuned.test_auc).cast(pl.Float64).alias("test_auc"),
            pl.lit(fold_score_train_test).cast(pl.Float64).alias("objective_score"),
            pl.lit(tuned.objective_score).cast(pl.Float64).alias("objective_score_val"),
        )

        fold_selections = select_top_n(fold_predictions, top_n=config.top_n)
        fold_returns = compute_monthly_portfolio_returns(fold_selections).with_columns(
            pl.lit(window.fold_index).cast(pl.Int64).alias("fold")
        )

        _save_fold_outputs(
            fold_dir=fold_dir,
            fold_predictions=fold_predictions,
            fold_selections=fold_selections,
            fold_returns=fold_returns,
            trials_rows=tuned.trials_df,
            best_params=tuned.best_params,
        )

        # Per-fold visual outputs
        fold_plot_assets: Dict[str, Any] = {"__label__": fold_label}

        learning_path = save_learning_curve(
            evals_result=tuned.evals_result,
            path=fold_dir / f"{fold_label}_learning_curve.png",
            fold_label=fold_label,
        )
        if learning_path is not None:
            fold_plot_assets["learning_curve"] = learning_path

        lift_path = save_lift_curve(
            y_true=y_test,
            y_score=tuned.y_test_proba,
            bins=config.lift_bins,
            path=fold_dir / f"{fold_label}_lift_curve.png",
            fold_label=fold_label,
        )
        if lift_path is not None:
            fold_plot_assets["lift_curve"] = lift_path

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

        shap_paths = generate_shap_plots(
            model=tuned.model,
            X_test=X_test,
            feature_names=features_used,
            predictions=tuned.y_test_proba,
            out_dir=fold_dir,
            fold_label=fold_label,
            max_samples=config.shap_sample_size,
            max_features=config.shap_top_features,
        )
        for key, value in shap_paths.items():
            fold_plot_assets[f"shap_{key}"] = value

        fold_assets.append(fold_plot_assets)

        fold_return_summary = _fold_summary_returns(fold_returns)
        fold_rows.append(
            {
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
                "train_auc": tuned.train_auc,
                "val_auc": tuned.val_auc,
                "test_auc": tuned.test_auc,
                "objective_score": fold_score_train_test,
                "objective_score_val": tuned.objective_score,
                **fold_return_summary,
            }
        )

        numeric_params = {
            k: float(v)
            for k, v in tuned.best_params.items()
            if isinstance(v, (int, float))
        }
        best_param_rows.append({"fold": window.fold_index, **numeric_params})

        all_predictions.append(fold_predictions)
        all_selections.append(fold_selections)
        all_returns.append(fold_returns)

        fold_elapsed = time.perf_counter() - fold_start
        fold_durations.append(fold_elapsed)
        avg_fold_duration = float(np.mean(fold_durations)) if fold_durations else 0.0
        remaining_folds = max(total_windows - window_position, 0)
        eta_pipeline_seconds = avg_fold_duration * remaining_folds

        if config.verbose:
            print(
                f"{fold_prefix} completed "
                f"score={fold_score_train_test:.4f} "
                f"(train_auc={tuned.train_auc:.4f}, val_auc={tuned.val_auc:.4f}, test_auc={tuned.test_auc:.4f}) "
                f"elapsed={_format_seconds(fold_elapsed)} "
                f"eta_pipeline={_format_seconds(eta_pipeline_seconds)} "
                f"finish_at~{_format_eta(eta_pipeline_seconds)}"
            )

    if all_predictions:
        predictions = pl.concat(all_predictions, how="vertical")
    else:
        predictions = pl.DataFrame(
            schema={
                "ticker": pl.Utf8,
                "year_month": pl.Date,
                "monthly_return": pl.Float64,
                "future_return": pl.Float64,
                "benchmark_future_return": pl.Float64,
                "prediction": pl.Float64,
                "target_label": pl.Int8,
                "fold": pl.Int64,
                "train_auc": pl.Float64,
                "val_auc": pl.Float64,
                "test_auc": pl.Float64,
                "objective_score": pl.Float64,
                "objective_score_val": pl.Float64,
            }
        )

    if all_selections:
        selections = pl.concat(all_selections, how="vertical")
    else:
        selections = predictions.head(0)

    if all_returns:
        monthly_returns = pl.concat(all_returns, how="vertical").sort(["year_month", "fold"])
    else:
        monthly_returns = pl.DataFrame(
            schema={
                "year_month": pl.Date,
                "portfolio_return": pl.Float64,
                "benchmark_return": pl.Float64,
                "active_return": pl.Float64,
                "hit_rate": pl.Float64,
                "n_positions": pl.Int64,
                "fold": pl.Int64,
            }
        )

    fold_metrics = pl.DataFrame(fold_rows) if fold_rows else pl.DataFrame(
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
            "train_auc": pl.Float64,
            "val_auc": pl.Float64,
            "test_auc": pl.Float64,
            "objective_score": pl.Float64,
            "objective_score_val": pl.Float64,
            "fold_portfolio_total_return": pl.Float64,
            "fold_benchmark_total_return": pl.Float64,
            "fold_active_total_return": pl.Float64,
            "fold_avg_hit_rate": pl.Float64,
            "fold_avg_positions": pl.Float64,
        }
    )

    best_params = pl.DataFrame(best_param_rows) if best_param_rows else pl.DataFrame(schema={"fold": pl.Int64})

    monthly_returns_for_kpi = (
        monthly_returns.select(
            ["year_month", "portfolio_return", "benchmark_return", "active_return", "hit_rate", "n_positions"]
        )
        .sort("year_month")
    )

    kpis = compute_backtest_kpis(
        monthly_returns=monthly_returns_for_kpi,
        risk_free_rate=config.risk_free_rate,
    )

    monthly_returns = sanitize_numeric_frame(monthly_returns)
    fold_metrics = sanitize_numeric_frame(fold_metrics)
    best_params = sanitize_numeric_frame(best_params)
    kpis = sanitize_numeric_frame(kpis)

    assert_no_numeric_na(kpis, context="kpis")

    # Global figures and report
    global_images: Dict[str, Path] = {}
    global_images.update(save_auc_score_overview(fold_metrics, figures_dir))
    hp_global = save_hyperparams_overview(best_params, figures_dir)
    for key, value in hp_global.items():
        global_images[f"hyperparam_{key}"] = value

    report_path = run_dir / "training_report.html"
    write_html_report(
        title=config.report_title,
        output_path=report_path,
        kpis=kpis,
        fold_metrics=fold_metrics,
        best_params=best_params,
        global_images=global_images,
        fold_assets=fold_assets,
    )

    total_elapsed = time.perf_counter() - pipeline_start
    if config.verbose:
        print(
            "[Pipeline] completed "
            f"elapsed={_format_seconds(total_elapsed)} "
            f"completed_folds={fold_metrics.height}/{total_windows} "
            f"report={report_path}"
        )

    # Consolidated artifacts
    paths = {
        "run_dir": run_dir,
        "model_frame": run_dir / "model_frame.parquet",
        "predictions": run_dir / "predictions.parquet",
        "selections": run_dir / "selections.parquet",
        "monthly_returns": run_dir / "monthly_returns.parquet",
        "fold_metrics": run_dir / "fold_metrics.parquet",
        "best_params": run_dir / "best_params.parquet",
        "kpis_parquet": run_dir / "kpis.parquet",
        "kpis_csv": run_dir / "kpis.csv",
        "report_html": report_path,
        "metadata": run_dir / "metadata.json",
    }

    model_frame.write_parquet(paths["model_frame"])
    predictions.write_parquet(paths["predictions"])
    selections.write_parquet(paths["selections"])
    monthly_returns.write_parquet(paths["monthly_returns"])
    fold_metrics.write_parquet(paths["fold_metrics"])
    best_params.write_parquet(paths["best_params"])
    kpis.write_parquet(paths["kpis_parquet"])
    kpis.write_csv(paths["kpis_csv"])

    metadata = {
        "created_at": datetime.now().isoformat(),
        "features_used": features_used,
        "dropped_features": dropped_features,
        "n_completed_folds": int(fold_metrics.height),
        "config": {
            "data_dir": str(config.data_dir),
            "output_dir": str(config.output_dir),
            "start_month": config.start_month,
            "n_folds": config.n_folds,
            "top_n": config.top_n,
            "prediction_threshold": config.prediction_threshold,
            "min_train_months": config.min_train_months,
            "missing_feature_threshold": config.missing_feature_threshold,
            "n_optuna_trials": config.n_optuna_trials,
            "optuna_lambda_gap": config.optuna_lambda_gap,
            "optuna_startup_trials": config.optuna_startup_trials,
            "verbose": config.verbose,
            "show_optuna_progress": config.show_optuna_progress,
            "optuna_progress_every": config.optuna_progress_every,
            "risk_free_rate": config.risk_free_rate,
            "xgb_params": config.xgb_params,
            "optuna_space": config.optuna_space,
        },
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return BacktestArtifacts(
        model_frame=model_frame,
        predictions=predictions,
        selections=selections,
        monthly_returns=monthly_returns,
        kpis=kpis,
        fold_metrics=fold_metrics,
        best_params=best_params,
        features_used=features_used,
        dropped_features=dropped_features,
        output_paths=paths,
    )
