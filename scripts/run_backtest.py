from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import polars as pl

from alpharank.backtest import (
    BacktestArtifacts,
    BacktestConfig,
    FundamentalFeatureConfig,
    LearningArtifacts,
    TechnicalFeatureConfig,
    run_backtest_from_learning,
    run_learning_phase,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_LEARNING_KPI_COLUMNS = (
    "fold",
    "objective_score",
    "objective_score_val",
    "train_auc",
    "val_auc",
    "test_auc",
    "train_average_precision",
    "val_average_precision",
    "test_average_precision",
    "train_logloss",
    "val_logloss",
    "test_logloss",
)

DEFAULT_BACKTEST_FOLD_KPI_COLUMNS = (
    "fold",
    "objective_score",
    "test_auc",
    "fold_portfolio_total_return",
    "fold_benchmark_total_return",
    "fold_active_total_return",
    "fold_avg_hit_rate",
    "fold_avg_positions",
)


def default_config(**overrides: Any) -> BacktestConfig:
    params: dict[str, Any] = {
        "data_dir": PROJECT_ROOT / "data",
        "output_dir": PROJECT_ROOT / "outputs",
        "start_month": "2000-01",
        "n_folds": 7,
        "top_n": 30,
        "outperformance_threshold": 0.15,
        "min_train_months": 24,
        "missing_feature_threshold": 0.05,
        "n_optuna_trials": 20,
        "optuna_lambda_gap": 5,
        "optuna_startup_trials": 20,
        "risk_free_rate": 0.02,
        "random_seed": 42,
        "verbose": True,
        "show_optuna_progress": True,
        "optuna_progress_every": 10,
        "technical_feature_config": TechnicalFeatureConfig(
            roc_windows=(1, 3, 6, 12),
            ema_pairs=((2, 6), (3, 6), (3, 12), (6, 12), (6, 18), (12, 24)),
            price_to_ema_spans=(3, 6, 12, 24),
            rsi_windows=(3, 6, 12, 24),
            rsi_ratio_pairs=((3, 12), (6, 24)),
            bollinger_windows=(6, 12),
            stochastic_windows=((6, 3), (12, 3)),
            range_windows=(6, 12),
            volatility_windows=(3, 6, 12),
            volatility_ratio_pairs=((3, 12), (6, 12)),
        ),
        "fundamental_feature_config": FundamentalFeatureConfig(
            quarterly_growth_lags=(1, 4, 12),
        ),
    }
    params.update(overrides)
    return BacktestConfig(**params)


def _config_to_metadata(config: BacktestConfig) -> dict[str, Any]:
    return {
        "data_dir": str(config.data_dir),
        "output_dir": str(config.output_dir),
        "start_month": config.start_month,
        "n_folds": config.n_folds,
        "top_n": config.top_n,
        "outperformance_threshold": config.outperformance_threshold,
        "prediction_threshold": config.prediction_threshold,
        "min_train_months": config.min_train_months,
        "missing_feature_threshold": config.missing_feature_threshold,
        "risk_free_rate": config.risk_free_rate,
        "n_optuna_trials": config.n_optuna_trials,
        "optuna_lambda_gap": config.optuna_lambda_gap,
        "optuna_startup_trials": config.optuna_startup_trials,
        "random_seed": config.random_seed,
        "shap_sample_size": config.shap_sample_size,
        "shap_top_features": config.shap_top_features,
        "calibration_buckets": config.calibration_buckets,
        "fold_min_train_rows": config.fold_min_train_rows,
        "fold_min_val_rows": config.fold_min_val_rows,
        "fold_min_test_rows": config.fold_min_test_rows,
        "report_title": config.report_title,
        "verbose": config.verbose,
        "show_optuna_progress": config.show_optuna_progress,
        "optuna_progress_every": config.optuna_progress_every,
        "save_optuna_all_plots": config.save_optuna_all_plots,
        "technical_feature_config": asdict(config.technical_feature_config),
        "fundamental_feature_config": asdict(config.fundamental_feature_config),
        "xgb_params": config.xgb_params,
        "optuna_space": config.optuna_space,
    }


def _config_from_metadata(config_data: dict[str, Any]) -> BacktestConfig:
    params = dict(config_data)
    params["data_dir"] = Path(params["data_dir"])
    params["output_dir"] = Path(params["output_dir"])
    params["technical_feature_config"] = TechnicalFeatureConfig(**params["technical_feature_config"])
    params["fundamental_feature_config"] = FundamentalFeatureConfig(**params["fundamental_feature_config"])
    return BacktestConfig(**params)


def _learning_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "model_frame": run_dir / "model_frame.parquet",
        "predictions": run_dir / "predictions.parquet",
        "fold_metrics": run_dir / "fold_metrics.parquet",
        "fold_index": run_dir / "fold_index.parquet",
        "best_params": run_dir / "best_params.parquet",
        "learning_metadata": run_dir / "learning_metadata.json",
        "metadata": run_dir / "metadata.json",
    }


def _resolve_run_dir(run_dir: str | Path) -> Path:
    resolved = Path(run_dir).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Run directory not found: {resolved}")
    return resolved


def save_learning_outputs(config: BacktestConfig, learning: LearningArtifacts) -> dict[str, Path]:
    paths = _learning_paths(learning.run_dir)
    learning.run_dir.mkdir(parents=True, exist_ok=True)
    learning.model_frame.write_parquet(paths["model_frame"])
    learning.predictions.write_parquet(paths["predictions"])
    learning.fold_metrics.write_parquet(paths["fold_metrics"])
    learning.fold_index.write_parquet(paths["fold_index"])
    learning.best_params.write_parquet(paths["best_params"])

    metadata = {
        "stage": "learning",
        "created_at": datetime.now().isoformat(),
        "run_dir": str(learning.run_dir),
        "features_used": learning.features_used,
        "dropped_features": learning.dropped_features,
        "n_completed_folds": int(learning.fold_metrics.height),
        "n_total_windows": int(learning.total_windows),
        "config": _config_to_metadata(config),
    }
    paths["learning_metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return paths


def load_config(run_dir: str | Path) -> BacktestConfig:
    resolved_run_dir = _resolve_run_dir(run_dir)
    paths = _learning_paths(resolved_run_dir)
    metadata_path = paths["learning_metadata"] if paths["learning_metadata"].exists() else paths["metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing learning metadata in {resolved_run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return _config_from_metadata(metadata["config"])


def run_learning(config: BacktestConfig | None = None) -> LearningArtifacts:
    active_config = config or default_config()
    learning = run_learning_phase(active_config)
    save_learning_outputs(active_config, learning)
    return learning


def load_learning(run_dir: str | Path) -> LearningArtifacts:
    resolved_run_dir = _resolve_run_dir(run_dir)
    paths = _learning_paths(resolved_run_dir)
    metadata_path = paths["learning_metadata"] if paths["learning_metadata"].exists() else paths["metadata"]
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing learning metadata in {resolved_run_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    fold_index = pl.read_parquet(paths["fold_index"])
    total_windows = int(metadata.get("n_total_windows") or fold_index.height)
    return LearningArtifacts(
        run_dir=resolved_run_dir,
        figures_dir=resolved_run_dir / "figures",
        model_frame=pl.read_parquet(paths["model_frame"]),
        predictions=pl.read_parquet(paths["predictions"]),
        fold_metrics=pl.read_parquet(paths["fold_metrics"]),
        fold_index=fold_index,
        best_params=pl.read_parquet(paths["best_params"]),
        features_used=list(metadata.get("features_used", [])),
        dropped_features=list(metadata.get("dropped_features", [])),
        fold_assets=[],
        shap_explanations=[],
        total_windows=total_windows,
    )


def learning_kpis(learning: LearningArtifacts | None = None, *, run_dir: str | Path | None = None) -> pl.DataFrame:
    active_learning = learning or load_learning(run_dir or latest_run_dir())
    cols = [col for col in DEFAULT_LEARNING_KPI_COLUMNS if col in active_learning.fold_metrics.columns]
    return active_learning.fold_metrics.select(cols).sort("fold")


def list_folds(learning: LearningArtifacts | None = None, *, run_dir: str | Path | None = None) -> pl.DataFrame:
    active_learning = learning or load_learning(run_dir or latest_run_dir())
    cols = [
        "fold",
        "status",
        "skip_reason",
        "train_month_start",
        "train_month_end",
        "val_month_start",
        "val_month_end",
        "test_month_start",
        "test_month_end",
        "train_rows",
        "val_rows",
        "test_rows",
    ]
    selected = [col for col in cols if col in active_learning.fold_index.columns]
    return active_learning.fold_index.select(selected).sort("fold")


def load_fold_predictions(run_dir: str | Path, fold: int) -> pl.DataFrame:
    resolved_run_dir = _resolve_run_dir(run_dir)
    fold_path = resolved_run_dir / f"fold_{int(fold):02d}" / "predictions.parquet"
    if not fold_path.exists():
        raise FileNotFoundError(f"Missing fold predictions: {fold_path}")
    return pl.read_parquet(fold_path)


def load_fold_monthly_returns(run_dir: str | Path, fold: int) -> pl.DataFrame:
    resolved_run_dir = _resolve_run_dir(run_dir)
    fold_path = resolved_run_dir / f"fold_{int(fold):02d}" / "monthly_returns.parquet"
    if not fold_path.exists():
        raise FileNotFoundError(
            f"Missing fold monthly returns: {fold_path}. Run `run_backtest(...)` first."
        )
    return pl.read_parquet(fold_path)


def run_backtest(
    config: BacktestConfig | None = None,
    *,
    learning: LearningArtifacts | None = None,
    run_dir: str | Path | None = None,
) -> BacktestArtifacts:
    if learning is None and run_dir is None:
        raise ValueError("Provide either `learning` or `run_dir`.")

    active_learning = learning or load_learning(run_dir)  # type: ignore[arg-type]
    active_config = config or load_config(active_learning.run_dir)
    return run_backtest_from_learning(active_config, active_learning)


def backtest_fold_kpis(
    artifacts: BacktestArtifacts | None = None,
    *,
    run_dir: str | Path | None = None,
) -> pl.DataFrame:
    if artifacts is None:
        if run_dir is None:
            run_dir = latest_run_dir()
        frame = pl.read_parquet(_resolve_run_dir(run_dir) / "fold_metrics.parquet")
    else:
        frame = artifacts.fold_metrics
    cols = [col for col in DEFAULT_BACKTEST_FOLD_KPI_COLUMNS if col in frame.columns]
    return frame.select(cols).sort("fold")


def latest_run_dir(output_dir: str | Path | None = None) -> Path:
    active_output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs"
    run_dirs = sorted(
        [path for path in active_output_dir.glob("xgboost_timefold_backtest_*") if path.is_dir()],
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No backtest run found under {active_output_dir}")
    return run_dirs[0]


def run(config: BacktestConfig | None = None) -> BacktestArtifacts:
    active_config = config or default_config()
    learning = run_learning(active_config)
    return run_backtest(active_config, learning=learning)


def main() -> BacktestArtifacts:
    artifacts = run()
    print("\n=== Backtest Completed ===")
    print(f"Run dir: {artifacts.output_paths['run_dir']}")
    print(f"Predictions rows: {artifacts.predictions.height}")
    print(f"Selections rows: {artifacts.selections.height}")
    print(f"Completed folds: {artifacts.fold_metrics.height}")
    print(f"KPI export: {artifacts.output_paths['kpis_parquet']}")
    print(f"Fold metrics: {artifacts.output_paths['fold_metrics']}")
    print(f"Debug predictions (long): {artifacts.output_paths['debug_predictions_long']}")
    print(f"Audit report: {artifacts.output_paths['backtest_audit_report']}")
    return artifacts


if __name__ == "__main__":
    main()
