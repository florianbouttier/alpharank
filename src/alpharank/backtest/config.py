from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


def default_xgb_params() -> Dict[str, Any]:
    """Default XGBoost hyperparameters for monthly cross-sectional classification."""
    return {
        "n_estimators": 500,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.8,
        "min_child_weight": 2.0,
        "gamma": 0.0,
        "reg_alpha": 0.0,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": ["auc", "logloss"],
        "n_jobs": -1,
        "random_state": 42,
    }


def default_optuna_space() -> Dict[str, Tuple[str, float, float]]:
    return {
        "n_estimators": ("int", 150, 900),
        "max_depth": ("int", 3, 10),
        "learning_rate": ("loguniform", 0.005, 0.25),
        "subsample": ("float", 0.5, 1.0),
        "colsample_bytree": ("float", 0.5, 1.0),
        "min_child_weight": ("float", 0.1, 10.0),
        "gamma": ("float", 0.0, 6.0),
        "reg_alpha": ("float", 0.0, 4.0),
        "reg_lambda": ("float", 0.0, 6.0),
    }


@dataclass
class BacktestConfig:
    data_dir: Path
    output_dir: Path
    start_month: str = "2006-01"
    n_folds: int = 10
    top_n: int = 20
    prediction_threshold: float = 0.02
    min_train_months: int = 24
    missing_feature_threshold: float = 0.35
    risk_free_rate: float = 0.02
    n_optuna_trials: int = 40
    optuna_lambda_gap: float = 0.2
    optuna_startup_trials: int = 12
    random_seed: int = 42
    shap_sample_size: int = 1000
    shap_top_features: int = 20
    lift_bins: int = 10
    fold_min_train_rows: int = 250
    fold_min_val_rows: int = 80
    fold_min_test_rows: int = 80
    report_title: str = "XGBoost Time-Fold Backtest Report"
    verbose: bool = True
    show_optuna_progress: bool = True
    optuna_progress_every: int = 1
    xgb_params: Dict[str, Any] = field(default_factory=default_xgb_params)
    optuna_space: Dict[str, Tuple[str, float, float]] = field(default_factory=default_optuna_space)
