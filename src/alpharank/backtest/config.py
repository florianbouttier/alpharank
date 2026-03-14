from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple


def _validate_positive_int_tuple(values: Tuple[int, ...], field_name: str) -> None:
    if any(value <= 0 for value in values):
        raise ValueError(f"{field_name} must contain only positive integers.")


def _validate_ordered_pairs(values: Tuple[Tuple[int, int], ...], field_name: str) -> None:
    for left, right in values:
        if left <= 0 or right <= 0:
            raise ValueError(f"{field_name} must contain only positive integers.")
        if left >= right:
            raise ValueError(f"{field_name} must use strictly increasing pairs, got {(left, right)!r}.")


def _validate_positive_pairs(values: Tuple[Tuple[int, int], ...], field_name: str) -> None:
    for left, right in values:
        if left <= 0 or right <= 0:
            raise ValueError(f"{field_name} must contain only positive integers.")


@dataclass(frozen=True)
class TechnicalFeatureConfig:
    roc_windows: Tuple[int, ...] = (1, 3, 6, 12)
    ema_pairs: Tuple[Tuple[int, int], ...] = ((2, 6), (3, 6), (3, 12), (6, 12), (6, 18), (12, 24))
    price_to_ema_spans: Tuple[int, ...] = (3, 6, 12, 24)
    rsi_windows: Tuple[int, ...] = (3, 6, 12, 24)
    rsi_ratio_pairs: Tuple[Tuple[int, int], ...] = ((3, 12), (6, 24))
    bollinger_windows: Tuple[int, ...] = (6, 12)
    stochastic_windows: Tuple[Tuple[int, int], ...] = ((6, 3), (12, 3))
    range_windows: Tuple[int, ...] = (6, 12)
    volatility_windows: Tuple[int, ...] = (3, 6, 12)
    volatility_ratio_pairs: Tuple[Tuple[int, int], ...] = ((3, 12), (6, 12))

    def __post_init__(self) -> None:
        _validate_positive_int_tuple(self.roc_windows, "roc_windows")
        _validate_ordered_pairs(self.ema_pairs, "ema_pairs")
        _validate_positive_int_tuple(self.price_to_ema_spans, "price_to_ema_spans")
        _validate_positive_int_tuple(self.rsi_windows, "rsi_windows")
        _validate_ordered_pairs(self.rsi_ratio_pairs, "rsi_ratio_pairs")
        _validate_positive_int_tuple(self.bollinger_windows, "bollinger_windows")
        _validate_positive_pairs(self.stochastic_windows, "stochastic_windows")
        _validate_positive_int_tuple(self.range_windows, "range_windows")
        _validate_positive_int_tuple(self.volatility_windows, "volatility_windows")
        _validate_ordered_pairs(self.volatility_ratio_pairs, "volatility_ratio_pairs")


@dataclass(frozen=True)
class FundamentalFeatureConfig:
    quarterly_growth_lags: Tuple[int, ...] = (1, 4, 12)

    def __post_init__(self) -> None:
        _validate_positive_int_tuple(self.quarterly_growth_lags, "quarterly_growth_lags")


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
    final_price_path: Path | None = None
    sp500_price_path: Path | None = None
    start_month: str = "2006-01"
    n_folds: int = 10
    top_n: int = 20
    outperformance_threshold: float = 0.0
    prediction_threshold: float | None = None
    min_train_months: int = 24
    missing_feature_threshold: float = 0.35
    risk_free_rate: float = 0.02
    n_optuna_trials: int = 40
    optuna_lambda_gap: float = 0.2
    optuna_startup_trials: int = 12
    random_seed: int = 42
    shap_sample_size: int = 1000
    shap_top_features: int = 20
    calibration_buckets: int = 20
    fold_min_train_rows: int = 250
    fold_min_val_rows: int = 80
    fold_min_test_rows: int = 80
    report_title: str = "XGBoost Time-Fold Overperformance Backtest Report"
    verbose: bool = True
    show_optuna_progress: bool = True
    optuna_progress_every: int = 1
    save_optuna_all_plots: bool = True
    technical_feature_config: TechnicalFeatureConfig = field(default_factory=TechnicalFeatureConfig)
    fundamental_feature_config: FundamentalFeatureConfig = field(default_factory=FundamentalFeatureConfig)
    xgb_params: Dict[str, Any] = field(default_factory=default_xgb_params)
    optuna_space: Dict[str, Tuple[str, float, float]] = field(default_factory=default_optuna_space)

    def __post_init__(self) -> None:
        if self.prediction_threshold is not None:
            self.outperformance_threshold = float(self.prediction_threshold)
