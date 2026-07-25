from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from alpharank.backtest.mlcraft_adapter import ensure_mlcraft_importable


@dataclass
class FittedBooster:
    model: Any
    method: str
    calibrator: IsotonicRegression | None
    features: tuple[str, ...]
    target_bounds: tuple[float, float] | None = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.method in {"classification", "teacher"}:
            raw = self.model.predict_proba(X)[:, 1]
            if self.calibrator is not None:
                return np.asarray(self.calibrator.predict(raw), dtype=float)
            return np.asarray(raw, dtype=float)
        return np.asarray(self.model.predict(X), dtype=float)


def _groups(frame: pl.DataFrame) -> list[int]:
    return (
        frame.group_by("decision_month", maintain_order=True)
        .len()
        .get_column("len")
        .cast(pl.Int64)
        .to_list()
    )


def _relevance(frame: pl.DataFrame, horizon: int) -> np.ndarray:
    rank = frame.get_column(f"future_excess_rank_{horizon}m").to_numpy()
    return np.clip(np.floor(np.nan_to_num(rank, nan=0.0) * 31.0), 0, 31).astype(np.int32)


def _target(
    frame: pl.DataFrame,
    *,
    method: str,
    horizon: int,
    positive_quantile: float,
    bounds: tuple[float, float] | None = None,
) -> tuple[np.ndarray, tuple[float, float] | None]:
    if method == "teacher":
        return frame.get_column("legacy_selected").to_numpy().astype(np.int8), None
    if method == "classification":
        rank = frame.get_column(f"future_excess_rank_{horizon}m").to_numpy()
        return (rank >= positive_quantile).astype(np.int8), None
    if method == "ranking":
        return _relevance(frame, horizon), None
    values = frame.get_column(f"future_excess_return_{horizon}m").to_numpy().astype(float)
    if bounds is None:
        bounds = tuple(np.nanquantile(values, [0.01, 0.99]).tolist())
    return np.clip(values, bounds[0], bounds[1]), bounds


def fit_booster(
    *,
    method: str,
    horizon: int,
    train_frame: pl.DataFrame,
    validation_frame: pl.DataFrame,
    X_train: np.ndarray,
    X_validation: np.ndarray,
    features: tuple[str, ...],
    positive_quantile: float,
    seed: int,
    num_boost_round: int,
    params: dict[str, Any] | None = None,
) -> FittedBooster:
    ensure_mlcraft_importable()
    from mlcraft.core.task import TaskSpec
    from mlcraft.models.factory import ModelFactory

    task_type = "classification" if method in {"classification", "teacher"} else method
    y_train, bounds = _target(
        train_frame,
        method=method,
        horizon=horizon,
        positive_quantile=positive_quantile,
    )
    y_validation, _ = _target(
        validation_frame,
        method=method,
        horizon=horizon,
        positive_quantile=positive_quantile,
        bounds=bounds,
    )
    model_params = {
        "eta": 0.04,
        "max_depth": 5,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.75,
        "lambda": 5.0,
        "alpha": 0.2,
        "nthread": 4,
        **(params or {}),
    }
    fit_params: dict[str, Any] = {
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": 25,
    }
    if method == "ranking":
        fit_params["group"] = _groups(train_frame)
        fit_params["eval_group"] = [_groups(validation_frame)]
    model = ModelFactory.create(
        "xgboost",
        task_spec=TaskSpec(task_type=task_type),
        model_params=model_params,
        fit_params=fit_params,
        random_state=seed,
    )
    model.fit(X_train, y_train, eval_set=[(X_validation, y_validation)])
    calibrator = None
    if method in {"classification", "teacher"} and np.unique(y_validation).size == 2:
        raw = model.predict_proba(X_validation)[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip").fit(raw, y_validation)
    return FittedBooster(
        model=model,
        method=method,
        calibrator=calibrator,
        features=features,
        target_bounds=bounds,
    )
