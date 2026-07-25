from __future__ import annotations

import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from alpharank.multihorizon.modeling import fit_booster
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.splits import PurgedCombinatorialMonthSplit


def _candidate_parameters(n_trials: int, seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    candidates = [{}]
    for _ in range(max(0, n_trials - 1)):
        candidates.append(
            {
                "eta": float(np.exp(rng.uniform(np.log(0.015), np.log(0.09)))),
                "max_depth": int(rng.integers(3, 8)),
                "min_child_weight": int(rng.integers(10, 61)),
                "subsample": float(rng.uniform(0.65, 1.0)),
                "colsample_bytree": float(rng.uniform(0.55, 1.0)),
                "lambda": float(np.exp(rng.uniform(np.log(1.0), np.log(20.0)))),
                "alpha": float(rng.uniform(0.0, 1.0)),
            }
        )
    return candidates


def tune_with_purged_cpcv(
    *,
    frame: pl.DataFrame,
    candidate_features: tuple[str, ...],
    method: str,
    horizon: int,
    n_trials: int,
    n_groups: int,
    test_group_count: int,
    missing_threshold: float,
    positive_quantile: float,
    num_boost_round: int,
    seed: int,
) -> tuple[dict, pl.DataFrame]:
    """Small, deterministic parameter search using only purged pre-test data."""

    if n_trials <= 1:
        return {}, pl.DataFrame({"trial": [0], "mean_score": [float("nan")], "params": ["{}"]})
    ordered = frame.sort(["decision_month", "ticker"])
    splitter = PurgedCombinatorialMonthSplit(
        ordered["decision_month"].to_list(),
        horizon=horizon,
        n_groups=n_groups,
        test_group_count=test_group_count,
    )
    rows: list[dict] = []
    best_params: dict = {}
    best_score = -np.inf
    for trial, params in enumerate(_candidate_parameters(n_trials, seed)):
        scores: list[float] = []
        for split_index, (train_idx, validation_idx) in enumerate(splitter.split(), start=1):
            train = ordered[train_idx]
            validation = ordered[validation_idx]
            preprocessor = fit_fold_preprocessor(
                train,
                candidate_features,
                max_missing_ratio=missing_threshold,
            )
            _, X_train = preprocessor.transform(train)
            _, X_validation = preprocessor.transform(validation)
            fitted = fit_booster(
                method=method,
                horizon=horizon,
                train_frame=train,
                validation_frame=validation,
                X_train=X_train,
                X_validation=X_validation,
                features=preprocessor.features,
                positive_quantile=positive_quantile,
                seed=seed + trial * 100 + split_index,
                num_boost_round=max(60, num_boost_round // 2),
                params=params,
            )
            prediction = fitted.predict(X_validation)
            if method == "teacher":
                target = validation["legacy_selected"].to_numpy()
                score = average_precision_score(target, prediction)
            else:
                target = validation[f"future_excess_return_{horizon}m"].to_numpy()
                score = spearmanr(target, prediction).statistic
            if np.isfinite(score):
                scores.append(float(score))
        mean_score = float(np.mean(scores)) if scores else -np.inf
        rows.append(
            {
                "trial": trial,
                "mean_score": mean_score,
                "params": repr(params),
            }
        )
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, pl.DataFrame(rows)
