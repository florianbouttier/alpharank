from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from alpharank.backtest.mlcraft_adapter import (
    ensure_mlcraft_importable,
    to_mlcraft_model_and_fit_params,
    to_mlcraft_search_space,
)
from alpharank.backtest.time_folds import CombinatorialPurgedGroupTimeSeriesSplit


@dataclass
class TunedModelResult:
    best_params: Dict[str, Any]
    train_auc: float
    val_auc: float
    test_auc: float
    objective_score: float
    train_size: int
    val_size: int
    test_size: int
    y_train_proba: np.ndarray
    y_val_proba: np.ndarray
    y_test_proba: np.ndarray
    evals_result: Dict[str, Dict[str, List[float]]]
    trials_df: List[Dict[str, Any]]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    model: Any
    study: Any | None


def safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    if y_true.size == 0:
        return 0.5
    unique = np.unique(y_true)
    if unique.size < 2:
        return 0.5
    try:
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return 0.5

def compute_classification_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        brier_score_loss,
        f1_score,
        log_loss,
        precision_score,
        recall_score,
    )

    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    if y_true.size == 0:
        return {
            "auc": 0.5,
            "average_precision": 0.0,
            "logloss": 0.0,
            "brier": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "accuracy": 0.0,
            "pred_positive_rate": 0.0,
            "realized_positive_rate": 0.0,
        }

    y_pred = (y_score >= threshold).astype(int)

    try:
        ap = float(average_precision_score(y_true, y_score))
    except Exception:
        ap = 0.0

    try:
        ll = float(log_loss(y_true, y_score, labels=[0, 1]))
    except Exception:
        ll = 0.0

    try:
        brier = float(brier_score_loss(y_true, y_score))
    except Exception:
        brier = 0.0

    metrics = {
        "auc": safe_auc(y_true, y_score),
        "average_precision": ap,
        "logloss": ll,
        "brier": brier,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "pred_positive_rate": float(np.mean(y_pred)),
        "realized_positive_rate": float(np.mean(y_true)),
    }

    sanitized: Dict[str, float] = {}
    for key, value in metrics.items():
        val = float(value)
        sanitized[key] = val if np.isfinite(val) else 0.0
    return sanitized


def _sample_params(
    trial: optuna.Trial,
    search_space: Dict[str, Tuple[str, float, float]],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, (ptype, low, high) in search_space.items():
        if ptype == "int":
            params[name] = trial.suggest_int(name, int(low), int(high))
        elif ptype == "loguniform":
            params[name] = trial.suggest_float(name, float(low), float(high), log=True)
        else:
            params[name] = trial.suggest_float(name, float(low), float(high))
    return params


def _format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, sec = divmod(rem, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def _fmt_metric(value: Any) -> str:
    try:
        value_float = float(value)
    except Exception:
        return "nan"
    if not np.isfinite(value_float):
        return "nan"
    return f"{value_float:.4f}"


def tune_and_fit_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    base_params: Dict[str, Any],
    search_space: Dict[str, Tuple[str, float, float]],
    n_trials: int,
    startup_trials: int,
    lambda_gap: float,
    seed: int,
    progress_label: str = "",
    show_progress: bool = True,
    progress_every: int = 1,
    train_groups: Sequence | None = None,
    cpcv_inner_groups: int = 5,
    cpcv_inner_test_groups: int = 1,
    cpcv_embargo_groups: int = 0,
) -> TunedModelResult:
    if np.unique(y_train).size < 2:
        fallback = np.full_like(y_train, fill_value=float(np.mean(y_train)), dtype=float)
        fallback_val = np.full_like(y_val, fill_value=float(np.mean(y_train)), dtype=float)
        fallback_test = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
        if show_progress:
            print(f"{progress_label} skipped tuning (single-class train target).")

        return TunedModelResult(
            best_params=base_params.copy(),
            train_auc=safe_auc(y_train, fallback),
            val_auc=safe_auc(y_val, fallback_val),
            test_auc=safe_auc(y_test, fallback_test),
            objective_score=safe_auc(y_val, fallback_val),
            train_size=int(y_train.size),
            val_size=int(y_val.size),
            test_size=int(y_test.size),
            y_train_proba=fallback,
            y_val_proba=fallback_val,
            y_test_proba=fallback_test,
            evals_result={},
            trials_df=[],
            train_metrics=compute_classification_metrics(y_train, fallback),
            val_metrics=compute_classification_metrics(y_val, fallback_val),
            test_metrics=compute_classification_metrics(y_test, fallback_test),
            model=None,
            study=None,
        )

    optimize_start = time.perf_counter()

    if show_progress:
        print(
            f"{progress_label} mlcraft tuning started: {n_trials} trials "
            f"(inner CV=CPCV, objective=roc_auc, penalty alpha={lambda_gap:.3f})"
        )

    ensure_mlcraft_importable()
    from mlcraft.core.task import TaskSpec
    from mlcraft.tuning.optuna_search import OptunaSearch

    model_params, fit_params = to_mlcraft_model_and_fit_params(base_params)
    mlcraft_space = to_mlcraft_search_space(search_space)
    if "num_boost_round" not in fit_params and "num_boost_round" not in mlcraft_space:
        fit_params["num_boost_round"] = int(base_params.get("n_estimators", 500))

    if train_groups is None:
        train_groups = np.arange(y_train.size)
    cv_splitter = CombinatorialPurgedGroupTimeSeriesSplit(
        train_groups,
        n_groups=min(int(cpcv_inner_groups), len(set(train_groups))),
        test_group_count=int(cpcv_inner_test_groups),
        embargo_groups=int(cpcv_embargo_groups),
    )

    search = OptunaSearch(
        task_spec=TaskSpec(task_type="classification"),
        model_type="xgboost",
        n_trials=n_trials,
        cv=cpcv_inner_groups,
        cv_splitter=cv_splitter,
        alpha=lambda_gap,
        random_state=seed,
        model_params=model_params,
        fit_params=fit_params,
        search_space=mlcraft_space,
    )
    result = search.run(X_train, y_train, X_test=X_val, y_test=y_val)
    final_model = result.final_model

    p_train = final_model.predict_proba(X_train)[:, 1]
    p_val = final_model.predict_proba(X_val)[:, 1]
    p_test = final_model.predict_proba(X_test)[:, 1]

    train_metrics = compute_classification_metrics(y_train, p_train)
    val_metrics = compute_classification_metrics(y_val, p_val)
    test_metrics = compute_classification_metrics(y_test, p_test)

    train_auc = train_metrics["auc"]
    val_auc = val_metrics["auc"]
    test_auc = test_metrics["auc"]
    objective_score = float(val_auc - lambda_gap * abs(train_auc - val_auc))

    if show_progress:
        total_elapsed = time.perf_counter() - optimize_start
        print(
            f"{progress_label} mlcraft tuning done: best_score={_fmt_metric(result.best_score)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} test_auc={test_auc:.4f} "
            f"elapsed={_format_seconds(total_elapsed)}"
        )

    trials_df: List[Dict[str, Any]] = []
    for tr in result.history:
        row: Dict[str, Any] = {
            "trial_number": tr.trial_number,
            "objective": tr.penalized_score,
            "train_auc": tr.train_metrics.get("roc_auc"),
            "val_auc": tr.val_metrics.get("roc_auc"),
            "score": tr.penalized_score,
            "state": "COMPLETE",
        }
        for k, v in tr.params.items():
            row[f"param_{k}"] = v
        trials_df.append(row)

    best_params = {
        **base_params,
        **result.best_params,
        "mlcraft_model_type": result.metadata.get("model_type", "xgboost"),
    }
    raw_model = getattr(final_model, "model_", final_model)

    return TunedModelResult(
        best_params=best_params,
        train_auc=train_auc,
        val_auc=val_auc,
        test_auc=test_auc,
        objective_score=objective_score,
        train_size=int(y_train.size),
        val_size=int(y_val.size),
        test_size=int(y_test.size),
        y_train_proba=p_train,
        y_val_proba=p_val,
        y_test_proba=p_test,
        evals_result={},
        trials_df=trials_df,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model=raw_model,
        study=result.study,
    )
