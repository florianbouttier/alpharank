from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import optuna
from optuna.samplers import TPESampler

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
    search_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for name, spec in search_space.items():
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(
                name,
                int(spec["low"]),
                int(spec["high"]),
                step=int(spec.get("step", 1)),
                log=bool(spec.get("log", False)),
            )
        elif ptype == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
                step=spec.get("step"),
            )
    return params


def _split_mlcraft_params(
    params: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    model_params: Dict[str, Any] = {}
    fit_params: Dict[str, Any] = {}
    for name, value in params.items():
        target = str(search_space.get(name, {}).get("target", "model"))
        if target == "fit":
            fit_params[name] = value
        else:
            model_params[name] = value
    return model_params, fit_params


def _top_n_precision_by_group(
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence,
    *,
    top_n: int,
) -> float:
    y_true = np.asarray(y_true).astype(float)
    y_score = np.asarray(y_score).astype(float)
    group_array = np.asarray(groups)
    if y_true.size == 0:
        return 0.0

    values: list[float] = []
    for group in np.unique(group_array):
        idx = np.flatnonzero(group_array == group)
        if idx.size == 0:
            continue
        order = idx[np.argsort(y_score[idx])[::-1]]
        selected = order[: max(1, min(int(top_n), order.size))]
        values.append(float(np.mean(y_true[selected])))
    return float(np.mean(values)) if values else 0.0


def _objective_metric(
    *,
    y_true: np.ndarray,
    y_score: np.ndarray,
    groups: Sequence | None,
    objective_name: str,
    top_n: int,
) -> float:
    if objective_name == "top_n_precision":
        if groups is None:
            return float(np.mean(y_true[np.argsort(y_score)[::-1][: max(1, min(int(top_n), y_true.size))]]))
        return _top_n_precision_by_group(y_true, y_score, groups, top_n=top_n)
    return safe_auc(y_true, y_score)


def _filter_enqueued_params(
    params: Dict[str, Any],
    search_space: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    filtered: Dict[str, Any] = {}
    for name, spec in search_space.items():
        if name not in params:
            continue
        value = params[name]
        if spec["type"] == "int":
            value = int(round(float(value)))
        elif spec["type"] == "float":
            value = float(value)
        filtered[name] = value
    return filtered


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
    top_n: int = 10,
    objective_name: str = "top_n_precision",
    warm_start_params: Sequence[Dict[str, Any]] | None = None,
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
    from mlcraft.models.factory import ModelFactory

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
    train_groups_array = np.asarray(train_groups)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed))
    for params in warm_start_params or []:
        enqueued = _filter_enqueued_params(params, mlcraft_space)
        if enqueued:
            study.enqueue_trial(enqueued)

    trial_rows: List[Dict[str, Any]] = []

    def _fit_model(params: Dict[str, Any], X_fit: np.ndarray, y_fit: np.ndarray):
        trial_model_params, trial_fit_params = _split_mlcraft_params(params, mlcraft_space)
        model = ModelFactory.create(
            "xgboost",
            task_spec=TaskSpec(task_type="classification"),
            model_params={**model_params, **trial_model_params},
            fit_params={**fit_params, **trial_fit_params},
            random_state=seed,
        )
        model.fit(X_fit, y_fit)
        return model

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample_params(trial, mlcraft_space)
        train_scores: list[float] = []
        val_scores: list[float] = []
        auc_train_scores: list[float] = []
        auc_val_scores: list[float] = []
        for inner_train_idx, inner_val_idx in cv_splitter.split(X_train, y_train):
            if np.unique(y_train[inner_train_idx]).size < 2:
                continue
            model = _fit_model(sampled, X_train[inner_train_idx], y_train[inner_train_idx])
            p_inner_train = model.predict_proba(X_train[inner_train_idx])[:, 1]
            p_inner_val = model.predict_proba(X_train[inner_val_idx])[:, 1]
            train_score = _objective_metric(
                y_true=y_train[inner_train_idx],
                y_score=p_inner_train,
                groups=train_groups_array[inner_train_idx],
                objective_name=objective_name,
                top_n=top_n,
            )
            val_score = _objective_metric(
                y_true=y_train[inner_val_idx],
                y_score=p_inner_val,
                groups=train_groups_array[inner_val_idx],
                objective_name=objective_name,
                top_n=top_n,
            )
            train_scores.append(train_score)
            val_scores.append(val_score)
            auc_train_scores.append(safe_auc(y_train[inner_train_idx], p_inner_train))
            auc_val_scores.append(safe_auc(y_train[inner_val_idx], p_inner_val))

        if not val_scores:
            return -1.0

        train_score_mean = float(np.mean(train_scores))
        val_score_mean = float(np.mean(val_scores))
        penalized = val_score_mean - lambda_gap * abs(train_score_mean - val_score_mean)
        trial.set_user_attr("train_objective", train_score_mean)
        trial.set_user_attr("val_objective", val_score_mean)
        trial.set_user_attr("train_auc", float(np.mean(auc_train_scores)))
        trial.set_user_attr("val_auc", float(np.mean(auc_val_scores)))
        trial.set_user_attr("score", penalized)
        return float(penalized)

    study.optimize(objective, n_trials=n_trials)

    best_params = dict(study.best_params)
    final_model = _fit_model(best_params, X_train, y_train)

    p_train = final_model.predict_proba(X_train)[:, 1]
    p_val = final_model.predict_proba(X_val)[:, 1]
    p_test = final_model.predict_proba(X_test)[:, 1]

    train_metrics = compute_classification_metrics(y_train, p_train)
    val_metrics = compute_classification_metrics(y_val, p_val)
    test_metrics = compute_classification_metrics(y_test, p_test)

    train_auc = train_metrics["auc"]
    val_auc = val_metrics["auc"]
    test_auc = test_metrics["auc"]
    train_objective = _objective_metric(
        y_true=y_train,
        y_score=p_train,
        groups=train_groups_array,
        objective_name=objective_name,
        top_n=top_n,
    )
    val_objective = _objective_metric(
        y_true=y_val,
        y_score=p_val,
        groups=None,
        objective_name=objective_name,
        top_n=top_n,
    )
    objective_score = float(val_objective - lambda_gap * abs(train_objective - val_objective))

    if show_progress:
        total_elapsed = time.perf_counter() - optimize_start
        print(
            f"{progress_label} mlcraft tuning done: best_score={_fmt_metric(study.best_value)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} test_auc={test_auc:.4f} "
            f"elapsed={_format_seconds(total_elapsed)}"
        )

    for tr in study.trials:
        row: Dict[str, Any] = {
            "trial_number": tr.number,
            "objective": tr.value,
            "train_auc": tr.user_attrs.get("train_auc"),
            "val_auc": tr.user_attrs.get("val_auc"),
            "train_objective": tr.user_attrs.get("train_objective"),
            "val_objective": tr.user_attrs.get("val_objective"),
            "score": tr.user_attrs.get("score", tr.value),
            "state": "COMPLETE",
        }
        for k, v in tr.params.items():
            row[f"param_{k}"] = v
        trial_rows.append(row)

    output_best_params = {
        **base_params,
        **best_params,
        "mlcraft_model_type": "xgboost",
        "optuna_objective": objective_name,
    }
    raw_model = getattr(final_model, "model_", final_model)

    return TunedModelResult(
        best_params=output_best_params,
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
        trials_df=trial_rows,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        model=raw_model,
        study=study,
    )
