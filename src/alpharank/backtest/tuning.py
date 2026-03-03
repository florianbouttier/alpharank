from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import optuna
import xgboost as xgb
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


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
    model: xgb.XGBClassifier


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
) -> TunedModelResult:
    if np.unique(y_train).size < 2:
        fallback = np.full_like(y_train, fill_value=float(np.mean(y_train)), dtype=float)
        fallback_val = np.full_like(y_val, fill_value=float(np.mean(y_train)), dtype=float)
        fallback_test = np.full_like(y_test, fill_value=float(np.mean(y_train)), dtype=float)
        if show_progress:
            print(f"{progress_label} skipped tuning (single-class train target).")

        fake_model = xgb.XGBClassifier(**base_params)
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
            model=fake_model,
        )

    def objective(trial: optuna.Trial) -> float:
        sampled = _sample_params(trial, search_space=search_space)
        params = {**base_params, **sampled}

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
        )

        p_train = model.predict_proba(X_train)[:, 1]
        p_val = model.predict_proba(X_val)[:, 1]

        train_auc = safe_auc(y_train, p_train)
        val_auc = safe_auc(y_val, p_val)

        score = val_auc - lambda_gap * abs(train_auc - val_auc)

        trial.set_user_attr("train_auc", train_auc)
        trial.set_user_attr("val_auc", val_auc)
        trial.set_user_attr("score", score)

        return float(score)

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    optimize_start = time.perf_counter()
    safe_every = max(1, int(progress_every))

    if show_progress:
        print(
            f"{progress_label} tuning started: {n_trials} trials "
            f"(objective = AUC_val - {lambda_gap:.3f} * |AUC_train - AUC_val|)"
        )

    def _progress_callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if not show_progress:
            return
        done = trial.number + 1
        if done % safe_every != 0 and done != n_trials:
            return

        elapsed = time.perf_counter() - optimize_start
        avg_per_trial = elapsed / max(done, 1)
        eta = avg_per_trial * max(n_trials - done, 0)

        train_auc = trial.user_attrs.get("train_auc")
        val_auc = trial.user_attrs.get("val_auc")
        score = trial.user_attrs.get("score", trial.value)

        gap_text = "nan"
        try:
            gap = abs(float(train_auc) - float(val_auc))
            gap_text = _fmt_metric(gap)
        except Exception:
            pass

        print(
            f"{progress_label} trial {done}/{n_trials} "
            f"score={_fmt_metric(score)} best={_fmt_metric(study.best_value)} "
            f"train_auc={_fmt_metric(train_auc)} val_auc={_fmt_metric(val_auc)} gap={gap_text} "
            f"elapsed={_format_seconds(elapsed)} eta={_format_seconds(eta)}"
        )

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=seed, n_startup_trials=startup_trials),
        pruner=MedianPruner(n_startup_trials=startup_trials),
    )
    study.optimize(objective, n_trials=n_trials, callbacks=[_progress_callback])

    best_trial = study.best_trial
    best_params = {**base_params, **best_trial.params}

    final_model = xgb.XGBClassifier(**best_params)
    final_model.fit(
        X_train,
        y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )

    p_train = final_model.predict_proba(X_train)[:, 1]
    p_val = final_model.predict_proba(X_val)[:, 1]
    p_test = final_model.predict_proba(X_test)[:, 1]

    train_auc = safe_auc(y_train, p_train)
    val_auc = safe_auc(y_val, p_val)
    test_auc = safe_auc(y_test, p_test)
    objective_score = float(val_auc - lambda_gap * abs(train_auc - val_auc))

    if show_progress:
        total_elapsed = time.perf_counter() - optimize_start
        print(
            f"{progress_label} tuning done: best_score={_fmt_metric(study.best_value)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} test_auc={test_auc:.4f} "
            f"elapsed={_format_seconds(total_elapsed)}"
        )

    trials_df: List[Dict[str, Any]] = []
    for tr in study.trials:
        row: Dict[str, Any] = {
            "trial_number": tr.number,
            "objective": tr.value,
            "train_auc": tr.user_attrs.get("train_auc"),
            "val_auc": tr.user_attrs.get("val_auc"),
            "score": tr.user_attrs.get("score"),
            "state": str(tr.state),
        }
        for k, v in tr.params.items():
            row[f"param_{k}"] = v
        trials_df.append(row)

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
        evals_result=final_model.evals_result(),
        trials_df=trials_df,
        model=final_model,
    )
