
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Optional
from .loaders import load_prepared_dataframe
from .preprocess import apply_user_preprocessing
from ..data_processor.datasets import make_rank_dataset
from ..ranker.models import make_xgb_ranker, fit_ranker, predict_scores
from ..ranker.portfolio import build_topk_per_month, evaluate_portfolio
from ..ranker.search_spaces import sample_xgb_space
from ..ranker.custom_stocks_metrics import get_objective_fn
from .utils.io import save_json

def _make_train_df(df: pd.DataFrame, year: int, date_col="year_month"):
    d = df.copy()
    per = pd.PeriodIndex(d[date_col].astype(str), freq="M")
    d[date_col] = per.astype(str)
    train_mask = per.to_timestamp(how="end").year <= year
    return d[train_mask].copy()

def _optuna_objective_builder(train_df, features, target, k_select, metric_name, metric_kwargs, n_windows=3, seed=123):
    rng = np.random.default_rng(seed)
    months = sorted(train_df["year_month"].unique())

    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_space(trial)
        vals = []
        for _ in range(n_windows):
            if len(months) < 4: break
            cut_idx = int(rng.integers(low=3, high=len(months)))
            tr_end = months[cut_idx-2]
            va_end = months[cut_idx-1]
            tr_mask = train_df["year_month"] <= tr_end
            va_mask = (train_df["year_month"] > tr_end) & (train_df["year_month"] <= va_end)
            tr = train_df[tr_mask].copy()
            va = train_df[va_mask].copy()
            if tr.empty or va.empty: continue

            X_tr, y_tr, g_tr, _ = make_rank_dataset(tr, features, target)
            X_va, y_va, g_va, meta_va = make_rank_dataset(va, features, target)

            booster_params, n_rounds = make_xgb_ranker(params)
            booster = fit_ranker(X_tr, y_tr, g_tr, X_valid=X_va, y_valid=y_va, group_valid=g_va,
                                 params=booster_params, num_boost_round=n_rounds, early_stopping_rounds=50)

            scores_va = predict_scores(booster, X_va)
            topk_va = build_topk_per_month(meta_va, scores_va, k=k_select)
            realized_va = va[["year_month","ticker",target]].rename(columns={target:"target"})
            met = evaluate_portfolio(topk_va, realized_va, bench_returns=None)
            obj_fn = get_objective_fn(metric_name)
            score = obj_fn(met["series"], bench=None, **(metric_kwargs or {}))
            vals.append(score)
        return float(np.median(vals)) if vals else -1e9

    return objective

def optimize_for_next_year(path_or_df,
                           target_year: int,
                           k_select: int = 10,
                           metric_name: str = "log_alpha",
                           metric_kwargs: Optional[Dict[str,Any]] = None,
                           n_trials: int = 60,
                           seed: int = 123,
                           params_out_dir: str = "outputs/params") -> Dict[str,Any]:
    df, features, target = load_prepared_dataframe(path_or_df)
    df = apply_user_preprocessing(df)
    df["year_month"] = pd.PeriodIndex(df["year_month"].astype(str), freq="M").astype(str)

    train_df = _make_train_df(df, year=target_year-1)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    objective = _optuna_objective_builder(train_df, features, target, k_select, metric_name, metric_kwargs, n_windows=3, seed=seed)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    best_params = study.best_params
    best = {
        "target_year": target_year,
        "metric": metric_name,
        "metric_kwargs": metric_kwargs or {},
        "best_value": float(study.best_value),
        "best_params": best_params,
        "n_trials": n_trials,
        "seed": seed
    }
    save_json(best, f"{params_out_dir}/params_{target_year}.json")
    return best
