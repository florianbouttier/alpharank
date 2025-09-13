
import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, Optional
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from ..data_processor.datasets import make_rank_dataset
from .models import make_xgb_ranker, fit_ranker, predict_scores
from .portfolio import build_topk_per_month, evaluate_portfolio
from .search_spaces import sample_xgb_space
from .custom_stocks_metrics import get_objective_fn
from .models import *

# %%
def scoring_function(results) -> float:
    """Build a unique KPIs after full fitting
    Can be outperformance, average of spearman correlation etc

    Args:
        results (_type_): _description_
    """
    scores = results.get("scores", None)
    return scores['spearman_correlation'].mean() if scores is not None else -1e9
    
def _optuna_objective_builder(df_train,df_returns,index, features, n_asset,target = 'future_return'):
    
    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_space(trial)
        res = train_fit(df_learning = df_train,
                       params = params,
                       df_returns= df_returns,
                       index = index,
                       n_asset = n_asset,
                       features = features,
                       target = target)
        return scoring_function(res)

    return objective

def optimize(df_train: pd.DataFrame,
             df_test: pd.DataFrame,
             df_returns: pd.DataFrame,
             index: IndexDataManager,
             features: list,
             n_asset: int = 10,
             target: str = 'future_return',
             n_trials: int = 60,
             seed: int = 123) -> Dict[str,Any]:
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    objective = _optuna_objective_builder(df_train,df_returns,index, features, n_asset, target)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    best_params = study.best_params
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)
    test = train_fit(df_learning = df_test,
                   params = best_params,
                   df_returns= df_returns,
                   index = index,
                   n_asset = n_asset,
                   features = features,
                   target = target)
    
    train = train_fit(df_learning = df_train,
                   params = best_params,
                   df_returns= df_test,
                   index = index,
                   n_asset = n_asset,
                   features = features,
                   target = target)
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
