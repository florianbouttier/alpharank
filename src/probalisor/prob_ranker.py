import numpy as np
import pandas as pd
import optuna
from typing import Dict, Any, List
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_slice
from ..data_processor.datasets import make_prob_dataset
from .models import make_xgb_classifier, fit_classifier, predict_scores,train_fit
from .portfolio import build_topk_per_month, evaluate_portfolio
from .search_spaces import sample_xgb_space


# %%
# ---------------------------
# Scoring function avec pénalité overfitting
# ---------------------------
def scoring_function(final_returns: pd.DataFrame,
                     metric_name: str = "precision",
                     penalty_weight: float = 0.5,
                     min_spread : float =  0.1) -> float:
    """
    Calcule le score final sur test, pénalisé par l'overfitting.
    
    final_returns : dataframe renvoyé par train_fit avec colonnes
        ['precision', 'recall', 'average_precision', 'roc_auc', ...,
         'train_precision', 'train_recall', ...]
    metric_name : nom de la métrique à scorer ('precision', 'precision@50', etc.)
    penalty_weight : pondération de la pénalité
    """
    test_val = final_returns['scores'][f"{metric_name}"].sum()/len(final_returns['scores'])
    train_val = final_returns['scores'][f"train_{metric_name}"].sum()/len(final_returns['scores'])
    
    minimum_spread = final_returns['scores']["spread"].min()
    overfit_penalty = penalty_weight * abs(train_val - test_val)
    
    final_score = test_val - overfit_penalty
    if minimum_spread < min_spread :
        final_score -= 1000
    return final_score

# ---------------------------
# Optuna objective builder
# ---------------------------
def _optuna_objective_builder(df_train, df_returns, index, features, n_asset,
                              target='future_return', metric_name='precision', penalty_weight=0.5,min_spread = 0.1):
    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_space(trial)
        final_returns = train_fit(
            df_learning=df_train,
            params=params,
            df_returns=df_returns,
            index=index,
            n_asset=n_asset,
            features=features,
            target=target
        )
        return scoring_function(final_returns, metric_name=metric_name, penalty_weight=penalty_weight,min_spread = min_spread)
    return objective

# ---------------------------
# Optimization
# ---------------------------
def optimize(
    df : pd.DataFrame,
    split_date : str,
    df_returns: pd.DataFrame,
    index,
    features: list,
    n_asset: int = 10,
    target: str = 'future_return',
    n_trials: int = 60,
    seed: int = 123,
    metric_name: str = "precision",
    penalty_weight: float = 0.5,
    min_spread : float =  0.1   
) -> Dict[str, Any]:
    
    df_train = df[df['year_month'] < split_date]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    objective = _optuna_objective_builder(df_train, df_returns, index, features,
                                          n_asset, target, metric_name, penalty_weight,min_spread)
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)

    # Refit sur test et train
    all_returns = train_fit(df_learning=df, 
                             params=best_params, 
                             df_returns=df_returns,
                             index=index, 
                             n_asset=n_asset, 
                             features=features, 
                             target=target)
    
    best = {
        "best_value": float(study.best_value),
        "study": study,
        "best_params": best_params,
        "n_trials": n_trials,
        "seed": seed,
        "metric_name": metric_name,
        "penalty_weight": penalty_weight,
        "all_returns" : all_returns
    }

    return best