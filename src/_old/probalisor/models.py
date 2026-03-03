
# %%
from typing import Dict, Tuple, Optional, List
from xgboost import XGBClassifier
import pandas as pd
from ..data_processor.index_manager import IndexDataManager
from ..data_processor.datasets import make_train_test_proba
from src.probalisor.prob_metrics import evaluate_classifier
from src.probalisor.shap_utils import run_shap_analysis
from .portfolio import build_topk_per_month, evaluate_portfolio
from tqdm import tqdm
import numpy as np

def make_xgb_classifier(space_params: Dict) -> Tuple[Dict, int]:
    params = dict(
        objective="binary:logistic",  # <-- ici changement principal
        tree_method="hist",
        n_estimators=space_params.get("n_estimators", 400),
        max_depth=space_params.get("max_depth", 5),
        learning_rate=space_params.get("eta", space_params.get("learning_rate", 0.06)),
        subsample=space_params.get("subsample", 0.8),
        colsample_bytree=space_params.get("colsample_bytree", 0.8),
        min_child_weight=space_params.get("min_child_weight", 10),
        gamma=space_params.get("gamma", 0.0),
        reg_lambda=space_params.get("lambda", space_params.get("reg_lambda", 1.0)),
        reg_alpha=space_params.get("alpha", space_params.get("reg_alpha", 0.0)),
        random_state=space_params.get("seed", 42),
        n_jobs=space_params.get("n_jobs", -1),
    )
    return params, params["n_estimators"]

def fit_classifier(X_tr, y_tr, params: Dict=None):
    """
    Fit an XGBClassifier model.
    """
    local_params = (params or {}).copy()
    model = XGBClassifier(**local_params)
    model.fit(X_tr, y_tr, verbose=False)
    return model

def predict_scores(model, X_te):
    return model.predict_proba(X_te)[:, 1]  

def train_fit_on_year_month(df_learning : pd.DataFrame,
                            params: Dict,
                            df_returns : pd.DataFrame,
                            index : IndexDataManager,
                            year_month_split: str,
                            n_asset : int,
                            features : List[str],
                            target : str = 'future_return',
                            plot_shap : bool = False) -> Dict[str, pd.DataFrame]:
    
    X_train,y_train,meta_train,X_test,y_test,meta_test,X_application = make_train_test_proba(df = df_learning,
                                                                                                features = features,
                                                                                                target = target,
                                                                                                year_month_split = year_month_split)
    booster_params, n_rounds = make_xgb_classifier(params)
    booster = fit_classifier(X_tr = X_train, 
                     y_tr = y_train, 
                     params=booster_params)
     
    if len(y_test) == 0 :
        prob_score = pd.DataFrame()
        
    else : 
        scores = predict_scores(booster, X_test)
        scores_train = predict_scores(booster, X_train)
        
        prob_score = evaluate_classifier(y_test, scores, top_k_list=[n_asset,50,100])
        prob_train = evaluate_classifier(y_train, scores_train, top_k_list=[n_asset,50,100])

        over_fitting = {}
        for k, test_val in prob_score.items():
            train_val = prob_train.get(k)
            if train_val is None:
                continue
            if test_val is None:
                over_fitting[f"overfitting_{k}"] = None
                continue
            if test_val == 0:
                over_fitting[f"overfitting_{k}"] = None
            else:
                over_fitting[f"overfitting_{k}"] = (train_val - test_val) / abs(test_val)

        train_prefixed = {f"train_{k}": v for k, v in prob_train.items()}

        prob_score.update(train_prefixed)
        prob_score.update(over_fitting)
        prob_score = pd.DataFrame([prob_score])
        prob_score['year_month'] = year_month_split
    
    applied_scores = predict_scores(booster, X_application[features])
    if plot_shap :
            run_shap_analysis(booster, X_application[features], top_n_heatmap=20)
    
    top_parameter = build_topk_per_month(
        df=X_application,
        scores=applied_scores,
        k=n_asset
    )
    top_parameter['year_month'] = top_parameter['year_month'] + 1 
    detailled_return = top_parameter.merge(df_returns, on=["year_month","ticker"], how="left")
    aggregated_return = (detailled_return.groupby("year_month")["monthly_return"].mean()).reset_index()
    aggregated_return['monthly_return'] = aggregated_return['monthly_return']-1
    aggregated_return = aggregated_return.merge(index.monthly_returns.rename(columns={'monthly_return' : 'monthly_return_index'}), on = 'year_month', how="left")
    return {
        "scores": prob_score,
        "detailled_return": detailled_return,
        "aggregated_return": aggregated_return,
    }

def train_fit(df_learning : pd.DataFrame,
              params: Dict,
              df_returns : pd.DataFrame,
              index : IndexDataManager,
              n_asset : int,
              features : List[str],
              target : str = 'future_return',
              plot_shap : bool = False) -> Dict[str, pd.DataFrame]:
    all_scores = []
    all_detailled_returns = []
    all_aggregated_returns = []
    years_months_splits = sorted(df_learning['year_month'].unique() , reverse=True)
    for year_month_split in tqdm(years_months_splits[:-4*12], desc="Training and evaluating over time"): 
        
        if (year_month_split == max(years_months_splits))  & plot_shap :
            plot_shap = True
        else : plot_shap = False
        out = train_fit_on_year_month(df_learning = df_learning,
                                      params = params,
                                      df_returns = df_returns,
                                      index = index,
                                      year_month_split = year_month_split,
                                      n_asset = n_asset,
                                      features = features,
                                      target = target,
                                      plot_shap = plot_shap)
        all_scores.append(out['scores'])
        all_detailled_returns.append(out['detailled_return'])
        all_aggregated_returns.append(out['aggregated_return'])
    
    all_scores = pd.concat(all_scores).reset_index(drop=True)
    all_detailled_returns = pd.concat(all_detailled_returns).reset_index(drop=True)
    all_aggregated_returns = pd.concat(all_aggregated_returns).reset_index(drop=True)
    
    return {
        "scores": all_scores,
        "detailled_return": all_detailled_returns,
        "aggregated_return": all_aggregated_returns,
    }
    

def perturb_params(params: dict, std_pct: float = 0.1) -> dict:
    """
    Crée un nouvel ensemble d'hyperparamètres en ajoutant un bruit gaussien
    autour de chaque paramètre numérique. L'écart-type est défini comme
    un pourcentage de la valeur actuelle (std = std_pct * valeur).

    Args:
        params (dict): dictionnaire des hyperparamètres actuels
        std_pct (float): pourcentage d'écart-type (ex: 0.1 = 10%)

    Returns:
        dict: nouveau dictionnaire de paramètres bruités
    """
    new_params = {}
    if std_pct is not None : 
        for k, v in params.items():
            if isinstance(v, (int, float)):
                std = std_pct * abs(v) if v != 0 else std_pct
                noise = np.random.normal(loc=0.0, scale=std)
                new_val = v + noise

            # Clamp selon le type d'hyperparam
                if k in ["subsample", "colsample_bytree"]:
                    new_val = float(np.clip(new_val, 0.1, 1.0))
                elif k == "learning_rate":
                    new_val = float(np.clip(new_val, 1e-5, 1.0))
                elif k in ["max_depth", "n_estimators"]:
                    new_val = int(max(1, round(new_val)))
                elif k in ["min_child_weight", "gamma", "lambda", "alpha"]:
                    new_val = float(max(0.0, new_val))

                new_params[k] = new_val
            else:
                new_params[k] = v
    else : 
        new_params = params
        new_params['seed'] = np.random.randint(0, 10000)
    return new_params