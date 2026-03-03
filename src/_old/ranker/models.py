
# %%
from typing import Dict, Tuple, Optional, List
from xgboost import XGBRanker
import pandas as pd
from ..data_processor.index_manager import IndexDataManager
from ..data_processor.datasets import make_train_test_ranked
from src.ranker.ranker_metrics import evaluate_ranking_model
from src.ranker.shap_utils import run_shap_analysis
from .portfolio import build_topk_per_month, evaluate_portfolio
from tqdm import tqdm

def make_xgb_ranker(space_params: Dict) -> Tuple[Dict, int]:
    params = dict(
        objective="rank:ndcg",
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

def fit_ranker(X_tr, y_tr, group_tr: List[int],
               params: Dict=None):
    """
    Fits an XGBRanker model with a crucial parameter fix for large ranks.
    """
    local_params = (params or {}).copy()
    local_params['ndcg_exp_gain'] = False
    model = XGBRanker(**local_params)
    model.fit(X_tr, y_tr, group=group_tr, verbose=False)
    return model

def predict_scores(model, X_te):
    return model.predict(X_te)

def train_fit_on_year_month(df_learning : pd.DataFrame,
                            params: Dict,
                            df_returns : pd.DataFrame,
                            index : IndexDataManager,
                            year_month_split: str,
                            n_asset : int,
                            features : List[str],
                            target : str = 'future_return',
                            plot_shap : bool = False) -> Dict[str, pd.DataFrame]:
    
    X_train,y_train,group_train,meta_train,X_test,y_test,group_test,meta_test,X_application = make_train_test_ranked(df = df_learning,
                                                                                                features = features,
                                                                                                target = target,
                                                                                                year_month_split = year_month_split)
    booster_params, n_rounds = make_xgb_ranker(params)
    booster = fit_ranker(X_tr = X_train, 
                     y_tr = y_train, 
                     group_tr = group_train, 
                     params=booster_params)
     
    if len(y_test) == 0 :
        ndcg_score = pd.DataFrame()
        
    else : 
        scores = predict_scores(booster, X_test)
        
        ndcg_score = evaluate_ranking_model(y_test, scores, top_k_list=[n_asset,50,100])
        ndcg_score = pd.DataFrame([ndcg_score])
        ndcg_score['year_month'] = year_month_split
    
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
        "scores": ndcg_score,
        "detailled_return": detailled_return,
        "aggregated_return": aggregated_return,
    }

def train_fit(df_learning : pd.DataFrame,
              params: Dict,
              df_returns : pd.DataFrame,
              index : IndexDataManager,
              n_asset : int,
              features : List[str],
              target : str = 'future_return') :
    all_scores = []
    all_detailled_returns = []
    all_aggregated_returns = []
    years_months_splits = sorted(df_learning['year_month'].unique() , reverse=True)
    for year_month_split in tqdm(years_months_splits[12:], desc="Training and evaluating over time"): 
        #print(f"Training and evaluating for year_month {year_month_split}...")
        if year_month_split == max(years_months_splits) :
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