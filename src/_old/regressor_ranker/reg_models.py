# %%
from typing import Dict, List
import numpy as np
import pandas as pd
import xgboost as xgb
from tqdm import tqdm
from ..probalisor.portfolio import build_topk_per_month
from ..probalisor.search_spaces import sample_xgb_space
from .reg_metrics import evaluate_regressor
import shap
import warnings
from ..probalisor.shap_utils import compute_shap_values, run_shap_analysis

def make_xgb_regressor(space_params: Dict) -> Dict:
    return dict(
        tree_method="hist",
        objective="reg:squarederror",  # baseline; custom obj overrides gradients
        n_estimators=space_params.get("n_estimators", 300),
        max_depth=space_params.get("max_depth", 6),
        learning_rate=space_params.get("learning_rate", 0.05),
        subsample=space_params.get("subsample", 0.8),
        colsample_bytree=space_params.get("colsample_bytree", 0.8),
        min_child_weight=space_params.get("min_child_weight", 5),
        gamma=space_params.get("gamma", 0.0),
        reg_lambda=space_params.get("lambda", space_params.get("reg_lambda", 1.0)),
        reg_alpha=space_params.get("alpha", space_params.get("reg_alpha", 0.0)),
        random_state=space_params.get("seed", 42),
        n_jobs=space_params.get("n_jobs", -1)
    )

def asymmetric_over_obj_old(alpha: float = 2.0):
    """
    Custom objective:
    L = 0.5 * (pred - y)^2 * (1 + alpha * I[pred > y])
    Over-predictions receive a multiplicative weight (1+alpha).
    """
    def _obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        diff = preds - y
        over = (diff > 0).astype(np.float32)
        w = 1.0 + alpha * over
        grad = diff * w
        hess = w  # derivative of diff wrt pred is 1
        return grad, hess
    return _obj

# Pseudo-code pour une version améliorée
def asymmetric_over_obj(alpha: float = 2.0, beta: float = 1.0):
    """
    Fonction de perte asymétrique améliorée.
    Pénalise plus les sur-prédictions (alpha), mais aussi les sous-prédictions
    avec un poids de base (beta) pour éviter de prédire des valeurs négatives.
    """
    def _obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        diff = preds - y
        
        # Le poids pour les sur-prédictions (pred > y)
        over_weight = (1.0 + alpha) * (diff > 0).astype(float)
        
        # Le poids pour les sous-prédictions (pred <= y)
        under_weight = (1.0 + beta) * (diff <= 0).astype(float)
        
        # On combine les deux poids
        total_weight = over_weight + under_weight
        
        grad = diff * total_weight
        hess = total_weight  # Le hessien est le poids, car la dérivée de diff est 1
        
        return grad, hess
    return _obj

def hinge_squared_obj(epsilon: float = 0.02, scale: float = 1.0):
    """
    Hinge (SVR-style) squared loss:
        loss = (max(0, |pred - y| - epsilon))^2
    - Pas de pénalité dans le tube |diff| <= epsilon
    - Pénalité quadratique au-delà (accentue les gros miss)
    epsilon : largeur du tube d'insensibilité
    scale   : facteur multiplicatif global
    """
    def _obj(preds: np.ndarray, dtrain: xgb.DMatrix):
        y = dtrain.get_label()
        diff = preds - y
        abs_diff = np.abs(diff)
        margin_excess = abs_diff - epsilon
        active = margin_excess > 0
        grad = np.zeros_like(diff)
        hess = np.full_like(diff, 1e-12)  # petite valeur pour stabilité
        # zone active
        grad[active] = 2.0 * margin_excess[active] * np.sign(diff[active]) * scale
        hess[active] = 2.0 * scale
        return grad, hess
    return _obj

def predict_scores(booster: xgb.Booster, X: pd.DataFrame):
    return booster.predict(xgb.DMatrix(X, feature_names=list(X.columns)))

def train_fit_on_year_month_reg(df_learning: pd.DataFrame,
                                params: Dict,
                                df_returns: pd.DataFrame,
                                index,
                                year_month_split,
                                features: List[str],
                                month_window: int = 1000,
                                n_asset: int = 10 ,
                                target: str = "future_return",
                                alpha_over_obj: float = 2.0,
                                compute_shap: bool = False,
                                shap_run_plots: bool = True,
                                shap_top_n: int = 20,
                                loss_type: str = "hinge",
                                hinge_epsilon: float = 0.02,
                                hinge_scale: float = 1.0):
    """
    Train until (exclusive) year_month_split, predict on that month, then
    build next-month portfolio and attach realized returns.
    If compute_shap=True: use probalisor.shap_utils (compute_shap_values + optional run_shap_analysis).
    loss_type:
        'hinge'       -> hinge_squared_obj (recommandé pour pénaliser gros miss)
        'asymmetric'  -> ancienne loss sur-prédiction
    """
    train = df_learning[df_learning['year_month'] < year_month_split].copy()
    train = train[train['year_month'] >= year_month_split - month_window]
    test = df_learning[df_learning['year_month'] == year_month_split].copy()
    if test.empty or train.empty:
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    # --- Data cleaning: drop NaN/Inf targets, sanitize features ---
    def _clean_block(block: pd.DataFrame) -> pd.DataFrame:
        if block.empty:
            return block
        # Replace inf in features with NaN
        block[features] = block[features].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN / Inf targets
        before = len(block)
        block = block[np.isfinite(block[target])]
        # Drop rows where target is NaN
        block = block[block[target].notna()]
        # Fill feature NaNs with column medians (computed on remaining rows)
        if not block.empty:
            med = block[features].median()
            block[features] = block[features].fillna(med)
        dropped = before - len(block)
        if dropped > 0:
            warnings.warn(f"[{year_month_split}] Dropped {dropped} rows due to invalid target/feature values.")
        return block

    train = _clean_block(train)
    test = _clean_block(test)

    if train.empty or test.empty:
        warnings.warn(f"[{year_month_split}] Empty train or test after cleaning, skipping.")
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    X_tr = train[features]
    y_tr = train[target].values
    X_te = test[features]
    y_te = test[target].values

    # Final guard on labels
    mask_tr = np.isfinite(y_tr)
    if not mask_tr.all():
        warnings.warn(f"[{year_month_split}] Dropping {(~mask_tr).sum()} non‑finite train labels post-clean.")
        X_tr = X_tr.iloc[mask_tr]
        y_tr = y_tr[mask_tr]
    mask_te = np.isfinite(y_te)
    if not mask_te.all():
        warnings.warn(f"[{year_month_split}] Dropping {(~mask_te).sum()} non‑finite test labels post-clean.")
        X_te = X_te.iloc[mask_te]
        y_te = y_te[mask_te]

    if len(y_tr) == 0 or len(y_te) == 0:
        warnings.warn(f"[{year_month_split}] No valid samples after final label filtering.")
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    booster_params = make_xgb_regressor(params)
    num_boost_round = booster_params.pop("n_estimators", 300)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=list(X_te.columns))

    # Sélection de la fonction objectif
    if loss_type == "hinge":
        obj_fn = hinge_squared_obj(epsilon=hinge_epsilon, scale=hinge_scale)
    elif loss_type == "asymmetric":
        obj_fn = asymmetric_over_obj(alpha_over_obj)
    else:
        # Fallback: use XGBoost's default RMSE (reg:squarederror)
        obj_fn = None

    booster = xgb.train(
        params=booster_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        obj=obj_fn,
        verbose_eval=False
    )

    preds_test = booster.predict(dtest)
    preds_train = booster.predict(dtrain)

    base_metrics_test = evaluate_regressor(y_te, preds_test)
    base_metrics_train = {f"train_{k}": v for k, v in evaluate_regressor(y_tr, preds_train).items()}

    # Overprediction ratio
    over_mask = preds_test > y_te
    if len(y_te) > 0:
        base_metrics_test["over_rate"] = float(np.mean(over_mask))
    if len(y_tr) > 0:
        base_metrics_train["train_over_rate"] = float(np.mean(preds_train > y_tr))

    metrics_row = {**base_metrics_test, **base_metrics_train}
    metrics_row["year_month"] = year_month_split
    scores_df = pd.DataFrame([metrics_row])

    # Build portfolio for next month (shift +1)
    # Select top n_asset by predicted return
    applied_scores = preds_test
    top_next = build_topk_per_month(
        df=test[['year_month', 'ticker'] + features].copy(),
        scores=applied_scores,
        k=n_asset
    )
    top_next['year_month'] = top_next['year_month'] + 1

    detailled = top_next.merge(df_returns, on=["year_month", "ticker"], how="left")
    agg = (detailled.groupby("year_month")["monthly_return"].mean()).reset_index()
    agg['monthly_return'] = agg['monthly_return'] - 1
    agg = agg.merge(index.monthly_returns.rename(columns={'monthly_return': 'monthly_return_index'}),
                    on='year_month', how='left')

    shap_values = None
    shap_interaction_values = None
    if compute_shap and not X_te.empty:
        try:
            shap_values, shap_interaction_values = compute_shap_values(model=booster, X=X_te)
            if shap_run_plots:
                # Fire standard analysis (plots only, no capture)
                run_shap_analysis(model=booster, X=X_te, top_n_heatmap=shap_top_n)
        except Exception as e:
            warnings.warn(f"[{year_month_split}] SHAP computation failed: {e}")

    return {
        "scores": scores_df,
        "detailled_return": detailled,
        "aggregated_return": agg,
        "model": booster,
        "shap_values": shap_values,                 # np.ndarray or None
        "shap_interaction_values": shap_interaction_values,  # np.ndarray or None
        "test_preds": pd.DataFrame({
            "ticker": test['ticker'].values,
            "year_month": test['year_month'].values,
            "y_true": y_te,
            "y_pred": preds_test
        })
    }

def train_fit_on_year_reg(df_learning: pd.DataFrame,
                                params: Dict,
                                df_returns: pd.DataFrame,
                                index,
                                year_split,
                                features: List[str],
                                year_window: int = 10,
                                n_asset: int = 10 ,
                                target: str = "future_return",
                                shap_top_n: int = 20,
                                compute_shap: bool = False,
                                shap_run_plots: bool = True):
    """
    Train until (exclusive) year_month_split, predict on that month, then
    build next-month portfolio and attach realized returns.
    If compute_shap=True: use probalisor.shap_utils (compute_shap_values + optional run_shap_analysis).
    loss_type:
        'hinge'       -> hinge_squared_obj (recommandé pour pénaliser gros miss)
        'asymmetric'  -> ancienne loss sur-prédiction
    """
    if 'year' not in df_learning.columns :
        #period year month to year 
        df_learning['year'] = df_learning['year_month'].astype(str).str[:4].astype(int)
        
    train = df_learning[df_learning['year'] < year_split].copy()
    train = train[train['year'] >= year_split - year_window]
    test = df_learning[df_learning['year'] == year_split].copy()
    if test.empty or train.empty:
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    # --- Data cleaning: drop NaN/Inf targets, sanitize features ---
    def _clean_block(block: pd.DataFrame) -> pd.DataFrame:
        if block.empty:
            return block
        # Replace inf in features with NaN
        block[features] = block[features].replace([np.inf, -np.inf], np.nan)
        # Drop rows with NaN / Inf targets
        before = len(block)
        block = block[np.isfinite(block[target])]
        # Drop rows where target is NaN
        block = block[block[target].notna()]
        # Fill feature NaNs with column medians (computed on remaining rows)
        if not block.empty:
            med = block[features].median()
            block[features] = block[features].fillna(med)
        dropped = before - len(block)
        if dropped > 0:
            warnings.warn(f"[{year_split}] Dropped {dropped} rows due to invalid target/feature values.")
        return block

    train = _clean_block(train)
    test = _clean_block(test)

    if train.empty or test.empty:
        warnings.warn(f"[{year_split}] Empty train or test after cleaning, skipping.")
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    X_tr = train[features]
    y_tr = train[target].values
    X_te = test[features]
    y_te = test[target].values

    # Final guard on labels
    mask_tr = np.isfinite(y_tr)
    if not mask_tr.all():
        warnings.warn(f"[{year_split}] Dropping {(~mask_tr).sum()} non‑finite train labels post-clean.")
        X_tr = X_tr.iloc[mask_tr]
        y_tr = y_tr[mask_tr]
    mask_te = np.isfinite(y_te)
    if not mask_te.all():
        warnings.warn(f"[{year_split}] Dropping {(~mask_te).sum()} non‑finite test labels post-clean.")
        X_te = X_te.iloc[mask_te]
        y_te = y_te[mask_te]

    if len(y_tr) == 0 or len(y_te) == 0:
        warnings.warn(f"[{year_split}] No valid samples after final label filtering.")
        return {
            "scores": pd.DataFrame(),
            "detailled_return": pd.DataFrame(),
            "aggregated_return": pd.DataFrame()
        }

    booster_params = make_xgb_regressor(params)
    num_boost_round = booster_params.pop("n_estimators", 300)

    dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=list(X_tr.columns))
    dtest = xgb.DMatrix(X_te, label=y_te, feature_names=list(X_te.columns))

    # Sélection de la fonction objectif
    loss_type = params.get("loss_type")
    epsilon = params.get("epsilon", 0.02)
    scale = params.get("scale", 1.0)
    
    if loss_type == "hinge":
        obj_fn = hinge_squared_obj(epsilon=epsilon, scale=scale)
    elif loss_type == "asymmetric":
        obj_fn = asymmetric_over_obj(scale)
    else:
        # Fallback: use XGBoost's default RMSE (reg:squarederror)
        obj_fn = None

    booster = xgb.train(
        params=booster_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        obj=obj_fn,
        verbose_eval=False
    )

    preds_test = booster.predict(dtest)
    preds_train = booster.predict(dtrain)

    base_metrics_test = evaluate_regressor(y_te, preds_test)
    base_metrics_train = {f"train_{k}": v for k, v in evaluate_regressor(y_tr, preds_train).items()}

    # Overprediction ratio
    over_mask = preds_test > y_te
    if len(y_te) > 0:
        base_metrics_test["over_rate"] = float(np.mean(over_mask))
    if len(y_tr) > 0:
        base_metrics_train["train_over_rate"] = float(np.mean(preds_train > y_tr))

    metrics_row = {**base_metrics_test, **base_metrics_train}
    metrics_row["year"] = year_split
    scores_df = pd.DataFrame([metrics_row])

    
    applied_scores = preds_test
    top_next = build_topk_per_month(
        df=test[['year_month', 'ticker'] + features].copy(),
        scores=applied_scores,
        k=n_asset
    )
    top_next['year_month'] = top_next['year_month'] + 1

    detailled = top_next.merge(df_returns, on=["year_month", "ticker"], how="left")
    agg = (detailled.groupby("year_month")["monthly_return"].mean()).reset_index()
    agg['monthly_return'] = agg['monthly_return'] - 1
    agg = agg.merge(index.monthly_returns.rename(columns={'monthly_return': 'monthly_return_index'}),
                    on='year_month', how='left')

    shap_values = None
    shap_interaction_values = None
    if compute_shap and not X_te.empty:
        try:
            shap_values, shap_interaction_values = compute_shap_values(model=booster, X=X_te)
            if shap_run_plots:
                # Fire standard analysis (plots only, no capture)
                run_shap_analysis(model=booster, X=X_te, top_n_heatmap=shap_top_n)
        except Exception as e:
            warnings.warn(f"[{year_split}] SHAP computation failed: {e}")

    return {
        "scores": scores_df,
        "detailled_return": detailled,
        "aggregated_return": agg,
        "model": booster,
        "shap_values": shap_values,                 # np.ndarray or None
        "shap_interaction_values": shap_interaction_values,  # np.ndarray or None
        "test_preds": pd.DataFrame({
            "ticker": test['ticker'].values,
            "year_month": test['year_month'].values,
            "y_true": y_te,
            "y_pred": preds_test
        })
    }


def train_fit_reg(df_learning: pd.DataFrame,
                  params: Dict,
                  df_returns: pd.DataFrame,
                  index,
                  n_asset: int,
                  features: List[str],
                  month_window : int = 1000,
                  target: str = "future_return",
                  alpha_over_obj: float = 2.0,
                  compute_shap_last: bool = False,
                  loss_type: str = "hinge",
                  hinge_epsilon: float = 0.02,
                  hinge_scale: float = 1.0):
    all_scores = []
    all_detailled = []
    all_agg = []
    all_preds = []
    months = sorted(df_learning['year_month'].unique(), reverse=True)
    for ym in tqdm(months[:-12], desc="Regressor walk"):
        compute_shap = compute_shap_last and (ym == months[0])
        out = train_fit_on_year_month_reg(
            df_learning=df_learning,
            params=params,
            df_returns=df_returns,
            index=index,
            year_month_split=ym,
            n_asset=n_asset,
            features=features,
            month_window = month_window,
            target=target,
            alpha_over_obj=alpha_over_obj,
            compute_shap=compute_shap,
            loss_type=loss_type,
            hinge_epsilon=hinge_epsilon,
            hinge_scale=hinge_scale
        )
        if not out["scores"].empty:
            all_scores.append(out["scores"])
        if not out["detailled_return"].empty:
            all_detailled.append(out["detailled_return"])
        if not out["aggregated_return"].empty:
            all_agg.append(out["aggregated_return"])
        if out.get("test_preds") is not None:
            all_preds.append(out["test_preds"])

    return {
        "scores": pd.concat(all_scores).reset_index(drop=True) if all_scores else pd.DataFrame(),
        "detailled_return": pd.concat(all_detailled).reset_index(drop=True) if all_detailled else pd.DataFrame(),
        "aggregated_return": pd.concat(all_agg).reset_index(drop=True) if all_agg else pd.DataFrame(),
        "predictions": pd.concat(all_preds).reset_index(drop=True) if all_preds else pd.DataFrame()
    }



# %%
