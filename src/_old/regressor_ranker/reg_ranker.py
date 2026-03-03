import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from .reg_models import train_fit_reg
from .reg_metrics import evaluate_regressor
from ..probalisor.search_spaces import sample_xgb_space
from sklearn.metrics import mean_squared_error
# %%
def regression_scoring(final_returns: Dict,
                       metric_return_col: str = "monthly_return",
                       penalty_weight: float = 1.0,
                       min_spread: float = 0.05) -> float:
    """
    Score = mean realized monthly_return - penalty_weight * (avg overprediction gap)
    Enforce prediction spread threshold.
    """
    preds_df = final_returns.get("predictions")
    preds_df = preds_df.copy()
    preds_df["year"] = preds_df["year_month"].astype(str).str[:4].astype(int)
    group_col = final_returns.get("group_col", "year")  # colonne de grouping optionnelle

    def ensure_df(x):
        if isinstance(x, pd.DataFrame):
            return x
        if isinstance(x, dict):
            return pd.DataFrame([x])
        return pd.DataFrame([{"mae": np.inf}])

    def build_regression_metrics(preds: pd.DataFrame,
                                 group_col: Optional[str] = "year_month",
                                 alpha_over: float = 2.0,
                                 n_lift_bins: int = 10) -> pd.DataFrame:
        
        if not group_col or group_col not in preds.columns:
            m = evaluate_regressor(preds["y_true"], preds["y_pred"],
                                   alpha_over=alpha_over, n_lift_bins=n_lift_bins)
            df_metrics = ensure_df(m)
            df_metrics.insert(0, group_col if group_col else "group", "ALL")
            return df_metrics

        rows = []
        for gval, gdf in preds.groupby(group_col):
            if gdf.empty:
                continue
            try:
                m = evaluate_regressor(gdf["y_true"], gdf["y_pred"],
                                       alpha_over=alpha_over, n_lift_bins=n_lift_bins)
                m_df = ensure_df(m)
                row = m_df.iloc[0].to_dict()
                row[group_col] = gval
                rows.append(row)
            except Exception:
                continue

        if rows:
            per_group_df = pd.DataFrame(rows)
            
            
            return pd.concat([per_group_df], ignore_index=True)
        else:
            # fallback global
            m = evaluate_regressor(preds["y_true"], preds["y_pred"],
                                   alpha_over=alpha_over, n_lift_bins=n_lift_bins)
            df_metrics = ensure_df(m)
            df_metrics.insert(0, group_col if group_col else "group", "ALL")
            return df_metrics

    obj = build_regression_metrics(preds = preds_df, group_col=group_col)
    value = np.mean(np.log(1+obj['calibration_corr']))
    mini = np.min(obj['calibration_corr'])
    value += penalty_weight*mini
    
    
    #agg = final_returns.get("aggregated_return")
    #agg['monthly_return_vs_index'] = (1+agg['monthly_return']) /(1+ agg['monthly_return_index'])-1
    #agg['monthly_return_vs_index'] = np.log(1+3*agg['monthly_return_vs_index'])
    
    #if agg is None or agg.empty:
    #    return -999.0
    return value

def _objective_builder(df_learning: pd.DataFrame,
                       df_returns: pd.DataFrame,
                       index,
                       features: List[str],
                       n_asset: int,
                       month_window: int,
                       target: str,
                       metric_return_col: str,
                       penalty_weight: float,
                       min_spread: float,
                       alpha_over_obj: float,
                       loss_type: str,
                       hinge_epsilon: float,
                       hinge_scale: float):
    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_space(trial)
        out = train_fit_reg(
            df_learning=df_learning,
            params=params,
            df_returns=df_returns,
            index=index,
            n_asset=n_asset,
            features=features,
            month_window = month_window,
            target=target,
            alpha_over_obj=alpha_over_obj,
            compute_shap_last=False,
            loss_type=loss_type,
            hinge_epsilon=hinge_epsilon,
            hinge_scale=hinge_scale
        )
        return regression_scoring(
            out,
            metric_return_col=metric_return_col,
            penalty_weight=penalty_weight,
            min_spread=min_spread
        )
    return objective

def optimize_regression(df: pd.DataFrame,
                        split_date: str,
                        df_returns: pd.DataFrame,
                        index,
                        features: List[str],
                        n_asset: int = 10,
                        month_window: int = 1000,
                        target: str = "future_return",
                        n_trials: int = 50,
                        seed: int = 123,
                        metric_return_col: str = "monthly_return",
                        penalty_weight: float = 1.0,
                        min_spread: float = 0.05,
                        alpha_over_obj: float = 2.0,
                        loss_type: str = "hinge",
                        hinge_epsilon: float = 0.02,
                        hinge_scale: float = 1.0) -> Dict[str, Any]:
    """
    Adds hinge loss support:
      loss_type='hinge' uses squared hinge tube (epsilon, scale)
      loss_type='asymmetric' keeps custom over-prediction penalty (alpha_over_obj)
    """
    # This code snippet is performing hyperparameter optimization using Optuna for a regression task.
    # Here's a breakdown of what each step is doing:
    df_learning = df[df['year_month'] < split_date].copy()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    objective = _objective_builder(df_learning,
                                   df_returns,
                                   index,
                                   features,
                                   n_asset,
                                   month_window,
                                   target,
                                   metric_return_col,
                                   penalty_weight,
                                   min_spread,
                                   alpha_over_obj,
                                   loss_type,
                                   hinge_epsilon,
                                   hinge_scale)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params

    # Refit on all data (train + "test" portion)
    all_out = train_fit_reg(
        df_learning=df,
        params=best_params,
        df_returns=df_returns,
        index=index,
        n_asset=n_asset,
        features=features,
        target=target,
        alpha_over_obj=alpha_over_obj,
        compute_shap_last=False,
        loss_type=loss_type,
        hinge_epsilon=hinge_epsilon,
        hinge_scale=hinge_scale
    )

    return {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "study": study,
        "all_returns": all_out,
        "config": {
            "metric_return_col": metric_return_col,
            "penalty_weight": penalty_weight,
            "min_spread": min_spread,
            "alpha_over_obj": alpha_over_obj,
            "n_asset": n_asset,
            "loss_type": loss_type,
            "hinge_epsilon": hinge_epsilon,
            "hinge_scale": hinge_scale
        }
    }
