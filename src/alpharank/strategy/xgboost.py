import optuna
import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Any, List, Tuple, Callable, Optional
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import copy
import random

from alpharank.strategy.base import BaseStrategy
from alpharank.data.datasets import clean_to_category, make_rank_dataset, make_prob_dataset
from alpharank.utils.metrics import evaluate_classifier
from alpharank.models.shap_analysis import compute_shap_values, run_shap_analysis, compute_best_shap
# %%
class XGBoostStrategy(BaseStrategy):
    """
    XGBoost-based strategy supporting Classification, Regression, and Ranking.
    """
    
    def __init__(self, mode: str = 'classification', params: Dict[str, Any] = None, n_simu: int = 1, std_perturb: float = 0.1):
        """
        Args:
            mode: 'classification', 'regression', or 'ranking'
            params: XGBoost hyperparameters
            n_simu: Number of simulations (perturbations) for robustness
            std_perturb: Standard deviation for parameter perturbation
        """
        super().__init__(name=f"XGBoost_{mode}", params=params)
        self.mode = mode.lower()
        self.n_simu = n_simu
        self.std_perturb = std_perturb
        self.models = {} # Store trained models per simulation
        
        if self.mode not in ['classification', 'regression', 'ranking']:
            raise ValueError(f"Invalid mode: {self.mode}. Must be 'classification', 'regression', or 'ranking'.")

    def _link_function(self, y: pd.Series) -> pd.Series:
        if self.mode == 'classification':
            return (y > 1).astype(int)
        elif self.mode == 'regression':
            return np.log(y.clip(lower=1e-8))
        elif self.mode == 'ranking':
             # For ranking, we typically use the raw return or a relevance score.
             # Here we assume the input y is the future return, and we'll process it in training.
             return y
        return y

    def _get_objective(self) -> str:
        if self.mode == 'classification':
            return 'binary:logistic'
        elif self.mode == 'regression':
            return 'reg:squarederror'
        elif self.mode == 'ranking':
            return 'rank:pairwise' # or rank:ndcg
        return ''

    def train(self, data: pd.DataFrame, target_col: str = 'future_return', features: List[str] = None, **kwargs) -> None:
        """
        Train the XGBoost model(s).
        """
        if features is None:
             features = [c for c in data.columns if c not in ['ticker', 'year_month', target_col, 'monthly_return']]
        
        self.features = features
        
        # Prepare data
        X = data[features]
        y = data[target_col]
        
        # Handle Ranking specific data prep (groups)
        if self.mode == 'ranking':
             # Ranking requires sorted data by group (year_month)
             data_sorted = data.sort_values('year_month')
             X = data_sorted[features]
             y = data_sorted[target_col]
             # We need to calculate group sizes for ranking
             # This is a simplified version; in a real scenario we might need to pass groups explicitly
             # or assume 'year_month' defines the groups.
             groups = data_sorted.groupby('year_month').size().to_numpy()
        
        # Train n_simu models
        base_params = self.params.copy()
        base_params['objective'] = self._get_objective()
        base_params['n_jobs'] = -1

        for i in range(self.n_simu):
            if i == 0:
                params = base_params
            else:
                params = self._perturb_params(base_params, self.std_perturb)
            
            if self.mode == 'classification':
                model = xgb.XGBClassifier(**params)
                y_proc = self._link_function(y)
                model.fit(X, y_proc, verbose=False)
            elif self.mode == 'regression':
                model = xgb.XGBRegressor(**params)
                y_proc = self._link_function(y)
                model.fit(X, y_proc, verbose=False)
            elif self.mode == 'ranking':
                model = xgb.XGBRanker(**params)
                # Ranking needs qid or group. 
                # XGBRanker in sklearn API uses `group` argument in fit, or `qid` in X (if supported)
                # We'll use the group argument.
                model.fit(X, y, group=groups, verbose=False)
                
            self.models[i] = model

    def predict(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Generate predictions. Aggregates predictions across simulations.
        """
        X = data[self.features]
        # Ensure categorical types match training if needed (using clean_to_category helper)
        X = clean_to_category(X) 
        
        preds_list = []
        for i, model in self.models.items():
            if self.mode == 'classification':
                pred = model.predict_proba(X)[:, 1]
            elif self.mode == 'regression':
                pred = np.exp(model.predict(X)) # Inverse of log link
            elif self.mode == 'ranking':
                pred = model.predict(X)
            preds_list.append(pred)
            
        # Average predictions
        avg_preds = np.mean(preds_list, axis=0)
        
        results = data[['ticker', 'year_month']].copy()
        results['prediction'] = avg_preds
        return results

    def _perturb_params(self, params: Dict, std: float) -> Dict:
        """Perturb hyperparameters for robustness."""
        new_params = copy.deepcopy(params)
        for k, v in new_params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if k in ['n_estimators', 'max_depth', 'random_state', 'n_jobs']:
                     continue # Don't perturb these typically
                
                noise = v * std * (2 * random.random() - 1)
                new_val = v + noise
                
                # Constraints
                if k in ['subsample', 'colsample_bytree']:
                    new_val = max(0.1, min(1.0, new_val))
                elif k in ['learning_rate', 'gamma', 'reg_lambda', 'reg_alpha', 'min_child_weight']:
                    new_val = max(0.0, new_val)
                
                if isinstance(v, int):
                    new_val = int(round(new_val))
                
                new_params[k] = new_val
        return new_params

    def _calculate_volatility_penalty(self, volatility: float, min_volatility: float, exponential_factor: float) -> float:
        """
        Calculate the volatility penalty.
        """
        if volatility < min_volatility:
            # Normalized difference [0, 1]
            diff_ratio = (min_volatility - volatility) / min_volatility
            # Exponential penalty
            return np.exp(exponential_factor * diff_ratio) - 1
        return 0.0

    def optimize_hyperparameters(self, 
                                 train_df: pd.DataFrame, 
                                 validation_df: pd.DataFrame, 
                                 hparam_space: Optional[Dict] = None, 
                                 n_trials: int = 20, 
                                 n_simu: int = 1,
                                 metric_col: str = 'precision@10',
                                 target_col: str = 'future_return',
                                 min_volatility: float = 0.005,
                                 exponential_factor: float = 5.0,
                                 n_startup_trials: int = 10,
                                 optuna_report_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Optimize hyperparameters using Optuna.
        
        Returns:
            Dict containing:
            - best_params: Dict of best hyperparameters
            - trained_model: The model trained with best params (self)
            - best_score: Best objective value (with penalty)
            - best_base_score: Best objective value (without penalty)
            - best_penalty: Penalty applied to the best trial
        """
        if hparam_space is None:
            hparam_space = {
                'n_estimators': ('int', 50, 1000),
                'max_depth': ('int', 3, 10),
                'learning_rate': ('loguniform', 0.001, 0.3),
                'subsample': ('float', 0.6, 1.0),
                'colsample_bytree': ('float', 0.6, 1.0),
                'min_child_weight': ('int', 1, 10),
                'gamma': ('float', 0.0, 5.0),
                'reg_alpha': ('float', 0.0, 10.0),
                'reg_lambda': ('float', 0.0, 10.0)
            }
        
        def objective(trial):
            params = {}
            for name, (ptype, low, high) in hparam_space.items():
                if ptype == "int": 
                    params[name] = trial.suggest_int(name, low, high)
                elif ptype == "loguniform":
                    params[name] = trial.suggest_float(name, low, high, log=True)
                else: 
                    params[name] = trial.suggest_float(name, low, high)
            
            # Setup temp strategy with these params
            temp_strat = XGBoostStrategy(mode=self.mode, params=params, n_simu=n_simu)
            
            # Train on train_df
            # Note: For ranking, we need to be careful with groups in CV. 
            # Here we assume simple train/val split is respected by the caller.
            temp_strat.train(train_df, target_col=target_col)
            
            # Predict on validation_df
            preds = temp_strat.predict(validation_df)

            volatility_preds = preds['prediction'].std()
            
            # Evaluate
            # We need to merge with actual returns/targets to evaluate
            val_with_preds = validation_df.merge(preds[['ticker', 'year_month', 'prediction']], on=['ticker', 'year_month'])
            
            # Calculate metric
            # This part needs to be flexible based on metric_col. 
            # For now, implementing a simple precision@k or IC logic.
            
            score = 0.0
            if 'precision' in metric_col:
                k = int(metric_col.split('@')[1])
                # Sort by prediction
                top_k = val_with_preds.sort_values('prediction', ascending=False).groupby('year_month').head(k)
                # Calculate precision (fraction of positives) - assumes binary target available or derived
                # If target is continuous return, precision might mean "fraction of positive returns" or "fraction outperforming index"
                # Let's assume we want to maximize the mean return of the top K for simplicity if metric is generic,
                # but if it's precision, we need a binary target.
                
                # Let's use the actual return to calculate a 'score'
                score = top_k['monthly_return'].mean() 
            
            elif metric_col == 'ic': # Information Coefficient
                score = val_with_preds['prediction'].corr(val_with_preds[target_col])
            
            else:
                # Default to mean return of top 10
                top_k = val_with_preds.sort_values('prediction', ascending=False).groupby('year_month').head(10)
                score = top_k['monthly_return'].mean()

            # Apply exponential penalty for low volatility
            penalty = self._calculate_volatility_penalty(volatility_preds, min_volatility, exponential_factor)
            final_score = score - penalty
            
            # Store attributes for retrieval
            trial.set_user_attr("base_score", score)
            trial.set_user_attr("penalty", penalty)
            
            return final_score

        study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42, n_startup_trials=n_startup_trials))
        study.optimize(objective, n_trials=n_trials)
        
        # Generate report if requested
        if optuna_report_path:
            try:
                from alpharank.visualization.optuna_report import generate_optuna_report
                generate_optuna_report(study, optuna_report_path)
            except ImportError:
                print("Could not import generate_optuna_report. Is plotly installed?")
            except Exception as e:
                print(f"Error generating Optuna report: {e}")
        
        best_trial = study.best_trial
        self.params = best_trial.params
        
        # Retrain with best params
        self.train(train_df, target_col=target_col)
        
        return {
            "best_params": best_trial.params,
            "trained_model": self,
            "best_score": best_trial.value,
            "best_base_score": best_trial.user_attrs.get("base_score"),
            "best_penalty": best_trial.user_attrs.get("penalty")
        }

    def get_feature_importance(self) -> pd.DataFrame:
        """Aggregate feature importance across models."""
        if not self.models:
            return pd.DataFrame()
            
        importances = []
        for i, model in self.models.items():
            # XGBoost feature importance
            # For sklearn API:
            if hasattr(model, 'feature_importances_'):
                imp = pd.Series(model.feature_importances_, index=self.features)
                importances.append(imp)
        
        if not importances:
            return pd.DataFrame()
            
        avg_imp = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
        return avg_imp.reset_index().rename(columns={'index': 'feature', 0: 'importance'})

    def run_shap(self, data: pd.DataFrame, top_n: int = 20) -> None:
        """Run SHAP analysis on the first model."""
        if not self.models:
            print("No models trained.")
            return
            
        # Use the first model for SHAP
        model = self.models[0]
        X = data[self.features]
        X = clean_to_category(X)
        
        run_shap_analysis(model, X, top_n_heatmap=top_n)
