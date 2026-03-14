import copy
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    mean_squared_error,
    ndcg_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from alpharank.strategy.base import BaseStrategy
from alpharank.data.datasets import clean_to_category
from alpharank.models.shap_analysis import run_shap_analysis
from alpharank.utils.xgboost_runtime import load_xgboost

xgb = load_xgboost()


class XGBoostModel(BaseStrategy):
    """
    Unified XGBoost model supporting classification, regression and ranking.
    Positioned under alpharank.models for consistency with the rest of the codebase.
    """

    def __init__(
        self,
        mode: str = "classification",
        params: Optional[Dict[str, Any]] = None,
        n_simu: int = 1,
        std_perturb: float = 0.1,
        threshold: float = 0.0,
    ):
        super().__init__(name=f"XGBoost_{mode}", params=params or {})
        self.mode = mode.lower()
        if self.mode not in {"classification", "regression", "ranking"}:
            raise ValueError("mode must be one of: classification, regression, ranking")
        self.n_simu = n_simu
        self.std_perturb = std_perturb
        self.threshold = threshold
        self.features: List[str] = []
        self.models: Dict[int, Any] = {}

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def train(
        self,
        data: pd.DataFrame,
        target_col: str = "future_return",
        features: Optional[List[str]] = None,
        group_col: str = "year_month",
    ) -> None:
        """Fit one or several (perturbed) XGBoost estimators."""
        self.features = features or self._default_features(data, target_col)
        data_sorted = data.sort_values(group_col).reset_index(drop=True)
        X = clean_to_category(data_sorted[self.features])
        y = self._prepare_target(data_sorted[target_col])

        base_params = {**self.params}
        base_params["objective"] = self._get_objective()
        base_params.setdefault("n_jobs", -1)

        for i in range(self.n_simu):
            params = self._perturb_params(base_params) if i else base_params
            model = self._build_estimator(params)

            if self.mode == "ranking":
                group_sizes = (
                    data_sorted[group_col].value_counts().sort_index().to_numpy()
                )
                model.fit(X, y, group=group_sizes, verbose=False)
            else:
                model.fit(X, y, verbose=False)

            self.models[i] = model

    def predict(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return averaged predictions over simulations."""
        if not self.models:
            raise RuntimeError("Model not trained. Call train() first.")

        X = clean_to_category(data[self.features])
        preds = []
        for model in self.models.values():
            if self.mode == "classification":
                preds.append(model.predict_proba(X)[:, 1])
            else:
                preds.append(model.predict(X))

        avg_pred = np.mean(preds, axis=0)
        out = data[["ticker", "year_month"]].copy()
        out["prediction"] = avg_pred
        return out

    def optimize_hyperparameters(
        self,
        data: pd.DataFrame,
        target_col: str = "future_return",
        features: Optional[List[str]] = None,
        group_col: str = "year_month",
        metric: Optional[str] = None,
        hparam_space: Optional[Dict[str, Tuple[str, float, float]]] = None,
        n_trials: int = 25,
        cv_folds: int = 4,
        n_startup_trials: int = 10,
        timeout: Optional[int] = None,
        optuna_report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Time-series cross-validated Optuna tuning.
        - Uses sklearn TimeSeriesSplit to respect chronology (important for finance).
        - Supports classification/regression/ranking.
        """
        metric = metric or self._default_metric()
        features = features or self._default_features(data, target_col)
        search_space = hparam_space or self._default_hparam_space()

        data_sorted = data.sort_values(group_col).reset_index(drop=True)
        X_all = clean_to_category(data_sorted[features])
        y_all = self._prepare_target(data_sorted[target_col])

        tscv = TimeSeriesSplit(n_splits=cv_folds)

        def objective(trial: optuna.Trial) -> float:
            params = self._sample_params(trial, search_space)
            params["objective"] = self._get_objective()
            params.setdefault("n_jobs", -1)

            fold_scores: List[float] = []
            for train_idx, val_idx in tscv.split(X_all):
                train_df = data_sorted.iloc[train_idx]
                val_df = data_sorted.iloc[val_idx]

                X_train, X_val = (
                    clean_to_category(train_df[features]),
                    clean_to_category(val_df[features]),
                )
                y_train = self._prepare_target(train_df[target_col])
                y_val = self._prepare_target(val_df[target_col])

                model = self._build_estimator(params)

                if self.mode == "ranking":
                    group_sizes = (
                        train_df[group_col].value_counts().sort_index().to_numpy()
                    )
                    model.fit(X_train, y_train, group=group_sizes, verbose=False)
                    preds = model.predict(X_val)
                    fold_scores.append(
                        self._ranking_score(
                            val_df[group_col].reset_index(drop=True),
                            y_val,
                            preds,
                            metric,
                        )
                    )
                else:
                    model.fit(X_train, y_train, verbose=False)
                    preds = (
                        model.predict_proba(X_val)[:, 1]
                        if self.mode == "classification"
                        else model.predict(X_val)
                    )
                    fold_scores.append(self._scalar_score(y_val, preds, metric))

            return float(np.nanmean(fold_scores))

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42, n_startup_trials=n_startup_trials),
            pruner=MedianPruner(n_startup_trials=n_startup_trials),
        )
        study.optimize(objective, n_trials=n_trials, timeout=timeout)

        best_params = study.best_trial.params
        self.params = best_params
        self.train(data, target_col=target_col, features=features, group_col=group_col)

        if optuna_report_path:
            try:
                from alpharank.visualization.optuna_report import (
                    generate_optuna_report,
                )

                generate_optuna_report(study, optuna_report_path)
            except Exception as exc:  # pragma: no cover
                print(f"Optuna report generation failed: {exc}")

        return {
            "best_params": best_params,
            "best_score": study.best_value,
            "trained_model": self,
        }

    @staticmethod
    def split_time_series(
        data: pd.DataFrame,
        date_col: str = "year_month",
        train_end: str = None,
        val_end: str = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Helper to create chronological train/val/test splits.
        Args:
            train_end: last date (inclusive) for training set (e.g., '2022-12').
            val_end: last date (inclusive) for validation set; test is after val_end.
        """
        if train_end is None or val_end is None:
            raise ValueError("train_end and val_end must be provided.")
        data_copy = data.copy()
        data_copy[date_col] = pd.to_datetime(data_copy[date_col]).dt.to_period("M")
        train = data_copy[data_copy[date_col] <= train_end]
        val = data_copy[(data_copy[date_col] > train_end) & (data_copy[date_col] <= val_end)]
        test = data_copy[data_copy[date_col] > val_end]
        return train, val, test

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _get_objective(self) -> str:
        if self.mode == "classification":
            return "binary:logistic"
        if self.mode == "regression":
            return "reg:squarederror"
        return "rank:pairwise"

    def _default_metric(self) -> str:
        if self.mode == "classification":
            return "roc_auc"
        if self.mode == "regression":
            return "rmse"
        return "spearman"

    def _default_features(self, data: pd.DataFrame, target_col: str) -> List[str]:
        excluded = {"ticker", "year_month", target_col, "monthly_return"}
        return [c for c in data.columns if c not in excluded]

    def _build_estimator(self, params: Dict[str, Any]):
        if self.mode == "classification":
            return xgb.XGBClassifier(**params)
        if self.mode == "regression":
            return xgb.XGBRegressor(**params)
        return xgb.XGBRanker(**params)

    def _prepare_target(self, y: pd.Series) -> pd.Series:
        if self.mode == "classification":
            if set(np.unique(y)) <= {0, 1}:
                return y
            return (y > self.threshold).astype(int)
        return y

    def _perturb_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        new_params = copy.deepcopy(params)
        for k, v in params.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                noise = v * self.std_perturb * (2 * np.random.rand() - 1)
                new_val = v + noise
                if k in {"subsample", "colsample_bytree"}:
                    new_val = float(np.clip(new_val, 0.1, 1.0))
                elif k in {"learning_rate", "gamma", "reg_lambda", "reg_alpha", "min_child_weight"}:
                    new_val = float(max(0.0, new_val))
                if isinstance(v, int):
                    new_val = int(max(1, round(new_val)))
                new_params[k] = new_val
        return new_params

    def _sample_params(
        self, trial: optuna.Trial, space: Dict[str, Tuple[str, float, float]]
    ) -> Dict[str, Any]:
        params = {}
        for name, (ptype, low, high) in space.items():
            if ptype == "int":
                params[name] = trial.suggest_int(name, int(low), int(high))
            elif ptype == "loguniform":
                params[name] = trial.suggest_float(name, low, high, log=True)
            else:
                params[name] = trial.suggest_float(name, low, high)
        return params

    def _default_hparam_space(self) -> Dict[str, Tuple[str, float, float]]:
        return {
            "n_estimators": ("int", 200, 800),
            "max_depth": ("int", 3, 10),
            "learning_rate": ("loguniform", 0.01, 0.3),
            "subsample": ("float", 0.5, 1.0),
            "colsample_bytree": ("float", 0.5, 1.0),
            "min_child_weight": ("float", 1.0, 10.0),
            "gamma": ("float", 0.0, 5.0),
            "reg_alpha": ("float", 0.0, 5.0),
            "reg_lambda": ("float", 0.0, 5.0),
        }

    def _scalar_score(self, y_true: pd.Series, preds: np.ndarray, metric: str) -> float:
        if metric == "roc_auc":
            return roc_auc_score(y_true, preds)
        if metric == "average_precision":
            return average_precision_score(y_true, preds)
        if metric == "rmse":
            return -float(np.sqrt(mean_squared_error(y_true, preds)))
        if metric == "mae":
            return -float(mean_absolute_error(y_true, preds))
        # default fallback
        return float(np.corrcoef(y_true, preds)[0, 1])

    def _ranking_score(
        self, groups: pd.Series, y_true: pd.Series, preds: np.ndarray, metric: str
    ) -> float:
        # Compute per-group score to respect grouping by date.
        scores: List[float] = []
        for g_val in groups.unique():
            mask = groups == g_val
            y_g = y_true[mask].to_numpy()
            p_g = preds[mask]
            if len(y_g) < 2:
                continue
            if metric.startswith("ndcg"):
                k = int(metric.split("@")[1]) if "@" in metric else min(10, len(y_g))
                scores.append(ndcg_score([y_g], [p_g], k=k))
            elif metric == "spearman":
                scores.append(pd.Series(p_g).corr(pd.Series(y_g), method="spearman"))
            else:
                scores.append(pd.Series(p_g).corr(pd.Series(y_g), method="pearson"))
        return float(np.nanmean(scores))

    # ------------------------------------------------------------------ #
    # Optional explainability
    # ------------------------------------------------------------------ #
    def run_shap(self, data: pd.DataFrame, top_n: int = 20) -> None:
        if not self.models:
            raise RuntimeError("No trained model to explain.")
        model = next(iter(self.models.values()))
        X = clean_to_category(data[self.features])
        run_shap_analysis(model, X, top_n_heatmap=top_n)
