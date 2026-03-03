
import os
import numpy as np
import pandas as pd

from alpharank.models.xgboost import XGBoostModel


def test_timeseries_hyperopt():
    """Simple smoke test for the Optuna + TimeSeriesSplit pipeline."""
    dates = pd.date_range(start="2020-01-01", periods=12, freq="M")
    tickers = [f"TICKER_{i}" for i in range(8)]

    rows = []
    for date in dates:
        for tic in tickers:
            rows.append(
                {
                    "year_month": date,
                    "ticker": tic,
                    "feature1": np.random.rand(),
                    "feature2": np.random.rand(),
                    "future_return": np.random.randn() * 0.05,
                    "monthly_return": np.random.randn() * 0.05,
                }
            )

    df = pd.DataFrame(rows)

    model = XGBoostModel(mode="classification")
    search_space = {
        "n_estimators": ("int", 10, 30),
        "max_depth": ("int", 2, 4),
        "learning_rate": ("float", 0.05, 0.2),
    }

    result = model.optimize_hyperparameters(
        data=df,
        target_col="future_return",
        metric="roc_auc",
        hparam_space=search_space,
        n_trials=3,
        cv_folds=3,
        optuna_report_path="optuna_report.html",
    )

    print("Best params:", result["best_params"])
    print("Best score:", result["best_score"])
    print("Report exists:", os.path.exists("optuna_report.html"))


if __name__ == "__main__":
    test_timeseries_hyperopt()
