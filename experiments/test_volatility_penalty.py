
import pandas as pd
import numpy as np
from alpharank.strategy.xgboost import XGBoostStrategy

def test_volatility_penalty():
    # Create dummy data
    dates = pd.date_range(start='2020-01-01', periods=12, freq='M')
    tickers = [f'TICKER_{i}' for i in range(10)]
    
    data = []
    for date in dates:
        for ticker in tickers:
            data.append({
                'year_month': date,
                'ticker': ticker,
                'feature1': np.random.rand(),
                'feature2': np.random.rand(),
                'future_return': np.random.randn() * 0.05,
                'monthly_return': np.random.randn() * 0.05
            })
    
    df = pd.DataFrame(data)
    
    # Split into train and validation
    train_df = df[df['year_month'] < '2020-10-01']
    val_df = df[df['year_month'] >= '2020-10-01']
    
    # Initialize strategy
    strategy = XGBoostStrategy(mode='classification')
    
    # Define a small search space for speed
    hparam_space = {
        'n_estimators': ('int', 10, 20),
        'max_depth': ('int', 2, 4),
        'learning_rate': ('float', 0.1, 0.2)
    }
    
    print("Running optimization with volatility penalty...")
    # Run optimization with penalty
    result = strategy.optimize_hyperparameters(
        train_df=train_df,
        validation_df=val_df,
        hparam_space=hparam_space,
        n_trials=5,
        min_volatility=0.1, # Set high to force penalty
        exponential_factor=5.0,
        n_startup_trials=2, # Small number for test
        optuna_report_path="optuna_report.html"
    )
    best_params = result['best_params']
    
    print("Best params found:", best_params)
    print("Best score:", result['best_score'])
    print("Best penalty:", result['best_penalty'])
    
    # Check if report exists
    import os
    if os.path.exists("optuna_report.html"):
        print("Optuna report generated successfully.")
    else:
        print("Optuna report NOT found.")
    print("Test completed successfully.")

if __name__ == "__main__":
    test_volatility_penalty()
