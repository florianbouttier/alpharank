import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.strategy.xgboost import XGBoostStrategy
from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor
from alpharank.utils.metrics import evaluate_classifier

def main():
    print("Running AlphaRank Backtest...")
    
    # Load data (Placeholder - replace with actual data loading logic)
    # For now, we assume data is available in a specific location or we mock it for the script structure
    # In a real scenario, we would load the parquet files here.
    
    # Example usage of XGBoostStrategy
    print("Initializing Strategy...")
    strategy = XGBoostStrategy(mode='classification', n_simu=3)
    
    # Mock data for demonstration
    dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    data = []
    for date in dates:
        for ticker in tickers:
            data.append({
                'year_month': date.to_period('M'),
                'ticker': ticker,
                'feature1': np.random.rand(),
                'feature2': np.random.rand(),
                'future_return': np.random.rand() - 0.5, # Random return
                'monthly_return': np.random.rand() - 0.5
            })
    df = pd.DataFrame(data)
    
    print("Training Strategy...")
    # Split train/test
    train_df = df[df['year_month'] < '2021-01']
    test_df = df[df['year_month'] >= '2021-01']
    
    strategy.train(train_df, target_col='future_return')
    
    print("Predicting...")
    predictions = strategy.predict(test_df)
    print(predictions.head())
    
    # Optimization example
    print("Optimizing Hyperparameters...")
    best_params = strategy.optimize_hyperparameters(
        train_df=train_df,
        validation_df=test_df, # In real usage, use a separate validation set
        hparam_space={
            'n_estimators': ('int', 50, 200),
            'learning_rate': ('float', 0.01, 0.3),
            'max_depth': ('int', 3, 10)
        },
        n_trials=2
    )
    print("Best Params:", best_params)

if __name__ == "__main__":
    main()
