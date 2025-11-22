import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor

# Simulate what happens in run_full_backtest.py
# Create some sample returns in multiplicative form
sample_returns = pd.Series([1.02, 0.98, 1.01, 0.99, 1.03])
print("Sample returns (multiplicative form):")
print(sample_returns)
print(f"Mean: {sample_returns.mean()}")
print(f"Mean - 1: {sample_returns.mean() - 1}")
print()

# What we should do instead
print("Correct approach - convert to decimal FIRST, then aggregate:")
decimal_returns = sample_returns - 1
print(f"Decimal returns: {decimal_returns.values}")
print(f"Mean of decimal returns: {decimal_returns.mean()}")
