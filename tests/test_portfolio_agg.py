
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Ensure src is in path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.strategy.legacy import StrategyLearner

def test_aggregation():
    print("=== Testing Portfolio Aggregation Logic ===")
    
    # Create Mock Data
    dates = [pd.Period('2024-01', freq='M'), pd.Period('2024-02', freq='M')]
    
    # Portfolio 1: Stocks A, B, C (C has NaN return)
    df1 = pd.DataFrame({
        'year_month': dates * 3,
        'ticker': ['A.US']*2 + ['B.US']*2 + ['C.US']*2,
        'dr': [1.05, 1.02, 1.03, 1.01, np.nan, np.nan], # C is missing return data
        'Sector': ['Tech']*2 + ['Fin']*2 + ['Energy']*2
    })
    
    # Portfolio 2: Stocks B, C, D (D is unique to Port 2)
    df2 = pd.DataFrame({
        'year_month': dates * 3,
        'ticker': ['B.US']*2 + ['C.US']*2 + ['D.US']*2,
        'dr': [1.03, 1.01, np.nan, np.nan, 1.10, 1.08],
        'Sector': ['Fin']*2 + ['Energy']*2 + ['Auto']*2
    })
    
    out1 = {'detailed': df1, 'aggregated': pd.DataFrame()}
    out2 = {'detailed': df2, 'aggregated': pd.DataFrame()}
    
    outputs = [out1, out2]
    
    # Test 1: Union Mode (Default)
    print("\n--- Test 1: Union Mode (Default) ---")
    res_union = StrategyLearner.aggregate_portfolios(outputs, mode='equal', union_mode=True)
    
    # Get portfolio for Jan 2024
    jan_port = StrategyLearner.get_portfolio_at_month(res_union, month=pd.Period('2024-01', freq='M'))
    jan_tickers = sorted(jan_port['ticker'].tolist())
    print(f"Jan Holdings (Union): {jan_tickers}")
    
    # Expected: A, B, C, D (D is in Port 2, A in Port 1, B/C in both)
    # Even C (with NaN dr) should be present in holdings
    expected = ['A.US', 'B.US', 'C.US', 'D.US']
    assert jan_tickers == expected, f"Expected {expected}, got {jan_tickers}"
    print("✓ Union logic correct: All stocks from both portfolios present")
    
    # Test 2: Intersection Mode
    print("\n--- Test 2: Intersection Mode ---")
    res_inter = StrategyLearner.aggregate_portfolios(outputs, mode='equal', union_mode=False)
    
    jan_port_inter = StrategyLearner.get_portfolio_at_month(res_inter, month=pd.Period('2024-01', freq='M'))
    jan_tickers_inter = sorted(jan_port_inter['ticker'].tolist())
    print(f"Jan Holdings (Intersection): {jan_tickers_inter}")
    
    # Expected: B, C (only stocks in BOTH)
    expected_inter = ['B.US', 'C.US']
    assert jan_tickers_inter == expected_inter, f"Expected {expected_inter}, got {jan_tickers_inter}"
    print("✓ Intersection logic correct: Only common stocks present")
    
    # Test 3: Frequency Weighting
    print("\n--- Test 3: Frequency Weighting ---")
    res_freq = StrategyLearner.aggregate_portfolios(outputs, mode='frequency', union_mode=True)
    jan_port_freq = StrategyLearner.get_portfolio_at_month(res_freq, month=pd.Period('2024-01', freq='M'))
    
    # B and C appear in 2 models -> Weight should be 2/2 = 1.0 (before normalization)
    # A and D appear in 1 model -> Weight should be 1/2 = 0.5
    
    # Check A
    w_A = jan_port_freq[jan_port_freq['ticker']=='A.US']['weight'].iloc[0]
    w_B = jan_port_freq[jan_port_freq['ticker']=='B.US']['weight'].iloc[0]
    
    print(f"Weight A (1 model): {w_A}")
    print(f"Weight B (2 models): {w_B}")
    
    assert w_A == 0.5, f"Expected weight 0.5 for A, got {w_A}"
    assert w_B == 1.0, f"Expected weight 1.0 for B, got {w_B}"
    print("✓ Frequency weights correct")
    
    print("\nSUCCESS: All aggregation tests passed!")

if __name__ == "__main__":
    test_aggregation()
