import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.data.processing import FundamentalProcessor

def verify_ratios():
    # Create dummy data
    dates = pd.date_range(start='2020-01-01', periods=8, freq='Q')
    ticker = 'TEST.US'
    
    data = {
        'ticker': [ticker] * 8,
        'date': dates,
        'filing_date': dates + pd.Timedelta(days=45),
        'reportDate': dates + pd.Timedelta(days=45),
    }
    
    # Balance Sheet
    balance = pd.DataFrame(data)
    balance['commonStockSharesOutstanding'] = 1000
    balance['totalStockholderEquity'] = 5000
    balance['netDebt'] = 1000
    balance['totalAssets'] = 10000
    balance['cashAndShortTermInvestments'] = 500
    
    # Income Statement
    income = pd.DataFrame(data)
    income['totalRevenue'] = [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700] # Growing
    income['grossProfit'] = income['totalRevenue'] * 0.4
    income['operatingIncome'] = income['totalRevenue'] * 0.2
    income['incomeBeforeTax'] = income['totalRevenue'] * 0.18
    income['netIncome'] = income['totalRevenue'] * 0.15
    income['ebit'] = income['totalRevenue'] * 0.2
    income['ebitda'] = income['totalRevenue'] * 0.25
    
    # Cash Flow
    cash = pd.DataFrame(data)
    cash['freeCashFlow'] = income['netIncome'] + 50
    
    # Earnings
    earnings = pd.DataFrame(data)
    earnings['epsActual'] = income['netIncome'] / 1000
    
    # Monthly Return (dummy)
    monthly_dates = pd.date_range(start='2020-01-01', periods=24, freq='M')
    monthly_return = pd.DataFrame({
        'ticker': [ticker] * 24,
        'date': monthly_dates,
        'last_close': 100,
        'monthly_return': 0.01
    })
    
    print("Calculating ratios...")
    result = FundamentalProcessor.calculate_all_ratios(
        balance_sheet=balance,
        cash_flow=cash,
        income_statement=income,
        earnings=earnings,
        monthly_return=monthly_return
    )
    
    print("\nColumns in result:")
    cols = result.columns.tolist()
    # print(cols)
    
    # Verify specific columns exist
    expected_cols = [
        'totalrevenue_rolling_lag1', 'totalrevenue_rolling_lag4',
        'netincome_rolling_lag1', 'netincome_rolling_lag4',
        'gross_margin_lag1', 'gross_margin_lag4',
        'totalrevenue_rolling_lag1_lag1' # Acceleration
    ]
    
    missing = [c for c in expected_cols if c not in cols]
    
    if missing:
        print(f"\nFAILED: Missing columns: {missing}")
    else:
        print("\nSUCCESS: All expected growth and acceleration columns found.")
        
    # Check values for revenue growth (lag1)
    # Revenue grows by 100 each quarter. TTM revenue will grow.
    # Just checking if values are not all NaN
    if result['totalrevenue_rolling_lag1'].isnull().all():
         print("WARNING: All values for totalrevenue_rolling_lag1 are NaN")
    else:
         print("Values calculated for totalrevenue_rolling_lag1.")

if __name__ == "__main__":
    verify_ratios()
