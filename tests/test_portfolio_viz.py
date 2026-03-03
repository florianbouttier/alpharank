
import pandas as pd
import numpy as np
from alpharank.visualization.plotting import PortfolioVisualizer
from alpharank.data.processing import FundamentalProcessor

def test_portfolio_report():
    print("Testing make_portfolio_report with raw data...")
    
    # 1. Mock Portfolio
    portfolio = pd.DataFrame({
        'ticker': ['AAPL.US', 'MSFT.US'],
        'weight_normalized': [0.6, 0.4],
        'Sector': ['Technology', 'Technology']
    })
    portfolio.attrs['month'] = '2023-12'

    # 2. Mock Price Data (Long format)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
    price_data = []
    for t in ['AAPL.US', 'MSFT.US']:
        df = pd.DataFrame({
            'date': dates,
            'ticker': t,
            'close': np.random.randn(len(dates)).cumsum() + 100,
            'adjusted_close': np.random.randn(len(dates)).cumsum() + 100
        })
        price_data.append(df)
    price_df = pd.concat(price_data)

    # 3. Mock Raw Fundamentals
    fund_dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='Q')
    
    # Balance Sheet
    bs = []
    for t in ['AAPL.US', 'MSFT.US']:
        df = pd.DataFrame({
            'ticker': t, 'date': fund_dates, 'filing_date': fund_dates,
            'commonStockSharesOutstanding': 1000, 'totalStockholderEquity': 5000,
            'netDebt': 1000, 'totalAssets': 10000, 'cashAndShortTermInvestments': 2000
        })
        bs.append(df)
    balance_sheet = pd.concat(bs)

    # Income Statement
    income = []
    for t in ['AAPL.US', 'MSFT.US']:
        df = pd.DataFrame({
            'ticker': t, 'date': fund_dates, 'filing_date': fund_dates,
            'totalRevenue': 2000, 'grossProfit': 1000, 'operatingIncome': 500,
            'incomeBeforeTax': 400, 'netIncome': 300, 'ebit': 500, 'ebitda': 600
        })
        income.append(df)
    income_statement = pd.concat(income)

    # Cash Flow
    cf = []
    for t in ['AAPL.US', 'MSFT.US']:
        df = pd.DataFrame({
            'ticker': t, 'date': fund_dates, 'filing_date': fund_dates,
            'freeCashFlow': 250
        })
        cf.append(df)
    cash_flow = pd.concat(cf)

    # Earnings
    earn = []
    for t in ['AAPL.US', 'MSFT.US']:
        df = pd.DataFrame({
            'ticker': t, 'date': fund_dates, 'reportDate': fund_dates,
            'epsActual': 2.5
        })
        earn.append(df)
    earnings = pd.concat(earn)

    # 4. Generate Report
    try:
        html = PortfolioVisualizer.make_portfolio_report(
            portfolio=portfolio,
            title="Test Report Libs",
            price_data=price_df,
            balance_sheet=balance_sheet,
            income_statement=income_statement,
            cash_flow=cash_flow,
            earnings=earnings
        )
        print("Report generated successfully!")
        
        # Validation
        if "Detailed Stock Analysis" not in html:
            print("FAILED: 'Detailed Stock Analysis' header missing")
        if "Financial Growth" not in html:
             print("FAILED: Financial Growth chart missing")
             
        with open("outputs/test_portfolio_viz_libs.html", "w") as f:
            f.write(html)
        print("Saved to outputs/test_portfolio_viz_libs.html")
        
    except Exception as e:
        print(f"FAILED with error: {e}")
        raise e

if __name__ == "__main__":
    test_portfolio_report()
