# %%
import os
import pandas as pd
from pathlib import Path

project_root = Path(__file__).parent.parent

from alpharank.data.processing import FundamentalProcessor, PricesDataPreprocessor
from alpharank.visualization.financial_comparison import FinancialReportGenerator

def load_data(data_dir: str):
    """Loads necessary parquet files."""
    print(f"Loading data from {data_dir}...")
    us_dir = os.path.join(data_dir, 'US')
    
    final_price = pd.read_parquet(os.path.join(us_dir, 'US_Finalprice.parquet'))
    income = pd.read_parquet(os.path.join(us_dir, 'US_Income_statement.parquet'))
    balance = pd.read_parquet(os.path.join(us_dir, 'US_Balance_sheet.parquet'))
    cash = pd.read_parquet(os.path.join(us_dir, 'US_Cash_flow.parquet'))
    earnings = pd.read_parquet(os.path.join(us_dir, 'US_Earnings.parquet'))
    
    return final_price, income, balance, cash, earnings

def process_data(final_price, income, balance, cash, earnings):
    """Calculates fundamentals and returns."""
    print("Processing data (calculating returns and ratios)...")
    
    # Calculate Monthly Returns
    monthly_return = PricesDataPreprocessor().calculate_monthly_returns(
        df=final_price.copy(), column_close='adjusted_close', column_date='date'
    )
    
    # Calculate Fundamentals
    fundamental = FundamentalProcessor().calculate_all_ratios(
        balance_sheet=balance.copy(),
        income_statement=income.copy(),
        cash_flow=cash.copy(),
        earnings=earnings.copy(),
        monthly_return=monthly_return.copy()
    )
    
    # Calculate Net Income Growth (TTM)
    # Formula: (Current TTM - Previous Year TTM) / abs(Previous Year TTM)
    # Assuming monthly data, lag is 12 months
    if 'netincome_rolling' in fundamental.columns:
        fundamental = fundamental.sort_values(['ticker', 'year_month'])
        fundamental['net_income_growth'] = fundamental.groupby('ticker')['netincome_rolling'].pct_change(periods=12)
    
    return fundamental

def main():
    # Script is in scripts/, so project root is parent of parent
    data_dir = project_root.parent / 'data'
    if not data_dir.exists():
        # Fallback if running from root
        data_dir = project_root / 'data'
        
    if not data_dir.exists():
        print(f"Error: Data directory not found at {data_dir}")
        return

    # 1. Load Real Data
    final_price, income, balance, cash, earnings = load_data(str(data_dir))
    
    # 2. Process Data
    df = process_data(final_price, income, balance, cash, earnings)
    
    # 3. Generate Report
    print("Generating Financial Report...")
    generator = FinancialReportGenerator(df)
    
    tickers = ['GEV.US', 'NVDA.US', 'MSTR.US', 'AMZN.US']
    # Corrected KPI names based on FundamentalProcessor output
    kpis = ['pnetresult', 'ps_ratio', 'ev_ebitda_ratio', 'gross_margin', 'netmargin', 'return_on_equity', 'return_on_assets', 'net_income_growth']
    
    # Ensure KPIs exist in dataframe
    available_kpis = [k for k in kpis if k in df.columns]
    if len(available_kpis) < len(kpis):
        print(f"Warning: Some KPIs were not found in data: {set(kpis) - set(available_kpis)}")
    
    html_report = generator.generate_report(
        tickers=tickers,
        kpis=available_kpis,
        start_date='2020-01-01',
        title="Real Data Financial Comparison"
    )
    
    output_path = "real_financial_report.html"
    with open(output_path, "w") as f:
        f.write(html_report)
        
    print(f"Report saved to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    main()

# %%
