import os
import sys
import pandas as pd
import numpy as np
import re
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

# Add src to python path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.strategy.legacy import StrategyLearner
from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor
from alpharank.features.indicators import TechnicalIndicators
from alpharank.visualization.plotting import StockComparisonPlotter

def main():
    # Setup paths
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print(f"Data directory not found at {data_dir}")
        return
    
    os.chdir(data_dir) # Legacy code relies on CWD for some reason, keeping it for safety or adjusting paths
    
    print("Loading data...")
    us_dir = data_dir / 'US'
    final_price = pd.read_parquet(us_dir / 'US_Finalprice.parquet')
    general = pd.read_parquet(us_dir / 'US_General.parquet')
    income_statement = pd.read_parquet(us_dir / 'US_Income_statement.parquet')
    balance_sheet = pd.read_parquet(us_dir / 'US_Balance_sheet.parquet')
    cash_flow = pd.read_parquet(us_dir / 'US_Cash_flow.parquet')
    earnings = pd.read_parquet(us_dir / 'US_Earnings.parquet')
    
    us_historical_company = pd.read_csv(us_dir / "SP500_Constituents.csv")
    sp500_price = pd.read_parquet(us_dir / 'SP500Price.parquet')
    
    # Preprocessing
    print("Preprocessing...")
    ticker_to_exclude = ['SII.US', 'CBE.US', 'TIE.US']
    for ticker in ticker_to_exclude:
        final_price = final_price[final_price['ticker'] != ticker]
        general = general[general['ticker'] != ticker]
        income_statement = income_statement[income_statement['ticker'] != ticker]
        balance_sheet = balance_sheet[balance_sheet['ticker'] != ticker]
        cash_flow = cash_flow[cash_flow['ticker'] != ticker]
        earnings = earnings[earnings['ticker'] != ticker]

    final_price['year_month'] = pd.to_datetime(final_price['date']).dt.to_period('M')
    us_historical_company['ticker'] = us_historical_company['Ticker'].apply(lambda x: re.sub(r'\.', '-', x) if isinstance(x, str) else x)
    us_historical_company['ticker'] = us_historical_company['ticker'] + '.US'
    us_historical_company['year_month'] = pd.to_datetime(us_historical_company['Date']).dt.to_period('M')

    index_data = IndexDataManager(
        daily_prices_df=sp500_price.copy(),
        components_df=us_historical_company.copy()
    )

    monthly_return = PricesDataPreprocessor().calculate_monthly_returns(df=final_price.copy(), column_close='adjusted_close', column_date='date')
    
    # Calculate prices vs index
    print("Calculating prices vs index...")
    sp500_price = sp500_price.rename(columns={'close': 'sp500_close'})
    
    final_price_vs_index = PricesDataPreprocessor.prices_vs_index(
        index=sp500_price.copy(),
        prices=final_price.copy(),
        column_close_index='sp500_close',
        column_close_prices='adjusted_close'
    )
    
    print("Calculating daily returns...")
    final_price_vs_index = PricesDataPreprocessor.compute_dr(
        df=final_price_vs_index, 
        column_date='date', 
        column_close='adjusted_close'
    )
    
    # Calculate Ratios
    print("Calculating Ratios...")
    stocks_selections = FundamentalProcessor().calculate_pe_ratios(
        balance=balance_sheet,
        earnings=earnings,
        cashflow=cash_flow,
        income=income_statement,
        earning_choice='netincome_rolling',
        monthly_return=monthly_return,
        list_date_to_maximise=['filing_date_income', 'filing_date_balance']
    )

    all_ratios = FundamentalProcessor().calculate_all_ratios(
        balance_sheet=balance_sheet.copy(),
        income_statement=income_statement.copy(),
        cash_flow=cash_flow.copy(),
        earnings=earnings.copy(),
        monthly_return=monthly_return.copy()
    )
    
    stocks_selections = (stocks_selections[(stocks_selections['pe'] < 100) & (stocks_selections['pe'] > 0)]
                        .dropna(subset=['pe', 'market_cap'])
                        .merge(us_historical_company[['year_month', 'ticker']],
                               how="inner",
                               left_on=['ticker', 'year_month'],
                               right_on=['ticker', 'year_month']))

    # Run Strategy
    print("Running Strategy Learning (Optuna)...")
    first_date = pd.Period("2006-01-01", freq='M')
    
    # Reduced trials for demo/test purposes
    optuna_output_1 = StrategyLearner.learning_process_optuna_full(
        prices=final_price_vs_index.copy(),
        index=index_data,
        first_date=first_date,
        stocks_filter=stocks_selections.copy(),
        sector=general[['ticker', 'Sector']].copy(),
        func_movingaverage=TechnicalIndicators.ema,
        n_trials=5, # Reduced from 20
        alpha=2,
        temp=10*12,
        mode="mean",
        seed=42
    )

    # Compare Models
    models = {
        'Legacy_Optuna': (optuna_output_1['aggregated']),
        'SP500': (index_data.monthly_returns)
    }
    
    # Save Legacy Returns for comparison
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    legacy_returns_path = output_dir / "legacy_returns.csv"
    optuna_output_1['aggregated'].to_csv(legacy_returns_path)
    print(f"Saved legacy returns to: {legacy_returns_path}")
    
    metrics, cumulative, correlation, worst_periods, figures = StrategyLearner.compare_models(models, start_year=2006)
    print("Metrics:", metrics)
    
    # Generate Report
    print("Generating Report...")
    data = optuna_output_1['detailled']
    last_year_month = data['year_month'].max()
    last_portfolio = data[data['year_month'] == last_year_month]
    
    tickers_list = sorted(last_portfolio['ticker'].unique().tolist())
    kpis = ['pebitda', 'gross_margin', 'netmargin', 'return_on_equity']
    
    plotter = StockComparisonPlotter(all_ratios.copy())
    report_html = plotter.make_report(
        tickers=tickers_list,
        kpis=kpis,
        normalize=True,
        smooth_span=3,
        iqr_multiplier=2.5,
        include_scatter_matrix=True
    )
    
    output_dir = project_root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "legacy_strategy_report.html"
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(report_html)
    print(f"Saved legacy report to: {out_file}")

if __name__ == "__main__":
    main()
