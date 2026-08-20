# %%
import os
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import List, Tuple

from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor
from alpharank.features.indicators import DecorrelatedIndicatorGenerator
from alpharank.data.datasets import prepare_data_for_xgboost
from alpharank.utils.data_utils import remove_columns_by_keywords
from alpharank.models.xgboost import XGBoostModel
import plotly.io as pio
import matplotlib.pyplot as plt

# %%
def load_data(data_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Loads all necessary parquet and csv files."""
    print(f"Loading data from {data_dir}...")
    us_dir = os.path.join(data_dir, 'US')
    
    final_price = pd.read_parquet(os.path.join(us_dir, 'US_Finalprice.parquet'))
    general = pd.read_parquet(os.path.join(us_dir, 'US_General.parquet'))
    income_statement = pd.read_parquet(os.path.join(us_dir, 'US_Income_statement.parquet'))
    balance_sheet = pd.read_parquet(os.path.join(us_dir, 'US_Balance_sheet.parquet'))
    cash_flow = pd.read_parquet(os.path.join(us_dir, 'US_Cash_flow.parquet'))
    earnings = pd.read_parquet(os.path.join(us_dir, 'US_Earnings.parquet'))
    
    us_historical_company = pd.read_csv(os.path.join(us_dir, "SP500_Constituents.csv"))
    sp500_price = pd.read_parquet(os.path.join(us_dir, 'SP500Price.parquet'))
    
    return final_price, general, income_statement, balance_sheet, cash_flow, earnings, us_historical_company, sp500_price

def preprocess_data(final_price, general, income_statement, balance_sheet, cash_flow, earnings, us_historical_company, sp500_price):
    """Preprocesses raw data into clean dataframes."""
    print("Preprocessing data...")
    
    # Exclude specific tickers
    ticker_to_exclude = ['SII.US', 'CBE.US', 'TIE.US']
    for ticker in ticker_to_exclude:
        final_price = final_price[final_price['ticker'] != ticker]
        general = general[general['ticker'] != ticker]
        income_statement = income_statement[income_statement['ticker'] != ticker]
        balance_sheet = balance_sheet[balance_sheet['ticker'] != ticker]
        cash_flow = cash_flow[cash_flow['ticker'] != ticker]
        earnings = earnings[earnings['ticker'] != ticker]

    # Date conversions
    final_price['year_month'] = pd.to_datetime(final_price['date']).dt.to_period('M')
    us_historical_company['ticker'] = us_historical_company['Ticker'].apply(lambda x: re.sub(r'\.', '-', x) if isinstance(x, str) else x)
    us_historical_company['ticker'] = us_historical_company['ticker'] + '.US'
    us_historical_company['year_month'] = pd.to_datetime(us_historical_company['Date']).dt.to_period('M')

    # Index Data
    index_data = IndexDataManager(
        daily_prices_df=sp500_price.copy(),
        components_df=us_historical_company.copy()
    )

    # Monthly Returns
    monthly_return = PricesDataPreprocessor().calculate_monthly_returns(
        df=final_price.copy(), column_close='adjusted_close', column_date='date'
    )

    # Fundamentals
    fundamental = FundamentalProcessor().calculate_all_ratios(
        balance_sheet=balance_sheet.copy(),
        income_statement=income_statement.copy(),
        cash_flow=cash_flow.copy(),
        earnings=earnings.copy(),
        monthly_return=monthly_return.copy()
    )
    
    # Drop unnecessary columns
    fundamental = fundamental.drop(
        columns=['date', 'last_close', 'monthly_return', 'quarter_end', 
                 'filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning'], 
        errors='ignore'
    )

    # Price vs Index
    final_price = PricesDataPreprocessor.prices_vs_index(
        index=index_data.daily_prices.copy(),
        prices=final_price.copy(),
        column_close_index='adjusted_close',
        column_close_prices='adjusted_close'
    )
    
    monthly_returns_vs_index = PricesDataPreprocessor.calculate_monthly_returns(
        df=final_price.copy(),
        column_date='date',
        column_close='close_vs_index'
    )

    return final_price, fundamental, monthly_returns_vs_index, index_data

def generate_indicators(final_price):
    """Generates technical indicators."""
    print("Generating technical indicators...")
    SEED_PAIRS = [(10, 100)]
    generator = DecorrelatedIndicatorGenerator(
        daily_prices_df=final_price.copy(),
        price_column='close_vs_index'
    )
    
    # Generate a small set for testing/demo purposes
    generator.generate_decorrelated_ema_ratios(
        seed_pairs=SEED_PAIRS,
        n_to_find=20, # Reduced for speed in this script
        correlation_threshold=0.98,
        max_tries=50
    )
    
    generator.generate_decorrelated_rsi(
        seed_params=[],
        n_to_find=20, # Reduced for speed
        correlation_threshold=0.98,
        max_tries=50
    )
    
    return generator.get_final_indicators()

# %%
def main():
    # 1. Setup paths
    # Assuming script is run from project root
    project_root = Path(__file__).parent.parent
    data_dir = project_root / 'data'
    
    if not data_dir.exists():
        print(f"Data directory not found at {data_dir}. Please ensure data is present.")
        return

    # 2. Load and Preprocess
    final_price, general, income, balance, cash, earnings, historical, sp500 = load_data(str(data_dir))
    final_price, fundamental, monthly_returns_vs_index, index_data = preprocess_data(
        final_price, general, income, balance, cash, earnings, historical, sp500
    )

    # 3. Feature Engineering
    technical_indicators_df = generate_indicators(final_price)
    
    # 4. Merge Data
    print("Merging data...")
    funda_joined = fundamental.merge(technical_indicators_df, on=['ticker', 'year_month'], how='left') \
                              .merge(monthly_returns_vs_index[['ticker', 'year_month', 'monthly_return']], on=['ticker', 'year_month'], how='left')
    
    funda_joined = remove_columns_by_keywords(funda_joined, ['_rolling', 'enterprise_value', 'market_cap'])
    
    # 5. Prepare for XGBoost
    print("Preparing dataset for XGBoost...")
    df_xg = prepare_data_for_xgboost(
        kpi_df=funda_joined,
        index=index_data,
        to_quantiles=True,
        treshold_percentage_missing=0.02
    )
    df_xg = df_xg[df_xg['year_month'] > '2000-01']
    
    # 6. Train Strategy
    print("Initializing XGBoost Model (Classification Mode)...")
    model = XGBoostModel(mode='classification', n_simu=30)
    
    # Split Data (Train until 2023, Test 2023+)
    split_date = '2023-01'
    df_xg['future_return'] = df_xg['future_return'] > 0.05
    train_df = df_xg[df_xg['year_month'] < split_date]
    test_df = df_xg[df_xg['year_month'] >= split_date]
    
    target_col = 'future_return'
    
    print(f"Training on data before {split_date}...")
    model.train(train_df, target_col=target_col)
    
    # 7. Predict and Evaluate
    print(f"Predicting on data from {split_date}...")
    predictions = model.predict(test_df)
    
    # Merge predictions with actual returns for evaluation
    results = predictions.merge(test_df[['ticker', 'year_month', 'monthly_return']], on=['ticker', 'year_month'])
    
    # Build Top 10 Portfolio
    print("Building Top 10 Portfolio...")
    top_10 = results.sort_values(['year_month', 'prediction'], ascending=[True, False]) \
                    .groupby('year_month').head(10)
    
    # All returns are now in standardized decimal format (0.02 for 2% gain)
    # from alpharank.utils.returns module
    portfolio_perf = top_10.groupby('year_month')['monthly_return'].mean()
    
    # Compare with Index
    # Index returns are also in decimal format from the standardized function
    index_perf = index_data.monthly_returns.set_index('year_month')['monthly_return']
    comparison = pd.DataFrame({
        'Portfolio': portfolio_perf,
        'Index': index_perf
    }).dropna()
    
    comparison['Active_Return'] = comparison['Portfolio'] - comparison['Index']
    
    print("\nPerformance Summary:")
    print(comparison.describe())
    
    # 8. Visualization and Comparison
    print("Generating Comprehensive Report...")
    
    from alpharank.utils.metrics import compare_models
    
    # Prepare models dictionary
    # Both are in standardized decimal format (0.02 for 2% gain)
    models = {
        'XGBoost_Strategy': portfolio_perf,
        'SP500': index_perf
    }
    
    # Try to load legacy returns
    legacy_path = project_root / "outputs" / "legacy_returns.csv"
    if legacy_path.exists():
        print(f"Loading legacy returns from {legacy_path}...")
        try:
            legacy_returns = pd.read_csv(legacy_path)
            # Ensure index is Period
            if 'year_month' in legacy_returns.columns:
                legacy_returns['year_month'] = pd.to_datetime(legacy_returns['year_month']).dt.to_period('M')
                legacy_returns = legacy_returns.set_index('year_month')
            
            # Find return column
            col = legacy_returns.columns[0] if len(legacy_returns.columns) > 0 else None
            if col:
                models['Legacy_Strategy'] = legacy_returns[col]
        except Exception as e:
            print(f"Failed to load legacy returns: {e}")
            
    # Run comparison
    metrics_df, cumulative, correlation, worst_periods, figures = compare_models(models, start_year=2006)
    
    print("\n--- Strategy Metrics ---")
    print(metrics_df)
    
    # Save Report
    report_path = project_root / 'backtest_report.html'
    
    # Save figures to HTML
    with open(report_path, 'w') as f:
        f.write("<html><body>")
        f.write("<h1>AlphaRank Strategy Backtest Report</h1>")
        
        f.write("<h2>Performance Metrics</h2>")
        f.write(metrics_df.to_html())
        
        f.write("<h2>Worst Periods</h2>")
        f.write(worst_periods.to_html())
        
        f.write("<h2>Plots</h2>")
        # Convert matplotlib figures to base64 for embedding
        import io
        import base64
        
        for name, fig in figures.items():
            if isinstance(fig, dict): # Handle dictionary of figures (heatmaps)
                f.write(f"<h3>{name}</h3>")
                for sub_name, sub_fig in fig.items():
                    buf = io.BytesIO()
                    sub_fig.savefig(buf, format='png')
                    buf.seek(0)
                    img_str = base64.b64encode(buf.read()).decode('utf-8')
                    f.write(f"<h4>{sub_name}</h4>")
                    f.write(f'<img src="data:image/png;base64,{img_str}" />')
                    plt.close(sub_fig)
            else:
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                img_str = base64.b64encode(buf.read()).decode('utf-8')
                f.write(f"<h3>{name}</h3>")
                f.write(f'<img src="data:image/png;base64,{img_str}" />')
                plt.close(fig)
                
        f.write("</body></html>")
        
    print(f"Report saved to {report_path}")

if __name__ == "__main__":
    main()
