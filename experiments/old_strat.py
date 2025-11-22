# %%
import copy
import os
import numpy as np
from numpy.random import seed
import pandas as pd
from datetime import timedelta
import scipy.stats as stats
from scipy import special
import re
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple,Optional,Callable
import xgboost as xgb
import shap
from tqdm import tqdm
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score,roc_auc_score,mean_squared_log_error,mean_squared_log_error,mean_squared_error,mean_absolute_error
from typing import Dict, Any, List
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway
from sklearn.metrics import mutual_info_score
from pathlib import Path
import random
os.chdir(str(Path(__file__).parent.parent))
%load_ext autoreload
%autoreload 2


from itertools import combinations
from src.probalisor.strategy_analysis import compare_models,compare_score_and_perf
from src.data_processor import technical_indicators
from src.data_processor import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor, TechnicalIndicators
from src.data_processor.utils import *
from src.data_processor.datasets import *
from src.data_processor.data_auditor import *
from src.probalisor.search_spaces import sample_xgb_space
from src.probalisor.models import *
from src.probalisor.prob_ranker import *
from src.probalisor.strategy_analysis import compare_models,compare_score_and_perf
from src.regressor_ranker.reg_ranker import *
from src.regressor_ranker.reg_metrics import *
from src.regressor_ranker.reg_models import *
from src.data_processor.datasets import make_rank_dataset
from src.trash.old_strategy import StrategyLearner
from matplotlib import patches as mpatches
from src.visualization.plotting import StockComparisonPlotter
# %% Import local modules
env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
data_dir = os.path.join(env_dir, 'data')
os.chdir(data_dir)

# %% US
final_price= pd.read_parquet('US_Finalprice.parquet')
general= pd.read_parquet('US/US_General.parquet')
income_statement = pd.read_parquet('US/US_Income_statement.parquet')
balance_sheet= pd.read_parquet('US/US_Balance_sheet.parquet')
cash_flow = pd.read_parquet('US/US_Cash_flow.parquet')
earnings= pd.read_parquet('US/US_Earnings.parquet')

us_historical_company = pd.read_csv("US/SP500_Constituents.csv")
sp500_price = pd.read_parquet('US/SP500Price.parquet')

index_data = IndexDataManager(
    daily_prices_df=sp500_price.copy(),
    components_df=us_historical_company.copy()
)
print(max(final_price['date']))
print(max(sp500_price['date']))
print(max(us_historical_company['Date']))

ticker_to_exclude = ['SII.US','CBE.US','TIE.US']
for ticker in ticker_to_exclude : 
    final_price = final_price[final_price['ticker'] != ticker]
    general = general[general['ticker'] != ticker]
    income_statement = income_statement[income_statement['ticker'] != ticker]           
    balance_sheet = balance_sheet[balance_sheet['ticker'] != ticker]
    cash_flow = cash_flow[cash_flow['ticker'] != ticker]
    earnings = earnings[earnings['ticker'] != ticker]


# %% Retreatement  
final_price['year_month'] = pd.to_datetime(final_price['date']).dt.to_period('M')  
us_historical_company['ticker'] = us_historical_company['Ticker'].apply(lambda x: re.sub(r'\.', '-', x) if isinstance(x, str) else x)
us_historical_company['ticker'] = us_historical_company['ticker'] + '.US'
us_historical_company['year_month'] = pd.to_datetime(us_historical_company['Date']).dt.to_period('M')

index_data = IndexDataManager(
    daily_prices_df=sp500_price.copy(),
    components_df=us_historical_company.copy()
)

monthly_return = PricesDataPreprocessor().calculate_monthly_returns(df = final_price.copy(),column_close = 'adjusted_close',column_date = 'date')
fundamental = FundamentalProcessor().calculate_all_ratios(balance_sheet=balance_sheet.copy(),
                                                         income_statement=income_statement.copy(),
                                                         cash_flow=cash_flow.copy(),
                                                         earnings=earnings.copy(),
                                                         monthly_return=monthly_return.copy())

fundamental = fundamental.drop(columns=['date','last_close','monthly_return','quarter_end', 'filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning'], errors='ignore')


final_price = PricesDataPreprocessor.prices_vs_index(index=index_data.daily_prices.copy(), 
                                               prices=final_price.copy(),
                                               column_close_index='adjusted_close',
                                               column_close_prices='adjusted_close')
final_price = PricesDataPreprocessor().compute_dr(df = final_price.copy(),
                                                  column_close = 'adjusted_close',
                                                  column_date = 'date')
monthly_returns_vs_index = PricesDataPreprocessor.calculate_monthly_returns(df=final_price.copy(),
                                                                    column_date='date',
                                                                    column_close='close_vs_index')

stocks_selections = FundamentalProcessor().calculate_pe_ratios(balance = balance_sheet,
                                                             earnings = earnings,
                                                             cashflow=cash_flow,
                                                             income=income_statement,
                                                             earning_choice= 'netincome_rolling',
                                                             monthly_return=monthly_return,
                                                             list_date_to_maximise = ['filing_date_income', 'filing_date_balance'])


all_ratios = FundamentalProcessor().calculate_all_ratios(balance_sheet=balance_sheet.copy(),
                                                         income_statement=income_statement.copy(),
                                                         cash_flow=cash_flow.copy(),
                                                         earnings=earnings.copy(),
                                                         monthly_return=monthly_return.copy())
stocks_selections = (stocks_selections[(stocks_selections['pe']<100) & (stocks_selections['pe']>0)]
                    .dropna(subset = ['pe', 'market_cap'])
                    .merge(us_historical_company[['year_month','ticker']],
                            how = "inner",
                            left_on = ['ticker','year_month'],
                            right_on = ['ticker','year_month']))

# %%
first_date = pd.Period("2006-01-01", freq='M')

optuna_output_1 = StrategyLearner.learning_process_optuna_full(
    prices = final_price.copy(),
    index = index_data,
    first_date = first_date,
    stocks_filter = stocks_selections.copy(),
    sector = general[['ticker','Sector']].copy(),
    func_movingaverage = TechnicalIndicators.ema,
    n_trials = 20, 
    alpha = 2,
    temp = 10*12, 
    mode = "mean",
    seed = 42)

# %%
optuna_output_2 = StrategyLearner.learning_process_optuna_full(
    prices = final_price.copy(),
    index = index_data,
    first_date = first_date,
    stocks_filter = stocks_selections.copy(),
    sector = general[['ticker','Sector']].copy(),
    func_movingaverage = TechnicalIndicators.ema,
    n_trials = 20, 
    alpha = 2,
    temp = 10*12, 
    mode = "mean",
    seed = 43)


optuna_output_3 = StrategyLearner.learning_process_optuna_full(
    prices = final_price.copy(),
    index = index_data,
    first_date = first_date,
    stocks_filter = stocks_selections.copy(),
    sector = general[['ticker','Sector']].copy(),
    func_movingaverage = TechnicalIndicators.ema,
    n_trials = 50, 
    alpha = 2,
    temp = 5*12, 
    mode = "mean",
    seed = 103,
    n_jobs=1)

optuna_output_4 = StrategyLearner.learning_process_optuna_full(
    prices = final_price.copy(),
    index = index_data,
    first_date = first_date,
    stocks_filter = stocks_selections.copy(),
    sector = general[['ticker','Sector']].copy(),
    func_movingaverage = TechnicalIndicators.ema,
    n_trials = 50, 
    alpha = 3,
    temp = 10*12, 
    mode = "mean",
    seed = 103,
    n_jobs=1)


    
technical_1 = StrategyLearner.learning_process_technical(
    prices = final_price.copy(), 
    index = index_data,
    stocks_filter = stocks_selections.copy(),
    sector = general[['ticker','Sector']].copy(),
    func_movingaverage = TechnicalIndicators.ema,
    liste_nlong = [50+80*i for i in range(5)], 
    liste_nshort = [1]+[5+10*i for i in range(5)], 
    liste_nasset = [20], 
    max_persector = 1, 
    final_nmaxasset = 10, 
    list_alpha = [round(1+0.2*i,1) for i in range(10)], 
    list_temp = [round(12*(4 + 2*i)) for i in range(5)], 
    mode = "mean",
    param_temp_lvl2 = 7*12, 
    param_alpha_lvl2 = 2)
# %%

models = {
    'optuna_1': (optuna_output_1['aggregated'] ),
    'optuna_2': (optuna_output_2['aggregated'] ),
    'technical_1': (technical_1['aggregated'] ),
    'SP500': (index_data.monthly_returns )
}


# Exécuter l'analyse
metrics, cumulative, correlation, worst_periods, figures = compare_models(models, start_year=2006)
print(metrics)
index_data.monthly_returns


# %% functions
def get_last_portfolio(list_optuna_output):
    import matplotlib.pyplot as plt

    all_portfolio = None
    for optuna_output in list_optuna_output:
        data = optuna_output['detailled']
        last_year_month = data['year_month'].max()
        last_portfolio = data[data['year_month'] == last_year_month]
        if all_portfolio is not None:
            all_portfolio = pd.concat([all_portfolio, last_portfolio], ignore_index=True)
        else:
            all_portfolio = last_portfolio.copy()

    # Summary per ticker and sector (occurrences across last portfolios)
    summary_portfolio = (
        all_portfolio.groupby(['ticker'], as_index=False)
        .agg(freq=('year_month', 'count'))
    )
    summary_portfolio = summary_portfolio.merge(
        all_portfolio[['ticker', 'Sector']].drop_duplicates(subset=['ticker']),
        how='left',
        on='ticker'
    )

    # Order by sector then frequency to group tickers of same sector together
    summary_portfolio = summary_portfolio.sort_values(
        by=['Sector', 'freq', 'ticker'], ascending=[True, False, True]
    ).reset_index(drop=True)

    # Colors: one color per sector, shared across its tickers
    sectors = summary_portfolio['Sector'].fillna('Unknown').unique().tolist()
    cmap = plt.cm.get_cmap('tab20', len(sectors))
    sector_colors = {s: cmap(i) for i, s in enumerate(sectors)}
    colors = summary_portfolio['Sector'].fillna('Unknown').map(sector_colors).tolist()

    # Pie data
    sizes = summary_portfolio['freq'].values
    labels = summary_portfolio['ticker'].values

    fig, ax = plt.subplots(figsize=(10, 10))
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct='%1.1f%%',
        startangle=90,
        counterclock=False,
        labeldistance=1.05,
        pctdistance=0.75,
        wedgeprops={'edgecolor': 'white', 'linewidth': 1.2}
    )
    ax.axis('equal')
    ax.set_title("Composition du dernier portefeuille (couleur = Secteur)")

    # Legend mapping sector -> color
    handles = [mpatches.Patch(color=sector_colors[s], label=s) for s in sectors]
    ax.legend(handles=handles, title='Secteur', loc='center left', bbox_to_anchor=(1.0, 0.5))

    plt.tight_layout()
    plt.show()
    return all_portfolio, summary_portfolio


A = get_last_portfolio([optuna_output_1])
# %%
tickers = A[0]['ticker'].nunique()
tickers_list = sorted(A[0]['ticker'].unique().tolist())
kpis = ['pebitda','gross_margin', 'netmargin', 'return_on_equity']
plotter = StockComparisonPlotter(all_ratios.copy())

# Create HTML report (normalized, smoothed, with IQR outlier clipping)
report_html = plotter.make_report(
    tickers=tickers_list,
    kpis=kpis,
    normalize=True,
    smooth_span=3,
    iqr_multiplier=2.5,
    include_scatter_matrix=True
)

# Save to disk and optionally display in notebook
output_dir = Path(env_dir) / "outputs"
output_dir.mkdir(parents=True, exist_ok=True)
out_file = output_dir / "kpi_report_last_portfolio.html"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(report_html)
print(f"Saved KPI report to: {out_file}")

try:
    from IPython.display import HTML, display
    display(HTML(report_html))
except Exception:
    pass
# %%
