

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


from src.data_processor import technical_indicators
from src.data_processor.technical_indicators import TechnicalIndicators
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
# %% Import local modules
env_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) if '__file__' in globals() else os.getcwd()
data_dir = os.path.join(env_dir, 'data')
os.chdir(data_dir)

# %% US
final_price= pd.read_parquet('US/US_Finalprice.parquet')
general= pd.read_parquet('US/US_General.parquet')
income_statement = pd.read_parquet('US/US_Income_statement.parquet')
balance_sheet= pd.read_parquet('US/US_Balance_sheet.parquet')
cash_flow = pd.read_parquet('US/US_Cash_flow.parquet')
earnings= pd.read_parquet('US/US_Earnings.parquet')

us_historical_company = pd.read_csv("US/SP500_Constituents.csv")
sp500_price = pd.read_parquet('US/SP500Price.parquet')

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
monthly_returns_vs_index = PricesDataPreprocessor.calculate_monthly_returns(df=final_price.copy(),
                                                                    column_date='date',
                                                                    column_close='close_vs_index')


from src.data_processor.technical_indicators import DecorrelatedIndicatorGenerator
import optuna
from optuna.samplers import RandomSampler
SEED_PAIRS = [(10, 100)] 
def generate_technical_indicator(df,seed_pairs,n_to_find,correlation_threshold,max_tries) : 
    
    generator = DecorrelatedIndicatorGenerator(
        daily_prices_df=df.copy(), 
        price_column='close_vs_index')
    
    generator.generate_decorrelated_ema_ratios(
        seed_pairs = seed_pairs,
        n_to_find = n_to_find,
        correlation_threshold = correlation_threshold,
        max_tries=max_tries
        ) 
    
    generator.generate_decorrelated_rsi(
        seed_params = [],
        n_to_find = n_to_find,
        correlation_threshold = correlation_threshold,
        max_tries=max_tries
        ) 
    #generator.generate_decorrelated_bollinger_bands(
    #    seed_params = [],
    #    n_to_find = n_to_find,
    #    correlation_threshold = correlation_threshold,
    #    max_tries = max_tries
    #    )
    #generator.generate_decorrelated_stochastic_oscillators(
    #    seed_params = [],
    #    n_to_find = n_to_find,
    #    correlation_threshold = correlation_threshold,
    #    max_tries = max_tries)

    technical_df = generator.get_final_indicators()
    return technical_df

technical_indicators_df = generate_technical_indicator(df=final_price.copy(),
                                                       seed_pairs=SEED_PAIRS,
                                                       n_to_find=100,
                                                       correlation_threshold=0.8,
                                                       max_tries=100)

funda_joined = fundamental.merge(technical_indicators_df, on=['ticker', 'year_month'], how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
#funda_joined = remove_columns_by_keywords(funda_joined,['_rolling','enterprise_value','market_cap'])
funda_joined = remove_columns_by_keywords(funda_joined,['_rolling','enterprise_value'])

df_xg = prepare_data_for_xgboost(kpi_df = funda_joined,
                                 index = index_data,
                                 to_quantiles = True,
                                 treshold_percentage_missing  =0.05)
df_xg = df_xg[df_xg['year_month'] > '2000-01']
df_xg['future_return_log'] = np.log(df_xg['future_return'])
df_xg = df_xg.dropna(subset=['future_return_log'])

data_quality(df_xg)
features = [col for col in df_xg.columns if col not in ['ticker', 'year_month', 'future_return','future_return_log', 'monthly_return']]

# %% Random Optuna trial (random sampler) to get one param set
params = sample_xgb_space(trial = None)

params = {'n_estimators': 5000,
 'learning_rate': 0.2,
 'max_depth': 5,
 'subsample': 0.8,
 'colsample_bytree': 0.8,
 'min_child_weight': 0.1,
 'gamma': 0.1,
 'lambda': 1,
 'alpha': 0}

final_returns  = train_fit_reg(
    df_learning=df_xg,
    params=  params,
    df_returns=monthly_return,
    target = 'future_return_log',
    month_window = 10*12,
    index=index_data,
    n_asset=10,
    loss_type = "other",
    features=features,
    alpha_over_obj=2,
    hinge_epsilon = 0.01,
    hinge_scale = 2.0,
    compute_shap_last=True)

final_returns2  = train_fit_reg(
    df_learning=df_xg,
    params=  params,
    df_returns=monthly_return,
    target = 'future_return_log',
    month_window = 10*12,
    index=index_data,
    n_asset=10,
    loss_type = "hinge",
    features=features,
    alpha_over_obj=2,
    hinge_epsilon = 0.01,
    hinge_scale = 2.0,
    compute_shap_last=True)

final_returns3  = train_fit_reg(
    df_learning=df_xg,
    params=  params,
    df_returns=monthly_return,
    target = 'future_return_log',
    month_window = 10*12,
    index=index_data,
    n_asset=10,
    loss_type = "asymmetric",
    features=features,
    alpha_over_obj=2,
    hinge_epsilon = 0.01,
    hinge_scale = 2.0,
    compute_shap_last=True)


final_returns['scores']
a = final_returns['predictions']
b = final_returns2['predictions']
c = final_returns3['predictions']
jj = a.merge(b,on = ["ticker","year_month"],suffixes=("_rmse","_hinge")).merge(c,on = ["ticker","year_month"],suffixes=("__","_asymetric"))

numeric_cols = jj.select_dtypes(include=[np.number]).columns
g = jj.groupby('year_month')[numeric_cols]
year_month_means = g.mean()
mins = g.min()
maxs = g.max()
denom = mins.abs().replace(0, np.nan)
range_ratio = (maxs - mins) / denom
year_month_stats = (
    year_month_means.add_suffix('_mean')
    .join(range_ratio.add_suffix('_range_ratio'))
)
jjj = jj[jj['year_month'] >= '2024-01']
a = evaluate_regressor(jjj['y_true'],jjj['y_pred_hinge'],alpha_over = 2,plot_lift = True,n_lift_bins=20)
b = evaluate_regressor(jjj['y_true'],jjj['y_pred'],alpha_over = 2,plot_lift = True,n_lift_bins=20)
c = evaluate_regressor(jjj['y_true'],jjj['y_pred_rmse'],alpha_over = 2,plot_lift = False,n_lift_bins=20)


all_returns = {}
all_returns['index'] = index_data.monthly_returns
all_returns['strategy'] = final_returns['aggregated_return']
all_returns['strategy_shocked'] = final_returns2['aggregated_return']
all_returns['strategy_shocked_2'] = final_returns3['aggregated_return']
#all_returns['strategy2'] = final_returns2['aggregated_return']
metrics, _, _, _, _ = compare_models(models_data=all_returns,start_year=2007)
print(metrics)

final_returns['predictions']
models = [final_returns,final_returns2]
def compare_shocked_models(models) : 
    df = pd.DataFrame()
    for i,model in enumerate(models) : 
        m = model['detailled_return'].copy()
        m= m[['year_month','ticker','score']]
        m['model'] = f'model_{i}'
        df = pd.concat([df,m], ignore_index=True)
    # Aggregate duplicates (same year_month, ticker, model) and reshape
    df_pivot = df.pivot_table(index=['year_month','ticker'],
                              columns='model',
                              values='score',
                              aggfunc='mean').reset_index()
    
    df_pivot['stock_in_common'] = df_pivot['model_0'].notna() & df_pivot['model_1'].notna()
    pi = df_pivot.groupby(['stock_in_common','year_month']).size().reset_index()
    pi = pi[pi['stock_in_common'] == True ]
    return pi

a = compare_shocked_models(models)
# %%

result = optimize_regression(
    df=df_xg,
    split_date="2025-01",
    df_returns=monthly_return,
    index=index_data,
    features=features,
    n_asset=10,
    month_window = 10*12,
    n_trials=50,
    target = 'future_return_log',
    metric_return_col = 'monthly_return_vs_index',
    penalty_weight = 0,
    min_spread = 0,
    alpha_over_obj = 0,
    loss_type =  "hinge",
    hinge_epsilon = 0.02,
    hinge_scale = 2
)

plot_optimization_history(result['study'])
plot_param_importances(result['study'])
plot_slice(result['study'])
print("Best params:", result["best_params"])


all_returns = {}
all_returns['index'] = index_data.monthly_returns
all_returns['strategy'] = result['all_returns']['aggregated_return']
#all_returns['strategy_shocked_2'] = final_returns3['aggregated_return']
#all_returns['strategy2'] = final_returns2['aggregated_return']
metrics, _, _, _, _ = compare_models(models_data=all_returns,start_year=2007)
print(metrics)

# %%
