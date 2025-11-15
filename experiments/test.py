

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


# %%
ye = 2023

def selection_from_vector(df_learning : pd.DataFrame,
                          vector : List[float],
                          index: IndexDataManager,
                          features : List[str]) :
    df_learning = df_learning.copy()
    if len(vector) != len(features) :
        raise ValueError("Vector length must match features length")
    if max(vector) > 1 or min(vector) < 0 :
        raise ValueError("Vector values must be in [0,1]")
    X = df_learning[features].copy()
    all_mask = []
    for i,feature in enumerate(features) :
        mask = X[feature] > vector[i]
        all_mask.append(mask)
    final_mask = np.all(np.array(all_mask),axis=0)
    df_selected = df_learning[final_mask][['ticker','year_month']]
    df_selected['year_month'] = df_selected['year_month'] + 1

    detailled = df_selected.merge(df_learning, on=["year_month", "ticker"], how="left")
    agg = (detailled.groupby("year_month")["monthly_return"].mean()).reset_index()
    agg['monthly_return'] = agg['monthly_return'] - 1
    agg = agg.merge(index.monthly_returns.rename(columns={'monthly_return': 'monthly_return_index'}),
                    on='year_month', how='left')
    return {"detailled": detailled,
            "aggregated_return": agg}
 
def filter_on_period(df: pd.DataFrame,
                     year_split: Union[int, str]) :
    if isinstance(year_split, int):
        if 'year' not in df.columns:
            df['year'] = df['year_month'].dt.year
        df = df[df['year'] == year_split]
    else : 
        df = df[df['year_month'] == year_split]
    return df

vector = [0.2 for _ in features]

A = selection_from_vector(df_learning = filter_on_period(df_xg,year_split=2023),
                      vector = vector,
                      index = index_data,
                      features = features)

        
def definition_penalty(p) :
    if p > 1 : p = 2-p
    return (1+np.tanh(10*(p-0.5)))/2

def regression_scoring(final_returns: Dict,
                       penalty_weight: float = 1.0) -> float:
    """
    Score = mean realized monthly_return - penalty_weight * (avg overprediction gap)
    Enforce prediction spread threshold.
    """
    preds_df = final_returns.get("scores" )
    p = definition_penalty(np.mean(preds_df['std_ratio']))
    p = 1-abs(p-1)
    ag = final_returns.get("aggregated_return")
    ag['over_perf'] = np.log(1+penalty_weight*((1+ag['monthly_return']) /(1+ ag['monthly_return_index'])-1))
    return (np.exp(np.sum(ag['over_perf'])))*p

def _objective_builder(df_learning: pd.DataFrame,
                       df_returns: pd.DataFrame,
                       index,
                       features: List[str],
                       n_asset: int,
                       target: str,
                       year_window: int,
                       penalty_weight: float):
    def objective(trial: optuna.Trial) -> float:
        params = sample_xgb_space(trial)
        out = train_fit_on_year_reg(
            df_learning=df_learning,
            params=params,
            df_returns=df_returns,
            year_split = ye,
            year_window = year_window,
            index=index,
            n_asset=n_asset,
            features=features,
            target=target,
            compute_shap=False)
        return regression_scoring(
            out,
            penalty_weight=penalty_weight
        )
    return objective

def optimize_regression(df: pd.DataFrame,
                        df_returns: pd.DataFrame,
                        index,
                        features: List[str],
                        n_asset: int = 10,
                        year_window: int = 10,
                        target: str = "future_return",
                        n_trials: int = 50,
                        seed: int = 123,
                        penalty_weight: float = 1.0) -> Dict[str, Any]:
    
    # This code snippet is performing hyperparameter optimization using Optuna for a regression task.
    # Here's a breakdown of what each step is doing:
    df_learning = df.copy()
    
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    objective = _objective_builder(df_learning,
                                   df_returns,
                                   index,
                                   features,
                                   n_asset,
                                   target,
                                   year_window,
                                   penalty_weight)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_params = study.best_params
    
    all_out_val = train_fit_on_year_reg(
        df_learning=df,
        params=best_params,
        df_returns=df_returns,
        year_split = ye,
        year_window = year_window,
        features=features,
        index=index_data,
        n_asset=n_asset,
        target=target,
        compute_shap=True,
        shap_run_plots = True)
    
    all_out_test = train_fit_on_year_reg(
        df_learning=df,
        params=best_params,
        df_returns=df_returns,
        year_split = ye+1,
        year_window = year_window,
        features=features,
        index=index_data,
        n_asset=n_asset,
        target=target,
        compute_shap=True,
        shap_run_plots = True)

    return {
        "best_value": float(study.best_value),
        "best_params": best_params,
        "study": study,
        "all_returns_val": all_out_val,
        "all_returns_test": all_out_test,
        "config": {
            "penalty_weight": penalty_weight,
            "n_asset": n_asset
        }
    }



warnings.filterwarnings("ignore",message="Calling float on a single element Series is deprecated")
warnings.simplefilter("ignore", category=FutureWarning)

result = optimize_regression(
    df=df_xg,
    df_returns=monthly_return,
    index=index_data,
    features=features,
    n_asset=20,
    year_window=  10,
    n_trials=1000,
    target = 'future_return_log',
    penalty_weight = 2
)

# %%
plot_optimization_history(result['study'])
plot_param_importances(result['study'])
plot_slice(result['study'])
print("Best params:", result["best_params"])
ret = result['all_returns']['aggregated_return']
ret['monthly_return'] = 1+ ret['monthly_return']
ret['monthly_return_index'] = 1+ ret['monthly_return_index']
ret['cum_return'] = ret['monthly_return'].cumprod()
ret['cum_return_index'] = ret['monthly_return_index'].cumprod() 
plt.figure(figsize=(10,5))
plt.plot(ret['year_month'].astype(str),ret['cum_return'], label='Strategy')
plt.plot(ret['year_month'].astype(str),ret['cum_return_index'], label='Index')
plt.xticks(rotation=90)
plt.legend()
plt.title('Cumulative Return vs Index')
plt.show()

all_returns = {}
all_returns['index'] = index_data.monthly_returns
all_returns['strategy'] = result['all_returns']['aggregated_return']
#all_returns['strategy_shocked_2'] = final_returns3['aggregated_return']
#all_returns['strategy2'] = final_returns2['aggregated_return']
metrics, _, _, _, _ = compare_models(models_data=all_returns,start_year=2007)
print(metrics)

# %%
A = train_fit_on_year_reg(
    df_learning=df_xg,
    params=result['best_params'],
    df_returns=monthly_return,
    year_split = 2023,
    year_window = 10,
    index=index_data,
    n_asset=10,
    features=features,
    target='future_return_log',
    alpha_over_obj=2.0,
    loss_type='asymetric',
    hinge_epsilon=0.02,
    hinge_scale=1.0,
    compute_shap=True
)


df_learning = df_xg[df_xg['year_month'] > '2023-01']
x = [0+0.002*i for i in range(5000)] 
for features in df_learning.columns : 
    if df_learning[features].dtype != 'object' and features not in ['future_return','future_return_log','monthly_return','year_month','ticker'] :


        liste = []
    
        for t in x:
            df_apps = df_learning[df_learning[features]> t]
            liste.append(np.mean(df_apps['future_return_log']))


        plt.plot(x,liste)
        plt.title(f"{features}")
        plt.show()
    else : 
        print(f"{features} non numerique")
        


