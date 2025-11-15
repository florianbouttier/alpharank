

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
SEED_PAIRS = [(10, 100),(1, 10)] 
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

def generate_inversed_indicators(df,
                                 features : List[str]) :
    df = df.copy()
    for feature in features :
        if feature in df.columns :
            df[f"{feature}_inversed"] = 1-df[feature]
    return df
technical_indicators_df = generate_technical_indicator(df=final_price.copy(),
                                                       seed_pairs=SEED_PAIRS,
                                                       n_to_find=100,
                                                       correlation_threshold=0.8,
                                                       max_tries=100)

funda_joined = fundamental.merge(technical_indicators_df, on=['ticker', 'year_month'], how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
#funda_joined = technical_indicators_df.merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
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
#df_xg = generate_inversed_indicators(df_xg,features)
#features = [col for col in df_xg.columns if col not in ['ticker', 'year_month', 'future_return','future_return_log', 'monthly_return']]
# %%
ye = 2023

def apply_from_vector(df_learning : pd.DataFrame,
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
        mask = X[feature] >= vector[i]
        all_mask.append(mask)
    final_mask = np.all(np.array(all_mask),axis=0)
    df_selected = df_learning[final_mask][['ticker','year_month']]
    df_selected['year_month'] = df_selected['year_month'] + 1

    detailled = df_selected.merge(df_learning, on=["year_month", "ticker"], how="left")
    agg = (
        detailled
        .groupby("year_month")
        .agg(
            monthly_return=("monthly_return", "mean"),
            n_assets=("ticker", "nunique")
        )
        .reset_index()
    )  
    agg['monthly_return'] = agg['monthly_return'] - 1
    index_df = index.monthly_returns
    index_df = index_df[index_df['year_month'].isin((df_learning['year_month']+1).unique())]
    agg = agg.merge(index_df.rename(columns={'monthly_return': 'monthly_return_index'}),
                    on='year_month', how='outer')
    agg['monthly_return'] = agg['monthly_return'].fillna(0)
    agg['n_assets'] = agg['n_assets'].fillna(0)
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

def sample_space(trial: optuna.Trial,
                 features: List[str]) -> List[float]:
    if trial is None:
        study = optuna.create_study()
        trial = study.ask()
        
    vector_dict = {} 
    for feature in features :
        v = trial.suggest_float(feature, 0.0, 1.0)
        vector_dict[feature] = v
    return vector_dict

def definition_penalty(p,tresh : int = 5,scale : int = 10) :
    a = scale*(p-tresh)
    
    return (1+np.tanh(a))/2

def regression_scoring(final_returns: Dict,
                       penalty_weight: float = 1.0) -> float:
    """
    Score = mean realized monthly_return - penalty_weight * (avg overprediction gap)
    Enforce prediction spread threshold.
    """
    aggregated = final_returns.get("aggregated_return")    
    aggregated['over_perf'] = np.log(1+penalty_weight*((1+aggregated['monthly_return']) /(1+ aggregated['monthly_return_index'])-1))
    minimum_asset = aggregated['n_assets'].min()
    
    p = definition_penalty(minimum_asset,tresh = 5,scale = 1)
    if aggregated is None or aggregated.empty:
        return 0  # Penalize empty results
    else : 
        return  (np.exp(np.sum(aggregated['over_perf'])))*p
    
def _objective_builder(df_learning: pd.DataFrame,
                       index,
                       features: List[str],
                       penalty_weight: float):
    def objective(trial: optuna.Trial) -> float:
        params = sample_space(trial,features)
        out = apply_from_vector(
            df_learning = df_learning,
            vector = list(params.values()),
            index = index,
            features = features)
        return regression_scoring(final_returns = out,penalty_weight=penalty_weight)
    return objective

def generate_initial_threshold_trials(feature_names, base_threshold=0.0, high_threshold=0.95):
    """
    Génère une liste de points de départ pour Optuna ou autre optimisation.
    
    feature_names : list[str]
        Les noms des colonnes j sur lesquelles tu mets des seuils.
    base_threshold : float
        La valeur de base (par défaut 0).
    high_threshold : float
        La valeur à utiliser quand une feature est "activée".
    
    Retourne : list[dict]
        Chaque dict est un set de seuils pour un trial.
    """
    trials = []
    
    # 1. Tout à base_threshold
    trials.append({j: base_threshold for j in feature_names})
    
    # 2. Un seul j à high_threshold
    for j in feature_names:
        trial = {k: base_threshold for k in feature_names}
        trial[j] = high_threshold
        trials.append(trial)
    
    # 3. Toutes les paires (j1, j2)
    for j1, j2 in combinations(feature_names, 2):
        trial = {k: base_threshold for k in feature_names}
        trial[j1] = high_threshold
        trial[j2] = high_threshold
        trials.append(trial)
    
    return trials

def optimize_regression(df: pd.DataFrame,
                        index,
                        features: List[str],
                        n_asset: int = 10,
                        year_month_window: int = 100,
                        n_trials: int = 50,
                        seed: int = 123,
                        penalty_weight: float = 1.0) -> Dict[str, Any]:
    
    split_month = df['year_month'].unique()[300]
    n_trials = 5000
    n_startup_trials = 100
    df_learning = df.copy()
    df_learning = df_learning[df_learning['year_month'] < split_month]
    df_learning = df_learning[df_learning['year_month'] > (split_month - year_month_window)]
    df_test = df.copy()
    df_test = df_test[df_test['year_month'] >= split_month]
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed,
                                                                   n_startup_trials = n_startup_trials))
    initial_trials = generate_initial_threshold_trials(features, base_threshold=0, high_threshold=0.95)
    for trial_params in initial_trials:
        study.enqueue_trial(trial_params)
    objective = _objective_builder(df_learning,index,features,penalty_weight)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    
    plot_optimization_history(study)
    plot_param_importances(study)
    plot_slice(study)

    best_params = study.best_params
    
    test = apply_from_vector(
            df_learning = df_test,
            vector = list(best_params.values()),
            index = index,
            features = features)
    test_naif = apply_from_vector(
            df_learning = df_test,
            vector = list(initial_trials[0].values()),
            index = index,
            features = features)
    

    out_best = apply_from_vector(
            df_learning = df_learning,
            vector = list(best_params.values()),
            index = index,
            features = features)
    
    out_naif = apply_from_vector(
            df_learning = df_learning,
            vector = list(initial_trials[0].values()),
            index = index,
            features = features)

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

def get_returns(final_returns) :
    if 'aggregated_return' in final_returns :
        final_returns = final_returns['aggregated_return']
    
    dict_return = {}
    dict_return['monthly_return'] = np.prod(1+final_returns['monthly_return'])
    dict_return['monthly_return_index'] = np.prod(1+final_returns['monthly_return_index'])
    dict_return['over_perf'] = dict_return['monthly_return'] / dict_return['monthly_return_index']-1
    return dict_return
# %%
df = df_xg.copy()
index = index_data
n_asset = 10
year_month_window = 100
n_trials = 50
seed = 123
penalty_weight = 1.0
# %%