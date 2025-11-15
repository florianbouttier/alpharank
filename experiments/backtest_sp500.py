# %%
import copy
import os
import numpy as np
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
from src.data_processor.datasets import make_rank_dataset
from src.data_processor.utils import *
from src.data_processor.datasets import *
from src.data_processor.data_auditor import *
from src.probalisor.search_spaces import sample_xgb_space
from src.probalisor.models import *
from src.probalisor.prob_ranker import *
from src.probalisor.strategy_analysis import compare_models,compare_score_and_perf
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
                                                       correlation_threshold=0.98,
                                                       max_tries=100)

funda_joined = fundamental.merge(technical_indicators_df, on=['ticker', 'year_month'], how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
funda_joined = remove_columns_by_keywords(funda_joined,['_rolling','enterprise_value','market_cap'])

df_xg = prepare_data_for_xgboost(kpi_df = funda_joined,
                                 index = index_data,
                                 to_quantiles = True,
                                 treshold_percentage_missing  =0.02)
df_xg = df_xg[df_xg['year_month'] > '2000-01']

data_quality(df_xg)
features = [col for col in df_xg.columns if col not in ['ticker', 'year_month', 'future_return', 'monthly_return']]

# %%

A = optimize(
    df = df_xg,
    split_date ='2025-01',
    df_returns =monthly_return,
    index = index_data,
    features = features,
    n_asset =  10,
    target =  'future_return',
    n_trials  = 10,
    seed =  123,
    metric_name = "precision@50",
    penalty_weight = 1,
    min_spread = 0.1
)

plot_optimization_history(A['study'])
plot_param_importances(A['study'])
plot_slice(A['study'])

all_return = A['all_returns']
all_return['scores']

def plt_validate_test(scores,split_date) : 

    def validate_test(scores: pd.DataFrame, split_date: str) -> pd.DataFrame:
        scores = scores.copy()
        metric_cols = [c for c in scores.columns if c not in ['year_month'] and not c.startswith('train_')]
        val_mask = scores['year_month'] < split_date
        test_mask = scores['year_month'] >= split_date

        val_means = scores.loc[val_mask, metric_cols].mean().reset_index()
        val_means.columns = ['metric', 'validation']
        test_means = scores.loc[test_mask, metric_cols].mean().reset_index()
        test_means.columns = ['metric', 'test']

        merged = pd.merge(val_means, test_means, on='metric')
        return merged

    scores_df = all_return['scores'] if 'all_return' in globals() else final_returns['scores']
    metrics_vt = validate_test(scores_df, split_date='2019-01')

    plot_df = metrics_vt.melt(id_vars='metric', value_vars=['validation', 'test'],
                              var_name='dataset', value_name='value')

    import plotly.express as px
    fig = px.bar(
        plot_df,
        x='metric',
        y='value',
        color='dataset',
        barmode='group',
        hover_data={'metric': True, 'dataset': True, 'value': ':.4f'}
    )
    fig.update_layout(
        title='Scores Validation vs Test (Interactif)',
        xaxis_title='Métrique',
        yaxis_title='Valeur',
        legend_title='Jeu',
        bargap=0.25
    )
    fig.show()

    metrics_vt


plt_validate_test(all_return['scores'],'2019-01')


all_returns = {}
all_returns['index'] = index_data.monthly_returns
all_returns['strategy'] = all_return['aggregated_return']
metrics, _, _, _, _ = compare_models(models_data=all_returns,start_year=2007)
print(metrics)

#%%----------------------------------------------------------------------------------------------------------------------------------------------------------------

params = A['best_params']
params_shocked = perturb_params(params,std_pct=0.05)
final_returns = train_fit(df_learning = df_xg,
                       params = params,
                       df_returns= monthly_return,
                       index = index_data,
                       n_asset = 20,
                       features = features)

final_returns2 = train_fit(df_learning = df_xg,
                       params = params_shocked,
                       df_returns= monthly_return,
                       index = index_data,
                       n_asset = 20,
                       features = features)

final_returns3 = train_fit(df_learning = df_xg,
                       params = perturb_params(params,std_pct=0.08),
                       df_returns= monthly_return,
                       index = index_data,
                       n_asset = 20,
                       features = features)

all_returns = {}
all_returns['index'] = index_data.monthly_returns
all_returns['strategy'] = final_returns['aggregated_return']
all_returns['strategy_shocked'] = final_returns2['aggregated_return']
all_returns['strategy_shocked_2'] = final_returns3['aggregated_return']
#all_returns['strategy2'] = final_returns2['aggregated_return']
metrics, _, _, _, _ = compare_models(models_data=all_returns,start_year=2007)
print(metrics)

sco = final_returns2['scores']

final_returns['aggregated_return']

# %%
final_returns['aggregated_return']['surperf'] = (1+final_returns['aggregated_return']['monthly_return']) / (1+final_returns['aggregated_return']['monthly_return_index']) -1
a = compare_score_and_perf(df_scores = final_returns['aggregated_return'],
                           df_perf = final_returns['scores'],
                           score_col = 'precision@50',
                           perf_col = 'surperf')


year = 2024
df_train = df_xg[df_xg['year_month'] < f'{year}-01']
df_test = df_xg[df_xg['year_month'] >= f'{year}-01']
features  = [col for col in df_xg.columns if col not in ['ticker', 'year_month', 'future_return', 'monthly_return']]



plot_kpis(df = fundamental.copy(),
          ticker = 'CHK.US',
          col_columns = 'netmargin',
          cols_lines = ['operatingincome_rolling','incomebeforetax_rolling','netincome_rolling','ebit_rolling','freecashflow_rolling','ebitda_rolling'],
          date_window = ('2015-01','2026-06'))

"""
def generate_monthly_technical_indicators_from_daily(daily_prices_df: pd.DataFrame,price_column: str,moving_average_pairs: List[Tuple[int, int]]) -> pd.DataFrame:
    
    Calcule des indicateurs techniques sur des données journalières et les
    échantillonne en mensuel (en prenant la dernière valeur du mois).

    Args:
        daily_prices_df (pd.DataFrame): DataFrame avec les prix JOURNALIERS.
                                        Doit contenir 'ticker', 'date', et la colonne de prix.
        price_column (str): Le nom de la colonne de prix (ex: 'close').
        moving_average_pairs (List[Tuple[int, int]]): Liste de paires (n_short, n_long).

    Returns:
        pd.DataFrame: Un DataFrame MENSUEL avec les indicateurs techniques par ticker.
    """
    #print("Calcul des indicateurs sur données journalières...")
    df = daily_prices_df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])

    for n_short, n_long in moving_average_pairs:
        grouped = df.groupby('ticker')[price_column]
        ema_short = grouped.transform(lambda x: TechnicalIndicators.ema(x, n=n_short))
        ema_long = grouped.transform(lambda x: TechnicalIndicators.ema(x, n=n_long))
        
        df[f'ema_ratio_{n_short}_{n_long}'] = ema_short / ema_long
        #df[f'macd_{n_short}_{n_long}'] = ema_short - ema_long

    # Échantillonnage en mensuel
    #print("Échantillonnage des indicateurs en mensuel...")
    df['year_month'] = df['date'].dt.to_period('M')
    
    # On garde la dernière valeur de chaque mois pour chaque ticker
    monthly_technicals = df.groupby(['ticker', 'year_month']).last().reset_index()
    
    # Garder uniquement les colonnes utiles
    cols_to_keep = ['ticker', 'year_month'] + [col for col in monthly_technicals.columns if 'ema_ratio' in col or 'macd' in col]
    
    return monthly_technicals[cols_to_keep]

def convert_to_quantiles(df: pd.DataFrame,exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Transforms the numerical feature columns of a DataFrame into their
    cross-sectional quantiles for each time period.

    Args:
        df (pd.DataFrame): The input DataFrame, must contain a 'year_month' column.
        exclude_cols (Optional[List[str]], optional): A list of numerical columns
            to exclude from the transformation. Defaults to None.

    Returns:
        pd.DataFrame: A new DataFrame with numerical features converted to quantiles (0.0 to 1.0).
    """
    print("Starting quantile transformation...")
    df_quantiles = df.copy()

    # Define a default list of columns that should never be transformed
    default_exclusions = ['year_month', 'monthly_return', 'future_return']
    if exclude_cols:
        default_exclusions.extend(exclude_cols)

    # Identify all numerical columns to be transformed
    cols_to_transform = df_quantiles.select_dtypes(include=np.number).columns.tolist()
    cols_to_transform = [col for col in cols_to_transform if col not in default_exclusions]

    print(f"Transforming {len(cols_to_transform)} numerical columns into quantiles...")

    # Group by each month, then apply the rank-to-quantile transformation on each column
    # .rank(pct=True) calculates the percentile rank, which is exactly the quantile.
    for col in cols_to_transform:
        df_quantiles[col] = df_quantiles.groupby('year_month')[col].transform(lambda x: x.rank(pct=True))

    print("Quantile transformation complete.")
    return df_quantiles

def generate_decorrelated_indicators_iteratively(daily_prices_df: pd.DataFrame,
                                                 price_column: str,
                                                 seed_pairs: List[Tuple[int, int]],
                                                 n_final_indicators: int,
                                                 correlation_threshold: float,
                                                 max_tries: int = 100) -> pd.DataFrame:
        """
        Génère un ensemble d'indicateurs techniques peu corrélés en utilisant une
        boucle itérative intelligente avec une barre de progression.
        """
        if not seed_pairs:
            raise ValueError("La liste 'seed_pairs' ne peut pas être vide.")

        # --- Étape 1: Initialiser avec les indicateurs de base ("seeds") ---
        selected_pairs = list(set(seed_pairs))
        final_indicators_df = generate_monthly_technical_indicators_from_daily(
            daily_prices_df=daily_prices_df,
            price_column=price_column,
            moving_average_pairs=selected_pairs
        )

        # --- Étape 2: Boucle pour trouver les indicateurs restants avec tqdm ---
        # On utilise une boucle for sur un range avec une description tqdm
        with tqdm(total=n_final_indicators, initial=len(selected_pairs), desc="Finding Decorrelated Indicators") as pbar:
            for _ in range(max_tries):
                if len(selected_pairs) >= n_final_indicators:
                    pbar.update(n_final_indicators - pbar.n) # S'assurer que la barre va à 100%
                    break

                # Générer une nouvelle paire candidate
                n_short = random.randint(2, 50)
                n_long = random.randint(n_short + 20, 250)
                new_pair = (n_short, n_long)

                if new_pair in selected_pairs:
                    continue

                new_indicator_df = generate_monthly_technical_indicators_from_daily(
                    daily_prices_df=daily_prices_df, price_column=price_column, moving_average_pairs=[new_pair]
                )
                new_indicator_name = f'ema_ratio_{n_short}_{n_long}'
                
                temp_df = pd.merge(final_indicators_df, new_indicator_df, on=['ticker', 'year_month'])
                
                selected_names = [f'ema_ratio_{s}_{l}' for s, l in selected_pairs]
                corr_matrix = temp_df[selected_names + [new_indicator_name]].corr().abs()
                max_corr_with_selected = corr_matrix.loc[selected_names, new_indicator_name].max()

                if max_corr_with_selected < correlation_threshold :
                    print(_," & ", max_corr_with_selected)
                    final_indicators_df = temp_df
                    selected_pairs.append(new_pair)
                    print(_)
                    
                    pbar.update(1) # Mettre à jour la barre de progression

        # --- Étape 3: Affichage final ---
        final_indicator_names = [f'ema_ratio_{s}_{l}' for s, l in selected_pairs]
        print("\n--- Indicateurs finaux sélectionnés ---")
        print(final_indicator_names)
        print("-" * 35)

        return final_indicators_df
    # ==============================================================================
    # MODULE 1: FONCTIONS DE MÉTRIQUES ET STRATÉGIE
# ==============================================================================
def link_function(y : pd.Series,model_type : str =  'classification') ->  pd.Series : 
    
    if np.min(y) < 0 : y = y+1
    if model_type == 'classification':
        y_linked = (y > 1).astype(int) # Assumes target is a price relative
    else: # regression
        y_linked = np.log(y.clip(lower=1e-8))
    return y_linked

def calculate_custom_score(portfolio_returns: pd.Series, benchmark_returns: pd.Series, alpha: float = 5.0) -> float:
    """Calculates your custom utility score vs. a benchmark."""
    
     # --- CORRECTION ICI ---
    # Si le benchmark est un DataFrame, on sélectionne la colonne 'monthly_return'.
    if isinstance(benchmark_returns, pd.DataFrame) :
        if 'monthly_return' in benchmark_returns.columns:
            benchmark_returns = benchmark_returns['monthly_return']
        else:
            # Fallback if the column name is different
            benchmark_returns = benchmark_returns.iloc[:, 0]
    if isinstance(portfolio_returns, pd.DataFrame) :
        if 'monthly_return' in portfolio_returns.columns:
            portfolio_returns = portfolio_returns['monthly_return']
        else:
            # Fallback if the column name is different
            portfolio_returns = portfolio_returns.iloc[:, 0]
    # --- FIN DE LA CORRECTION ---
    
    returns = pd.DataFrame({'portfolio': portfolio_returns, 'benchmark': benchmark_returns}).dropna()
    if returns['portfolio'].prod() < 0.0001 : returns['portfolio'] += 1
    if returns['benchmark'].prod() < 0.0001 : returns['benchmark'] += 1
    if returns.empty: return -999.0
    excess_return = returns['portfolio'] / returns['benchmark']-1
    utility_arg = 1 + alpha * excess_return
    utility_arg[utility_arg <= 0] = 1e-6
    return np.log(utility_arg).sum()

def wgenerate_portfolio_returns(all_predictions_df: pd.DataFrame, actual_returns_df: pd.DataFrame, n: int = 10,mode :str =  "exact") -> Tuple[pd.DataFrame,pd.DataFrame] :
    """Simulates a top N long-only strategy and calculates its monthly returns."""
    predictions,actual_returns_df = all_predictions_df.copy(),actual_returns_df.copy()
    predictions['year_month'] = pd.to_datetime(predictions['year_month'].astype(str))
    
    if mode == "exact":
        #In this case, top n of prediction across all simu. Insure to have only n assset
        top_n_portfolios = predictions.sort_values('prediction', ascending=False).groupby(['year_month']).head(n)
    else : 
        top_n_portfolios = predictions.sort_values('prediction', ascending=False).groupby(['year_month','simu']).head(n)[['ticker','year_month']].drop_duplicates()
    
    #top_n_portfolios = predictions.sort_values('prediction', ascending=False).groupby(['year_month','simu']).head(n)
    
    
    top_n_portfolios['holding_month'] = (top_n_portfolios['year_month'] + pd.DateOffset(months=1)).dt.to_period('M')
    returns_to_join = actual_returns_df[['ticker', 'year_month', 'monthly_return']].copy()
    strategy_realized_returns = pd.merge(top_n_portfolios, returns_to_join, left_on=['ticker', 'holding_month'], right_on=['ticker', 'year_month'], how='left')
    strategy_performance = strategy_realized_returns.groupby('holding_month').agg({
        'monthly_return': 'mean',
        'ticker': 'count'
    }).rename(columns={'ticker': 'n_assets'})
    strategy_performance.index.name = 'year_month'
    strategy_performance['monthly_return'] = strategy_performance['monthly_return']-1
    
    return strategy_performance.reset_index(),top_n_portfolios

def calculate_discrimination_penalty(predictions: np.ndarray,threshold: float = 0.00001,max_penalty: float = 1000,steepness: float = 1/0.00001) -> float:
    """
    Calculates a penalty for models with low prediction variance using an inverted sigmoid function.

    Args:
        predictions (np.ndarray): The array of model predictions.
        threshold (float): The standard deviation below which the penalty starts to apply.
        max_penalty (float): The maximum penalty to apply when std dev is very low.
        steepness (float): Controls how sharp the transition is around the threshold.

    Returns:
        float: The calculated penalty value (between 0 and max_penalty).
    """
    if predictions is None or len(predictions) < 2:
        return max_penalty

    std_dev = np.std(predictions)
    
    # Inverted sigmoid function: penalty is high when std_dev is low, and near zero when high.
    penalty = min(np.exp(-steepness * (std_dev - threshold)),max_penalty)
    
    return penalty

def generate_shocked_hyperparameters(hyperparameters : dict,std : float = 0.1) : 
    shocked_hyperparameters = copy.deepcopy(hyperparameters)
    for keys in  shocked_hyperparameters.keys() : 
        if keys  in ['n_estimators','max_depth'] : 
            value = shocked_hyperparameters[keys]
            alea = 2* value * std* (2*random.random()-1)
            shocked_hyperparameters[keys] = int(value + alea)
            
        
        if keys  in ['learning_rate', 'subsample', 'colsample_bytree', 'gamma', 'reg_lambda'] : 
            value = shocked_hyperparameters[keys]
            alea =  value * std* (2*random.random()-1)
            shocked_hyperparameters[keys] = max(value + alea,0)
        else : pass
    return shocked_hyperparameters
    
# ==============================================================================
# MODULE 2: OPTIMISATION DES HYPERPARAMÈTRES
# ==============================================================================
def objective_for_period(trial: optuna.Trial, train_df: pd.DataFrame, validation_df: pd.DataFrame, hparam_space: Dict, model_type: str, metric_function: Callable, metric_kwargs: Dict,random_state = None) -> float:
    
    params = {}
    for name, (ptype, low, high) in hparam_space.items():
        if ptype == "int": params[name] = trial.suggest_int(name, low, high)
        else: params[name] = trial.suggest_float(name, low, high, log=(name == 'learning_rate'))
    
    params.update({'n_jobs': -1, 'enable_categorical': True, 'early_stopping_rounds': 1500,'random_state': random_state})
    target_column, identifier_cols = 'future_return', ['ticker', 'year_month']
    categorical_columns = [c for c in train_df.select_dtypes(include=['object', 'category']).columns if c not in identifier_cols]
    numeric_feature_cols = [c for c in train_df.select_dtypes(include=np.number).columns if c not in [target_column, 'monthly_return', 'future_return']]
    feature_columns = numeric_feature_cols + categorical_columns
    
    for col in categorical_columns:
        # 1. Apprendre toutes les catégories possibles depuis le jeu d'entraînement
        known_categories = train_df[col].dropna().unique()
        
        # 2. Créer un type Catégoriel avec ces catégories
        cat_dtype = pd.api.types.CategoricalDtype(categories=known_categories, ordered=False)
        
        # 3. Appliquer ce type de manière consistente aux deux DataFrames
        train_df[col] = train_df[col].astype(cat_dtype)
        validation_df[col] = validation_df[col].astype(cat_dtype)
        # Toute catégorie dans validation_df qui n'était pas dans train_df deviendra NaN, ce qui est correct.
    
    
    X_train, X_val = train_df[feature_columns], validation_df[feature_columns]
    
    if model_type == 'classification':
        y_train, y_val = (train_df[target_column] > 1).astype(int), (validation_df[target_column] > 1).astype(int)
        params.update({'objective': 'binary:logistic', 'eval_metric': 'auc','enable_categorical': True})
        model = xgb.XGBClassifier(**params)
    else:
        y_train, y_val = np.log(train_df[target_column].clip(lower=1e-8)), np.log(validation_df[target_column].clip(lower=1e-8))
        params.update({'objective': 'reg:squarederror','enable_categorical': True})
        model = xgb.XGBRegressor(**params)
        
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    if model_type == 'classification': 
        predictions = model.predict_proba(X_val)[:, 1]
    else: 
        predictions = np.exp(model.predict(X_val))
    
    pred_df = validation_df[['ticker', 'year_month', 'monthly_return']].copy()
    pred_df['prediction'] = predictions
    
    top_n_portfolio = pred_df.sort_values('prediction', ascending=False).groupby('year_month').head(10)
    
    portfolio_monthly_returns = top_n_portfolio.groupby('year_month')['monthly_return'].mean()

    final_metric_kwargs = {'portfolio_returns': portfolio_monthly_returns}
    
    if metric_kwargs :
        final_metric_kwargs.update(metric_kwargs)

    if 'benchmark_returns' in final_metric_kwargs:
        benchmark_df = final_metric_kwargs['benchmark_returns']
        aligned_benchmark = benchmark_df.set_index('year_month').loc[portfolio_monthly_returns.index]
        final_metric_kwargs['benchmark_returns'] = aligned_benchmark.iloc[:, 0]
        
        
    #print('Disctimination:', calculate_discrimination_penalty(predictions))
    #print('std:', np.std(predictions))
    return metric_function(**final_metric_kwargs) - calculate_discrimination_penalty(predictions)

def find_best_hyperparameters_for_year(df: pd.DataFrame, hpo_train_end: pd.Period, hpo_validation_end: pd.Period, hparam_space: Dict, n_trials: int, model_type: str, metric_function: Callable, metric_kwargs: Dict, warm_start_params: Dict = None,random_state = None) -> Dict:
    """Runs an Optuna study for a single year with an optional warm start."""
    train_df = df[df['year_month'] <= hpo_train_end].copy()
    validation_df = df[(df['year_month'] > hpo_train_end) & (df['year_month'] <= hpo_validation_end)].copy()
    if train_df.empty or validation_df.empty: return None
    try : 
        study = optuna.create_study(direction="maximize",pruner=MedianPruner(n_warmup_steps=5),sampler = TPESampler(seed=random_state))
    except ValueError as e:
        study = optuna.create_study(direction="maximize",sampler = TPESampler(seed=random_state))
        
    if warm_start_params:
        print(f"Warm starting HPO with previous best params: {warm_start_params}")
        study.enqueue_trial(warm_start_params)
    study.optimize(lambda trial: objective_for_period(trial, train_df, validation_df, hparam_space, model_type, metric_function, metric_kwargs,random_state), n_trials=n_trials)
    return study.best_params

# ==============================================================================
# MODULE 3: MOTEUR DE MODÉLISATION MENSUEL (NOUVELLE FONCTION MODULAIRE)
# ==============================================================================
def train_and_predict_monthly(train_df : pd.DataFrame, predict_df : pd.DataFrame, model_type : str, xgb_params : Dict,n_simu : int,std : float = 0.1,plot_shap : bool = False) -> Tuple[pd.DataFrame,pd.DataFrame, Dict,Dict]:
    """
    Handles the entire train and predict process for a single month, including
    simulations with shocked hyperparameters and SHAP value calculation.
    """
    target_column = 'future_return'
    y_train = link_function(train_df[target_column], model_type)
    
    final_model_params = xgb_params.copy()
    final_model_params.pop('early_stopping_rounds', None)
    
    predictions_total = pd.DataFrame()
    models = {}
    dict_shap_values = {}

    for i in range(n_simu): 
        if i == 0: 
            final_model_params_loop = final_model_params
        else: 
            final_model_params_loop = generate_shocked_hyperparameters(final_model_params, std=std)
        
            
        X_train_loop = train_df.drop(columns=[target_column, 'ticker', 'year_month','monthly_return'])
        X_test_loop = clean_to_category(predict_df[X_train_loop.columns]) # Ensure consistent columns and types
        
        if model_type == 'classification':
            final_model_params_loop.update({'objective': 'binary:logistic', 'enable_categorical': True})
            model = xgb.XGBClassifier(**final_model_params_loop)
            model.fit(X_train_loop, y_train, verbose=False)
            models[i] = model
            
            
            
            predictions = model.predict_proba(X_test_loop)[:, 1]
            dict_shap_values[i] = compute_shap_values(model,X_test_loop)
            
            result_df_loop = predict_df[['ticker', 'year_month']].copy()
            result_df_loop['prediction'] = predictions
            result_df_loop['simu'] = i
            predictions_total = pd.concat([predictions_total, result_df_loop], axis=0)
        
        else: 
            final_model_params_loop.update({'objective': 'reg:squarederror', 'enable_categorical': True})
            model = xgb.XGBRegressor(**final_model_params_loop)
            model.fit(X_train_loop, y_train, verbose=False)
            models[i] = model
            
            predictions = np.exp(model.predict(X_test_loop))
            
            dict_shap_values[i] = compute_shap_values(model,X_test_loop)
            
            result_df_loop = predict_df[['ticker', 'year_month']].copy()
            result_df_loop['prediction'] = predictions
            result_df_loop['simu'] = i
            predictions_total = pd.concat([predictions_total, result_df_loop], axis=0)
    
    for keys,shapito in dict_shap_values.items() :
        if keys == 0 : 
            final_shap = shapito
        else : 
            final_shap.base_values += shapito.base_values
            final_shap.values += shapito.values
    if final_shap is not None :   
        final_shap.base_values = final_shap.base_values/len(dict_shap_values.items())
        final_shap.values = final_shap.values/len(dict_shap_values.items())
        best_tickers = compute_best_shap(final_shap, predict_df, top_n=10)
    
    
    
    if plot_shap : run_shap_analysis(final_shap,top_n_variables=30,top_n_waterfall=10,df=predict_df)


   
    return predictions_total,best_tickers, models,dict_shap_values

def shap_importance_df(shap_object) -> pd.DataFrame :
    vals = np.abs(shap_object.values).mean(axis=0)
    importance_df = pd.DataFrame({
        'feature': shap_object.feature_names,
        'importance': vals
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    return importance_df
  
def compute_shap_values(model: Any,df: pd.DataFrame) : 
        feature_columns = model.feature_names_in_

        original_data_aligned = df[feature_columns].copy() # Utiliser une copie pour éviter les warnings
    
        original_data_aligned = clean_to_category(original_data_aligned)

        explainer = shap.TreeExplainer(model)
        shap_object = explainer(original_data_aligned)
        return shap_object

def expit(x: np.ndarray) -> np.ndarray:
    """
    Computes the logistic function (expit) for an array of values.
    
    Args:
        x (np.ndarray): Input array of values.

    Returns:
        np.ndarray: The computed expit values.
    """
    return 1 / (1 + np.exp(-x))
def compute_best_shap(shap_object: Any, df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """
    Computes the best SHAP values and returns a DataFrame with the top N features.
    
    Args:
        shap_object (Any): The SHAP object containing the values.
        df (pd.DataFrame): The original DataFrame with feature values.
        top_n (int): The number of top features to return.

    Returns:
        pd.DataFrame: A DataFrame with the top N features and their SHAP values.
    """
    sum_values =  shap_object.base_values+shap_object.values.sum(axis = 1)
    top_n_index = np.argsort(sum_values)[-top_n:][::-1]
    tickers = []
    for  best_prediction_index in top_n_index : 
        ticker_loop = df['ticker'].iloc[best_prediction_index.astype(int)]
        shap_values = sum_values[best_prediction_index]
        prediction = expit(shap_values)
        df_tickers_loop = pd.DataFrame({
            'ticker': [ticker_loop],
            'shap_value': [shap_values],
            'prediction': [prediction]
        })
        tickers.append(df_tickers_loop)
    return pd.concat(tickers, ignore_index=True)

def run_shap_analysis(shap_object : Any,top_n_variables: int = 10,top_n_waterfall : int = 10, df: pd.DataFrame = None) -> Any :
    """
    Exécute une analyse SHAP complète à partir de valeurs pré-calculées,
    de manière autonome et sans dépendre d'une classe.

    Args:
        model: Le modèle XGBoost entraîné. Il est nécessaire pour obtenir
               l'espérance (base_value) et la liste officielle des features.
        shap_values_df (pd.DataFrame): Le DataFrame contenant les valeurs SHAP pré-calculées.
        original_data_df (pd.DataFrame): Le DataFrame original (ex: X_test)
                                         avec les valeurs des features.
        cat_features (List[str]): La liste des noms des colonnes catégorielles.
        top_n (int): Le nombre de features à analyser en détail.

    Returns:
        pd.DataFrame: Un DataFrame contenant l'importance de chaque feature.
    """

    importance_df = shap_importance_df(shap_object)
    shap.plots.beeswarm(shap_object, max_display=top_n_variables)
    plt.show()
    
    top_features = importance_df['feature'].head(top_n_variables).tolist()
    
    for feature in top_features:
        shap.plots.scatter(shap_object[:, feature], color=shap_object, show=False)
        plt.show()
        
    sum_values =  shap_object.base_values+shap_object.values.sum(axis = 1)
    top_n_index = np.argsort(sum_values)[-top_n_waterfall:][::-1]
    tickers = []
    for  best_prediction_index in top_n_index : 
        
        ticker_loop = df['ticker'].iloc[best_prediction_index.astype(int)]
        tickers.append(ticker_loop)
        plt.title(f"SHAP Waterfall Plot for {ticker_loop}")
        shap_explanation_for_best_instance = shap_object[best_prediction_index]
        shap.plots.waterfall(shap_explanation_for_best_instance, max_display=top_n_waterfall, show=False)
        plt.show()

        
    return shap_object,tickers

def run_backtest_loop(df: pd.DataFrame, model_type: str, xgb_params: Dict,n_simu : int,std : float = 0.1) -> Tuple[pd.DataFrame,pd.DataFrame, Dict,Any]:
    """
    Exécute la boucle de backtest walk-forward de manière correcte.
    """
    all_predictions = []
    all_predictions_shap= []
    all_models = {}
    df = df[df['year_month'] > '2000-01'].copy()
    date_range = sorted(df['year_month'].unique())
    #date_range.append(np.max(date_range)+1)

    for cutoff_period in tqdm(date_range, desc="Walk-Forward Backtest"):
        if cutoff_period == max(date_range): plot_shap = True
        else : plot_shap = False
        
        train_df = df[df['year_month'] < cutoff_period].copy()
        predict_df = df[df['year_month'] == cutoff_period].copy()

        if train_df.empty or predict_df.empty or train_df['year_month'].nunique() < 12 :
            continue
            
        # 2. Appel de la fonction modulaire d'entraînement et de prédiction
        results_loop,results_loop_shap,models,shap_dict = train_and_predict_monthly(train_df, 
                                                           predict_df, 
                                                           model_type, 
                                                           xgb_params,
                                                           n_simu,
                                                           std,
                                                           plot_shap = plot_shap)
        results_loop['year_month'] = cutoff_period
        results_loop_shap['year_month'] = cutoff_period
        all_predictions_shap.append(results_loop_shap)
        all_predictions.append(results_loop)
        all_models[cutoff_period] = models
            
            
        
    return pd.concat(all_predictions, ignore_index=True),pd.concat(all_predictions_shap, ignore_index=True),all_models
# ==============================================================================
# MODULE 4: ORCHESTRATEUR DE BACKTEST FINAL
# ==============================================================================
def run_full_backtest(df: pd.DataFrame, backtest_start_year: int, backtest_end_year: int, train_years: int, validation_years: int, hparam_space: Dict, n_trials: int, model_type: str, metric_function: Callable, metric_kwargs: Dict,random_state = None) -> Tuple[pd.DataFrame, Dict]:
    """
    The main orchestrator for the full walk-forward backtest.
    """
    all_predictions = []
    hyperparameter_history = {}
    best_params_previous_year = None

    for year in tqdm(range(backtest_start_year, backtest_end_year + 1), desc="Annual Backtest & HPO Loop"):
        print(f"\n===== Processing Year: {year} =====")
        
        # 1. Annual Hyperparameter Optimization with Warm Start
        hpo_validation_end = pd.Period(f'{year-1}-12', freq='M')
        #hpo_train_end = hpo_validation_end - pd.DateOffset(years=validation_years)
        hpo_train_end = hpo_validation_end - (12 * validation_years)
        best_params = find_best_hyperparameters_for_year(df, hpo_train_end, hpo_validation_end, hparam_space, n_trials, model_type, metric_function, metric_kwargs, warm_start_params=best_params_previous_year,random_state =random_state)
        
        if best_params is None:
            print(f"Using previous year's params for year {year}")
            best_params = best_params_previous_year
        
        hyperparameter_history[year] = best_params
        best_params_previous_year = best_params

        # 2. Monthly Retraining & Prediction for the current year
        print(f"--- Running monthly backtest for year {year} ---")
        for month in range(1, 13):
            cutoff_period = pd.Period(f'{year}-{month:02d}', freq='M')
            monthly_train_df = df[df['year_month'] < cutoff_period].copy()
            predict_df = df[df['year_month'] == cutoff_period].copy()
            if monthly_train_df.empty or predict_df.empty: continue
            
            # CALL THE NEW MODULAR FUNCTION
            result_df,_ = train_and_predict_monthly(monthly_train_df, predict_df, model_type, best_params)
            all_predictions.append(result_df)

    return pd.concat(all_predictions, ignore_index=True), hyperparameter_history

def plot_hyperparameter_stability(hparam_history_df: pd.DataFrame):
    """
    Analyzes and visualizes the stability of hyperparameters over time.

    Args:
        hparam_history_df (pd.DataFrame): DataFrame where the index is the year
                                          and columns are the hyperparameter names.
    """
    # Exclude non-numeric columns that Optuna might add
    params_to_plot = hparam_history_df.select_dtypes(include=np.number).columns.tolist()
    
    if not params_to_plot:
        print("No numeric hyperparameters to plot.")
        return

    # Determine the grid size for the subplots
    n_params = len(params_to_plot)
    n_cols = 3  # Display 3 plots per row
    n_rows = math.ceil(n_params / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
    # Flatten the axes array for easy iteration, handling the case of a single row/col
    axes = np.array(axes).flatten()

    print("--- Hyperparameter Stability Analysis ---")
    for i, param in enumerate(params_to_plot):
        ax = axes[i]
        
        # Plot the evolution of the hyperparameter
        ax.plot(hparam_history_df.index, hparam_history_df[param], marker='o', linestyle='-')
        
        # Calculate and display the mean as a horizontal line
        mean_val = hparam_history_df[param].mean()
        ax.axhline(mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}')
        
        ax.set_title(f"Evolution of '{param}'", fontsize=14)
        ax.set_xlabel("Year")
        ax.set_ylabel("Value")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend()

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()


# ==============================================================================
# SOLUTION 2: FONCTION D'OPTIMISATION AVEC VALIDATION CROISÉE
# ==============================================================================

def objective_for_period_stable_v2(trial: optuna.Trial, train_df: pd.DataFrame, 
                                  validation_df: pd.DataFrame, hparam_space: Dict, 
                                  model_type: str, metric_function: Callable, 
                                  metric_kwargs: Dict, random_state=None,n_simu : int = 10,std : float = 0.02,n_asset : int = 10) -> float:
    """
    Version simplifiée et plus robuste sans validation croisée complexe
    """
    params = {}
    for name, (ptype, low, high) in hparam_space.items():
        if ptype == "int": 
            params[name] = trial.suggest_int(name, low, high)
        else: 
            params[name] = trial.suggest_float(name, low, high, log=(name == 'learning_rate'))
    
    params.update({
        'n_jobs': -1, 
        'enable_categorical': True, 
        'early_stopping_rounds': 50,
        'random_state': random_state
    })
    
    target_column = 'future_return'
    identifier_cols = ['ticker', 'year_month']
    categorical_columns = [c for c in train_df.select_dtypes(include=['object', 'category']).columns 
                          if c not in identifier_cols]
    numeric_feature_cols = [c for c in train_df.select_dtypes(include=np.number).columns 
                           if c not in [target_column, 'monthly_return', 'future_return']]
    feature_columns = numeric_feature_cols + categorical_columns
    
    # Préparation des données catégorielles
    for col in categorical_columns:
        known_categories = train_df[col].dropna().unique()
        cat_dtype = pd.api.types.CategoricalDtype(categories=known_categories, ordered=False)
        train_df[col] = train_df[col].astype(cat_dtype)
        validation_df[col] = validation_df[col].astype(cat_dtype)
    
    X_train, X_val = train_df[feature_columns], validation_df[feature_columns]
    predictions_total = pd.DataFrame()
    result_df = validation_df[['ticker', 'year_month']].copy()
    models = {}
    for i in range(n_simu) : 
        if i == 0 : final_model_params_loop = params
        else : final_model_params_loop = generate_shocked_hyperparameters(params,std = std)
        
        if model_type == 'classification':
            y_train = link_function(train_df[target_column])
            y_val = link_function(validation_df[target_column])
            params.update({'objective': 'binary:logistic', 'eval_metric': 'auc'})
            final_model_params_loop.update({'objective': 'binary:logistic', 'enable_categorical': True})
            model = xgb.XGBClassifier(**final_model_params_loop)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            models[i] = model
            #x_test = validation_df[train_df.columns]
            predictions = model.predict_proba(X_val)[:, 1]
            result_df_loop = result_df.copy()
            result_df_loop['prediction'] = predictions
            result_df_loop['simu'] = i
            predictions_total = pd.concat([predictions_total,result_df_loop],axis = 0)
        
        else:
            y_train = link_function(train_df[target_column].clip(lower=1e-8))
            y_val = link_function(validation_df[target_column].clip(lower=1e-8))
            params.update({'objective': 'reg:squarederror'})
            final_model_params_loop.update({'objective': 'reg:squarederror', 'enable_categorical': True})
            model = xgb.XGBRegressor(**final_model_params_loop)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  verbose=False)
            models[i] = model
            #x_test = validation_df[train_df.columns]
            predictions = np.exp(model.predict(X_val))
            result_df_loop = result_df.copy()
            result_df_loop['prediction'] = predictions
            result_df_loop['simu'] = i
            predictions_total = pd.concat([predictions_total,result_df_loop],axis = 0)
    
    
    #top_n_portfolio = predictions_total.sort_values('prediction', ascending=False).groupby('year_month').head(20)
    
    top_n_portfolio = predictions_total.sort_values('prediction', ascending=False).groupby(['year_month','simu']).head(n_asset)[['ticker','year_month']].drop_duplicates()
    top_n_portfolio['year_month'] = pd.to_datetime(top_n_portfolio['year_month'].astype(str))
    top_n_portfolio['holding_month'] = (top_n_portfolio['year_month'] + pd.DateOffset(months=1)).dt.to_period('M')
    returns_to_join = validation_df[['ticker', 'year_month', 'monthly_return']].copy()
    strategy_realized_returns = pd.merge(top_n_portfolio, returns_to_join, left_on=['ticker', 'holding_month'], right_on=['ticker', 'year_month'], how='left')
    portfolio_monthly_returns = strategy_realized_returns.groupby('holding_month').agg({
        'monthly_return': 'mean',
        'ticker': 'count'
    }).rename(columns={'ticker': 'n_assets'})
    portfolio_monthly_returns.index.name = 'year_month'
    portfolio_monthly_returns['monthly_return'] = portfolio_monthly_returns['monthly_return']-1
    #portfolio_monthly_returns = portfolio_monthly_returns.reset_index()
    portfolio_monthly_returns.dropna(subset=['monthly_return'], inplace=True)
    
    
    # Vérification que nous avons des données
    if len(portfolio_monthly_returns) == 0:
        return -999.0
    
    final_metric_kwargs = {'portfolio_returns': portfolio_monthly_returns}
    if metric_kwargs:
        final_metric_kwargs.update(metric_kwargs)
    
    if 'benchmark_returns' in final_metric_kwargs:
        benchmark_df = final_metric_kwargs['benchmark_returns']
        aligned_benchmark = benchmark_df.set_index('year_month').loc[portfolio_monthly_returns.index]
        final_metric_kwargs['benchmark_returns'] = aligned_benchmark.iloc[:, 0]
    
    try:
        final_score = metric_function(**final_metric_kwargs)
    except Exception as e:
        print(f"Error in metric calculation: {e}")
        return -999.0
    
    # Pénalité de discrimination
    discrimination_penalty = calculate_discrimination_penalty(predictions) #Compute discrimination in prediciton have no volatility, and then randomise output
    
    return final_score - discrimination_penalty

# ==============================================================================
# SOLUTION 3: FONCTION D'OPTIMISATION AVEC ENSEMBLE DE MODÈLES
# ==============================================================================

def find_best_hyperparameters_for_year_stable(df: pd.DataFrame, hpo_train_end: pd.Period, 
                                             hpo_validation_end: pd.Period, hparam_space: Dict, 
                                             n_trials: int, model_type: str, 
                                             metric_function: Callable, metric_kwargs: Dict, 
                                             warm_start_params: Dict = None, 
                                             random_state=None,n_simu : int = 10,std : float = 0.02,n_asset : int = 10) -> Dict:
    """
    Version stable et robuste pour l'optimisation des hyperparamètres
    """
    train_df = df[df['year_month'] < hpo_train_end].copy()
    validation_df = df[(df['year_month'] >= hpo_train_end) & (df['year_month'] < hpo_validation_end)].copy()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if train_df.empty or validation_df.empty:
        return None
    
    try:
        study = optuna.create_study(
            direction="maximize",
            pruner=MedianPruner(n_warmup_steps=10, n_startup_trials=5),
            sampler=TPESampler(seed=random_state, n_startup_trials=20)
        )
    except ValueError:
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=random_state, n_startup_trials=20)
        )
    
    if warm_start_params:
        #print(f"Warm starting HPO with previous best params: {warm_start_params}")
        study.enqueue_trial(warm_start_params)
    
    # Utiliser la fonction objective simplifiée
    study.optimize(
        lambda trial: objective_for_period_stable_v2(
            trial, train_df, validation_df, hparam_space, 
            model_type, metric_function, metric_kwargs, random_state,
            n_simu  = n_simu,std = std,n_asset =n_asset
        ), 
        n_trials=n_trials
    )
    try : 
        plot_optimization_history(study).show()
    except Exception as e:  
        print(f"Error during optimization history plot: {e}")
    
    try : plot_param_importances(study).show()
    except Exception as e:  
        print(f"Error during optimization history plot: {e}")
    
    try : plot_parallel_coordinate(study).show()
    except Exception as e:  
        print(f"Error during optimization history plot: {e}")
    return study.best_params

# ==============================================================================
# SOLUTION 4: BACKTEST AVEC PARAMÈTRES PLUS STABLES
# ==============================================================================

def run_full_backtest_stable(df: pd.DataFrame, backtest_start_year: int, 
                            backtest_end_year: int, train_years: int, 
                            validation_years: int, hparam_space: Dict, 
                            n_trials: int, model_type: str, 
                            metric_function: Callable, metric_kwargs: Dict, 
                            random_state=None, stability_mode=True,n_simu : int = 10,std : float = 0.05,n_asset : int =  10) -> Tuple[pd.DataFrame, Dict,pd.DataFrame]:
    """
    Version plus stable du backtest avec options de stabilité
    """
    all_predictions = []
    all_shap_values = []
    hyperparameter_history = {}
    best_params_previous_year = None
    
    
    for year in tqdm(range(backtest_start_year, backtest_end_year + 1), 
                     desc="Annual Backtest & HPO Loop"):
        print(f"\n===== Processing Year: {year} =====")
        
        hpo_validation_end = pd.Period(f'{year-1}-12', freq='M')
        hpo_train_end = hpo_validation_end - (12 * validation_years)
        
        if stability_mode:
            # Utiliser la version simplifiée et robuste
            best_params = find_best_hyperparameters_for_year_stable(
                df, hpo_train_end, hpo_validation_end, hparam_space, 
                n_trials, model_type, metric_function, metric_kwargs,
                warm_start_params=best_params_previous_year, 
                random_state=random_state,
                n_simu =  n_simu,
                std = std,
                n_asset = n_asset)
        else:
            # Version originale
            best_params = find_best_hyperparameters_for_year(
                df, hpo_train_end, hpo_validation_end, hparam_space, 
                n_trials, model_type, metric_function, metric_kwargs,
                warm_start_params=best_params_previous_year, 
                random_state=random_state
            )
        
        if best_params is None:
            print(f"Using previous year's params for year {year}")
            best_params = best_params_previous_year
        
        hyperparameter_history[year] = best_params
        best_params_previous_year = best_params

        # Prédictions mensuelles
        print(f"--- Running monthly backtest for year {year} ---")
        for month in range(1, 13):
            #cutoff_period = pd.Period(f'{year}-{month:02d}', freq='M')
            cutoff_period = pd.Period(f'{year}-{month:02d}', freq='M')-1 #major change. Hyper parameter had last data from future retun on year-1 12. Last seen data where eof. So we can train first untile end of year -1 to fit on year month 1
            monthly_train_df = df[df['year_month'] < cutoff_period].copy()
            predict_df = df[df['year_month'] == cutoff_period].copy()
            
            if monthly_train_df.empty or predict_df.empty:
                continue
            
            result_df, _ ,shap_values = train_and_predict_monthly(
                monthly_train_df, predict_df, model_type, best_params,n_simu = n_simu,std =  std)
            shap_values['year_month'] = cutoff_period
            all_predictions.append(result_df)
            all_shap_values.append(shap_values)

    return pd.concat(all_predictions, ignore_index=True), hyperparameter_history,pd.concat(all_shap_values, ignore_index=True)

# ==============================================================================
# SOLUTION 5: EXEMPLE D'UTILISATION AVEC PARAMÈTRES STABLES
# ==============================================================================


# %% data preparation
funda_kpis = generate_all_kpis(
     balance_sheet=balance_sheet,
     cash_flow=cash_flow,
     income_statement=income_statement,
     earnings=earnings,
     monthly_return=monthly_return)

funda_kpis = funda_kpis.drop(columns=['date','last_close','monthly_return','quarter_end', 'filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning'], errors='ignore')

final_price = PricesDataPreprocessor.prices_vs_index(index=index_data.daily_prices.copy(), 
                                               prices=final_price.copy(),
                                               column_close_index='adjusted_close',
                                               column_close_prices='adjusted_close')
monthly_returns_vs_index = PricesDataPreprocessor.calculate_monthly_returns(df=final_price.copy(),
                                                                    column_date='date',
                                                                    column_close='close_vs_index')
"""
technical_kpis = generate_monthly_technical_indicators_from_daily(final_price.copy(),
                                                               price_column = 'close_vs_index',
                                                               moving_average_pairs=[(1, 10),
                                                                                     (1, 5),
                                                                                     (1, 15),
                                                                                     #(1, 20),
                                                                                    (10, 100),
                                                                                    (20, 200),
                                                                                    #(20, 100),
                                                                                    (10, 20)
                                                                                    ])"""

SEED_PAIRS = [(1, 10), (10, 100)] # Vos indicateurs de base
N_INDICATORS = 15
CORR_THRESHOLD = 0.95

# 2. Appelez la nouvelle fonction
technical_kpis = generate_decorrelated_indicators_iteratively(
    daily_prices_df=final_price.copy(),
    price_column='close_vs_index',
    seed_pairs=SEED_PAIRS,
    n_final_indicators=N_INDICATORS,
    correlation_threshold=CORR_THRESHOLD,
    max_tries = 200
)

funda_x_technical_joined = funda_kpis.merge(technical_kpis, on=['ticker', 'year_month'], how='outer').merge(general[['ticker','GicGroup']], on='ticker', how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
funda_joined = funda_kpis.merge(general[['ticker','GicGroup']], on='ticker', how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')
technical_joined = technical_kpis.merge(general[['ticker','GicGroup']], on='ticker', how='left').merge(monthly_returns_vs_index[['ticker','year_month','monthly_return']], on=['ticker', 'year_month'], how='left')

funda_x_technical_joined =  prepare_data_for_xgboost(funda_x_technical_joined,index = index_data,to_quantiles=True,treshold_percentage_missing=0.10)
funda_joined =  prepare_data_for_xgboost(funda_joined,index = index_data,to_quantiles=True,treshold_percentage_missing=0.10)
technical_joined =  prepare_data_for_xgboost(technical_joined,index = index_data,to_quantiles=True,treshold_percentage_missing=0.10)

def correlation_ratio(categories, measurements):
    """
    Calcule le rapport de corrélation (eta) pour une association
    entre une variable catégorielle et une variable numérique.
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(cat_num):
        cat_measures = measurements[f_cat == i]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures) if n_array[i] > 0 else 0
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    
    numerator = np.sum(np.multiply(n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)))
    denominator = np.sum(np.power(np.subtract(measurements, y_total_avg), 2))
    
    return np.sqrt(numerator / denominator) if denominator else 0.0

def cramers_v(x, y):
    """
    Calcule le V de Cramér pour une association entre deux variables catégorielles.
    """
    confusion_matrix = pd.crosstab(x, y)
    if confusion_matrix.shape[0] == 1 or confusion_matrix.shape[1] == 1:
        return 1.0 # Association parfaite si une variable n'a qu'une seule catégorie
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Correction pour le biais
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    
    if min((kcorr-1), (rcorr-1)) == 0:
        return 0.0
    
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def matrice_correlation_mixte(df, plot=True, n=10):
    """
    Calcule et affiche la matrice de corrélation pour des types de données mixtes.

    Args:
        df (pd.DataFrame): Le DataFrame d'entrée.
        plot (bool, optional): Si True, affiche la liste des N plus grandes
                               corrélations et une heatmap. Par défaut True.
        n (int, optional): Le nombre de plus grandes corrélations à afficher
                           dans la liste. Par défaut 10.

    Returns:
        pd.DataFrame: Une matrice de corrélation carrée.
    """
    cols = df.columns
    n_cols = len(cols)
    corr_matrix = pd.DataFrame(np.zeros((n_cols, n_cols)), index=cols, columns=cols)
    
    numerical_cols = df.select_dtypes(include=np.number).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    for i in range(n_cols):
        for j in range(i, n_cols):
            col1_name = cols[i]
            col2_name = cols[j]
            
            if col1_name == col2_name:
                corr = 1.0
            elif col1_name in numerical_cols and col2_name in numerical_cols:
                corr = df[col1_name].corr(df[col2_name])
            elif col1_name in categorical_cols and col2_name in numerical_cols:
                corr = correlation_ratio(df[col1_name], df[col2_name])
            elif col1_name in numerical_cols and col2_name in categorical_cols:
                corr = correlation_ratio(df[col2_name], df[col1_name])
            else: # Catégorielle - Catégorielle
                corr = cramers_v(df[col1_name], df[col2_name])

            corr_matrix.loc[col1_name, col2_name] = corr
            corr_matrix.loc[col2_name, col1_name] = corr

    if plot:
        # Extraire les N plus grandes corrélations (en ignorant la diagonale)
        # On ne garde que le triangle supérieur pour éviter les doublons
        sol = (corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                      .stack()
                      .sort_values(key=abs, ascending=False))
        
        print(f"--- Top {n} des plus grandes corrélations (valeur absolue) ---")
        print(sol.head(n).round(2))
        print("-" * 50)

        # Créer la heatmap
        plt.figure(figsize=(30, 30))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Matrice de Corrélation Mixte', fontsize=15)
        plt.show()

    return corr_matrix


matrice_corr = matrice_correlation_mixte(technical_joined.drop(columns=['ticker', 'year_month', 'monthly_return', 'future_return'], errors='ignore'))



# %%
hyperparametres_1 = {    
    'objective': 'binary:logistic',
    'n_estimators': 100,
    'learning_rate': 0.01,      # Un peu plus rapide
    'max_depth': 6,             # Permettre des arbres plus profonds
    'subsample': 0.5,
    'colsample_bytree': 0.7,
    'gamma': 0.3,                 # Moins de pénalité pour créer des branches
    'reg_lambda': 1,            
    'n_jobs': -1,
    'enable_categorical': True
}

hyperparametres_2 = {    
    'objective': 'binary:logistic',
    'n_estimators': 50,
    'learning_rate': 0.05,      # Un peu plus rapide
    'max_depth': 10,             # Permettre des arbres plus profonds
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0,                 # Moins de pénalité pour créer des branches
    'reg_lambda': 1,            
    'n_jobs': -1,
    'enable_categorical': True
}

test_1 = run_backtest_loop(df = funda_x_technical_joined,model_type = 'classification',xgb_params = hyperparametres_1,n_simu=20,std = 0.05)
# %%
test_2 = run_backtest_loop(df = funda_x_technical_joined,model_type = 'classification',xgb_params = hyperparametres_2,n_simu=20,std = 0.05)



return_test11,portfolios11 = generate_portfolio_returns(test_1[0], monthly_return, n=10,mode = 'exact')
return_test12,portfolios12 = generate_portfolio_returns(test_1[0], monthly_return, n=10,mode = 'inexact')
return_test13,portfolios13 = generate_portfolio_returns(test_1[1], monthly_return, n=10,mode = 'exact')


return_test21,portfolios21 = generate_portfolio_returns(test_2[0], monthly_return, n=10,mode = 'exact')
return_test22,portfolios22 = generate_portfolio_returns(test_2[0], monthly_return, n=10,mode = 'inexact')
return_test23,portfolios23 = generate_portfolio_returns(test_2[1], monthly_return, n=10,mode = 'exact')

models_to_compare = {'test11': return_test11,
                     'test12': return_test12,
                     'test13': return_test13,
                     'test21': return_test21,
                     'test22': return_test22,
                     'test23': return_test23,
                     'SP500': index_data.monthly_returns}
evaluator = ModelEvaluator()
metrics, _, _, _, _ = evaluator.compare_models(models_data=models_to_compare, start_year = 2000)
plt.show()
# %% 
HPARAM_SPACE_STABLE = {
    "n_estimators": ("int", 40, 100),           # Plus restreint autour de 50
    "learning_rate": ("float", 0.01, 0.10),   # Plus restreint autour de 0.05
    "max_depth": ("int", 3, 10),               # Plus restreint autour de 4
    "subsample": ("float", 0.5, 0.9),         # Plus restreint autour de 0.8
    "colsample_bytree": ("float", 0.7, 0.9),  # Plus restreint autour de 0.8
    "gamma": ("float", 0.0, 2.0),             # Beaucoup plus restreint
    "reg_lambda": ("float", 0.3, 1.5)         # Plus restreint autour de 0.5
}

STABLE_CONFIG = {
    'hparam_space': HPARAM_SPACE_STABLE,
    'n_trials': 100,  # Plus d'essais pour la stabilité
    'stability_mode': True,
    'metric_function': calculate_custom_score,
    'metric_kwargs': {'benchmark_returns': index_data.monthly_returns, 'alpha': 2}
}


list_random_states = [42, 123,32]  # Liste des états aléatoires à tester
all_returns = {}
hparam_log = pd.DataFrame()
for rs in list_random_states:
    print("Processing RS -",rs )

    final_predictions_loop, hparam_log_loop,all_shap = run_full_backtest_stable(
        df=technical_joined,
        backtest_start_year=2010,
        backtest_end_year=2025,
        train_years=15,
        validation_years=1,
        hparam_space=STABLE_CONFIG['hparam_space'],
        n_trials=STABLE_CONFIG['n_trials'],
        model_type='classification',
        metric_function=STABLE_CONFIG['metric_function'],
        metric_kwargs=STABLE_CONFIG['metric_kwargs'],
        random_state=rs,
        n_simu= 10,
        std=0.02,
        n_asset=10)
    
    
    hparam_log_loop = pd.DataFrame.from_dict(hparam_log_loop, orient='index')
    hparam_log_loop['random_state'] = rs
    strategy_returns = generate_portfolio_returns(final_predictions_loop, monthly_return, n=3,mode = 'notexact')
    
    hparam_log = pd.concat([hparam_log, hparam_log_loop], axis=0)
    all_returns[rs] = strategy_returns
    plot_hyperparameter_stability(hparam_log_loop)
    


all_returns['index'] = index_data.monthly_returns
evaluator = ModelEvaluator()
metrics, _, _, _, _ = evaluator.compare_models(models_data=all_returns,start_year=2012)
print(metrics)
"""