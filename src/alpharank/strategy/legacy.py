import pandas as pd
import numpy as np
#from .data_preprocessing import DataPreprocessor
#from .fundamentals import FundamentalAnalyzer
from alpharank.data.processing import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor
from alpharank.features.indicators import TechnicalIndicators
import datetime
from scipy import stats
from tqdm import tqdm
from datetime import *
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import warnings
from alpharank.utils.frame_backend import (
    Backend,
    ensure_backend_name,
    normalize_year_month_to_period,
    normalize_year_month_to_timestamp,
    require_polars,
    to_pandas,
    to_polars,
)

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None

warnings.filterwarnings("ignore", category=FutureWarning)
# %%
class StrategyLearner:
    """Apprentissage et backtest des stratégies techniques et fondamentales."""
    @staticmethod
    def learning_one_technicalparameter(df,column_price, historical_company,n_long, n_short, func_movingaverage, n_asset):
        df['year_month'] = df['date'].dt.to_period('M')
        df_save = df.copy()
        for n, col in [(n_short, 'short_movingaverage'), (n_long, 'long_movingaverage')]:
            df[col] = df.groupby('ticker')[column_price].transform(lambda x: func_movingaverage(x, n=n))
        
        df = df.groupby('ticker').apply(lambda x: x.iloc[n_long:], include_groups=False).reset_index(drop=False)
        df = df.dropna(subset=['short_movingaverage', 'long_movingaverage'])
        #pd.options.mode.chained_assignment = None  # default='warn'
        df.loc[:, 'mtr'] = df['short_movingaverage'] / df['long_movingaverage']

        df = df.merge(historical_company, on=['year_month', 'ticker'], how='inner')
        df = df.reset_index(drop=True)

        max_date_indices = df.groupby(['year_month', 'ticker'])['date'].idxmax()
        df = df.loc[max_date_indices.values]

        grouped = df.groupby('year_month')
        df = df.assign(
            mean=grouped['mtr'].transform('mean'),
            ecart=grouped['mtr'].transform('std'),
            quantile_mtr=lambda x: stats.norm.cdf(x['mtr'], x['mean'], x['ecart'])
        )
        
        df = df.sort_values(by=['year_month', 'mtr'], ascending=[True, False]).groupby('year_month').head(n_asset)
        df = df.assign(n_long=n_long, n_short=n_short, n_asset=n_asset)[['year_month', 'date', 'ticker', 'n_long', 'n_short', 'n_asset', 'mtr', 'quantile_mtr']]
        
        df_return = (df_save.groupby(['year_month', 'ticker'])['dr'].prod()
                        .reset_index()
                        .sort_values(by=['ticker', 'year_month'])
                        .assign(application_month=lambda x: x.groupby('ticker')['year_month'].shift(1)))
        
        # Adding fake application month because we don't know the retun of the last date.
        max_application_month = df_return['application_month'].max()
        if pd.notna(max_application_month):
            additional_rows = df_return[df_return['application_month'] == max_application_month].copy()
            additional_rows['application_month'] = max_application_month + 1
            additional_rows['dr'] = np.nan
            df_return = pd.concat([df_return, additional_rows], ignore_index=True)
        final_return = (df_return
                .merge(df, 
                       left_on=['application_month', 'ticker'], 
                       right_on=['year_month', 'ticker'], 
                       how='inner',
                       suffixes=('_return', '_signal'))  
                .rename(columns={"year_month_return": "year_month"}) 
                [['year_month', 'application_month', 'date', 'ticker', 'dr', 'n_long', 'n_short', 'n_asset', 'mtr', 'quantile_mtr']]
               )

        return final_return, df

    @staticmethod
    def fiting(df, column_price, historical_company, n_long, n_short, func_movingaverage, n_asset, backend: Backend = "polars"):
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for StrategyLearner.fiting.")

        require_polars()
        df_pd = to_pandas(df)
        df_pd['date'] = pd.to_datetime(df_pd['date'], errors='coerce')
        if 'year_month' in df_pd.columns:
            df_pd = normalize_year_month_to_timestamp(df_pd, col='year_month')
        else:
            df_pd['year_month'] = pd.to_datetime(df_pd['date']).dt.to_period('M').dt.to_timestamp(how='start')
        hc_pd = normalize_year_month_to_timestamp(to_pandas(historical_company), col='year_month')
        hc_pd = hc_pd[['year_month', 'ticker']].drop_duplicates()

        pl_base = to_polars(df_pd).sort(['ticker', 'date'])
        pl_df = pl_base.clone()
        if func_movingaverage == TechnicalIndicators.ema:
            pl_df = pl_df.with_columns([
                pl.col(column_price).ewm_mean(span=n_short, adjust=False).over('ticker').alias('short_movingaverage'),
                pl.col(column_price).ewm_mean(span=n_long, adjust=False).over('ticker').alias('long_movingaverage'),
            ])
        elif func_movingaverage == TechnicalIndicators.sma:
            pl_df = pl_df.with_columns([
                pl.col(column_price).rolling_mean(window_size=n_short, min_samples=1).over('ticker').alias('short_movingaverage'),
                pl.col(column_price).rolling_mean(window_size=n_long, min_samples=1).over('ticker').alias('long_movingaverage'),
            ])
        else:
            # Fallback for custom kernels while keeping a polars-first path.
            tmp = to_pandas(pl_df)
            for n, col in [(n_short, 'short_movingaverage'), (n_long, 'long_movingaverage')]:
                tmp[col] = tmp.groupby('ticker', sort=False)[column_price].transform(lambda x: func_movingaverage(x, n=n))
            pl_df = to_polars(tmp)

        pl_df = (
            pl_df.with_row_index('_row_idx')
            .with_columns((pl.col('_row_idx').rank(method='ordinal').over('ticker') - 1).alias('_cumcount'))
            .filter(pl.col('_cumcount') >= n_long)
            .drop(['_row_idx', '_cumcount'])
            .filter(pl.col('short_movingaverage').is_not_null() & pl.col('long_movingaverage').is_not_null())
            .with_columns((pl.col('short_movingaverage') / pl.col('long_movingaverage')).alias('mtr'))
            .join(to_polars(hc_pd), on=['year_month', 'ticker'], how='inner')
        )

        value_cols = [c for c in pl_df.columns if c not in ['year_month', 'ticker']]
        pl_df = (
            pl_df.group_by(['year_month', 'ticker'], maintain_order=True)
            .agg([pl.col(c).sort_by('date').last().alias(c) for c in value_cols])
        )
        stats_df = pl_df.group_by('year_month').agg([
            pl.col('mtr').mean().alias('mean'),
            pl.col('mtr').std().alias('ecart'),
        ])
        pl_df = pl_df.join(stats_df, on='year_month', how='left')

        quantiles = stats.norm.cdf(
            pl_df['mtr'].to_numpy(),
            pl_df['mean'].to_numpy(),
            pl_df['ecart'].to_numpy(),
        )
        pl_df = (
            pl_df.with_columns(pl.Series('quantile_mtr', quantiles))
            .sort(['year_month', 'mtr', 'ticker'], descending=[False, True, False])
            .with_columns(pl.col('mtr').rank(method='ordinal', descending=True).over('year_month').alias('_rank_month'))
            .filter(pl.col('_rank_month') <= n_asset)
            .drop('_rank_month')
            .select(['year_month', 'date', 'ticker', 'mtr', 'quantile_mtr'])
            .with_columns([
                pl.lit(n_long).alias('n_long'),
                pl.lit(n_short).alias('n_short'),
                pl.lit(n_asset).alias('n_asset'),
                pl.col('year_month').dt.offset_by('1mo').alias('year_month'),
            ])
        )

        df_return = (
            pl_base.with_columns((1 + pl.col('dr')).alias('dr_factor'))
            .group_by(['year_month', 'ticker'])
            .agg(pl.col('dr_factor').product().alias('dr'))
            .sort(['ticker', 'year_month'])
            .select(['year_month', 'ticker', 'dr'])
        )
        final_return = pl_df.join(df_return, on=['year_month', 'ticker'], how='left')
        out = normalize_year_month_to_period(to_pandas(final_return), col='year_month')
        return out
    
    @staticmethod
    def return_from_training(
        df_fiting,
        stocks_filter,
        sector,
        index,
        alpha,
        temp,
        mode,
        params,
        backend: Backend = "polars",
        score_only: bool = False,
    ):
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for StrategyLearner.return_from_training.")
        stocks_filter = stocks_filter.copy()
        df_fiting = df_fiting.copy()
        sector = sector.copy()
        stocks_filter['year_month'] = stocks_filter['year_month']+1
        require_polars()
        fit_df = normalize_year_month_to_timestamp(df_fiting, col="year_month")
        filter_df = normalize_year_month_to_timestamp(stocks_filter[['year_month', 'ticker']], col="year_month")
        index_monthly = normalize_year_month_to_timestamp(index.monthly_returns[['year_month', 'monthly_return']], col="year_month")
        pl_fit = to_polars(fit_df)
        pl_filter = to_polars(filter_df)
        pl_sector = to_polars(sector[['ticker', 'Sector']])

        selected = (
            pl_fit.join(pl_filter, on=['year_month', 'ticker'], how='inner')
            .join(pl_sector, on='ticker', how='left')
            .sort(['year_month', 'n_long', 'n_short', 'n_asset', 'Sector', 'quantile_mtr', 'ticker'], descending=[False, False, False, False, False, True, False])
            .with_columns(
                pl.col('quantile_mtr').rank(method='ordinal', descending=True).over(
                    ['year_month', 'n_long', 'n_short', 'n_asset', 'Sector']
                ).alias('_rk_sector')
            )
            .filter(pl.col('_rk_sector') <= int(params['n_max_per_sector']))
            .sort(['year_month', 'n_long', 'n_short', 'n_asset', 'quantile_mtr', 'ticker'], descending=[False, False, False, False, True, False])
            .with_columns(
                pl.col('quantile_mtr').rank(method='ordinal', descending=True).over(
                    ['year_month', 'n_long', 'n_short', 'n_asset']
                ).alias('_rk_asset')
            )
            .filter(pl.col('_rk_asset') <= int(params['n_asset']))
            .drop(['_rk_sector', '_rk_asset'])
            .with_columns(
                (
                    pl.col('n_long').cast(pl.Float64).cast(pl.Utf8)
                    + pl.lit('-')
                    + pl.col('n_short').cast(pl.Float64).cast(pl.Utf8)
                    + pl.lit('-')
                    + pl.col('n_asset').cast(pl.Utf8)
                ).alias('model')
            )
        )
        summary = (
            selected
            .group_by(['year_month', 'model'])
            .agg(
                pl.col('dr').mean().alias('dr'),
                pl.col('dr').len().alias('n'),
            )
            .join(to_polars(index_monthly), on='year_month', how='left')
            .with_columns([
                pl.col('monthly_return').alias('monthly_return_index'),
                (pl.col('dr') / (1 + pl.col('monthly_return'))).alias('monthly_return_vs_index'),
                (pl.col('dr') - 1).alias('monthly_return'),
            ])
        )
        summary = summary.select(
            'year_month',
            'model',
            'monthly_return',
            'n',
            'monthly_return_index',
            'monthly_return_vs_index',
        ).filter(pl.col('monthly_return_vs_index').is_not_null())

        if summary.height == 0:
            return {
                'aggregated': pd.DataFrame(),
                'detailed': pd.DataFrame(),
                'score': float('-inf'),
            }

        score_arr = ModelEvaluator.score(summary['monthly_return_vs_index'].to_numpy(), alpha=alpha)
        summary = summary.with_columns(pl.Series('score', score_arr))
        lvl1_bestmodel_loop = ModelEvaluator.best_model_date(
            data=summary,
            mode=mode,
            param_temp=temp,
            param_alpha=alpha,
            backend="polars",
        )
        best_score = float(lvl1_bestmodel_loop['score'].to_list()[0]) if lvl1_bestmodel_loop.height > 0 else float('-inf')

        if score_only:
            return {'aggregated': pd.DataFrame(), 'detailed': pd.DataFrame(), 'score': best_score}

        all_return_monthly_afterselection = normalize_year_month_to_period(to_pandas(selected), 'year_month')
        all_return_monthly_afterselection_summarised = normalize_year_month_to_period(to_pandas(summary), 'year_month')
        return {
            'aggregated': all_return_monthly_afterselection_summarised,
            'detailed': all_return_monthly_afterselection,
            'score': best_score,
        }
    
    @staticmethod
    def learning_process_technical(prices,index, stocks_filter, sector, func_movingaverage, liste_nlong, liste_nshort, liste_nasset, max_persector, final_nmaxasset, list_alpha, list_temp, mode, param_temp_lvl2, param_alpha_lvl2):
        historical_company = index.components
        index_price = index.monthly_returns
        parameters = pd.DataFrame([(n_long, n_short, n_asset) 
                               for n_long in liste_nlong 
                               for n_short in liste_nshort 
                               for n_asset in liste_nasset 
                               if n_long > 3 * n_short],
        columns=['n_long', 'n_short', 'n_asset'])
        all_return_monthly = pd.DataFrame()
        prices = prices.dropna(subset=['close_vs_index']).sort_values(by=['ticker', 'date'])

        for i in tqdm(range(len(parameters)), desc="Processing level 0 analysis"):
            row = parameters.iloc[i]
            result = StrategyLearner().fiting(
                df=prices.copy(),
                column_price = 'close_vs_index',
                historical_company=historical_company,
                n_long=row['n_long'],
                n_short=row['n_short'],
                func_movingaverage=func_movingaverage,  
                n_asset=row['n_asset'])
            
            all_return_monthly = pd.concat([all_return_monthly, result], ignore_index=True)
        
        stocks_filter['year_month'] = stocks_filter['year_month']+1
        pd.options.mode.chained_assignment = None  # default='warn'
        all_return_monthly_afterselection = (all_return_monthly
                                                .merge(stocks_filter[['year_month', 'ticker']],
                                                        how="inner",
                                                        left_on=['year_month', 'ticker'],
                                                        right_on=['year_month', 'ticker'])
                                                .merge(sector, on="ticker", how="left")
                                                .sort_values('quantile_mtr', ascending=False)
                                                .groupby(['year_month', 'n_long', 'n_short', 'n_asset', 'Sector'], group_keys=False)
                                                .apply(lambda g: g.head(max_persector), include_groups=True)
                                                .sort_values('quantile_mtr', ascending=False)
                                                .groupby(['year_month', 'n_long', 'n_short', 'n_asset'], group_keys=False)
                                                .apply(lambda g: g.head(final_nmaxasset), include_groups=True))
        
        all_return_monthly_afterselection['model'] = all_return_monthly_afterselection['n_long'].astype(float).astype(str) + "-" + all_return_monthly_afterselection['n_short'].astype(float).astype(str) + "-" + all_return_monthly_afterselection['n_asset'].astype(str)
        all_return_monthly_afterselection_summarised = (
            all_return_monthly_afterselection
            .groupby(['year_month', 'model'], as_index=False)
            .agg(
                dr=('dr', 'mean'),
                n=('dr', 'size')
                )
            .merge(
                index_price[['year_month', 'monthly_return']], 
                on='year_month', 
                how='left'
                    )
            .assign(total_return=lambda x: x['dr'] / (1+x['monthly_return']))
                                                )
        lvl1_bestmodel = []
        for alpha in tqdm(list_alpha , desc="Processing level 1 analysis"):
            for temp in list_temp:
                lvl1_bestmodel_loop = ModelEvaluator.bestmodel(
                    data=all_return_monthly_afterselection_summarised,
                    mode=mode,
                    param_temp=temp,
                    param_alpha=alpha)
                lvl1_bestmodel.append(lvl1_bestmodel_loop)
        
        lvl1_bestmodel = (pd.concat(lvl1_bestmodel, ignore_index=True)
                            .rename(columns={"model": "model_lvl0"})
                            .assign(model_lvl1=lambda x: x['param_alpha'].astype(str) + "-" + x['param_temp'].astype(str))
                            .drop(columns=['score', 'param_alpha', 'param_temp'])
                            )
        
        lvl1_return = (all_return_monthly_afterselection_summarised[['model', 'year_month', 'total_return']]
                        .merge(lvl1_bestmodel,
                                left_on=["model", "year_month"],
                                right_on=["model_lvl0", "year_month"],
                                how="inner")
                        .drop(columns=['model'])
                        .rename(columns={"model_lvl1": "model"})
                        )
        
        lvl2_bestmodel = (ModelEvaluator.bestmodel(data=lvl1_return,
                                mode=mode,
                                param_temp=param_temp_lvl2,
                                param_alpha=param_alpha_lvl2)[['year_month', 'model']]
                            .rename(columns={"model": "model_lvl1"})
                            .merge(lvl1_bestmodel,
                                how="left",
                                on=["year_month", "model_lvl1"])
                            .merge(all_return_monthly_afterselection_summarised[['year_month', 'model', 'dr', 'monthly_return', 'total_return']],
                                how="left",
                                left_on=['year_month', 'model_lvl0'],
                                right_on=['year_month', 'model'])
                            .drop(columns=['model']))
        
        detail_components = (lvl2_bestmodel[['year_month', 'model_lvl0', 'model_lvl1']]
                    .merge(all_return_monthly_afterselection[['year_month', 'ticker', 'model', 'Sector', 'dr']],
                            left_on=['year_month', 'model_lvl0'],
                            right_on=['year_month', 'model'],
                            how='left'))
                    
        lvl2_bestmodel = lvl2_bestmodel.rename(columns={"year_month": "year_month",
                                                        "dr" : "monthly_return",
                                                        "monthly_return" : "monthly_return_index",
                                                        "total_return" : "relative_monthly_return"})
        
        detail_components = detail_components.rename(columns={"year_month": "year_month",
                                                              "dr" : "monthly_return"})
        #detail_components = detail_components.rename(columns={"year_month": "year_month",
        #                                                      "dr" : "monthly_return"})
        # Removed unused computation of last_view_technical and tickers to avoid KeyError when the dataframe is empty.
        lvl2_bestmodel['monthly_return'] = lvl2_bestmodel['monthly_return'].fillna(0)-1
        lvl2_bestmodel['relative_monthly_return'] = lvl2_bestmodel['relative_monthly_return'].fillna(0)-1
        dict_return = {'aggregated' : lvl2_bestmodel,
                'detailed' : detail_components}
        
        return dict_return
    
    @staticmethod
    def sample_space(trial: optuna.Trial):
        if trial is None:
            study = optuna.create_study()
            trial = study.ask()
            
        return {
            "n_long": trial.suggest_int("n_long", 50, 400),
            "n_short": trial.suggest_int("n_short", 1, 100),
            "n_asset": trial.suggest_int("n_asset", 5, 30),
            "n_max_per_sector": trial.suggest_int("n_max_per_sector", 1,2),
            }
    
    @staticmethod
    def _objective_builder(df: pd.DataFrame,
                       index,
                       column_price : str,
                       func_movingaverage,
                       stocks_filter,
                       sector,
                       alpha,
                       temp,
                       mode,
                       backend: Backend = "polars"):
        def objective(trial: optuna.Trial) -> float:
            params = StrategyLearner().sample_space(trial)
            all_return_monthly = StrategyLearner().fiting(
                df = df.copy(),
                column_price = column_price,
                historical_company = index.components,
                n_long = params['n_long'], 
                n_short = params['n_short'], 
                func_movingaverage = func_movingaverage, 
                n_asset = 30,
                backend=backend,
            )
            
            output = StrategyLearner.return_from_training(df_fiting=all_return_monthly.copy(),
                                                        stocks_filter=stocks_filter.copy(),
                                                        sector=sector.copy(),
                                                        index=index,
                                                        alpha=alpha,
                                                        temp=temp,
                                                        mode=mode,
                                                        params=params,
                                                        backend=backend,
                                                        score_only=True)
            
            return output['score']
        return objective

    @staticmethod
    def learning_process_optuna_split(prices,split,index, stocks_filter, sector, func_movingaverage,n_trials, alpha,temp, mode,seed,n_jobs = 1, backend: Backend = "polars"):
       
        prices = prices.copy()
        stocks_filter = stocks_filter.copy()
        sector = sector.copy()
        
        train_prices = prices[prices['year_month'] < split].copy()
        train_prices = train_prices.dropna(subset=['close_vs_index']).sort_values(by=['ticker', 'date'])
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize",
                                    sampler=optuna.samplers.TPESampler(seed=seed))
        objective = StrategyLearner()._objective_builder(df = train_prices.copy(),
                                                         index = index,
                                                         column_price='close_vs_index',
                                                         func_movingaverage=func_movingaverage,
                                                         stocks_filter=stocks_filter.copy(),
                                                         sector=sector,
                                                         alpha=alpha,
                                                         temp=temp,
                                                         mode=mode,
                                                         backend=backend)
                                    
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True,n_jobs = n_jobs)
        #ploting optuna
        #optuna.visualization.plot_optimization_history(study).show()
        #optuna.visualization.plot_slice(study).show()
        #optuna.visualization.plot_param_importances(study).show()
        
        best_params = study.best_params
        full_learning = StrategyLearner().fiting(
                df = prices.copy(),
                column_price = 'close_vs_index',
                historical_company = index.components,
                n_long = best_params['n_long'], 
                n_short = best_params['n_short'], 
                func_movingaverage = func_movingaverage, 
                n_asset = 30,
                backend=backend,
            )
        output = StrategyLearner.return_from_training(df_fiting=full_learning.copy(),
                                                        stocks_filter=stocks_filter.copy(),
                                                        sector=sector.copy(),
                                                        index=index,
                                                        alpha=alpha,
                                                        temp=temp,
                                                        mode=mode,
                                                        params=best_params,
                                                        backend=backend)
        aggregated = output['aggregated']
        detailled = output['detailed']
        aggregated = aggregated[aggregated['year_month'] > split]
        detailled = detailled[detailled['year_month'] > split]
        return {'aggregated': aggregated,
                'detailled': detailled,
                'study': study}   
        
    @staticmethod
    def learning_process_optuna_full(prices,index, first_date,stocks_filter, sector, func_movingaverage,n_trials,alpha,temp,mode,seed,n_jobs : int = 1, backend: Backend = "polars"):
       
        prices = prices.copy()
        stocks_filter = stocks_filter.copy()
        sector = sector.copy()
        prices = prices.dropna(subset=['close_vs_index']).sort_values(by=['ticker', 'date'])
        prices = normalize_year_month_to_period(prices, col='year_month')
        first_date = first_date if isinstance(first_date, pd.Period) else pd.Period(first_date, freq='M')
        
        list_spliting_date_end_of_year = sorted(prices['year_month'].unique())
        list_spliting_date_end_of_year = [d for d in list_spliting_date_end_of_year if d.month == 1]
        list_spliting_date_end_of_year = [d for d in list_spliting_date_end_of_year if d >= first_date]
        
        detailled_portfolio = pd.DataFrame()
        aggregated_portfolio = pd.DataFrame()
        dict_study = {}
        list_spliting_date_end_of_year = sorted(list_spliting_date_end_of_year)
        for date in tqdm(list_spliting_date_end_of_year, desc="Processing full optuna analysis") : 
            
            lpos = StrategyLearner().learning_process_optuna_split(prices=prices.copy(),
                                                                  split=date,
                                                                  index=index,
                                                                  stocks_filter=stocks_filter.copy(),
                                                                  sector=sector.copy(),
                                                                  func_movingaverage=func_movingaverage,
                                                                  n_trials=n_trials,
                                                                  alpha=alpha,
                                                                  temp=temp,
                                                                  mode=mode,
                                                                  seed=seed,
                                                                  n_jobs = n_jobs,
                                                                  backend=backend)
            detailled = lpos['detailled']
            aggregated = lpos['aggregated']
            #print(min(detailled['year_month']), max(detailled['year_month']))
            #print(min(aggregated['year_month']), max(aggregated['year_month']))
            
            min_date = min(detailled['year_month']) if not detailled.empty else pd.Period('2100-01', freq='M')
            if len(detailled_portfolio) > 0:
                detailled_portfolio = detailled_portfolio[detailled_portfolio['year_month'] < min_date] # We will update detailled portfolio so we erase
                aggregated_portfolio = aggregated_portfolio[aggregated_portfolio['year_month'] < min_date] # We will update detailled portfolio so we erase
            
            detailled_portfolio = pd.concat([detailled_portfolio, detailled], ignore_index=True)
            aggregated_portfolio = pd.concat([aggregated_portfolio, aggregated], ignore_index=True)
            dict_study[date] = lpos['study']
        return {'aggregated': aggregated_portfolio,
                'detailled': detailled_portfolio,
                'studies': dict_study}
    

    @staticmethod
    def aggregate_portfolios(
        optuna_outputs: List[Dict[str, Any]],
        mode: str = 'equal',
        index: 'IndexDataManager' = None,
        union_mode: bool = True,
        backend: Backend = "polars",
    ) -> Dict[str, pd.DataFrame]:
        """
        Aggregate multiple optuna portfolio outputs into a single combined portfolio.
        
        Args:
            optuna_outputs: List of optuna output dicts, each containing 'detailled' and 'aggregated' keys
            mode: Weighting mode:
                - 'equal': All stocks weighted equally regardless of how many models picked them
                - 'frequency': Stocks weighted by how many models selected them
            index: Optional IndexDataManager for computing returns vs index
            union_mode: If True (default), stocks present in ANY model are included (Union).
                        If False, only stocks present in ALL models are included (Intersection).
                
        Returns:
            Dict with 'detailed' and 'aggregated' DataFrames
        """
        if not optuna_outputs:
            raise ValueError("optuna_outputs list cannot be empty")
        
        if mode not in ['equal', 'frequency']:
            raise ValueError(f"mode must be 'equal' or 'frequency', got '{mode}'")
        
        backend_name = ensure_backend_name(backend, default="polars")
        n_models = len(optuna_outputs)

        if backend_name == "polars":
            require_polars()
            all_detailed_pl = []
            for i, output in enumerate(optuna_outputs):
                detailed_key = 'detailed' if 'detailed' in output else 'detailled'
                df = output[detailed_key]
                if isinstance(df, pd.DataFrame):
                    pldf = to_polars(normalize_year_month_to_timestamp(df, col='year_month'))
                else:
                    pldf = df.clone()
                    if 'year_month' in pldf.columns:
                        ym_dtype = pldf.schema.get('year_month')
                        if ym_dtype == pl.Date:
                            pass
                        elif ym_dtype == pl.Datetime:
                            pldf = pldf.with_columns(pl.col('year_month').dt.truncate('1mo').alias('year_month'))
                        else:
                            pldf = pldf.with_columns(
                                pl.col('year_month').cast(pl.Utf8).str.strptime(pl.Date, format='%Y-%m-%d', strict=False).alias('year_month')
                            )
                all_detailed_pl.append(pldf.with_columns(pl.lit(i).alias('source_model')))
            pld = pl.concat(all_detailed_pl, how='vertical_relaxed')
            if not union_mode:
                valid_keys = (
                    pld.group_by(['year_month', 'ticker'])
                    .agg(pl.col('source_model').n_unique().alias('_models'))
                    .filter(pl.col('_models') == n_models)
                    .select(['year_month', 'ticker'])
                )
                pld = pld.join(valid_keys, on=['year_month', 'ticker'], how='inner')

            detailed_holdings = (
                pld.group_by(['year_month', 'ticker'])
                .agg(
                    pl.col('dr').mean().alias('dr'),
                    pl.col('source_model').n_unique().alias('n_models'),
                    pl.col('Sector').first().alias('Sector'),
                )
                .with_columns(
                    (pl.lit(1.0) if mode == 'equal' else (pl.col('n_models') / n_models)).alias('weight')
                )
                .with_columns((pl.col('weight') / pl.col('weight').sum().over('year_month')).alias('weight_normalized'))
            )
            perf_data = (
                detailed_holdings
                .filter(pl.col('dr').is_not_null())
                .with_columns((pl.col('weight') / pl.col('weight').sum().over('year_month')).alias('weight_perf'))
                .with_columns((pl.col('dr') * pl.col('weight_perf')).alias('weighted_dr'))
            )
            aggregated = (
                perf_data.group_by('year_month')
                .agg(
                    pl.col('weighted_dr').sum().alias('monthly_return'),
                    pl.col('ticker').count().alias('n'),
                    pl.col('n_models').mean().alias('avg_models_per_stock'),
                )
                .with_columns((pl.col('monthly_return') - 1).alias('monthly_return'))
            )
            if index is not None:
                idx = normalize_year_month_to_timestamp(index.monthly_returns[['year_month', 'monthly_return']], col='year_month')
                idx_pl = to_polars(idx).rename({'monthly_return': 'monthly_return_index'})
                aggregated = (
                    aggregated.join(idx_pl, on='year_month', how='left')
                    .with_columns(((1 + pl.col('monthly_return')) / (1 + pl.col('monthly_return_index'))).alias('monthly_return_vs_index'))
                )
            detailed_output = detailed_holdings.select(['year_month', 'ticker', 'dr', 'n_models', 'Sector', 'weight', 'weight_normalized'])
            detailed_output = detailed_output.with_columns((pl.col('dr') - 1).alias('dr'))
            return {
                'detailed': normalize_year_month_to_period(to_pandas(detailed_output), 'year_month'),
                'aggregated': normalize_year_month_to_period(to_pandas(aggregated), 'year_month'),
            }

        raise ValueError("Pandas backend is disabled for StrategyLearner.aggregate_portfolios.")
    
    @staticmethod
    def get_portfolio_at_month(
        portfolio_output: Dict[str, Any],
        month: Optional[pd.Period] = None
    ) -> pd.DataFrame:
        """
        Get portfolio holdings for a specific month.
        
        Args:
            portfolio_output: Output from learning_process_optuna_full or aggregate_portfolios.
                              Must contain 'detailled' or 'detailed' key.
            month: Target month as pd.Period. If None, uses the latest month.
            
        Returns:
            DataFrame with columns: ticker, Sector, weight, weight_normalized, monthly_return
            Sorted by weight_normalized descending.
        """
        # Get detailed dataframe (handle both spellings)
        if 'detailed' in portfolio_output:
            df = portfolio_output['detailed'].copy()
        elif 'detailled' in portfolio_output:
            df = portfolio_output['detailled'].copy()
        else:
            raise ValueError("portfolio_output must contain 'detailed' or 'detailled' key")
        
        if df.empty:
            raise ValueError("Portfolio is empty")
        
        # Determine target month
        if month is None:
            month = df['year_month'].max()
        
        # Filter to target month
        df_month = df[df['year_month'] == month].copy()
        
        if df_month.empty:
            available = df['year_month'].unique()[:5]
            raise ValueError(f"No data for month {month}. Available months: {list(available)}...")
        
        # Handle weight column - if it doesn't exist, create equal weights
        if 'weight' not in df_month.columns:
            df_month['weight'] = 1.0
        
        # Calculate normalized weights (sum to 1)
        total_weight = df_month['weight'].sum()
        if total_weight > 0:
            df_month['weight_normalized'] = df_month['weight'] / total_weight
        else:
            df_month['weight_normalized'] = 1.0 / len(df_month)
        
        # Get return column (try 'dr' or 'monthly_return')
        if 'dr' in df_month.columns:
            return_col = 'dr'
        elif 'monthly_return' in df_month.columns:
            return_col = 'monthly_return'
        else:
            return_col = None
        
        # Build output
        output_cols = ['ticker']
        if 'Sector' in df_month.columns:
            output_cols.append('Sector')
        output_cols.extend(['weight', 'weight_normalized'])
        if return_col:
            df_month['monthly_return'] = df_month[return_col]
            output_cols.append('monthly_return')
        if 'n_models' in df_month.columns:
            output_cols.append('n_models')
        
        result = df_month[output_cols].copy()
        result = result.sort_values('weight_normalized', ascending=False).reset_index(drop=True)
        
        # Add metadata
        result.attrs['month'] = month
        result.attrs['total_stocks'] = len(result)
        
        return result

    def learning_fundamental(balance, cashflow, income, earnings, general, monthly_return, historical_company, col_learning, earning_choice, list_date_to_maximise_earning_choice, tresh, n_max_sector, list_kpi_toinvert=['pe'], list_kpi_toincrease=['totalrevenue_rolling', 'grossprofit_rolling', 'operatingincome_rolling', 'incomebeforeTax_rolling', 'netincome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freecashflow_rolling', 'epsactual_rolling'], list_ratios_toincrease=['roic', 'netmargin'], list_kpi_toaccelerate=['epsactual_rolling'], list_lag_increase=[1, 4, 4*5], list_ratios_to_augment=["roic_lag4", "roic_lag1", "netmargin_lag4"], list_date_to_maximise=['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']):
        ratios = FundamentalAnalyzer.calculate_fundamental_ratios(balance=balance,
                                                                  cashflow=cashflow,
                                                                  income=income,
                                                                  earnings=earnings,
                                                                  list_kpi_toincrease=list_kpi_toincrease,
                                                                  list_ratios_toincrease=list_ratios_toincrease,
                                                                  list_kpi_toaccelerate=list_kpi_toaccelerate,
                                                                  list_lag_increase=list_lag_increase,
                                                                  list_ratios_to_augment=list_ratios_to_augment,
                                                                  list_date_to_maximise=list_date_to_maximise)
        pe = FundamentalAnalyzer.calculate_pe_ratios(balance=balance, 
                                                     earnings=earnings, 
                                                     cashflow=cashflow,
                                                     income=income, 
                                                     earning_choice=earning_choice,
                                                     monthly_return=monthly_return,
                                                     list_date_to_maximise=list_date_to_maximise_earning_choice)
        ratios['year_month'] = ratios['date'].dt.to_period('M')
        final_merged = []
        list_date_loop = sorted(ratios[ratios['year_month'] >= '2000-01'].dropna(subset=['year_month'])['year_month'].unique())
        list_date_loop.append(max(list_date_loop) + 1)
        for date_loop in list_date_loop:
            if not(pd.isna(date_loop)):
                historical_company_loop = historical_company[historical_company['year_month'] < date_loop]
                historical_company_loop = historical_company_loop[historical_company_loop['year_month'] == historical_company_loop['year_month'].max()]['ticker'].unique()
                
                ratios_loop = ratios[ratios['year_month'] < date_loop]
                ratios_loop = ratios_loop[(ratios_loop['date'] == ratios_loop.groupby('ticker')['date'].transform('max')) & 
                                          (ratios_loop['ticker'].isin(historical_company_loop))]
                ratios_loop['date_diff_ratios'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(ratios_loop['date'])).dt.days
                
                pe_loop = pe[pe['year_month'] < date_loop]
                pe_loop = pe_loop[(pe_loop['year_month'] == pe_loop.groupby('ticker')['year_month'].transform('max')) & 
                                          (pe_loop['ticker'].isin(historical_company_loop))]
                pe_loop['date_diff_pe'] = ((date_loop-1).to_timestamp(how='end') - pd.to_datetime(pe_loop['year_month'].dt.to_timestamp(how='end'))).dt.days
                
                merge_loop = pe_loop.merge(ratios_loop, on='ticker')
                for kpi_toinvert in list_kpi_toinvert:
                    merge_loop[f"{kpi_toinvert}_inverted"] = 1/(merge_loop[kpi_toinvert]+0.00001)

                merge_loop = merge_loop[['ticker']+col_learning]
                for c in col_learning:
                    merge_loop[f"{c}_quantile"] = ModelEvaluator.ranking(merge_loop[c], tresh)
                merge_loop['rank'] = merge_loop.filter(regex='_quantile$').prod(axis=1)
                merge_loop = (
                    merge_loop.merge(general, on='ticker')
                    .sort_values(by='rank', ascending=False)
                    .assign(one=1)
                    .assign(one=lambda x: x.groupby('Sector')['one'].cumsum())
                    .loc[lambda x: x['one'] <= n_max_sector]
                    .drop(columns='one')
                    .loc[lambda x: x['rank'] > 0]  
                    .assign(year_month=date_loop)
                    )
                final_merged.append(merge_loop)

        final_result = pd.concat(final_merged, ignore_index=True)
        
        return_model = final_result.merge(monthly_return,
                           how='left',
                           on=['ticker', 'year_month'])
        
        result_summarised = return_model.groupby(['year_month']).agg(
                                monthly_return=('monthly_return', 'mean'),
                                n=('monthly_return', 'count')).reset_index()
        
        dict_return = {'aggregated' : result_summarised,
                'detailed' : return_model}

        return dict_return

    @staticmethod
    def return_benchmark(self, prices, historical_company, index_price, stocks_filter, sector):
        prices_dr = (prices
                       .sort_values('year_month')
                       .groupby(['ticker'])
                       .apply(lambda x: x.assign(dr=x['close'] / x['close'].shift(1)), include_groups=False)
                       .reset_index()
                       )
        benchmark_base = (prices_dr
                  .merge(historical_company,
                         how="inner",
                         on=['year_month', 'ticker'])
                  .groupby(['ticker', 'year_month'])
                  .agg({'dr': 'prod'})
                  .reset_index()
                  .groupby(['year_month'])
                  .agg({'dr': 'mean'})
                  .reset_index()
                  .assign(model='base')
                 )
        
        benchmark_after_selection = (prices_dr
                                              .merge(stocks_filter[['year_month', 'ticker']],
                                                     how="inner",
                                                     left_on=['year_month', 'ticker'],
                                                     right_on=['year_month', 'ticker'])
                                              .merge(sector, on="ticker", how="left")
                                              .groupby(['ticker', 'year_month'])
                                              .agg({'dr': 'prod'})
                    
                                              .reset_index()
                                              .groupby(['year_month'])
                                              .agg({'dr': 'mean'})
                                              .reset_index()
                                              .assign(model='after_selection')
                                             )
        bench = pd.concat([benchmark_base,
                           benchmark_after_selection,
                           (index_price
                            .rename(columns={"dr_sp500": "dr"})
                            .assign(model='index'))])

        return bench

    @staticmethod
    def scoping_fundamental(self, balance, cashflow, income, earnings, list_kpi_toincrease=['totalrevenue_rolling', 'grossprofit_rolling', 'operatingincome_rolling', 'incomebeforeTax_rolling', 'netincome_rolling', 'ebit_rolling', 'ebitda_rolling', 'freecashflow_rolling', 'epsactual_rolling'], list_ratios_toincrease=['roic', 'netmargin'], list_kpi_toaccelerate=['epsactual_rolling'], list_lag_increase=[1, 4, 4*5], list_ratios_to_augment=["roic_lag4", "roic_lag1", "netmargin_lag4"], list_date_to_maximise=['filing_date_income', 'filing_date_cash', 'filing_date_balance', 'filing_date_earning']):
        balance_clean = (
            balance[['ticker', 'date', 'filing_date', 'commonstocksharesoutstanding', 'totalstockholderequity', 'netdebt']]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('q'),
                filing_date_balance=lambda x: pd.to_datetime(x['filing_date'])
            )
            .sort_values('filing_date_balance')
            .groupby(['ticker', 'quarter_end'])
            .last()
            .reset_index()
            .drop(columns=['filing_date'])
        )
        for columns in ['totalstockholderequity', 'netdebt', 'commonstocksharesoutstanding']:
            balance_clean[f"{columns}_rolling"] = balance_clean.sort_values(['ticker', 'filing_date_balance']).groupby('ticker')[columns].transform(lambda x: TechnicalIndicators.custom_sma(x, n=4))

        earnings_clean = (
            earnings[['ticker', 'date', 'reportdate', 'epsactual']]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('q'),
                filing_date_earning=lambda x: pd.to_datetime(x['reportdate'])
            )
            .sort_values('filing_date_earning')
            .groupby(['ticker', 'quarter_end'])
            .last()
            .reset_index()
            .drop(columns=['reportdate'])
            .dropna(subset=['epsactual'])
        )
        earnings_clean['epsactual_rolling'] = earnings_clean.sort_values(['ticker', 'filing_date_earning']).groupby('ticker')['epsactual'] \
                                                .transform(lambda x: 4 * TechnicalIndicators.custom_sma(x, n=4))

        columns_to_annualise_income = ['totalrevenue', 'grossprofit', 'operatingincome', 
                           'incomebeforeTax', 'netincome', 'ebit', 'ebitda']
         
        income_clean = (
             income[['ticker', 'date', 'filing_date', 'totalrevenue', 'grossprofit', 'operatingincome', 'incomebeforeTax', 'netincome', 'ebit', 'ebitda']]
             .assign(
                 quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('q'),
                 filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
             .sort_values('filing_date_income')
             .groupby(['ticker', 'quarter_end'])
             .last()
             .reset_index()
             .drop(columns=['filing_date'])
         )
        for columns in columns_to_annualise_income:
            income_clean[f"{columns}_rolling"] = income_clean.sort_values(['ticker', 'filing_date_income']).groupby('ticker')[columns].transform(lambda x: 4 * TechnicalIndicators.custom_sma(x, n=4))
                                              
        cash_clean = (
             cashflow[['ticker', 'date', 'filing_date', 'freecashflow']]
             .assign(
                 quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('q'),
                 filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
             .sort_values('filing_date_cash')
             .groupby(['ticker', 'quarter_end'])
             .last()
             .reset_index()
             .drop(columns=['filing_date'])
         )
        
        for columns in ['freecashflow']:
            cash_clean[f"{columns}_rolling"] = cash_clean.sort_values(['filing_date_cash']).groupby('ticker')[columns].transform(lambda x: 4 * TechnicalIndicators.custom_sma(x, n=4))                                        

        funda = (income_clean
                 .merge(cash_clean, on=['ticker', 'quarter_end'], how='outer')
                 .merge(balance_clean, on=['ticker', 'quarter_end'], how='outer')
                 .merge(earnings_clean[['ticker', 'quarter_end', 'filing_date_earning', 'epsactual', 'epsactual_rolling']], on=['ticker', 'quarter_end'], how='outer')
                 .assign(netmargin=lambda x: x['ebit_rolling'] / x['totalrevenue_rolling'])
                 .assign(roic=lambda x: x['ebit_rolling'] / (x['totalstockholderequity_rolling'] + x['netdebt_rolling'].fillna(0)))
                 .assign(ebitpershare_rolling=lambda x: x['ebit_rolling'] / (x['commonstocksharesoutstanding_rolling'].fillna(0)))
                 .assign(ebitdapershare_rolling=lambda x: x['ebitda_rolling'] / (x['commonstocksharesoutstanding_rolling'].fillna(0)))
                 .assign(netincomepershare_rolling=lambda x: x['netincome_rolling'] / (x['commonstocksharesoutstanding_rolling'].fillna(0)))
                 .assign(fcfpershare_rolling=lambda x: x['freecashflow_rolling'] / (x['commonstocksharesoutstanding_rolling'].fillna(0)))
                 )
        view = funda[['date', 'ticker', 'epsactual_rolling',
                      'ebitpershare_rolling', 'ebitdapershare_rolling',
                      'netincomepershare_rolling', 'fcfpershare_rolling']]
        return view




# %%
class ModelEvaluator:
    """Sélection du meilleur modèle et métriques de performance."""
    @staticmethod
    def score(x, alpha):
        x = x - 1
        sco = np.log(1 + alpha * x)
        sco = np.where(np.isnan(sco), -np.inf, sco)
        return sco

    @staticmethod
    def ranking(values, thresh):
        quantiles = stats.rankdata(values.fillna(values.min() - 1), method="average") / len(values)
        quantiles = np.where(quantiles == 0, -np.inf, quantiles)
        ranked_values = quantiles * (np.maximum(0, quantiles - thresh) > 0).astype(int)
        return ranked_values

    @staticmethod
    def best_model_date(data, mode: str, param_alpha: float, param_temp: int, backend: Backend = "polars"):
            backend_name = ensure_backend_name(backend, default="polars")
            if backend_name != "polars":
                raise ValueError("Pandas backend is disabled for ModelEvaluator.best_model_date.")

            require_polars()
            pld = data.clone() if isinstance(data, pl.DataFrame) else to_polars(data)

            if pld.height == 0:
                return pl.DataFrame({
                    'model': ['none'],
                    'score': [float('-inf')],
                    'param_alpha': [param_alpha],
                    'param_temp': [param_temp],
                })

            if 'year_month' in pld.columns:
                pld = pld.sort(['model', 'year_month'], descending=[False, True])
            else:
                pld = pld.sort(['model'])

            rows = []
            for key, g in pld.group_by('model', maintain_order=True):
                model = key[0] if isinstance(key, tuple) else key
                score_value = TechnicalIndicators.decreasing_sum(
                    g['score'].to_numpy(),
                    halfPeriod=param_temp,
                    mode=mode,
                )
                rows.append({'model': model, 'score': float(score_value)})

            if not rows:
                return pl.DataFrame({
                    'model': ['none'],
                    'score': [float('-inf')],
                    'param_alpha': [param_alpha],
                    'param_temp': [param_temp],
                })

            summarized_data = pl.DataFrame(rows)
            max_score = summarized_data.select(pl.col('score').max()).item()
            if max_score is None or np.isnan(max_score):
                return pl.DataFrame({
                    'model': ['none'],
                    'score': [float('-inf')],
                    'param_alpha': [param_alpha],
                    'param_temp': [param_temp],
                })

            return (
                summarized_data
                .filter(pl.col('score') == max_score)
                .sort('model')
                .head(1)
                .with_columns(
                    pl.lit(param_alpha).alias('param_alpha'),
                    pl.lit(param_temp).alias('param_temp'),
                )
            )
    @staticmethod
    def bestmodel(data, mode, param_temp, param_alpha):
        
        data['score'] = ModelEvaluator.score(data['total_return'], alpha=param_alpha)
        def best_model_date(data, date, mode, param_alpha, param_temp):
            filtered_data = data[data['year_month'] < date]
            summarized_data = (filtered_data
                    .sort_values(by='year_month', ascending=False) 
                    .groupby(['model'], group_keys=False)
                    .apply(lambda x: TechnicalIndicators.decreasing_sum(x['score'], halfPeriod=param_temp, mode=mode), include_groups=False)
                    .rename('score')
                    .reset_index(name='score') 
                    )
            best_model = (summarized_data[summarized_data['score'] == summarized_data['score'].max()].sample(1)
                          .assign(
                                  year_month=date,
                                  param_alpha=param_alpha,
                                  param_temp=param_temp
                                  )
                          )
            return best_model
        
        results = []
        data = data.sort_values(['year_month'])
        list_date = data['year_month'].unique().tolist()
        list_date.append(pd.to_datetime(datetime.now()).to_period('M') + 1)
        list_date = list_date[1:]
        for date_loop in list_date:
            results_loop = best_model_date(data=data,
                                           date=date_loop,
                                           mode=mode,
                                           param_temp=param_temp,
                                           param_alpha=param_alpha)
            results.append(results_loop)
        
        results = pd.concat(results, ignore_index=True) 
        return results

    @staticmethod
    def compare_models(models_data, start_year=None, end_year=None, risk_free_rate=0.02):
        from alpharank.strategy.analytics import PerformanceAnalyzer
        
        processed_data = {}
        # Use None to track min/max dates dynamically
        global_min_date = None
        global_max_date = None
        
        # 1. Process and Filter Data
        for model_name, df in models_data.items():
            df_copy = df.copy()
            # Logic to find return column
            return_cols = [col for col in df_copy.columns if 'return' in col.lower()]
            if return_cols:
                return_col = return_cols[0]
            else:
                if 'monthly_return' in df_copy.columns:
                    return_col = 'monthly_return'
                elif len(df_copy.columns) > 1:
                    return_col = df_copy.columns[1]
                else:
                    print(f"Warning: Could not identify returns column for {model_name}. Skipping.")
                    continue
            
            # Ensure Period Index
            if not isinstance(df_copy['year_month'].iloc[0], pd.Period):
                try:
                    df_copy['year_month'] = df_copy['year_month'].dt.to_period('M')
                except:
                    df_copy['year_month'] = pd.to_datetime(df_copy['year_month']).dt.to_period('M')
            
            # Filter by date
            if start_year:
                df_copy = df_copy[df_copy['year_month'].dt.year >= start_year]
            if end_year:
                df_copy = df_copy[df_copy['year_month'].dt.year <= end_year]
                
            if df_copy.empty:
                print(f"Warning: No data available for {model_name} in selected period")
                continue
                
            # Update global date range
            model_min = df_copy['year_month'].min()
            model_max = df_copy['year_month'].max()
            
            if global_min_date is None or model_min < global_min_date:
                global_min_date = model_min
            if global_max_date is None or model_max > global_max_date:
                global_max_date = model_max
            
            # Store Series (indexed by Period)
            processed_data[model_name] = df_copy.set_index('year_month')[return_col]
        
        # 2. Create Aligned DataFrame
        all_returns = pd.DataFrame(processed_data)
        all_returns = all_returns.sort_index()
        
        # 3. Calculate Global Metrics via Analyzer
        metrics = {}
        for model in processed_data:
             series = processed_data[model]
             if series.empty: continue
             
             m_dict = PerformanceAnalyzer.calculate_metrics(series, risk_free_rate)
             
             # Add specific metrics requiring original data (like avg stocks)
             n_stocks_avg = None
             if model in models_data: 
                orig_df = models_data[model]
                if 'n' in orig_df.columns:
                     n_stocks_avg = orig_df['n'].mean()
             m_dict['Number of Stocks (Avg)'] = n_stocks_avg
             
             # Start/End dates
             m_dict['Start Date'] = series.index.min().strftime('%y-%m')
             m_dict['End Date'] = series.index.max().strftime('%y-%m')
             
             # Add windowed CAGR which isn't in standard single-series calculator by default or requires extra logic
             # We can just add it here manually or extend Analyzer. 
             # For now, let's keep it simple and use logic similar to before, but verify dates.
             total_years = (series.index.max() - series.index.min()).n / 12
             for yr in [3, 5, 10]:
                 if total_years >= yr:
                     # Get last N years
                     subset = series.iloc[-12*yr:]
                     m_dict[f'CAGR ({yr}Y)'] = (1 + subset).prod() ** (1/yr) - 1
                 else:
                     m_dict[f'CAGR ({yr}Y)'] = None

             metrics[model] = m_dict

        metrics_df = pd.DataFrame(metrics).T
        
        # Format Metrics DF for display
        pct_cols = ['Total Return', 'CAGR', 'Monthly Mean', 'Monthly Volatility', 
                    'Annualized Volatility', 'Max Drawdown', 'Positive Periods %',
                    'CAGR (3Y)', 'CAGR (5Y)', 'CAGR (10Y)']
        for col in pct_cols:
            if col in metrics_df.columns:
                 metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        
        float_cols = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']
        for col in float_cols:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2f}" if pd.notnull(x) and not np.isinf(x) else "N/A")

        # 4. Generate Visualization Data (CRITICAL: Convert Periods to Timestamps)
        cumulative_returns = PerformanceAnalyzer.calculate_cumulative_returns(all_returns, fill_missing=True)
        # Convert index to timestamp for Plotly
        cumulative_returns.index = cumulative_returns.index.to_timestamp()
        
        drawdowns_df = PerformanceAnalyzer.calculate_drawdowns(cumulative_returns)
        # Index is already timestamp from cumulative_returns
        
        annual_returns_df = PerformanceAnalyzer.get_annual_returns(all_returns).T
        # Annual returns index is integers (years), which is fine for Plotly
        
        correlation_matrix = all_returns.corr()
        
        # Comprehensive Metrics Grids
        # 1. Cumulative from Start Year (e.g. 2010->End, 2011->End)
        cumulative_metrics_dict = PerformanceAnalyzer.calculate_metrics_by_start_year(all_returns, risk_free_rate)
        
        # 2. Discrete Annual Metrics (e.g. 2010, 2011)
        annual_metrics_dict = PerformanceAnalyzer.calculate_annual_metrics(all_returns, risk_free_rate)
        
        # Worst Periods
        worst_periods_df = PerformanceAnalyzer.calculate_worst_periods(all_returns)

        # Monthly Returns Dict - Convert each series index to timestamp
        monthly_returns_dict = {}
        for col in all_returns.columns:
            s_clean = all_returns[col].dropna()
            # Convert index
            s_clean.index = s_clean.index.to_timestamp()
            monthly_returns_dict[col] = s_clean
            
        return metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, drawdowns_df, annual_returns_df, cumulative_metrics_dict, annual_metrics_dict, monthly_returns_dict

    @staticmethod
    def calculate_cagr_by_year(returns_df):
        years = sorted(set(returns_df.index.year))
        cagr_results = {}
        for model in returns_df.columns:
            model_cagr = {}
            for start_year in years:
                filtered_returns = returns_df.loc[returns_df.index.year >= start_year, model]
                if len(filtered_returns) < 12:
                    model_cagr[start_year] = np.nan
                    continue
                total_return = (1 + filtered_returns).prod() - 1
                years_count = len(filtered_returns) / 12
                cagr = (1 + total_return) ** (1 / years_count) - 1
                model_cagr[start_year] = cagr
            cagr_results[model] = model_cagr
        cagr_df = pd.DataFrame(cagr_results)
        cagr_df = cagr_df.dropna(how='all')
        return cagr_df


# %%
