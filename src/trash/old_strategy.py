import pandas as pd
import numpy as np
#from .data_preprocessing import DataPreprocessor
#from .fundamentals import FundamentalAnalyzer
from ..data_processor import IndexDataManager, PricesDataPreprocessor, FundamentalProcessor, TechnicalIndicators
import datetime
from scipy import stats
from tqdm import tqdm
from datetime import *
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import optuna
import seaborn as sns
import warnings
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
    def fiting(df, column_price, historical_company, n_long, n_short, func_movingaverage, n_asset):
        df = df.copy()
        df_save = df.copy()
        df['year_month'] = df['date'].dt.to_period('M')

        df = df.sort_values(['ticker', 'date'])
        for n, col in [(n_short, 'short_movingaverage'), (n_long, 'long_movingaverage')]:
            df[col] = (
                df.groupby('ticker', sort=False)[column_price]
                .transform(lambda x: func_movingaverage(x, n=n))
            )

        # Supprime les n premiers points par ticker sans apply
        df = df[df.groupby('ticker').cumcount() >= n_long]

        # Nettoyage et ratio
        df = df.dropna(subset=['short_movingaverage', 'long_movingaverage'])
        df['mtr'] = df['short_movingaverage'] / df['long_movingaverage']

        # Jointure sur historique
        df = df.merge(historical_company, on=['year_month', 'ticker'], how='inner')

        # Dernière date par (year_month, ticker)
        idx = df.groupby(['year_month', 'ticker'])['date'].idxmax()
        df = df.loc[idx].reset_index(drop=True)

        # Moyenne et écart-type groupés → merge
        stats_df = (
            df.groupby('year_month', as_index=False)['mtr']
            .agg(mean='mean', ecart='std')
        )
        df = df.merge(stats_df, on='year_month', how='left')
        df['quantile_mtr'] = stats.norm.cdf(df['mtr'], df['mean'], df['ecart'])

        # Sélection top actifs
        df = (
            df.sort_values(['year_month', 'mtr'], ascending=[True, False])
            .groupby('year_month')
            .head(n_asset)
        )[['year_month', 'date', 'ticker', 'mtr', 'quantile_mtr']]

        df['n_long'] = n_long
        df['n_short'] = n_short
        df['n_asset'] = n_asset
        df['year_month'] = df['year_month'] + 1

        # Calcul du rendement mensuel
        df_return = (
            df_save.groupby(['year_month', 'ticker'], as_index=False)['dr']
            .prod()
            .sort_values(['ticker', 'year_month'])
        )[['year_month', 'ticker', 'dr']]

        # Merge final
        final_return = df.merge(
            df_return,
            on=['year_month', 'ticker'],
            how='left'
        )

        return final_return
    
    @staticmethod
    def return_from_training(df_fiting,stocks_filter,sector, index, alpha, temp, mode,params) : 
        stocks_filter = stocks_filter.copy()
        df_fiting = df_fiting.copy()
        sector = sector.copy()
        stocks_filter['year_month'] = stocks_filter['year_month']+1
        
        all_return_monthly_afterselection = (df_fiting
                                            .merge(stocks_filter[['year_month', 'ticker']],
                                                    how="inner",
                                                    left_on=['year_month', 'ticker'],
                                                    right_on=['year_month', 'ticker'])
                                            .merge(sector[['ticker','Sector']], on="ticker", how="left")  # only needed cols to keep year_month
                                            .sort_values('quantile_mtr', ascending=False)
                                            .groupby(['year_month', 'n_long', 'n_short', 'n_asset', 'Sector'], group_keys=False)
                                            .apply(lambda g: g.head(params['n_max_per_sector']))
                                            .sort_values('quantile_mtr', ascending=False)
                                            .groupby(['year_month', 'n_long', 'n_short', 'n_asset'], group_keys=False)
                                            .apply(lambda g: g.head(params["n_asset"])))
    
        all_return_monthly_afterselection['model'] = all_return_monthly_afterselection['n_long'].astype(float).astype(str) + "-" + all_return_monthly_afterselection['n_short'].astype(float).astype(str) + "-" + all_return_monthly_afterselection['n_asset'].astype(str)
        all_return_monthly_afterselection_summarised = (
            all_return_monthly_afterselection
            .groupby(['year_month', 'model'], as_index=False)
            .agg(
                dr=('dr', 'mean'),
                n=('dr', 'size')
                )
            .merge(
                index.monthly_returns[['year_month', 'monthly_return']], 
                on='year_month', 
                how='left'
                    )
            .assign(monthly_return_vs_index=lambda x: x['dr'] / (1+x['monthly_return']),
                    dr=lambda x: x['dr'] -1 )
            .rename(columns={"monthly_return": "monthly_return_index",
                             "dr" : "monthly_return"}))
    
        
        all_return_monthly_afterselection_summarised = all_return_monthly_afterselection_summarised.dropna(subset=['monthly_return_vs_index'])
        
        all_return_monthly_afterselection_summarised['score'] = ModelEvaluator.score(all_return_monthly_afterselection_summarised['monthly_return_vs_index'], alpha = alpha)
        
        lvl1_bestmodel_loop = ModelEvaluator.best_model_date(
            data=all_return_monthly_afterselection_summarised,
            mode=mode,
            param_temp=temp,
            param_alpha=alpha)
                
        dict_return = {'aggregated' : all_return_monthly_afterselection_summarised,
                        'detailed' : all_return_monthly_afterselection,
                        'score' : lvl1_bestmodel_loop['score'].values[0]}        
        return dict_return
    
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
    
    
    def sample_space(trial: optuna.Trial):
        if trial is None:
            study = optuna.create_study()
            trial = study.ask()
            
        return {
            "n_long": trial.suggest_int("n_long", 50, 400),
            "n_short": trial.suggest_int("n_short", 1, 100),
            "n_asset": trial.suggest_int("n_asset", 10, 30),
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
                       mode):
        def objective(trial: optuna.Trial) -> float:
            params = StrategyLearner().sample_space(trial)
            all_return_monthly = StrategyLearner().fiting(
                df = df.copy(),
                column_price = column_price,
                historical_company = index.components,
                n_long = params['n_long'], 
                n_short = params['n_short'], 
                func_movingaverage = func_movingaverage, 
                n_asset = 30
            )
            
            output = StrategyLearner.return_from_training(df_fiting=all_return_monthly.copy(),
                                                        stocks_filter=stocks_filter.copy(),
                                                        sector=sector.copy(),
                                                        index=index,
                                                        alpha=alpha,
                                                        temp=temp,
                                                        mode=mode,
                                                        params=params)
            
            return output['score']
        return objective

    @staticmethod
    def learning_process_optuna_split(prices,split,index, stocks_filter, sector, func_movingaverage,n_trials, alpha,temp, mode,seed,n_jobs = 1):
       
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
                                                         mode=mode)
                                    
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
                n_asset = 30
            )
        output = StrategyLearner.return_from_training(df_fiting=full_learning.copy(),
                                                        stocks_filter=stocks_filter.copy(),
                                                        sector=sector.copy(),
                                                        index=index,
                                                        alpha=alpha,
                                                        temp=temp,
                                                        mode=mode,
                                                        params=best_params)
        aggregated = output['aggregated']
        detailled = output['detailed']
        aggregated = aggregated[aggregated['year_month'] > split]
        detailled = detailled[detailled['year_month'] > split]
        return {'aggregated': aggregated,
                'detailled': detailled,
                'study': study}   
        
    @staticmethod
    def learning_process_optuna_full(prices,index, first_date,stocks_filter, sector, func_movingaverage,n_trials,alpha,temp,mode,seed,n_jobs : int = 1):
       
        prices = prices.copy()
        stocks_filter = stocks_filter.copy()
        sector = sector.copy()
        prices = prices.dropna(subset=['close_vs_index']).sort_values(by=['ticker', 'date'])
        
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
                                                                  n_jobs = n_jobs)
            detailled = lpos['detailled']
            aggregated = lpos['aggregated']
            print(min(detailled['year_month']), max(detailled['year_month']))
            print(min(aggregated['year_month']), max(aggregated['year_month']))
            
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
    def best_model_date(data : pd.DataFrame, mode : str, param_alpha : float, param_temp : int):
            data = data.copy()
            summarized_data = (
            data.sort_values(by='year_month', ascending=False)
                .groupby('model', group_keys=False)
                .apply(
                    lambda x: TechnicalIndicators.decreasing_sum(
                    x['score'], halfPeriod=param_temp, mode=mode
                    ),include_groups=False
                        )
                .to_frame('score')       # transforme la Series en DataFrame avec une colonne "score"
                .reset_index()           # remet "model" comme colonne normale
                    )
            best_model = (summarized_data[summarized_data['score'] == summarized_data['score'].max()].sample(1)
                          .assign(
                                  param_alpha=param_alpha,
                                  param_temp=param_temp
                                  )
                          )
            return best_model
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
        processed_data = {}
        min_date = pd.Period('2000-01', freq='M')
        max_date = pd.Period('2040-01', freq='M')
        
        for model_name, df in models_data.items():
            df_copy = df.copy()
            return_cols = [col for col in df_copy.columns if 'return' in col.lower()]
            if return_cols:
                return_col = return_cols[0]
            else:
                if 'monthly_return' in df_copy.columns:
                    return_col = 'monthly_return'
                elif len(df_copy.columns) > 1:
                    return_col = df_copy.columns[1]
                else:
                    raise ValueError(f"Could not identify returns column for model {model_name}")
            
            if not isinstance(df_copy['year_month'].iloc[0], pd.Period):
                try:
                    df_copy['year_month'] = df_copy['year_month'].dt.to_period('m')
                except:
                    df_copy['year_month'] = pd.to_datetime(df_copy['year_month']).dt.to_period('m')
            
            if start_year:
                df_copy = df_copy[df_copy['year_month'].dt.year >= start_year]
            if end_year:
                df_copy = df_copy[df_copy['year_month'].dt.year <= end_year]
                
            if df_copy.empty:
                print(f"Warning: No data available for {model_name} in selected time period")
                continue
                
            min_date = min(min_date, df_copy['year_month'].min())
            max_date = max(max_date, df_copy['year_month'].max())
            
            returns_series = df_copy.set_index('year_month')[return_col]
            processed_data[model_name] = returns_series
        
        all_returns = pd.DataFrame(processed_data)
        correlation_matrix = all_returns.corr()
        all_returns_ts = all_returns.copy()
        all_returns_ts.index = all_returns_ts.index.to_timestamp()
        all_returns_ts.sort_index(inplace=True)
        cumulative_returns = (all_returns_ts + 1).cumprod()
        worst_periods = {}
        for model in all_returns.columns:
            model_returns = all_returns[model].reset_index()
            model_returns['year'] = model_returns['year_month'].dt.year
            model_returns['year_month'] = model_returns['year_month'].dt.month
            worst_month_idx = model_returns[model].idxmin()
            worst_month = model_returns.loc[worst_month_idx]
            worst_month_date = worst_month['year_month']
            worst_month_return = worst_month[model]
            annual_returns = model_returns.groupby('year')[model].apply(
                lambda x: np.prod(1 + x) - 1
            )
            worst_year_idx = annual_returns.idxmin()
            worst_year_return = annual_returns.loc[worst_year_idx]
            worst_periods[model] = {
                'Worst Month': f"{worst_month_date}: {worst_month_return:.2%}",
                'Worst Year': f"{worst_year_idx}: {worst_year_return:.2%}"
            }
        
        worst_periods_df = pd.DataFrame(worst_periods).T
        annual_returns_data = {}
        for model in all_returns.columns:
            model_returns = all_returns[model].reset_index()
            model_returns['year'] = model_returns['year_month'].dt.year
            annual_returns = model_returns.groupby('year')[model].apply(
                lambda x: np.prod(1 + x) - 1
            )
            annual_returns_data[model] = annual_returns
        
        annual_returns_df = pd.DataFrame(annual_returns_data)
        metrics = {}
        for model in all_returns.columns:
            model_returns = all_returns[model].dropna()
            model_returns_ts = all_returns_ts[model].dropna()
            if len(model_returns) < 12:
                print(f"Warning: Insufficient data for {model}")
                continue
                
            total_months = len(model_returns)
            total_years = total_months / 12
            total_return = cumulative_returns[model].iloc[-1] - 1
            annualized_return = (1 + total_return) ** (1 / total_years) - 1
            monthly_mean = model_returns.mean()
            monthly_std = model_returns.std()
            annualized_vol = monthly_std * np.sqrt(12)
            sharpe = (annualized_return - risk_free_rate) / annualized_vol
            rolling_max = cumulative_returns[model].cummax()
            drawdown = (cumulative_returns[model] / rolling_max - 1)
            max_drawdown = drawdown.min()
            positive_months = (model_returns > 0).sum() / total_months
            cagr_3yr = None
            cagr_5yr = None
            cagr_10yr = None
            if total_years >= 3:
                returns_3yr = model_returns_ts.iloc[-36:]
                cagr_3yr = (1 + returns_3yr).prod() ** (1/3) - 1
            if total_years >= 5:
                returns_5yr = model_returns_ts.iloc[-60:]
                cagr_5yr = (1 + returns_5yr).prod() ** (1/5) - 1
            if total_years >= 10:
                returns_10yr = model_returns_ts.iloc[-120:]
                cagr_10yr = (1 + returns_10yr).prod() ** (1/10) - 1
            metrics[model] = {
                'Start Date': min_date.strftime('%y-%m'),
                'End Date': max_date.strftime('%y-%m'),
                'Total Return': total_return,
                'CAGR': annualized_return,
                'CAGR (3Y)': cagr_3yr,
                'CAGR (5Y)': cagr_5yr,
                'CAGR (10Y)': cagr_10yr,
                'Monthly Mean': monthly_mean,
                'Monthly Volatility': monthly_std,
                'Annualized Volatility': annualized_vol,
                'Sharpe Ratio': sharpe,
                'Max Drawdown': max_drawdown,
                'Positive Months %': positive_months,
                'Number of Stocks (Avg)': models_data[model]['n'].mean() if 'n' in models_data[model].columns else None
            }
        
        metrics_df = pd.DataFrame(metrics).T
        for col in ['Total Return', 'CAGR', 'CAGR (3Y)', 'CAGR (5Y)', 'CAGR (10Y)', 
                    'Monthly Mean', 'Monthly Volatility', 'Annualized Volatility', 
                    'Max Drawdown', 'Positive Months %']:
            if col in metrics_df.columns:
                metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
        
        if 'Sharpe Ratio' in metrics_df.columns:
            metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")
        
        cagr_by_year = ModelEvaluator.calculate_cagr_by_year(all_returns)
        fig = plt.figure(figsize=(20, 16))
        plt.subplots_adjust(wspace=0.3, hspace=0.3)
        ax1 = plt.subplot(2, 2, 1)
        cumulative_returns.plot(ax=ax1)
        ax1.set_title('Cumulative Returns', fontsize=16)
        ax1.set_ylabel('Value of $1 Investment', fontsize=12)
        ax1.grid(True)
        ax1.set_yscale('log')
        ax1.legend(fontsize=10)
        ax2 = plt.subplot(2, 2, 2)
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    mask=mask, vmin=-1, vmax=1, ax=ax2)
        ax2.set_title('Return Correlation Matrix', fontsize=16)
        ax3 = plt.subplot(2, 2, 3)
        sns.heatmap(cagr_by_year, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax3)
        ax3.set_title('CAGR by Start Year', fontsize=16)
        ax3.set_ylabel('Start Year', fontsize=12)
        ax3.set_xlabel('Model', fontsize=12)
        ax4 = plt.subplot(2, 2, 4)
        sns.heatmap(annual_returns_df, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax4)
        ax4.set_title('Annual Returns by Year', fontsize=16)
        ax4.set_ylabel('Year', fontsize=12)
        ax4.set_xlabel('Model', fontsize=12)
        plt.tight_layout()
        individual_heatmaps = {}
        for model in all_returns.columns:
            fig_heatmap = ModelEvaluator.plot_monthly_returns_heatmap(all_returns[model], model)
            individual_heatmaps[model] = fig_heatmap
        
        figures = {
            'Main Figure': fig,
            'Monthly Heatmaps': individual_heatmaps
        }
        return metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, figures

    @staticmethod
    def plot_monthly_returns_heatmap(returns_series, model_name):
        df = returns_series.reset_index()
        df['year'] = df['year_month'].dt.year
        df['year_month'] = df['year_month'].dt.month
        heatmap_data = df.pivot(index='year', columns='year_month', values=model_name)
        fig, ax = plt.subplots(figsize=(12, 8))
        cmap = sns.diverging_palette(10, 133, as_cmap=True)
        sns.heatmap(heatmap_data, 
                    cmap=cmap, 
                    center=0,
                    annot=True, 
                    fmt='.1%', 
                    linewidths=.5, 
                    ax=ax,
                    cbar_kws={'label': 'Monthly Return'})
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        ax.set_xticklabels(month_names)
        ax.set_title(f'Monthly Returns Heatmap - {model_name}', fontsize=16)
        plt.tight_layout()
        return fig

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
