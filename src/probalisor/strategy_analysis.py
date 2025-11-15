
import pandas as pd
import numpy as np

import datetime
from scipy import stats
from tqdm import tqdm
from datetime import *
import matplotlib.pyplot as plt
import seaborn as sns
# %%

def compare_models(models_data, start_year=None, end_year=None, risk_free_rate=0.02):
        
        common_date = []
        for model in models_data:
            df = models_data[model]
            if 'year_month' not in df.columns:
                raise ValueError(f"Data for model {model} must contain 'year_month' column")
            if not isinstance(df['year_month'].iloc[0], pd.Period):
                try:
                    df['year_month'] = df['year_month'].dt.to_period('m')
                except:
                    df['year_month'] = pd.to_datetime(df['year_month']).dt.to_period('m')
            common_date.append(set(df['year_month'].unique()))
        common_dates = set.intersection(*common_date)
        if not common_dates:
            raise ValueError("No common dates found across all models")
            
            
        processed_data = {}
        
        
        for model_name, df in models_data.items():
            df_copy = df.copy()
            df_copy = df_copy[df_copy['year_month'].isin(common_dates)]
            return_cols = [col for col in df_copy.columns if 'return' in col.lower()]
            if return_cols :
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
        
        cagr_by_year = calculate_cagr_by_year(all_returns)
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
            fig_heatmap = plot_monthly_returns_heatmap(all_returns[model], model)
            individual_heatmaps[model] = fig_heatmap
        
        figures = {
            'Main Figure': fig,
            'Monthly Heatmaps': individual_heatmaps
        }
        return metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, figures

    
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

def compare_score_and_perf(df_scores: pd.DataFrame, df_perf: pd.DataFrame, score_col: str, perf_col: str):
    merged = df_scores.merge(df_perf, on=["year_month"], how="inner")
    merged = merged.dropna(subset=[score_col, perf_col])
    if merged.empty:
        print("No overlapping data between scores and performance.")
        return None
    spearman_corr = stats.spearmanr(merged[score_col], merged[perf_col])
    pearson_corr = stats.pearsonr(merged[score_col], merged[perf_col])
    
    plt.plot(merged[score_col], merged[perf_col], 'o')
    plt.xlabel(score_col)
    plt.ylabel(perf_col)
    plt.title(f'Scatter plot of {perf_col} vs {score_col}')
    plt.grid(True)
    plt.show()
    return {
        "spearman_correlation": spearman_corr.correlation,
        "spearman_pvalue": spearman_corr.pvalue,
        "pearson_correlation": pearson_corr[0],
        "pearson_pvalue": pearson_corr[1],
        "n_samples": len(merged)
    }
