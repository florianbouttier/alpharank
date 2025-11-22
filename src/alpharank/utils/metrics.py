import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import precision_score, recall_score, average_precision_score, roc_auc_score
from typing import Dict, List, Optional, Any, Tuple

def evaluate_classifier(
    y_true,
    y_pred_scores,
    threshold: float = 0.5,
    top_k_list: List[int] = None
) -> Dict[str, float]:
    """
    Évalue un classifieur binaire avec un focus sur la précision des prédictions positives.
    
    y_true : array-like (0/1)
    y_pred_scores : array-like (probas ou scores)
    threshold : seuil de décision pour classer en positif
    top_k_list : liste d'entiers pour calculer precision@k
    """
    # Conversion en arrays 1D
    y_true = np.asarray(y_true).ravel()
    y_pred_scores = np.asarray(y_pred_scores).ravel()

    if y_true.shape[0] != y_pred_scores.shape[0]:
        raise ValueError(
            f"Incohérence : y_true contient {y_true.shape[0]} échantillons, "
            f"y_pred_scores en contient {y_pred_scores.shape[0]}"
        )

    # Prédictions binaires
    y_pred_labels = (y_pred_scores >= threshold).astype(int)

    results = {}
    results["precision"] = precision_score(y_true, y_pred_labels, zero_division=0)
    results["recall"] = recall_score(y_true, y_pred_labels, zero_division=0)
    results["average_precision"] = average_precision_score(y_true, y_pred_scores)
    results["roc_auc"] = roc_auc_score(y_true, y_pred_scores)
    
    results['spread'] = np.max(y_pred_scores) - np.min(y_pred_scores)

    # Precision@k si demandé
    if top_k_list is not None:
        sorted_idx = np.argsort(y_pred_scores)[::-1]
        for k in top_k_list:
            if k > len(sorted_idx):
                raise ValueError(f"k={k} est supérieur au nombre d'échantillons ({len(sorted_idx)})")
            top_k_idx = sorted_idx[:k]
            results[f"precision@{k}"] = y_true[top_k_idx].sum() / k

    return results

def compare_models(models_data: Dict[str, pd.Series], start_year: Optional[int] = None, end_year: Optional[int] = None, risk_free_rate: float = 0.02):
    """
    Compares multiple strategy models against benchmarks and each other.
    Generates metrics, cumulative returns, correlations, and plots.
    
    Args:
        models_data: Dictionary where keys are model names and values are Series of monthly returns.
                     The Series index or a column must be 'year_month' (Period or datetime).
        start_year: Filter data starting from this year.
        end_year: Filter data up to this year.
        risk_free_rate: Risk-free rate for Sharpe ratio calculation.
        
    Returns:
        metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, figures
    """
    
    # Standardize input data to a common DataFrame
    processed_data = {}
    common_dates = None

    for model_name, data in models_data.items():
        # Handle different input formats (Series vs DataFrame)
        if isinstance(data, pd.Series):
            df = data.to_frame(name='monthly_return')
            if not isinstance(data.index, pd.PeriodIndex):
                 # Try to infer index from name or reset index
                 if 'year_month' in data.index.names:
                     df = df.reset_index()
                 else:
                     # Assume index is date-like
                     df.index = pd.to_datetime(df.index).to_period('M')
                     df.index.name = 'year_month'
                     df = df.reset_index()
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"Data for {model_name} must be Series or DataFrame")

        # Ensure 'year_month' column exists
        if 'year_month' not in df.columns:
            # Check index
            if isinstance(df.index, pd.PeriodIndex):
                df = df.reset_index()
                df.rename(columns={'index': 'year_month'}, inplace=True)
            else:
                raise ValueError(f"Data for {model_name} must contain 'year_month' column or index")

        # Ensure 'year_month' is Period('M')
        if not isinstance(df['year_month'].iloc[0], pd.Period):
            try:
                df['year_month'] = pd.to_datetime(df['year_month']).dt.to_period('M')
            except Exception as e:
                print(f"Error converting dates for {model_name}: {e}")
                continue

        # Identify return column
        return_cols = [col for col in df.columns if 'return' in col.lower() and 'monthly' in col.lower()]
        if return_cols:
            return_col = return_cols[0]
        elif 'monthly_return' in df.columns:
            return_col = 'monthly_return'
        elif len(df.columns) > 1:
            # Fallback: assume second column if not obvious
            return_col = df.columns[1]
        else:
             # Fallback: assume first column if it's the only one (besides year_month)
             cols = [c for c in df.columns if c != 'year_month']
             if cols:
                 return_col = cols[0]
             else:
                raise ValueError(f"Could not identify returns column for model {model_name}")

        # Filter by year
        if start_year:
            df = df[df['year_month'].dt.year >= start_year]
        if end_year:
            df = df[df['year_month'].dt.year <= end_year]

        if df.empty:
            print(f"Warning: No data available for {model_name} in selected time period")
            continue

        # Store as Series indexed by year_month
        processed_data[model_name] = df.set_index('year_month')[return_col]
        
        # Track common dates
        dates = set(processed_data[model_name].index)
        if common_dates is None:
            common_dates = dates
        else:
            common_dates = common_dates.intersection(dates)

    if not processed_data:
        raise ValueError("No valid data found for any model")
        
    if not common_dates:
        print("Warning: No common dates found across all models. Using all available data (outer join).")
    
    # Combine into single DataFrame
    all_returns = pd.DataFrame(processed_data)
    # Align to common dates if strict comparison needed, or keep all (filling NaNs with 0 or dropping?)
    # Legacy code did intersection:
    if common_dates:
        all_returns = all_returns.loc[sorted(list(common_dates))]
    
    all_returns = all_returns.sort_index()
    
    # --- Analysis ---
    correlation_matrix = all_returns.corr()
    
    # Cumulative Returns
    # Convert PeriodIndex to Timestamp for plotting
    all_returns_ts = all_returns.copy()
    all_returns_ts.index = all_returns_ts.index.to_timestamp()
    cumulative_returns = (all_returns_ts + 1).cumprod()

    # Worst Periods
    worst_periods = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].reset_index()
        model_returns['year'] = model_returns['year_month'].dt.year
        
        # Worst Month
        worst_month_idx = model_returns[model].idxmin()
        worst_month_val = model_returns.loc[worst_month_idx, model]
        worst_month_date = model_returns.loc[worst_month_idx, 'year_month']
        
        # Worst Year
        annual_returns = model_returns.groupby('year')[model].apply(lambda x: np.prod(1 + x) - 1)
        worst_year_idx = annual_returns.idxmin()
        worst_year_val = annual_returns.loc[worst_year_idx]
        
        worst_periods[model] = {
            'Worst Month': f"{worst_month_date}: {worst_month_val:.2%}",
            'Worst Year': f"{worst_year_idx}: {worst_year_val:.2%}"
        }
    worst_periods_df = pd.DataFrame(worst_periods).T

    # Metrics Calculation
    metrics = {}
    for model in all_returns.columns:
        model_returns = all_returns[model].dropna()
        if len(model_returns) < 12:
            continue
            
        total_months = len(model_returns)
        total_years = total_months / 12
        
        total_return = (1 + model_returns).prod() - 1
        annualized_return = (1 + total_return) ** (1 / total_years) - 1
        
        monthly_mean = model_returns.mean()
        monthly_std = model_returns.std()
        annualized_vol = monthly_std * np.sqrt(12)
        
        sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol != 0 else np.nan
        
        # Max Drawdown
        cum_ret = (1 + model_returns).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret / rolling_max) - 1
        max_drawdown = drawdown.min()
        
        positive_months = (model_returns > 0).sum() / total_months
        
        # CAGRs
        cagr_3yr = cagr_5yr = cagr_10yr = None
        if total_years >= 3:
            ret_3yr = model_returns.iloc[-36:]
            cagr_3yr = (1 + ret_3yr).prod() ** (1/3) - 1
        if total_years >= 5:
            ret_5yr = model_returns.iloc[-60:]
            cagr_5yr = (1 + ret_5yr).prod() ** (1/5) - 1
        if total_years >= 10:
            ret_10yr = model_returns.iloc[-120:]
            cagr_10yr = (1 + ret_10yr).prod() ** (1/10) - 1
            
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
            'Positive Months %': positive_months
        }

    metrics_df = pd.DataFrame(metrics).T
    
    # Formatting
    format_cols = ['Total Return', 'CAGR', 'CAGR (3Y)', 'CAGR (5Y)', 'CAGR (10Y)', 
                   'Monthly Mean', 'Monthly Volatility', 'Annualized Volatility', 
                   'Max Drawdown', 'Positive Months %']
    for col in format_cols:
        if col in metrics_df.columns:
            metrics_df[col] = metrics_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
    if 'Sharpe Ratio' in metrics_df.columns:
        metrics_df['Sharpe Ratio'] = metrics_df['Sharpe Ratio'].apply(lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A")

    # Plots
    cagr_by_year = calculate_cagr_by_year(all_returns)
    annual_returns_df = all_returns.groupby(all_returns.index.year).apply(lambda x: (1 + x).prod() - 1)

    fig = plt.figure(figsize=(20, 16))
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    # 1. Cumulative
    ax1 = plt.subplot(2, 2, 1)
    cumulative_returns.plot(ax=ax1)
    ax1.set_title('Cumulative Returns', fontsize=16)
    ax1.set_ylabel('Value of $1 Investment', fontsize=12)
    ax1.grid(True)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    
    # 2. Correlation
    ax2 = plt.subplot(2, 2, 2)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                mask=mask, vmin=-1, vmax=1, ax=ax2)
    ax2.set_title('Return Correlation Matrix', fontsize=16)
    
    # 3. CAGR by Start Year
    ax3 = plt.subplot(2, 2, 3)
    sns.heatmap(cagr_by_year, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax3)
    ax3.set_title('CAGR by Start Year', fontsize=16)
    ax3.set_ylabel('Start Year', fontsize=12)
    
    # 4. Annual Returns
    ax4 = plt.subplot(2, 2, 4)
    sns.heatmap(annual_returns_df, annot=True, fmt=".1%", cmap="RdYlGn", ax=ax4)
    ax4.set_title('Annual Returns by Year', fontsize=16)
    ax4.set_ylabel('Year', fontsize=12)
    
    plt.tight_layout()
    
    individual_heatmaps = {}
    for model in all_returns.columns:
        individual_heatmaps[model] = plot_monthly_returns_heatmap(all_returns[model], model)

    figures = {
        'Main Figure': fig,
        'Monthly Heatmaps': individual_heatmaps
    }
    
    return metrics_df, cumulative_returns, correlation_matrix, worst_periods_df, figures

def plot_monthly_returns_heatmap(returns_series, model_name):
    df = returns_series.to_frame(name=model_name)
    # Ensure index is PeriodIndex
    if not isinstance(df.index, pd.PeriodIndex):
         df.index = pd.to_datetime(df.index).to_period('M')
         
    df['year'] = df.index.year
    df['month'] = df.index.month
    
    heatmap_data = df.pivot(index='year', columns='month', values=model_name)
    
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
    # Ensure xticks match data columns (1-12)
    ax.set_xticks(np.arange(len(month_names)) + 0.5)
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
    
    plt.figure()
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
