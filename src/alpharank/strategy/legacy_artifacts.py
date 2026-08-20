"""Comparison tables and chart-ready data for Legacy evaluation artifacts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from alpharank.strategy.analytics import PerformanceAnalyzer


def compare_models(
    models_data,
    start_year=None,
    end_year=None,
    risk_free_rate=0.02,
):
    """Build the historical Legacy comparison artifact payload."""

    processed_data = {}
    for model_name, df in models_data.items():
        df_copy = df.copy()
        return_cols = [col for col in df_copy.columns if "return" in col.lower()]
        if return_cols:
            return_col = return_cols[0]
        elif "monthly_return" in df_copy.columns:
            return_col = "monthly_return"
        elif len(df_copy.columns) > 1:
            return_col = df_copy.columns[1]
        else:
            print(f"Warning: Could not identify returns column for {model_name}. Skipping.")
            continue

        if not isinstance(df_copy["year_month"].iloc[0], pd.Period):
            try:
                df_copy["year_month"] = df_copy["year_month"].dt.to_period("M")
            except Exception:
                df_copy["year_month"] = pd.to_datetime(df_copy["year_month"]).dt.to_period("M")

        if start_year:
            df_copy = df_copy[df_copy["year_month"].dt.year >= start_year]
        if end_year:
            df_copy = df_copy[df_copy["year_month"].dt.year <= end_year]
        if df_copy.empty:
            print(f"Warning: No data available for {model_name} in selected period")
            continue

        processed_data[model_name] = df_copy.set_index("year_month")[return_col]

    all_returns = pd.DataFrame(processed_data).sort_index()
    metrics = {}
    for model in processed_data:
        series = processed_data[model]
        if series.empty:
            continue
        model_metrics = PerformanceAnalyzer.calculate_metrics(series, risk_free_rate)
        n_stocks_avg = None
        if model in models_data and "n" in models_data[model].columns:
            n_stocks_avg = models_data[model]["n"].mean()
        model_metrics["Number of Stocks (Avg)"] = n_stocks_avg
        model_metrics["Start Date"] = series.index.min().strftime("%y-%m")
        model_metrics["End Date"] = series.index.max().strftime("%y-%m")

        total_years = (series.index.max() - series.index.min()).n / 12
        for years in [3, 5, 10]:
            if total_years >= years:
                subset = series.iloc[-12 * years :]
                model_metrics[f"CAGR ({years}Y)"] = (1 + subset).prod() ** (1 / years) - 1
            else:
                model_metrics[f"CAGR ({years}Y)"] = None
        metrics[model] = model_metrics

    metrics_df = pd.DataFrame(metrics).T
    percentage_columns = [
        "Total Return",
        "CAGR",
        "Monthly Mean",
        "Monthly Volatility",
        "Annualized Volatility",
        "Max Drawdown",
        "Positive Periods %",
        "CAGR (3Y)",
        "CAGR (5Y)",
        "CAGR (10Y)",
    ]
    for column in percentage_columns:
        if column in metrics_df.columns:
            metrics_df[column] = metrics_df[column].apply(
                lambda value: f"{value:.2%}" if pd.notnull(value) else "N/A"
            )
    for column in ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio"]:
        if column in metrics_df.columns:
            metrics_df[column] = metrics_df[column].apply(
                lambda value: f"{value:.2f}" if pd.notnull(value) and not np.isinf(value) else "N/A"
            )

    cumulative_returns = PerformanceAnalyzer.calculate_cumulative_returns(
        all_returns, fill_missing=True
    )
    cumulative_returns.index = cumulative_returns.index.to_timestamp()
    drawdowns_df = PerformanceAnalyzer.calculate_drawdowns(cumulative_returns)
    annual_returns_df = PerformanceAnalyzer.get_annual_returns(all_returns).T
    correlation_matrix = all_returns.corr()
    cumulative_metrics_dict = PerformanceAnalyzer.calculate_metrics_by_start_year(
        all_returns, risk_free_rate
    )
    annual_metrics_dict = PerformanceAnalyzer.calculate_annual_metrics(all_returns, risk_free_rate)
    worst_periods_df = PerformanceAnalyzer.calculate_worst_periods(all_returns)
    monthly_returns_dict = {}
    for column in all_returns.columns:
        clean_series = all_returns[column].dropna()
        clean_series.index = clean_series.index.to_timestamp()
        monthly_returns_dict[column] = clean_series

    return (
        metrics_df,
        cumulative_returns,
        correlation_matrix,
        worst_periods_df,
        drawdowns_df,
        annual_returns_df,
        cumulative_metrics_dict,
        annual_metrics_dict,
        monthly_returns_dict,
    )


def calculate_cagr_by_year(returns_df):
    """Calculate each model CAGR for every available starting year."""

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
            model_cagr[start_year] = (1 + total_return) ** (1 / years_count) - 1
        cagr_results[model] = model_cagr
    return pd.DataFrame(cagr_results).dropna(how="all")
