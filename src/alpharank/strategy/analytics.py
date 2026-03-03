import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union

class PerformanceAnalyzer:
    """
    Centralized engine for calculating financial performance metrics and time series analysis.
    """

    @staticmethod
    def calculate_cumulative_returns(returns_df: pd.DataFrame, fill_missing: bool = True) -> pd.DataFrame:
        """
        Calculates cumulative returns from a DataFrame of periodic returns.
        
        Args:
            returns_df: DataFrame where each column is a strategy/asset returns series.
            fill_missing: If True, fills NaNs with 0 before compounding (flat performance).
                          Essential for comparing strategies with different start dates.
        """
        df = returns_df.copy()
        if fill_missing:
            df = df.fillna(0)
        return (1 + df).cumprod()

    @staticmethod
    def calculate_drawdowns(cumulative_returns_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates drawdown series from cumulative returns.
        """
        rolling_max = cumulative_returns_df.cummax()
        drawdowns = (cumulative_returns_df / rolling_max) - 1
        return drawdowns

    @staticmethod
    def calculate_metrics(series: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 12) -> Dict[str, float]:
        """
        Computes a comprehensive suite of financial KPIs for a single returns series.
        """
        series = series.dropna()
        if series.empty:
            return {}

        # Time Calculation
        n_periods = len(series)
        total_years = n_periods / periods_per_year
        if total_years < 1/periods_per_year: 
             # Avoid huge numbers for very short periods
             total_years = 1/periods_per_year

        # 1. Return Metrics
        total_return = (1 + series).prod() - 1
        cagr = (1 + total_return) ** (1 / total_years) - 1
        
        # 2. Volatility Metrics
        std_dev = series.std()
        annualized_vol = std_dev * np.sqrt(periods_per_year)
        
        # 3. Risk-Adjusted Return
        sharpe = (cagr - risk_free_rate) / annualized_vol if annualized_vol != 0 else np.nan
        
        # Sortino
        downside_returns = series[series < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year)
        sortino = (cagr - risk_free_rate) / downside_std if downside_std > 0 else np.nan
        
        # 4. Drawdown Analysis
        # Re-calculate specific cumulative for this series (independent of others)
        wealth = (1 + series).cumprod()
        peaks = wealth.cummax()
        drawdown = (wealth - peaks) / peaks
        max_drawdown = drawdown.min()
        
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else np.nan
        
        # Max Drawdown Duration
        is_underwater = drawdown < 0
        underwater_groups = (is_underwater != is_underwater.shift()).cumsum()
        underwater_periods = is_underwater.groupby(underwater_groups).sum()
        max_dd_duration = underwater_periods.max() if not underwater_periods.empty else 0
        
        # 5. Consistency
        positive_months_pct = (series > 0).mean()

        return {
            'Total Return': total_return,
            'CAGR': cagr,
            'Annualized Volatility': annualized_vol,
            'Sharpe Ratio': sharpe,
            'Sortino Ratio': sortino,
            'Calmar Ratio': calmar,
            'Max Drawdown': max_drawdown,
            'Max DD Duration': max_dd_duration, # In periods (months)
            'Positive Periods %': positive_months_pct
        }

    @staticmethod
    def calculate_metrics_by_start_year(all_returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, pd.DataFrame]:
        """
        Calculates KPIs for every model for every possible start year (Start Year -> End).
        Returns a dictionary where keys are Metric Names (e.g. 'CAGR', 'Sharpe Ratio')
        and values are DataFrames (rows=Start Year, cols=Model).
        """
        years = sorted(list(set(all_returns.index.year)))
        models = all_returns.columns
        
        # Initialize storage for each metric
        metric_names = ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Sortino Ratio', 'Annualized Volatility', 'Calmar Ratio']
        metric_grids = {m: pd.DataFrame(index=years, columns=models) for m in metric_names}
        
        for model in models:
            for start_year in years:
                # Filter data from start_year to simple end
                subset = all_returns[model].loc[all_returns.index.year >= start_year].dropna()
                
                if len(subset) < 3:
                     continue
                    
                metrics = PerformanceAnalyzer.calculate_metrics(subset, risk_free_rate=risk_free_rate)
                
                for m in metric_names:
                    if m in metrics:
                        metric_grids[m].loc[start_year, model] = metrics[m]
                
        return metric_grids

    @staticmethod
    def calculate_annual_metrics(all_returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, pd.DataFrame]:
        """
        Calculates KPIs for each discrete calendar year.
        Returns a dictionary of DataFrames (rows=Year, cols=Model).
        """
        years = sorted(list(set(all_returns.index.year)))
        models = all_returns.columns
        
        metric_names = ['CAGR', 'Sharpe Ratio', 'Max Drawdown', 'Sortino Ratio', 'Annualized Volatility', 'Calmar Ratio']
        metric_grids = {m: pd.DataFrame(index=years, columns=models) for m in metric_names}
        
        for model in models:
            for year in years:
                # Filter data strictly for this year
                subset = all_returns[model].loc[all_returns.index.year == year].dropna()
                
                if len(subset) < 3: # Need some data to calculate metrics
                    continue
                
                metrics = PerformanceAnalyzer.calculate_metrics(subset, risk_free_rate=risk_free_rate)
                
                for m in metric_names:
                    if m in metrics:
                         metric_grids[m].loc[year, model] = metrics[m]
                         
        return metric_grids
    
    @staticmethod
    def get_annual_returns(all_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Computes calendar year returns for each model.
        """
        # Ensure we group by year. If index is Period, attribute is .year
        # If timestamp, .year works too.
        yrs = all_returns.index.year
        return all_returns.groupby(yrs).apply(lambda x: (1 + x).prod() - 1)

    @staticmethod
    def calculate_worst_periods(all_returns: pd.DataFrame) -> pd.DataFrame:
        """
        Identifies the worst month and worst year for each model.
        Returns a DataFrame with 'Worst Month' and 'Worst Year' as formatted strings.
        """
        worst_periods = {}
        # Calculate annual returns first
        annual = PerformanceAnalyzer.get_annual_returns(all_returns)
        
        for model in all_returns.columns:
            series = all_returns[model].dropna()
            if series.empty: continue
            
            # Worst Month
            worst_month_idx = series.idxmin()
            worst_month_val = series[worst_month_idx]
            
            # Format date. If Period, just str. If Timestamp, strftime.
            if hasattr(worst_month_idx, 'strftime'):
                date_str = worst_month_idx.strftime('%Y-%m')
            else:
                date_str = str(worst_month_idx)
                
            worst_m_str = f"{date_str}: {worst_month_val:.2%}"
            
            # Worst Year
            if model in annual.columns:
                ann_series = annual[model].dropna()
                if not ann_series.empty:
                    worst_year = ann_series.idxmin()
                    worst_year_val = ann_series[worst_year]
                    worst_y_str = f"{worst_year}: {worst_year_val:.2%}"
                else:
                    worst_y_str = "N/A"
            else:
                worst_y_str = "N/A"
                
            worst_periods[model] = {
                'Worst Month': worst_m_str,
                'Worst Year': worst_y_str
            }
            
        return pd.DataFrame(worst_periods).T
