"""
Centralized return calculation module.

This module provides atomic, readable functions for computing returns in a standardized format.
All returns are calculated as decimal percentages (e.g., 0.02 for 2% gain, -0.02 for 2% loss).

This ensures consistency across the entire codebase and eliminates confusion between
multiplicative (1.02) and decimal (0.02) return formats.
"""

import pandas as pd
import numpy as np
from typing import Optional


def calculate_simple_return(current_price: float, previous_price: float) -> float:
    """
    Calculate simple return between two prices.
    
    Args:
        current_price: Current price
        previous_price: Previous price
        
    Returns:
        Decimal return (e.g., 0.02 for 2% gain, -0.02 for 2% loss)
        
    Example:
        >>> calculate_simple_return(102, 100)
        0.02
        >>> calculate_simple_return(98, 100)
        -0.02
    """
    if previous_price == 0 or pd.isna(previous_price) or pd.isna(current_price):
        return np.nan
    return (current_price - previous_price) / previous_price


def calculate_daily_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    date_column: str = 'date',
    ticker_column: Optional[str] = 'ticker',
    return_column: str = 'daily_return'
) -> pd.DataFrame:
    """
    Calculate daily returns for a DataFrame of prices.
    
    Args:
        df: DataFrame containing price data
        price_column: Name of the column containing prices
        date_column: Name of the column containing dates
        ticker_column: Name of the column containing ticker symbols (None for index data)
        return_column: Name of the output column for returns
        
    Returns:
        DataFrame with daily returns added as a new column
        
    Note:
        Returns are in decimal format (0.02 for 2% gain)
    """
    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    result = result.sort_values(by=[ticker_column, date_column] if ticker_column else [date_column])
    
    if ticker_column:
        # Group by ticker and calculate returns
        result[return_column] = result.groupby(ticker_column)[price_column].pct_change()
    else:
        # Single series (e.g., index)
        result[return_column] = result[price_column].pct_change()
    
    return result


def calculate_monthly_returns(
    df: pd.DataFrame,
    price_column: str = 'close',
    date_column: str = 'date',
    ticker_column: Optional[str] = 'ticker',
    return_column: str = 'monthly_return'
) -> pd.DataFrame:
    """
    Calculate monthly returns from daily price data.
    
    Args:
        df: DataFrame containing daily price data
        price_column: Name of the column containing prices
        date_column: Name of the column containing dates
        ticker_column: Name of the column containing ticker symbols (None for index data)
        return_column: Name of the output column for returns
        
    Returns:
        DataFrame with year_month, date (last date of month), last_close, and monthly returns
        
    Note:
        Returns are in decimal format (0.02 for 2% gain)
        Uses end-of-month prices to calculate month-over-month returns
    """
    result = df.copy()
    result[date_column] = pd.to_datetime(result[date_column])
    result['year_month'] = result[date_column].dt.to_period('M')
    
    if ticker_column:
        # Group by ticker and year_month, take last price and date of each month
        monthly_data = (
            result.groupby([ticker_column, 'year_month'])
            .agg({
                price_column: 'last',
                date_column: 'last'
            })
            .reset_index()
        )
        # Calculate returns within each ticker
        monthly_data[return_column] = (
            monthly_data.groupby(ticker_column)[price_column].pct_change()
        )
        # Rename price column to 'last_close' for backward compatibility
        monthly_data = monthly_data.rename(columns={price_column: 'last_close'})
    else:
        # Single series (e.g., index)
        monthly_data = (
            result.groupby('year_month')
            .agg({
                price_column: 'last',
                date_column: 'last'
            })
            .reset_index()
        )
        monthly_data[return_column] = monthly_data[price_column].pct_change()
        # For index, keep the original column name
        if price_column != 'last_close':
            monthly_data = monthly_data.rename(columns={price_column: 'last_close'})
    
    return monthly_data


def aggregate_returns(
    returns: pd.Series,
    method: str = 'mean'
) -> float:
    """
    Aggregate multiple returns.
    
    Args:
        returns: Series of decimal returns
        method: Aggregation method ('mean', 'median', 'compound')
        
    Returns:
        Aggregated return in decimal format
        
    Note:
        For 'compound', this calculates the geometric mean return
    """
    if method == 'mean':
        return returns.mean()
    elif method == 'median':
        return returns.median()
    elif method == 'compound':
        # Compound returns: (1+r1)*(1+r2)*...*(1+rn) - 1
        return (1 + returns).prod() - 1
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def convert_to_cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Convert a series of periodic returns to cumulative returns.
    
    Args:
        returns: Series of decimal returns
        
    Returns:
        Series of cumulative returns
        
    Example:
        >>> returns = pd.Series([0.02, 0.03, -0.01])
        >>> convert_to_cumulative_returns(returns)
        0    1.020000
        1    1.050600
        2    1.040094
        dtype: float64
    """
    return (1 + returns).cumprod()


def annualize_return(total_return: float, num_periods: int, periods_per_year: int = 12) -> float:
    """
    Annualize a total return.
    
    Args:
        total_return: Total return over the period (decimal format)
        num_periods: Number of periods
        periods_per_year: Number of periods in a year (12 for monthly, 252 for daily)
        
    Returns:
        Annualized return (CAGR)
        
    Example:
        >>> annualize_return(0.20, 24, 12)  # 20% over 24 months
        0.0954451...
    """
    years = num_periods / periods_per_year
    return (1 + total_return) ** (1 / years) - 1
