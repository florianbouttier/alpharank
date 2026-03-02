"""
Centralized return calculation module.

This module provides atomic, readable functions for computing returns in a standardized format.
All returns are calculated as decimal percentages (e.g., 0.02 for 2% gain, -0.02 for 2% loss).

This ensures consistency across the entire codebase and eliminates confusion between
multiplicative (1.02) and decimal (0.02) return formats.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from alpharank.utils.frame_backend import (
    Backend,
    ensure_backend_name,
    normalize_year_month_to_period,
    require_polars,
    to_pandas,
    to_polars,
)

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None


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
    df,
    price_column: str = 'close',
    date_column: str = 'date',
    ticker_column: Optional[str] = 'ticker',
    return_column: str = 'daily_return',
    backend: Optional[Backend] = None,
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
    backend_name = ensure_backend_name(backend, default="polars")
    if backend_name != "polars":
        raise ValueError("Pandas backend is disabled for returns.calculate_daily_returns.")

    require_polars()
    pldf = to_polars(df).with_columns(pl.col(date_column).cast(pl.Date, strict=False))
    sort_cols = [ticker_column, date_column] if ticker_column else [date_column]
    pldf = pldf.sort(sort_cols)
    if ticker_column:
        pldf = pldf.with_columns(
            pl.col(price_column).pct_change().over(ticker_column).alias(return_column)
        )
    else:
        pldf = pldf.with_columns(pl.col(price_column).pct_change().alias(return_column))
    return to_pandas(pldf)


def calculate_monthly_returns(
    df,
    price_column: str = 'close',
    date_column: str = 'date',
    ticker_column: Optional[str] = 'ticker',
    return_column: str = 'monthly_return',
    backend: Optional[Backend] = None,
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
    backend_name = ensure_backend_name(backend, default="polars")
    if backend_name != "polars":
        raise ValueError("Pandas backend is disabled for returns.calculate_monthly_returns.")

    require_polars()
    pldf = to_polars(df).with_columns(pl.col(date_column).cast(pl.Date, strict=False))
    pldf = pldf.with_columns(pl.col(date_column).dt.truncate("1mo").alias("year_month"))
    if ticker_column:
        monthly = (
            pldf.sort([ticker_column, date_column])
            .group_by([ticker_column, "year_month"])
            .agg(
                pl.col(price_column).last().alias("last_close"),
                pl.col(date_column).last().alias(date_column),
            )
            .sort([ticker_column, "year_month"])
            .with_columns(pl.col("last_close").pct_change().over(ticker_column).alias(return_column))
        )
    else:
        monthly = (
            pldf.sort([date_column])
            .group_by(["year_month"])
            .agg(
                pl.col(price_column).last().alias("last_close"),
                pl.col(date_column).last().alias(date_column),
            )
            .sort(["year_month"])
            .with_columns(pl.col("last_close").pct_change().alias(return_column))
        )
    result = to_pandas(monthly)
    result = normalize_year_month_to_period(result, "year_month")
    return result


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
