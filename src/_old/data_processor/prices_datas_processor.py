import pandas as pd
from datetime import datetime

class PricesDataPreprocessor:
    """Nettoyage et mise en forme des séries de prix."""
    @staticmethod
    def augment_prices(df: pd.DataFrame, columns_to_augment: str, column_date: str = 'date') -> pd.DataFrame:
        """
        Fill missing dates in price data by forward-filling values for each ticker.
        
        Args:
            df: DataFrame containing price data with date and ticker columns
            columns_to_augment: Name of the column(s) to forward-fill for missing dates
            columns_date: Name of the date column (default: 'date')
        
        Returns:
            DataFrame with complete date range for all tickers, missing values forward-filled
        """
        # Convert date column to datetime format
        df[column_date] = pd.to_datetime(df[column_date])
        
        # Create complete date range from min to max date in the dataset
        all_dates = pd.date_range(start=df[column_date].min(), end=df[column_date].max())
        
        # Restructure data: pivot by ticker, reindex with full date range, then unpivot
        df_full = df.set_index([column_date, 'ticker']).unstack('ticker').reindex(all_dates).stack('ticker', dropna=False).reset_index().rename(columns={'level_0': 'date'})
        
        # Forward-fill missing values for each ticker separately
        df_full[columns_to_augment] = df_full.groupby('ticker')[columns_to_augment].ffill()
        
        # Return only the relevant columns
        df_full = df_full[[column_date, 'ticker']+ columns_to_augment]
        return df_full

    @staticmethod
    def compute_dr(df: pd.DataFrame, column_date: str = 'date', column_close: str = 'close') -> pd.DataFrame:
        """
        Compute daily returns (dr) for each ticker in the prices DataFrame.
        
        Args:
            prices: DataFrame containing price data with date and ticker columns
            column_date: Name of the date column (default: 'date')
            column_close: Name of the close price column (default: 'close')
        
        Returns:
            DataFrame with daily returns added as a new column
        """
        
        df[column_date] = pd.to_datetime(df[column_date])
        df.sort_values(by=['ticker', column_date], inplace=True)
        df['close_lag'] = df.groupby('ticker')[column_close].shift(1)
        df['dr'] = df[column_close] / df['close_lag']
        
        return df.drop(columns=['close_lag'], errors='ignore')  

    @staticmethod
    def prices_vs_index(index: pd.DataFrame, prices: pd.DataFrame, column_close_index: str, column_close_prices: str) -> pd.DataFrame:
        """
        Calculate relative performance of prices against a benchmark index.
        This function merges price data with index data and computes the relative performance
        by dividing prices by the corresponding index values. It also calculates daily returns
        for the relative performance metric.
        
        Args:
            index (pd.DataFrame): DataFrame containing index/benchmark data with date and price columns
            prices (pd.DataFrame): DataFrame containing stock/asset prices with date and price columns
            column_close_index (str): Name of the closing price column in the index DataFrame
            column_close_prices (str): Name of the closing price column in the prices DataFrame
            
        Returns:
            pd.DataFrame: Augmented prices DataFrame with additional columns:
                - close_vs_index: Ratio of asset price to index price
                - dr_vs_index: Daily returns of the relative performance metric
                - Additional augmented columns from the index data
                
        Notes:
            - If column names are identical, the index column is renamed with '_index' suffix
            - The function adds a 'ticker' column to index data if not present
            - Date columns are converted to datetime format for proper merging
            - Uses left join to preserve all price data points
            - Computes augmented price features and daily returns for relative performance
        """
        
        if column_close_index == column_close_prices:
            index.rename(columns={column_close_index: column_close_index + "_index"}, inplace=True)
            column_close_index = column_close_index + "_index"
            
        if 'ticker' not in index.columns:
            index['ticker'] = 'index'

        index = PricesDataPreprocessor.augment_prices(df=index.copy(), 
                                              columns_to_augment=[column_close_index], 
                                              column_date='date')
        
        index = index.drop(['ticker'], axis=1, errors='ignore')
        prices['date'] = pd.to_datetime(prices['date'])
        prices_augmented = prices.merge(index, how="left", on="date")
        prices_augmented['close_vs_index'] = prices_augmented[column_close_prices] / prices_augmented[column_close_index]
        
        prices_augmented = PricesDataPreprocessor.compute_dr(df=prices_augmented.copy(),
                                                     column_date='date',
                                                     column_close='close_vs_index')
        prices_augmented.rename(columns={"dr": "dr_vs_index"}, inplace=True)
        
        return prices_augmented
    
    @staticmethod
    def calculate_monthly_returns(df  : pd.DataFrame,column_date: str = 'date',column_close: str = 'close') -> pd.DataFrame:
        df = df.copy()
        df[column_date] = pd.to_datetime(df[column_date])
        df['year_month'] = df[column_date].dt.to_period('M')
        last_rows = df.groupby(['ticker', 'year_month']).apply(lambda x: x.loc[x[column_date].idxmax()],include_groups = False)
        last_closes_df = last_rows[[column_date,column_close]].reset_index()
        last_closes_df = PricesDataPreprocessor.compute_dr(df =last_closes_df.copy(),
                                                     column_date=column_date,
                                                     column_close=column_close)
        last_closes_df = (last_closes_df
                          .rename(columns={'dr': 'monthly_return', column_close: 'last_close'}))
        return last_closes_df
