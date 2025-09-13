import pandas as pd
from typing import Optional

class IndexDataManager:
    """
    A generic class to encapsulate and manage data for a market index.

    This class holds the daily prices, historical components, and monthly
    returns for a given index, facilitating its use in backtesting pipelines.
    """

    def __init__(self,
                 daily_prices_df: pd.DataFrame,
                 components_df: pd.DataFrame,
                 monthly_returns_df: Optional[pd.DataFrame] = None):
        """
        Initializes the IndexDataManager object.

        Args:
            daily_prices_df (pd.DataFrame): A DataFrame containing 'date' and 'close' columns.
            components_df (pd.DataFrame): A DataFrame containing the historical components
                                          of the index (e.g., ticker, add_date, remove_date).
            monthly_returns_df (Optional[pd.DataFrame], optional):
                A DataFrame with monthly returns. If not provided, it will be
                calculated automatically from the daily prices. Defaults to None.
        """
        if 'date' not in daily_prices_df.columns or 'close' not in daily_prices_df.columns:
            raise ValueError("The 'daily_prices_df' DataFrame must contain 'date' and 'close' columns.")

        self.daily_prices = daily_prices_df.copy()
        self.components = components_df.copy()

        if monthly_returns_df is not None:
            self.monthly_returns = monthly_returns_df.copy()
        else:
            print("Monthly returns not provided. Calculating from daily prices...")
            self.monthly_returns = self._calculate_monthly_returns(self.daily_prices)

    def _calculate_monthly_returns(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates monthly returns from a daily prices DataFrame.
        """
        df = daily_prices.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date')

        # Resample to monthly frequency, taking the last price of each month
        monthly_prices = df['close'].resample('M').last()

        # Calculate the percentage return
        monthly_returns = monthly_prices.pct_change().to_frame(name='monthly_return')

        # Convert the index to a period for joining purposes
        monthly_returns.index = monthly_returns.index.to_period('M')
        monthly_returns.index.name = 'year_month'
        
        return monthly_returns.reset_index()

    def __repr__(self) -> str:
        """Provides a string representation of the object."""
        return (f"IndexDataManager(\n"
                f"  daily_prices: {self.daily_prices.shape[0]} rows,\n"
                f"  monthly_returns: {self.monthly_returns.shape[0]} rows,\n"
                f"  components: {self.components.shape[0]} rows\n"
                f")")