import numpy as np
import pandas as pd
import itertools
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
import random
# %%
class TechnicalIndicators:
    """
    A library of static methods for calculating various technical indicators.
    This class serves as a collection of calculation functions.
    """
    @staticmethod
    def ema(series: pd.Series, n: int) -> pd.Series:
        """Calculates the Exponential Moving Average (EMA)."""
        return series.ewm(span=n, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, n: int) -> pd.Series:
        """Calculates the Simple Moving Average (SMA)."""
        return series.rolling(window=n, min_periods=1).mean()

    @staticmethod
    def rsi(series: pd.Series, n: int = 14) -> pd.Series:
        """Calculates the Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).ewm(alpha=1/n, adjust=False).mean()
        loss = (-delta.where(delta < 0, 0)).ewm(alpha=1/n, adjust=False).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def roc(series: pd.Series, n: int = 12) -> pd.Series:
        """Calculates the Rate of Change (ROC)."""
        return (series / series.shift(n)) - 1

    @staticmethod
    def bollinger_percent_b(series: pd.Series, n: int = 20, n_std: float = 2.0) -> pd.Series:
        """Calculates the %B value of Bollinger Bands."""
        middle_band = TechnicalIndicators.sma(series, n)
        std_dev = series.rolling(window=n).std()
        upper_band = middle_band + (std_dev * n_std)
        lower_band = middle_band - (std_dev * n_std)
        band_width = upper_band - lower_band
        return (series - lower_band) / band_width.replace(0, np.nan)

    @staticmethod
    def stochastic_oscillator_d(series: pd.Series, n: int = 14, d_window: int = 3) -> pd.Series:
        """Calculates the smoothed Stochastic Oscillator (%D)."""
        low_n = series.rolling(window=n).min()
        high_n = series.rolling(window=n).max()
        range_n = high_n - low_n
        percent_k = 100 * ((series - low_n) / range_n.replace(0, np.nan))
        percent_d = TechnicalIndicators.sma(percent_k, d_window)
        return percent_d
        
    @staticmethod
    def increase(values, n, diff=True, annual_base=4):
        """Calculates the increase in values over a specified period."""
        values = pd.Series(values)
        if diff:
            return values - values.shift(n)
        v0, v1 = values.shift(n), values
        denom = v0.abs()
        growth = np.where(denom == 0, np.nan, (v1 - v0) / denom)
        base = 1 + growth
        result = np.where(base > 0, base ** (annual_base / n) - 1, base)
        return pd.Series(result, index=values.index)

    @staticmethod
    def augmenting_ratios(data: pd.DataFrame, kpi_list: List[str], date_col: str) -> pd.DataFrame:
        """
        Augments the DataFrame with 'days since last negative' for specified KPIs.
        
        For each KPI in kpi_list, it calculates the number of days since the value was last negative.
        This effectively measures the duration of a positive streak or 'time since recovery'.
        
        Args:
            data: Input DataFrame (must contain 'ticker', date_col, and kpi columns)
            kpi_list: List of KPI column names to analyze
            date_col: Name of the date column
            
        Returns:
            DataFrame with new columns: {kpi}_days_increase
        """
        # Ensure data is sorted
        data = data.sort_values(['ticker', date_col], ascending=[True, True])
        
        def days_since_last_negative(group, kpi):
            # Find dates where KPI is negative
            # .where replaces False (non-negative) with NaN
            # .ffill propagates the last negative date forward
            last_neg = group[date_col].where(group[kpi] < 0).ffill()
            
            # Calculate days difference between current date and last negative date
            return (group[date_col] - last_neg).dt.days
        
        for kpi in kpi_list:
            if kpi not in data.columns:
                continue
                
            # Apply calculation per ticker
            # group_keys=False prevents the index from being modified by groupby
            data[f"{kpi}_days_increase"] = (
                data.groupby('ticker', group_keys=False)
                    .apply(lambda x: days_since_last_negative(x, kpi))
            )
            
            # Fill NaNs (never negative or start of series) with 0 or appropriate value
            # Here we leave as NaN or 0 depending on preference. 
            # If it's never been negative, the subtraction might be NaN.
            # Usually, we might want to fill with days since start, but let's stick to the core logic.
            
        return data
    
    @staticmethod
    def decreasing_sum(vals, halfPeriod, mode):
        """
        The function `decreasing_sum` calculates a weighted sum of values based on a specified mode and
        half-period.
        
        :param vals: The `vals` parameter in the `decreasing_sum` function represents a list of values
        for which you want to calculate a weighted sum based on the specified mode and halfPeriod. These
        values could be numerical data points, time series values, or any other type of data that you
        want to apply a
        :param halfPeriod: The `halfPeriod` parameter in the `decreasing_sum` function represents half
        of the period over which the weights are calculated. It is used in various weight calculation
        formulas based on the `mode` specified. The choice of `halfPeriod` affects how the weights are
        distributed and how they contribute to
        :param mode: The `mode` parameter in the `decreasing_sum` function determines the type of
        weighting function to use for calculating the weighted sum of the input values. The available
        modes are:
        :return: The function `decreasing_sum` returns the weighted sum of the input values `vals` based
        on the specified mode of weighting. The weighted sum is calculated using different weighting
        schemes such as exponential, tanh, special, linear, quadratic, sigmoidal, and mean. The function
        computes the weights based on the specified mode, normalizes the weights, and then calculates
        the dot product of the weights and
        """
        n = len(vals)
        if n == 0:
            return 0.0
        weight = np.zeros(n)
        if mode == "exponential":
            p = np.log(2) / halfPeriod
            weight = np.exp(-p * np.arange(n))
        elif mode == "tanh":
            p = np.log(3) / (2 * halfPeriod)
            weight = 1 - np.tanh(p * np.arange(n))
        elif mode == "special":
            alpha = halfPeriod
            weight = np.maximum(
                1 - (1 + (1 + alpha * np.arange(n)) * (np.log(1 + alpha * np.arange(n)) - 1) / (alpha**2)),
                0
            )
        elif mode == "linear":
            weight = np.maximum(1 - np.arange(n) / halfPeriod, 0)
        elif mode == "quadratic":
            weight = np.maximum(1 - (np.arange(n) / halfPeriod)**2, 0)
        elif mode == "sigmoidal":
            k = np.log(3) / halfPeriod
            weight = 1 / (1 + np.exp(k * (np.arange(n) - halfPeriod)))
        elif mode == "mean":
            len1 = min(halfPeriod, n)
            weight[:len1] = 1
        else:
            raise ValueError(f"unknown mode: {mode}.")
        wsum = np.sum(weight)
        weight = np.divide(weight, wsum) if wsum != 0 else np.full(n, 1.0 / n)
        return np.dot(weight, vals)


class DecorrelatedIndicatorGenerator:
    """
    Generates sets of decorrelated indicators using a random search methodology
    for each indicator family, starting from a mandatory set of seed indicators.
    """
    def __init__(self, daily_prices_df: pd.DataFrame, price_column: str, seed: int = None):
        """
        Initializes the generator.
        """
        self.price_column = price_column
        self.df = daily_prices_df.copy()
        self.df['date'] = pd.to_datetime(self.df['date'])
        self.df = self.df.sort_values(['ticker', 'date'])

        self.final_indicators_df = self.df[['ticker', 'date']].copy()
        self.final_indicators_df['year_month'] = self.final_indicators_df['date'].dt.to_period('M')

        if seed is not None:
            random.seed(seed)

    def _add_if_decorrelated(self, new_indicator_daily: pd.Series, new_indicator_name: str, threshold: float) -> bool:
        """Helper to check correlation and add a new indicator."""
        if self.final_indicators_df.shape[1] <= 3: # Only contains ticker, date, year_month
            self.final_indicators_df[new_indicator_name] = new_indicator_daily
            return True

        temp_df = self.final_indicators_df.copy()
        temp_df[new_indicator_name] = new_indicator_daily
        monthly_temp = temp_df.groupby(['ticker', 'year_month']).last()
        
        numeric_cols = monthly_temp.select_dtypes(include=np.number).columns.tolist()
        corr_matrix = monthly_temp[numeric_cols].corr().abs()
        
        max_corr = corr_matrix[new_indicator_name].drop(new_indicator_name).max()

        if max_corr < threshold:
            self.final_indicators_df[new_indicator_name] = new_indicator_daily
            return True
        return False

    # --- Calculation Helpers ---
    def _calculate_ema_ratio(self, n_short, n_long):
        self.df.sort_values(['ticker', 'date'])
        ema_short = self.df.groupby('ticker')[self.price_column].transform(lambda x: TechnicalIndicators.ema(x, n=n_short))
        ema_long = self.df.groupby('ticker')[self.price_column].transform(lambda x: TechnicalIndicators.ema(x, n=n_long))
        return ema_short / ema_long

    def _calculate_rsi(self, n):
        self.df.sort_values(['ticker', 'date'])
        return self.df.groupby('ticker')[self.price_column].transform(lambda x: TechnicalIndicators.rsi(x, n=n))

    def _calculate_bollinger_band(self, n, n_std):
        self.df.sort_values(['ticker', 'date'])
        return self.df.groupby('ticker')[self.price_column].transform(lambda x: TechnicalIndicators.bollinger_percent_b(x, n=n, n_std=n_std))

    def _calculate_stochastic_oscillator(self, n, d_window):
        self.df.sort_values(['ticker', 'date'])
        return self.df.groupby('ticker')[self.price_column].transform(lambda x: TechnicalIndicators.stochastic_oscillator_d(x, n=n, d_window=d_window))
        
    # --- Decorrelation Search Functions ---
    def generate_decorrelated_ema_ratios(self, seed_pairs: List[Tuple[int, int]], n_to_find: int, correlation_threshold: float, max_tries: int = 200):
        print(f"\nSearching for {n_to_find} decorrelated EMA Ratios...")
        
        # Add seed pairs unconditionally
        for n_short, n_long in seed_pairs:
            name = f'ema_ratio_{n_short}_{n_long}'
            if name in self.final_indicators_df.columns: continue
            new_indicator = self._calculate_ema_ratio(n_short, n_long)
            self._add_if_decorrelated(new_indicator, name, 1.0) # Threshold of 1 ensures it's always added
        
        found_count = self.final_indicators_df.shape[1] - 3 # ticker, date, year_month
        with tqdm(total=n_to_find, initial=found_count) as pbar:
            for _ in range(max_tries):
                if found_count >= n_to_find: break
                
                n_short = random.randint(1, 100)
                n_long = random.randint(n_short + 10, 500)
                name = f'ema_ratio_{n_short}_{n_long}'

                if name in self.final_indicators_df.columns: continue
                
                new_indicator = self._calculate_ema_ratio(n_short, n_long)
                if self._add_if_decorrelated(new_indicator, name, correlation_threshold):
                    found_count += 1
                    pbar.update(1)

    def generate_decorrelated_rsi(self, seed_params: List[Dict], n_to_find: int, correlation_threshold: float, max_tries: int = 100):
        print(f"\nSearching for {n_to_find} decorrelated RSI indicators...")

        for params in seed_params:
            n = params
            name = f'rsi_{n}'
            if name in self.final_indicators_df.columns: continue
            new_indicator = self._calculate_rsi(n)
            self._add_if_decorrelated(new_indicator, name, 1.0)

        found_count = self.final_indicators_df.shape[1] - 3
        with tqdm(total=n_to_find, initial=found_count) as pbar:
            for _ in range(max_tries):
                if found_count >= n_to_find: break
                
                n = random.randint(5, 50)
                name = f'rsi_{n}'

                if name in self.final_indicators_df.columns: continue
                
                new_indicator = self._calculate_rsi(n)
                if self._add_if_decorrelated(new_indicator, name, correlation_threshold):
                    found_count += 1
                    pbar.update(1)

    def generate_decorrelated_bollinger_bands(self, seed_params: List[Dict], n_to_find: int, correlation_threshold: float, max_tries: int = 100):
        print(f"\nSearching for {n_to_find} decorrelated Bollinger Band %B indicators...")

        for params in seed_params:
            n, n_std = params['n'], params['n_std']
            name = f'bollinger_b_{n}_{n_std:.1f}'
            if name in self.final_indicators_df.columns: continue
            new_indicator = self._calculate_bollinger_band(n, n_std)
            self._add_if_decorrelated(new_indicator, name, 1.0)
        
        found_count = self.final_indicators_df.shape[1] - 3
        with tqdm(total=n_to_find, initial=found_count) as pbar:
            for _ in range(max_tries):
                if found_count >= n_to_find: break
                
                n = random.randint(10, 100)
                n_std = random.choice([1.5, 2.0, 2.5])
                name = f'bollinger_b_{n}_{n_std:.1f}'

                if name in self.final_indicators_df.columns: continue

                new_indicator = self._calculate_bollinger_band(n, n_std)
                if self._add_if_decorrelated(new_indicator, name, correlation_threshold):
                    found_count += 1
                    pbar.update(1)
    
    def generate_decorrelated_stochastic_oscillators(self, seed_params: List[Dict], n_to_find: int, correlation_threshold: float, max_tries: int = 100):
        print(f"\nSearching for {n_to_find} decorrelated Stochastic Oscillator %D indicators...")

        for params in seed_params:
            n, d_window = params['n'], params['d_window']
            name = f'stoch_d_{n}_{d_window}'
            if name in self.final_indicators_df.columns: continue
            new_indicator = self._calculate_stochastic_oscillator(n, d_window)
            self._add_if_decorrelated(new_indicator, name, 1.0)

        found_count = self.final_indicators_df.shape[1] - 3
        with tqdm(total=n_to_find, initial=found_count) as pbar:
            for _ in range(max_tries):
                if found_count >= n_to_find: break
                
                n = random.randint(10, 50)
                d_window = random.randint(3, 10)
                name = f'stoch_d_{n}_{d_window}'

                if name in self.final_indicators_df.columns: continue

                new_indicator = self._calculate_stochastic_oscillator(n, d_window)
                if self._add_if_decorrelated(new_indicator, name, correlation_threshold):
                    found_count += 1
                    pbar.update(1)

    def get_final_indicators(self) -> pd.DataFrame:
        """
        Resamples the final selected daily indicators to monthly frequency
        and returns the result.
        """
        print("\n--- Final Selected Indicators ---")
        selected_names = self.final_indicators_df.columns.drop(['ticker', 'date', 'year_month'])
        print(list(selected_names))
        print("-" * 35)
        
        monthly_df = self.final_indicators_df.groupby(['ticker', 'year_month']).last().reset_index()
        return monthly_df.drop(columns='date')