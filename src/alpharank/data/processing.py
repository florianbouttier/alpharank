import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Sequence, Union
from alpharank.features.indicators import TechnicalIndicators
from alpharank.utils.frame_backend import (
    Backend,
    ensure_backend_name,
    normalize_year_month_to_period,
    normalize_year_month_to_timestamp,
    require_polars,
    to_pandas,
    to_polars,
)

try:
    import polars as pl
except Exception:  # pragma: no cover - optional dependency
    pl = None

class IndexDataManager:
    """
    A generic class to encapsulate and manage data for a market index.

    This class holds the daily prices, historical components, and monthly
    returns for a given index, facilitating its use in backtesting pipelines.
    """

    def __init__(self,
                 daily_prices_df: pd.DataFrame,
                 components_df: pd.DataFrame,
                 monthly_returns_df: Optional[pd.DataFrame] = None,
                 backend: Backend = "polars"):
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

        if pl is not None and isinstance(daily_prices_df, pl.DataFrame):
            self.daily_prices = daily_prices_df.clone()
        else:
            self.daily_prices = daily_prices_df.copy()
        if pl is not None and isinstance(components_df, pl.DataFrame):
            self.components = components_df.clone()
        else:
            self.components = components_df.copy()
        self.backend = backend

        if monthly_returns_df is not None:
            if pl is not None and isinstance(monthly_returns_df, pl.DataFrame):
                self.monthly_returns = monthly_returns_df.clone()
            else:
                self.monthly_returns = monthly_returns_df.copy()
        else:
            print("Monthly returns not provided. Calculating from daily prices...")
            self.monthly_returns = self._calculate_monthly_returns(self.daily_prices)

    def _calculate_monthly_returns(self, daily_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates monthly returns from a daily prices DataFrame.
        
        Returns:
            DataFrame with year_month and monthly returns (decimal format: 0.02 for 2% gain)
        """
        from alpharank.utils.returns import calculate_monthly_returns
        
        return calculate_monthly_returns(
            df=daily_prices,
            price_column='close',
            date_column='date',
            ticker_column=None,  # Index data doesn't have tickers
            return_column='monthly_return',
            backend=self.backend,
        )

    def __repr__(self) -> str:
        """Provides a string representation of the object."""
        return (f"IndexDataManager(\n"
                f"  daily_prices: {self.daily_prices.shape[0]} rows,\n"
                f"  monthly_returns: {self.monthly_returns.shape[0]} rows,\n"
                f"  components: {self.components.shape[0]} rows\n"
                f")")

class PricesDataPreprocessor:
    """Nettoyage et mise en forme des séries de prix."""
    @staticmethod
    def augment_prices(
        df: Union[pd.DataFrame, "pl.DataFrame"],  # type: ignore[name-defined]
        columns_to_augment: Union[str, Sequence[str]],
        column_date: str = 'date',
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Fill missing dates in price data by forward-filling values for each ticker.
        
        Args:
            df: DataFrame containing price data with date and ticker columns
            columns_to_augment: Name of the column(s) to forward-fill for missing dates
            columns_date: Name of the date column (default: 'date')
        
        Returns:
            DataFrame with complete date range for all tickers, missing values forward-filled
        """
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for PricesDataPreprocessor.augment_prices.")
        cols = [columns_to_augment] if isinstance(columns_to_augment, str) else list(columns_to_augment)

        require_polars()
        pldf = to_polars(df).with_columns(pl.col(column_date).cast(pl.Date, strict=False))
        min_date = pldf.select(pl.col(column_date).min()).item()
        max_date = pldf.select(pl.col(column_date).max()).item()
        dates = pl.date_range(min_date, max_date, interval="1d", eager=True).alias(column_date)
        date_df = pl.DataFrame({column_date: dates})
        ticker_df = pldf.select("ticker").unique()
        skeleton = ticker_df.join(date_df, how="cross").sort(["ticker", column_date])
        joined = (
            skeleton.join(pldf.select(["ticker", column_date] + cols), on=["ticker", column_date], how="left")
            .sort(["ticker", column_date])
            .with_columns([pl.col(c).forward_fill().over("ticker").alias(c) for c in cols])
            .select([column_date, "ticker"] + cols)
        )
        out = to_pandas(joined)
        return out

    @staticmethod
    def compute_dr(
        df: Union[pd.DataFrame, "pl.DataFrame"],  # type: ignore[name-defined]
        column_date: str = 'date',
        column_close: str = 'close',
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Compute daily returns (dr) for each ticker in the prices DataFrame.
        
        Args:
            df: DataFrame containing price data with date and ticker columns
            column_date: Name of the date column (default: 'date')
            column_close: Name of the close price column (default: 'close')
        
        Returns:
            DataFrame with daily returns added as 'dr' column (decimal format: 0.02 for 2% gain)
        """
        from alpharank.utils.returns import calculate_daily_returns
        
        return calculate_daily_returns(
            df=df,
            price_column=column_close,
            date_column=column_date,
            ticker_column='ticker',
            return_column='dr',
            backend=backend,
        )  

    @staticmethod
    def prices_vs_index(
        index: Union[pd.DataFrame, "pl.DataFrame"],  # type: ignore[name-defined]
        prices: Union[pd.DataFrame, "pl.DataFrame"],  # type: ignore[name-defined]
        column_close_index: str,
        column_close_prices: str,
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
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
        
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for PricesDataPreprocessor.prices_vs_index.")
        idx = to_pandas(index)
        px = to_pandas(prices)
        if column_close_index == column_close_prices:
            idx = idx.rename(columns={column_close_index: column_close_index + "_index"})
            column_close_index = column_close_index + "_index"

        if 'ticker' not in idx.columns:
            idx['ticker'] = 'index'

        idx = PricesDataPreprocessor.augment_prices(
            df=idx.copy(),
            columns_to_augment=[column_close_index],
            column_date='date',
            backend=backend_name,
        )
        idx = idx.drop(['ticker'], axis=1, errors='ignore')
        require_polars()
        pl_px = to_polars(px).with_columns(pl.col("date").cast(pl.Date, strict=False))
        pl_idx = to_polars(idx).with_columns(pl.col("date").cast(pl.Date, strict=False))
        prices_augmented = to_pandas(
            pl_px.join(pl_idx, on="date", how="left").with_columns(
                (pl.col(column_close_prices) / pl.col(column_close_index)).alias("close_vs_index")
            )
        )
        prices_augmented["date"] = pd.to_datetime(prices_augmented["date"], errors="coerce")

        prices_augmented = PricesDataPreprocessor.compute_dr(
            df=prices_augmented.copy(),
            column_date='date',
            column_close='close_vs_index',
            backend=backend_name,
        )
        prices_augmented.rename(columns={"dr": "dr_vs_index"}, inplace=True)
        return prices_augmented
    
    @staticmethod
    def calculate_monthly_returns(
        df: Union[pd.DataFrame, "pl.DataFrame"],  # type: ignore[name-defined]
        column_date: str = 'date',
        column_close: str = 'close',
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Calculate monthly returns from daily price data.
        
        Args:
            df: DataFrame containing daily price data
            column_date: Name of the date column
            column_close: Name of the close price column
            
        Returns:
            DataFrame with year_month and monthly returns (decimal format: 0.02 for 2% gain)
        """
        from alpharank.utils.returns import calculate_monthly_returns as calc_monthly_returns
        
        return calc_monthly_returns(
            df=df,
            price_column=column_close,
            date_column=column_date,
            ticker_column='ticker',
            return_column='monthly_return',
            backend=backend,
        )

class FundamentalProcessor :
    """
    A professional-grade analyzer for processing and calculating financial metrics.

    This class provides a suite of static methods designed to clean, merge, and analyze
    raw financial statement data (Income, Balance Sheet, Cash Flow) from sources
    like EODHD. It calculates Trailing Twelve Month (TTM) figures, computes a wide
    array of fundamental and valuation ratios, and prepares the data for
    quantitative analysis and backtesting.
    """

    @staticmethod
    def calculate_fundamental_ratios(
        balance: pd.DataFrame, 
        cashflow: pd.DataFrame, 
        income: pd.DataFrame, 
        earnings: pd.DataFrame,
        list_kpi_toincrease: List[str], 
        list_ratios_toincrease: List[str],
        list_kpi_toaccelerate: List[str], 
        list_lag_increase: List[int],
        list_ratios_to_augment: List[str], 
        list_date_to_maximise: List[str],
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Processes raw financial statements to compute TTM metrics and fundamental ratios.

        This method takes quarterly data, selects the latest filing for each period,
        calculates Trailing Twelve Month (TTM) figures for key metrics, and then
        computes a comprehensive set of profitability, solvency, and efficiency ratios.
        It also calculates growth and acceleration metrics as specified.

        Args:
            balance (pd.DataFrame): DataFrame with quarterly balance sheet data.
            cashflow (pd.DataFrame): DataFrame with quarterly cash flow data.
            income (pd.DataFrame): DataFrame with quarterly income statement data.
            earnings (pd.DataFrame): DataFrame with quarterly earnings (EPS) data.
            list_kpi_toincrease (List[str]): List of KPIs for wfrom .fundamental_analyzer import FundamentalAnalyzerhich to calculate growth rate.
            list_ratios_toincrease (List[str]): List of ratios for which to calculate the simple difference.
            list_kpi_toaccelerate (List[str]): List of KPIs for which to calculate acceleration.
            list_lag_increase (List[int]): List of lag periods for growth/acceleration calculations.
            list_ratios_to_augment (List[str]): List of ratios for augmenting factor calculation.
            list_date_to_maximise (List[str]): Date columns to determine the final report date.

        Returns:
            pd.DataFrame: A DataFrame with tickers and quarterly dates, containing calculated TTM
                          metrics and fundamental ratios.
        """
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for FundamentalProcessor.calculate_fundamental_ratios.")

        require_polars()
        balance_cols_to_roll = ['totalStockholderEquity', 'netDebt', 'commonStockSharesOutstanding', 'totalAssets', 'cashAndShortTermInvestments']
        income_cols_to_annualize = ['totalRevenue', 'grossProfit', 'operatingIncome', 'incomeBeforeTax', 'netIncome', 'ebit', 'ebitda']

        pl_balance = (
            to_polars(balance[['ticker', 'date', 'filing_date', 'commonStockSharesOutstanding', 'totalStockholderEquity', 'netDebt', 'totalAssets', 'cashAndShortTermInvestments']])
            .with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Date, strict=False).alias('date'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date'),
                pl.col('commonStockSharesOutstanding').cast(pl.Float64, strict=False).alias('commonStockSharesOutstanding'),
                pl.col('totalStockholderEquity').cast(pl.Float64, strict=False).alias('totalStockholderEquity'),
                pl.col('netDebt').cast(pl.Float64, strict=False).alias('netDebt'),
                pl.col('totalAssets').cast(pl.Float64, strict=False).alias('totalAssets'),
                pl.col('cashAndShortTermInvestments').cast(pl.Float64, strict=False).alias('cashAndShortTermInvestments'),
                pl.col('date').cast(pl.Date, strict=False).dt.truncate('1q').alias('quarter_end'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date_balance'),
            ])
            .sort(['ticker', 'filing_date_balance'])
            .group_by(['ticker', 'quarter_end'])
            .agg([
                pl.col('filing_date_balance').last().alias('filing_date_balance'),
                pl.col('commonStockSharesOutstanding').last().alias('commonStockSharesOutstanding'),
                pl.col('totalStockholderEquity').last().alias('totalStockholderEquity'),
                pl.col('netDebt').last().alias('netDebt'),
                pl.col('totalAssets').last().alias('totalAssets'),
                pl.col('cashAndShortTermInvestments').last().alias('cashAndShortTermInvestments'),
            ])
            .sort(['ticker', 'filing_date_balance'])
        )
        pl_balance = pl_balance.with_columns([
            pl.col(col).rolling_mean(window_size=4, min_samples=1).over('ticker').alias(f"{col.lower()}_rolling")
            for col in balance_cols_to_roll
        ])

        pl_earnings = (
            to_polars(earnings[['ticker', 'date', 'reportDate', 'epsActual']])
            .with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Date, strict=False).alias('date'),
                pl.col('reportDate').cast(pl.Date, strict=False).alias('reportDate'),
                pl.col('epsActual').cast(pl.Float64, strict=False).alias('epsActual'),
                pl.col('date').cast(pl.Date, strict=False).dt.truncate('1q').alias('quarter_end'),
                pl.col('reportDate').cast(pl.Date, strict=False).alias('filing_date_earning'),
            ])
            .sort(['ticker', 'filing_date_earning'])
            .group_by(['ticker', 'quarter_end'])
            .agg([
                pl.col('filing_date_earning').last().alias('filing_date_earning'),
                pl.col('epsActual').last().alias('epsActual'),
            ])
            .filter(pl.col('epsActual').is_not_null())
            .sort(['ticker', 'filing_date_earning'])
            .with_columns(
                (pl.col('epsActual').rolling_mean(window_size=4, min_samples=1).over('ticker') * 4.0).alias('epsactual_rolling')
            )
        )

        pl_income = (
            to_polars(income[['ticker', 'date', 'filing_date'] + income_cols_to_annualize])
            .with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Date, strict=False).alias('date'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date'),
                pl.col('date').cast(pl.Date, strict=False).dt.truncate('1q').alias('quarter_end'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date_income'),
            ])
            .sort(['ticker', 'filing_date_income'])
            .group_by(['ticker', 'quarter_end'])
            .agg(
                [pl.col('filing_date_income').last().alias('filing_date_income')]
                + [pl.col(col).last().alias(col) for col in income_cols_to_annualize]
            )
            .sort(['ticker', 'filing_date_income'])
        )
        pl_income = pl_income.with_columns([
            pl.col(col).cast(pl.Float64, strict=False).alias(col)
            for col in income_cols_to_annualize
        ])
        pl_income = pl_income.with_columns([
            (pl.col(col).rolling_mean(window_size=4, min_samples=1).over('ticker') * 4.0).alias(f"{col.lower()}_rolling")
            for col in income_cols_to_annualize
        ])

        pl_cash = (
            to_polars(cashflow[['ticker', 'date', 'filing_date', 'freeCashFlow']])
            .with_columns([
                pl.col('ticker').cast(pl.Utf8),
                pl.col('date').cast(pl.Date, strict=False).alias('date'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date'),
                pl.col('freeCashFlow').cast(pl.Float64, strict=False).alias('freeCashFlow'),
                pl.col('date').cast(pl.Date, strict=False).dt.truncate('1q').alias('quarter_end'),
                pl.col('filing_date').cast(pl.Date, strict=False).alias('filing_date_cash'),
            ])
            .sort(['ticker', 'filing_date_cash'])
            .group_by(['ticker', 'quarter_end'])
            .agg([
                pl.col('filing_date_cash').last().alias('filing_date_cash'),
                pl.col('freeCashFlow').last().alias('freeCashFlow'),
            ])
            .sort(['ticker', 'filing_date_cash'])
            .with_columns(
                (pl.col('freeCashFlow').rolling_mean(window_size=4, min_samples=1).over('ticker') * 4.0).alias('freecashflow_rolling')
            )
        )

        funda = (
            pl_income
            .join(pl_cash, on=['ticker', 'quarter_end'], how='full', coalesce=True)
            .join(pl_balance, on=['ticker', 'quarter_end'], how='full', coalesce=True)
            .join(
                pl_earnings.select(['ticker', 'quarter_end', 'filing_date_earning', 'epsActual', 'epsactual_rolling']),
                on=['ticker', 'quarter_end'],
                how='full',
                coalesce=True,
            )
            .sort(['ticker', 'quarter_end'])
            .with_columns([
                (pl.col('netincome_rolling') / pl.col('totalrevenue_rolling')).alias('netmargin'),
                (pl.col('ebit_rolling') / pl.col('totalrevenue_rolling')).alias('ebitmargin'),
                (pl.col('ebitda_rolling') / pl.col('totalrevenue_rolling')).alias('ebitdamargin'),
                (pl.col('ebit_rolling') / (pl.col('totalstockholderequity_rolling') + pl.col('netdebt_rolling').fill_null(0))).alias('roic'),
                (pl.col('ebit_rolling') / pl.col('commonstocksharesoutstanding_rolling').fill_null(0)).alias('ebitpershare_rolling'),
                (pl.col('ebitda_rolling') / pl.col('commonstocksharesoutstanding_rolling').fill_null(0)).alias('ebitdapershare_rolling'),
                (pl.col('netincome_rolling') / pl.col('commonstocksharesoutstanding_rolling').fill_null(0)).alias('netincomepershare_rolling'),
                (pl.col('freecashflow_rolling') / pl.col('commonstocksharesoutstanding_rolling').fill_null(0)).alias('fcfpershare_rolling'),
                (pl.col('grossprofit_rolling') / pl.col('totalrevenue_rolling')).alias('gross_margin'),
                (pl.col('netincome_rolling') / pl.col('totalassets_rolling')).alias('return_on_assets'),
                (pl.col('netincome_rolling') / pl.col('totalstockholderequity_rolling')).alias('return_on_equity'),
                (pl.col('netdebt_rolling').fill_null(0) / pl.col('totalstockholderequity_rolling')).alias('debt_to_equity'),
                (pl.col('totalrevenue_rolling') / pl.col('totalassets_rolling')).alias('asset_turnover'),
            ])
        )
        numeric_cols = [
            c for c, d in funda.schema.items()
            if d in {
                pl.Float32, pl.Float64, pl.Int8, pl.Int16, pl.Int32, pl.Int64,
                pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            }
        ]
        if numeric_cols:
            funda = funda.with_columns([
                pl.when(pl.col(c).cast(pl.Float64).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
                for c in numeric_cols
            ])

        if not list_lag_increase:
            list_lag_increase = [1, 4]
        if not list_kpi_toincrease:
            kpi_candidates = ['totalrevenue_rolling', 'ebit_rolling', 'netincome_rolling', 'freecashflow_rolling', 'ebitda_rolling']
            list_kpi_toincrease = [c for c in kpi_candidates if c in funda.columns]
        if not list_kpi_toaccelerate:
            list_kpi_toaccelerate = list_kpi_toincrease[:]
        if not list_ratios_toincrease:
            ratio_candidates = ['gross_margin', 'netmargin', 'return_on_equity', 'debt_to_equity']
            list_ratios_toincrease = [c for c in ratio_candidates if c in funda.columns]
        if not list_ratios_to_augment:
            list_ratios_to_augment = []

        for col in list_kpi_toincrease:
            if col not in funda.columns:
                continue
            for lag in list_lag_increase:
                prev = pl.col(col).shift(lag).over('ticker')
                growth = (pl.col(col) - prev) / prev.abs()
                base = 1 + growth
                funda = funda.with_columns(
                    pl.when(prev.abs() == 0)
                    .then(None)
                    .otherwise(
                        pl.when(base > 0).then(base.pow(4.0 / lag) - 1).otherwise(base)
                    )
                    .alias(f"{col}_lag{lag}")
                )

        for col in list_ratios_toincrease:
            if col not in funda.columns:
                continue
            for lag in list_lag_increase:
                funda = funda.with_columns(
                    (pl.col(col) - pl.col(col).shift(lag).over('ticker')).alias(f"{col}_lag{lag}")
                )

        for col in list_kpi_toaccelerate:
            if col not in funda.columns:
                continue
            for lag in list_lag_increase:
                lag_col = f"{col}_lag{lag}"
                if lag_col in funda.columns:
                    funda = funda.with_columns(
                        (pl.col(lag_col) - pl.col(lag_col).shift(1).over('ticker')).alias(f"{col}_lag{lag}_lag1")
                    )

        filing_cols = [c for c in funda.columns if c.startswith('filing_date_')]
        date_cols = [c for c in (list_date_to_maximise or []) if c in funda.columns] or filing_cols
        funda = funda.drop(['date', 'date_x', 'date_y'], strict=False)
        if date_cols:
            funda = funda.with_columns(pl.max_horizontal([pl.col(c) for c in date_cols]).alias('date'))

        if list_ratios_to_augment:
            funda_pd = to_pandas(funda)
            funda_pd = TechnicalIndicators.augmenting_ratios(funda_pd, list_ratios_to_augment, 'date')
            funda = to_polars(funda_pd)

        funda = funda.drop(balance_cols_to_roll + ['epsActual', 'freeCashFlow'] + income_cols_to_annualize, strict=False)
        return to_pandas(funda)

    @staticmethod
    def calculate_pe_ratios(
        balance: pd.DataFrame, 
        earnings: pd.DataFrame, 
        cashflow: pd.DataFrame, 
        income: pd.DataFrame, 
        earning_choice: str, 
        monthly_return: pd.DataFrame, 
        list_date_to_maximise: List[str],
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Calculates valuation ratios by merging fundamental data with market price data.

        This method first generates fundamental ratios using `calculate_fundamental_ratios`.
        It then merges this data with a time series of market prices (e.g., daily or monthly
        returns) and calculates key valuation ratios like P/E, P/S, P/B, and EV/EBITDA.
        The data is returned as a monthly series, using the last trading day of each month.

        Args:
            balance (pd.DataFrame): DataFrame with quarterly balance sheet data.
            earnings (pd.DataFrame): DataFrame with quarterly earnings (EPS) data.
            cashflow (pd.DataFrame): DataFrame with quarterly cash flow data.
            income (pd.DataFrame): DataFrame with quarterly income statement data.
            earning_choice (str): The column name for the earnings metric to be used (e.g., 'epsactual_rolling').
            monthly_return (pd.DataFrame): DataFrame containing price history, must have 'ticker', 'date', 'last_close'.
            list_date_to_maximise (List[str]): Date columns to determine the final report date.

        Returns:
            pd.DataFrame: A DataFrame with one row per ticker per month, containing key
                          valuation ratios and market cap data.
        """
        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for FundamentalProcessor.calculate_pe_ratios.")

        # --- 1. Get Base Fundamental Data ---
        fundamental = FundamentalProcessor.calculate_fundamental_ratios(
            balance=balance, cashflow=cashflow, income=income, earnings=earnings,
            list_kpi_toincrease=[], list_ratios_toincrease=[],
            list_kpi_toaccelerate=[], list_lag_increase=[],
            list_ratios_to_augment=[], list_date_to_maximise=list_date_to_maximise,
            backend=backend_name,
        )

        # --- 2. Select Earnings Metric ---
        if earning_choice != 'epsactual_rolling':
            fundamental['rolling_epsactual'] = fundamental[earning_choice] / fundamental['commonstocksharesoutstanding_rolling']
        else:
            fundamental['rolling_epsactual'] = fundamental['epsactual_rolling']
        
        # --- 3. Merge Fundamental Data with Price Data ---
        funda_cols = [
            'ticker', 'date', 'rolling_epsactual', 'commonstocksharesoutstanding_rolling',
            'totalrevenue_rolling', 'totalstockholderequity_rolling', 'netdebt_rolling',
            'cashandshortterminvestments_rolling', 'ebitda_rolling'
        ]
        
        output_cols = ['ticker', 'year_month', 'pe', 'ps_ratio', 'pb_ratio', 'ev_ebitda_ratio', 'market_cap']
        ffill_cols = [
            'last_close', 'rolling_epsactual', 'commonstocksharesoutstanding_rolling',
            'totalrevenue_rolling', 'totalstockholderequity_rolling', 'netdebt_rolling',
            'cashandshortterminvestments_rolling', 'ebitda_rolling'
        ]

        require_polars()
        mret = monthly_return.copy()
        mret = mret.drop(columns=['year_month'], errors='ignore')
        mret['date'] = pd.to_datetime(mret['date'], errors='coerce').dt.normalize()
        fcopy = fundamental.copy()
        fcopy['date'] = pd.to_datetime(fcopy['date'], errors='coerce').dt.normalize()
        pl_monthly = to_polars(mret).with_columns(pl.col('date').cast(pl.Date))
        pl_funda = to_polars(fcopy[funda_cols]).with_columns(pl.col('date').cast(pl.Date))
        merged = (
            pl_monthly.join(pl_funda, on=['ticker', 'date'], how='full', coalesce=True)
            .sort(['ticker', 'date'])
            .with_columns([pl.col(c).forward_fill().over('ticker').alias(c) for c in ffill_cols])
            .filter(pl.col('last_close').is_not_null() & pl.col('rolling_epsactual').is_not_null())
            .with_columns([
                (pl.col('last_close') * pl.col('commonstocksharesoutstanding_rolling').cast(pl.Float64)).alias('market_cap'),
            ])
            .with_columns([
                (pl.col('market_cap') + pl.col('netdebt_rolling').fill_null(0)).alias('enterprise_value'),
                (pl.col('last_close') / pl.col('rolling_epsactual')).alias('pe'),
                (pl.col('market_cap') / pl.col('totalrevenue_rolling')).alias('ps_ratio'),
                (pl.col('market_cap') / pl.col('totalstockholderequity_rolling')).alias('pb_ratio'),
                pl.col('date').dt.truncate('1mo').alias('year_month'),
            ])
            .with_columns((pl.col('enterprise_value') / pl.col('ebitda_rolling')).alias('ev_ebitda_ratio'))
            .with_columns([
                pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
                for c in ['pe', 'ps_ratio', 'pb_ratio', 'ev_ebitda_ratio', 'market_cap']
            ])
            .sort(['ticker', 'year_month', 'date'])
            .group_by(['ticker', 'year_month'])
            .agg(pl.all().last())
        )
        result = to_pandas(merged.select(output_cols))
        result = normalize_year_month_to_period(result, 'year_month')
        return result

    @staticmethod
    def calculate_all_ratios(
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        income_statement: pd.DataFrame,
        earnings: pd.DataFrame,
        monthly_return: pd.DataFrame,
        backend: Optional[Backend] = None,
    ) -> pd.DataFrame:
        """
        Build a unified monthly dataset:
        - Fundamental ratios (TTM and base ratios)
        - Valuation ratios (Price/EBIT, Price/EBITDA, Price/NetIncome, P/S, P/B, EV/EBITDA)
        """
        # Define standard financial KPIs for growth and acceleration
        kpis_growth = [
            'totalrevenue_rolling', 'netincome_rolling', 'ebitda_rolling', 
            'ebit_rolling', 'freecashflow_rolling', 'epsactual_rolling'
        ]
        
        ratios_diff = [
            'gross_margin', 'netmargin', 'return_on_equity', 
            'return_on_assets', 'debt_to_equity'
        ]
        
        kpis_accel = [
            'totalrevenue_rolling', 'netincome_rolling', 'ebitda_rolling', 'epsactual_rolling'
        ]

        backend_name = ensure_backend_name(backend, default="polars")
        if backend_name != "polars":
            raise ValueError("Pandas backend is disabled for FundamentalProcessor.calculate_all_ratios.")

        # Compute fundamentals with dynamic defaults (no hard-coded lists)
        fundamentals_df = FundamentalProcessor.calculate_fundamental_ratios(
            balance=balance_sheet,
            cashflow=cash_flow,
            income=income_statement,
            earnings=earnings,
            list_kpi_toincrease=kpis_growth,
            list_ratios_toincrease=ratios_diff,
            list_kpi_toaccelerate=kpis_accel,
            list_lag_increase=[1, 4],
            list_ratios_to_augment=[],
            list_date_to_maximise=[],            # auto-select filing_date_* columns
            backend=backend_name,
        )

        require_polars()
        mret = monthly_return.copy()
        fdf = fundamentals_df.copy()
        mret['date'] = pd.to_datetime(mret['date'], errors='coerce').dt.normalize()
        fdf['date'] = pd.to_datetime(fdf['date'], errors='coerce').dt.normalize()
        if 'quarter_end' in fdf.columns:
            fdf = normalize_year_month_to_timestamp(fdf, col='quarter_end')

        pl_mret = to_polars(mret).with_columns(pl.col('date').cast(pl.Date))
        pl_fdf = to_polars(fdf).with_columns(pl.col('date').cast(pl.Date))
        ffill_cols = [c for c in fdf.columns if c not in ['ticker', 'date']]

        combined = (
            pl_mret.join(pl_fdf, on=['ticker', 'date'], how='full', coalesce=True)
            .sort(['ticker', 'date'])
            .with_columns([pl.col(c).forward_fill().over('ticker').alias(c) for c in ffill_cols])
        )
        if 'quarter_end' in combined.columns:
            combined = combined.filter(pl.col('quarter_end').is_not_null())

        combined = combined.with_columns([
            (pl.col('last_close') * pl.col('commonstocksharesoutstanding_rolling').cast(pl.Float64)).alias('market_cap'),
        ]).with_columns([
            (
                pl.col('market_cap')
                + pl.col('netdebt_rolling').fill_null(0)
                - pl.col('cashandshortterminvestments_rolling').fill_null(0)
            ).alias('enterprise_value'),
        ]).with_columns([
            (pl.col('market_cap') / pl.col('ebit_rolling')).alias('pebit'),
            (pl.col('market_cap') / pl.col('ebitda_rolling')).alias('pebitda'),
            (pl.col('market_cap') / pl.col('netincome_rolling')).alias('pnetresult'),
            (pl.col('market_cap') / pl.col('freecashflow_rolling')).alias('pfcf'),
            (pl.col('market_cap') / pl.col('totalrevenue_rolling')).alias('ps_ratio'),
            (pl.col('market_cap') / pl.col('totalstockholderequity_rolling')).alias('pb_ratio'),
            (pl.col('enterprise_value') / pl.col('ebitda_rolling')).alias('ev_ebitda_ratio'),
            pl.col('date').dt.truncate('1mo').alias('year_month'),
        ]).with_columns([
            pl.when(pl.col(c).is_infinite()).then(None).otherwise(pl.col(c)).alias(c)
            for c in ['market_cap', 'enterprise_value', 'pebit', 'pebitda', 'pnetresult', 'pfcf', 'ps_ratio', 'pb_ratio', 'ev_ebitda_ratio']
        ])

        final_df = (
            combined.sort(['ticker', 'year_month', 'date'])
            .group_by(['ticker', 'year_month'])
            .agg(pl.all().last())
        )
        out = to_pandas(final_df)
        out = normalize_year_month_to_period(out, 'year_month')
        return out
