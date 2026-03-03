import pandas as pd
import numpy as np
from datetime import timedelta
from typing import List

# Assuming the TechnicalIndicators class is in a file within the same package
from .technical_indicators import TechnicalIndicators


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
        list_date_to_maximise: List[str]
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
        # --- 1. Clean and Prepare Balance Sheet Data ---
        balance_clean = (
            balance[['ticker', 'date', 'filing_date', 'commonStockSharesOutstanding', 'totalStockholderEquity', 'netDebt', 'totalAssets', 'cashAndShortTermInvestments']]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
                filing_date_balance=lambda x: pd.to_datetime(x['filing_date'])
            )
            .sort_values('filing_date_balance')
            .groupby(['ticker', 'quarter_end'])
            .last() # Keep only the latest filing for each quarter
            .reset_index()
            .drop(columns=['filing_date'])
        )
        # Calculate TTM (rolling 4-quarter average/sum) for key balance sheet items
        balance_cols_to_roll = ['totalStockholderEquity', 'netDebt', 'commonStockSharesOutstanding', 'totalAssets', 'cashAndShortTermInvestments']
        for col in balance_cols_to_roll:
            balance_clean[f"{col.lower()}_rolling"] = balance_clean.sort_values('filing_date_balance').groupby('ticker')[col].transform(lambda x: TechnicalIndicators.sma(x, n=4))

        # --- 2. Clean and Prepare Earnings Data ---
        earnings_clean = (
            earnings[['ticker', 'date', 'reportDate', 'epsActual']]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
                filing_date_earning=lambda x: pd.to_datetime(x['reportDate'])
            )
            .sort_values('filing_date_earning')
            .groupby(['ticker', 'quarter_end'])
            .last()
            .reset_index()
            .drop(columns=['reportDate'])
            .dropna(subset=['epsActual'])
        )
        # Calculate TTM EPS
        earnings_clean['epsactual_rolling'] = earnings_clean.sort_values('filing_date_earning').groupby('ticker')['epsActual'] \
                                                .transform(lambda x: 4 * TechnicalIndicators.sma(x, n=4))

        # --- 3. Clean and Prepare Income Statement Data ---
        income_cols_to_annualize = ['totalRevenue', 'grossProfit', 'operatingIncome', 'incomeBeforeTax', 'netIncome', 'ebit', 'ebitda']
        income_clean = (
            income[['ticker', 'date', 'filing_date'] + income_cols_to_annualize]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
                filing_date_income=lambda x: pd.to_datetime(x['filing_date']))
            .sort_values('filing_date_income')
            .groupby(['ticker', 'quarter_end'])
            .last()
            .reset_index()
            .drop(columns=['filing_date'])
        )
        # Calculate TTM for income statement items
        for col in income_cols_to_annualize:
            income_clean[f"{col.lower()}_rolling"] = income_clean.sort_values('filing_date_income').groupby('ticker')[col].transform(lambda x: 4 * TechnicalIndicators.sma(x, n=4))

        # --- 4. Clean and Prepare Cash Flow Data ---
        cash_clean = (
            cashflow[['ticker', 'date', 'filing_date', 'freeCashFlow']]
            .assign(
                quarter_end=lambda x: pd.to_datetime(x['date']).dt.to_period('Q'),
                filing_date_cash=lambda x: pd.to_datetime(x['filing_date']))
            .sort_values('filing_date_cash')
            .groupby(['ticker', 'quarter_end'])
            .last()
            .reset_index()
            .drop(columns=['filing_date'])
        )
        # Calculate TTM for cash flow items
        for col in ['freeCashFlow']:
            cash_clean[f"{col.lower()}_rolling"] = cash_clean.sort_values('filing_date_cash').groupby('ticker')[col].transform(lambda x: 4 * TechnicalIndicators.sma(x, n=4))

        # --- 5. Merge Data and Calculate Ratios ---
        funda = (income_clean
                 .merge(cash_clean, on=['ticker', 'quarter_end'], how='outer')
                 .merge(balance_clean, on=['ticker', 'quarter_end'], how='outer')
                 .merge(earnings_clean[['ticker', 'quarter_end', 'filing_date_earning', 'epsActual', 'epsactual_rolling']], on=['ticker', 'quarter_end'], how='outer')
                 # Calculate ratios using the TTM figures
                 .assign(
                     # Existing Ratios
                     netmargin=lambda x: x['netincome_rolling'] / x['totalrevenue_rolling'],
                     ebitmargin=lambda x: x['ebit_rolling'] / x['totalrevenue_rolling'],
                     ebitdamargin=lambda x: x['ebitda_rolling'] / x['totalrevenue_rolling'],
                     
                     roic=lambda x: x['ebit_rolling'] / (x['totalstockholderequity_rolling'] + x['netdebt_rolling'].fillna(0)),
                     ebitpershare_rolling=lambda x: x['ebit_rolling'] / x['commonstocksharesoutstanding_rolling'].fillna(0),
                     ebitdapershare_rolling=lambda x: x['ebitda_rolling'] / x['commonstocksharesoutstanding_rolling'].fillna(0),
                     netincomepershare_rolling=lambda x: x['netincome_rolling'] / x['commonstocksharesoutstanding_rolling'].fillna(0),
                     fcfpershare_rolling=lambda x: x['freecashflow_rolling'] / x['commonstocksharesoutstanding_rolling'].fillna(0),
                     # New Recommended Ratios
                     gross_margin=lambda x: x['grossprofit_rolling'] / x['totalrevenue_rolling'],
                     return_on_assets=lambda x: x['netincome_rolling'] / x['totalassets_rolling'],
                     return_on_equity=lambda x: x['netincome_rolling'] / x['totalstockholderequity_rolling'],
                     debt_to_equity=lambda x: x['netdebt_rolling'].fillna(0) / x['totalstockholderequity_rolling'],
                     asset_turnover=lambda x: x['totalrevenue_rolling'] / x['totalassets_rolling']
                 ))
        
        # Replace potential division-by-zero infinities with NaN
        funda.replace([np.inf, -np.inf], np.nan, inplace=True)

        # --- Dynamic defaults if lists are not provided ---
        if not list_lag_increase:
            list_lag_increase = [1, 4]
        # KPIs where we want growth/acceleration (rolling fundamentals)
        if not list_kpi_toincrease:
            kpi_candidates = ['totalrevenue_rolling', 'ebit_rolling', 'netincome_rolling',
                              'freecashflow_rolling', 'ebitda_rolling']
            list_kpi_toincrease = [c for c in kpi_candidates if c in funda.columns]
        if not list_kpi_toaccelerate:
            list_kpi_toaccelerate = list_kpi_toincrease[:]
        # Ratios where we want simple change
        if not list_ratios_toincrease:
            ratio_candidates = ['gross_margin', 'netmargin', 'return_on_equity','debt_to_equity']
            list_ratios_toincrease = [c for c in ratio_candidates if c in funda.columns]
        # Ratios to augment
        if not list_ratios_to_augment:
            list_ratios_to_augment = []

        # --- 6. Calculate Growth and Acceleration Metrics (using dynamic lists) ---
        for col in list_kpi_toincrease:
            for lag in list_lag_increase:
                funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: TechnicalIndicators.increase(x, lag, diff=False))

        for col in list_ratios_toincrease:
            for lag in list_lag_increase:
                funda[f"{col}_lag{lag}"] = funda.groupby('ticker')[col].transform(lambda x: TechnicalIndicators.increase(x, lag, diff=True))

        for col in list_kpi_toaccelerate:
            for lag in list_lag_increase:
                funda[f"{col}_lag{lag}_lag1"] = funda.groupby('ticker')[col].transform(
                    lambda x: TechnicalIndicators.increase(TechnicalIndicators.increase(x, lag, diff=False), 1, diff=True))

        # --- 7. Finalize DataFrame with automatic filing-date max (no hard-coded list) ---
        filing_cols = [c for c in funda.columns if c.startswith('filing_date_')]
        date_cols = [c for c in (list_date_to_maximise or []) if c in funda.columns] or filing_cols
        funda = (funda
                 .drop(columns=['date', 'date_x', 'date_y'], errors='ignore')
                 .assign(date=funda[date_cols].max(axis=1)))

        if list_ratios_to_augment:
            funda = TechnicalIndicators.augmenting_ratios(funda, list_ratios_to_augment, 'date')
            
        funda = funda.drop(columns=balance_cols_to_roll + ['epsActual','freeCashFlow']+ income_cols_to_annualize)

        return funda

    @staticmethod
    def calculate_pe_ratios(
        balance: pd.DataFrame, 
        earnings: pd.DataFrame, 
        cashflow: pd.DataFrame, 
        income: pd.DataFrame, 
        earning_choice: str, 
        monthly_return: pd.DataFrame, 
        list_date_to_maximise: List[str]
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
        # --- 1. Get Base Fundamental Data ---
        fundamental = FundamentalProcessor.calculate_fundamental_ratios(
            balance=balance, cashflow=cashflow, income=income, earnings=earnings,
            list_kpi_toincrease=[], list_ratios_toincrease=[],
            list_kpi_toaccelerate=[], list_lag_increase=[],
            list_ratios_to_augment=[], list_date_to_maximise=list_date_to_maximise
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
        
        monthly_return['date'] = pd.to_datetime(monthly_return['date'])
        price_merge = (monthly_return
                       .merge(fundamental[funda_cols], on=['ticker', 'date'], how='outer')
                       .sort_values(by=['ticker', 'date']))
        
        # Forward-fill fundamental data to align with daily/monthly prices
        # This assumes the last reported fundamental data is valid until a new report is released.
        ffill_cols = [
            'last_close', 'rolling_epsactual', 'commonstocksharesoutstanding_rolling',
            'totalrevenue_rolling', 'totalstockholderequity_rolling', 'netdebt_rolling',
            'cashandshortterminvestments_rolling', 'ebitda_rolling'
        ]
        price_merge[ffill_cols] = price_merge.groupby('ticker')[ffill_cols].ffill()
        price_merge = price_merge.dropna(subset=['last_close', 'rolling_epsactual'])

        # --- 4. Calculate Valuation Ratios ---
        price_merge['market_cap'] = price_merge['last_close'] * pd.to_numeric(price_merge['commonstocksharesoutstanding_rolling'])
        
        # Enterprise Value (EV) = Market Cap + Net Debt
        price_merge['enterprise_value'] = price_merge['market_cap'] + price_merge['netdebt_rolling'].fillna(0)
        
        # Standard Valuation Ratios
        price_merge['pe'] = price_merge['last_close'] / price_merge['rolling_epsactual']
        price_merge['ps_ratio'] = price_merge['market_cap'] / price_merge['totalrevenue_rolling']
        price_merge['pb_ratio'] = price_merge['market_cap'] / price_merge['totalstockholderequity_rolling']
        price_merge['ev_ebitda_ratio'] = price_merge['enterprise_value'] / price_merge['ebitda_rolling']
        
        # Replace potential infinities from division-by-zero with NaN
        price_merge.replace([np.inf, -np.inf], np.nan, inplace=True)

        # --- 5. Resample to Last Day of the Month ---
        price_merge['year_month'] = pd.to_datetime(price_merge['date']).dt.to_period('M')
        price_merge_last_day = price_merge.groupby(['ticker', 'year_month']).last().reset_index()

        output_cols = [
            'ticker', 'year_month', 'pe', 'ps_ratio', 'pb_ratio', 
            'ev_ebitda_ratio', 'market_cap'
        ]
        
        return price_merge_last_day[output_cols]

    @staticmethod
    def calculate_all_ratios(
        balance_sheet: pd.DataFrame,
        cash_flow: pd.DataFrame,
        income_statement: pd.DataFrame,
        earnings: pd.DataFrame,
        monthly_return: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build a unified monthly dataset:
        - Fundamental ratios (TTM and base ratios)
        - Valuation ratios (Price/EBIT, Price/EBITDA, Price/NetIncome, P/S, P/B, EV/EBITDA)
        """
        # Compute fundamentals with dynamic defaults (no hard-coded lists)
        fundamentals_df = FundamentalProcessor.calculate_fundamental_ratios(
            balance=balance_sheet,
            cashflow=cash_flow,
            income=income_statement,
            earnings=earnings,
            list_kpi_toincrease=[],              # dynamic defaults inside
            list_ratios_toincrease=[],
            list_kpi_toaccelerate=[],
            list_lag_increase=[],
            list_ratios_to_augment=[],
            list_date_to_maximise=[]            # auto-select filing_date_* columns
        )

        # Merge with prices and compute valuations
        monthly_return = monthly_return.copy()
        monthly_return['date'] = pd.to_datetime(monthly_return['date'])
        fundamentals_df['date'] = pd.to_datetime(fundamentals_df['date'])

        combined_df = (monthly_return
                       .merge(fundamentals_df, on=['ticker', 'date'], how='outer')
                       .sort_values(['ticker', 'date']))

        ffill_cols = [c for c in fundamentals_df.columns if c not in ['ticker', 'date']]
        combined_df[ffill_cols] = combined_df.groupby('ticker')[ffill_cols].ffill()

        if 'quarter_end' in combined_df.columns:
            combined_df.dropna(subset=['quarter_end'], inplace=True)

        combined_df = combined_df.assign(
            market_cap=lambda x: x['last_close'] * pd.to_numeric(x.get('commonstocksharesoutstanding_rolling')),
            enterprise_value=lambda x: x['market_cap'] + x.get('netdebt_rolling', 0).fillna(0) - x.get('cashandshortterminvestments_rolling', 0).fillna(0),
            pebit=lambda x: x['market_cap'] / x.get('ebit_rolling'),
            pebitda=lambda x: x['market_cap'] / x.get('ebitda_rolling'),
            pnetresult=lambda x: x['market_cap'] / x.get('netincome_rolling'),
            pfcf=lambda x: x['market_cap'] / x.get('freecashflow_rolling'),
            ps_ratio=lambda x: x['market_cap'] / x.get('totalrevenue_rolling'),
            pb_ratio=lambda x: x['market_cap'] / x.get('totalstockholderequity_rolling'),
            ev_ebitda_ratio=lambda x: x['enterprise_value'] / x.get('ebitda_rolling'),
        )

        combined_df.replace([np.inf, -np.inf], np.nan, inplace=True)
        combined_df['year_month'] = pd.to_datetime(combined_df['date']).dt.to_period('M')
        final_df = combined_df.groupby(['ticker', 'year_month']).last().reset_index()

        return final_df