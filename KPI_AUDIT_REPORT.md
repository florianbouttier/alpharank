# FundamentalProcessor Audit Report

## 1. Overview
The `FundamentalProcessor` class (in `src/alpharank/data/processing.py`) is responsible for transforming raw quarterly financial statements into a structured, monthly time-series of fundamental KPIs and valuation ratios for each ticker.

It handles:
1.  **Data Cleaning**: Selecting latest filings, handling duplicates.
2.  **TTM Calculation**: Converting quarterly data into Trailing Twelve Month (TTM) figures.
3.  **Ratio Computation**: Calculating profitability, efficiency, solvency, and valuation ratios.
4.  **Growth & Acceleration**: Computing annualized growth rates and acceleration metrics.
5.  **Alignment**: Merging fundamental data with market price data and forward-filling to create a monthly dataset.

## 2. Processing Logic

### 2.1. TTM (Trailing Twelve Months) Calculation
The processor converts quarterly data into TTM figures to remove seasonality and provide a full-year view.

*   **Method**: It uses a rolling sum/average over the last 4 quarters.
*   **Implementation**: `4 * SMA(n=4)` (Simple Moving Average of the last 4 quarters).
    *   *Note*: For flow items (Revenue, Income), `4 * Average` is equivalent to `Sum` of the last 4 quarters.
    *   *Note*: For stock items (Assets, Equity), this effectively smooths the balance sheet over the last year.

### 2.2. Financial Ratios
Ratios are calculated using the TTM figures to ensure consistency (e.g., Income TTM / Revenue TTM).

### 2.3. Growth and Acceleration
*   **Growth (`_lagX`)**: Calculated using `TechnicalIndicators.increase(diff=False)`.
    *   **Logic**: `((Current / Previous) ^ (4/n)) - 1`
    *   **Key Feature**: Growth rates are **annualized**. A 1-quarter growth is annualized to a yearly rate.
*   **Acceleration (`_lagX_lag1`)**: The quarter-over-quarter change in the growth rate.
    *   **Logic**: `Growth_Current - Growth_Previous` (Simple difference of rates).

### 2.4. Valuation Ratios
Valuation ratios merge fundamental data with market prices (`last_close` of the month).
*   **Market Cap**: `Price * Shares Outstanding`
*   **Enterprise Value (EV)**: `Market Cap + Net Debt - Cash`

## 3. KPI Dictionary

The following KPIs are generated and available for analysis/backtesting.

### 3.1. Rolling Fundamentals (TTM)
| KPI Name | Source | Description |
| :--- | :--- | :--- |
| `totalrevenue_rolling` | Income | Total Revenue (TTM) |
| `grossprofit_rolling` | Income | Gross Profit (TTM) |
| `operatingincome_rolling` | Income | Operating Income (TTM) |
| `incomebeforetax_rolling` | Income | Income Before Tax (TTM) |
| `netincome_rolling` | Income | Net Income (TTM) |
| `ebit_rolling` | Income | EBIT (TTM) |
| `ebitda_rolling` | Income | EBITDA (TTM) |
| `freecashflow_rolling` | Cash Flow | Free Cash Flow (TTM) |
| `epsactual_rolling` | Earnings | Earnings Per Share (TTM) |
| `commonstocksharesoutstanding_rolling` | Balance | Shares Outstanding (Smoothed) |
| `totalstockholderequity_rolling` | Balance | Shareholders' Equity (Smoothed) |
| `netdebt_rolling` | Balance | Net Debt (Smoothed) |
| `totalassets_rolling` | Balance | Total Assets (Smoothed) |
| `cashandshortterminvestments_rolling` | Balance | Cash & Equivalents (Smoothed) |

### 3.2. Fundamental Ratios
| KPI Name | Formula | Category |
| :--- | :--- | :--- |
| `netmargin` | Net Income / Revenue | Profitability |
| `ebitmargin` | EBIT / Revenue | Profitability |
| `ebitdamargin` | EBITDA / Revenue | Profitability |
| `gross_margin` | Gross Profit / Revenue | Profitability |
| `roic` | EBIT / (Equity + Net Debt) | Efficiency |
| `return_on_assets` | Net Income / Total Assets | Efficiency |
| `return_on_equity` | Net Income / Equity | Efficiency |
| `debt_to_equity` | Net Debt / Equity | Solvency |
| `asset_turnover` | Revenue / Total Assets | Efficiency |

### 3.3. Per Share Metrics
| KPI Name | Formula |
| :--- | :--- |
| `ebitpershare_rolling` | EBIT / Shares |
| `ebitdapershare_rolling` | EBITDA / Shares |
| `netincomepershare_rolling` | Net Income / Shares |
| `fcfpershare_rolling` | FCF / Shares |

### 3.4. Valuation Ratios
| KPI Name | Formula | Description |
| :--- | :--- | :--- |
| `market_cap` | Price * Shares | Market Capitalization |
| `enterprise_value` | Mkt Cap + Net Debt - Cash | Enterprise Value |
| `pe` (or `pnetresult`) | Price / EPS | Price to Earnings |
| `ps_ratio` | Mkt Cap / Revenue | Price to Sales |
| `pb_ratio` | Mkt Cap / Equity | Price to Book |
| `pebit` | Mkt Cap / EBIT | Price to EBIT |
| `pebitda` | Mkt Cap / EBITDA | Price to EBITDA |
| `pfcf` | Mkt Cap / FCF | Price to Free Cash Flow |
| `ev_ebitda_ratio` | EV / EBITDA | EV to EBITDA |

### 3.5. Growth & Acceleration Metrics
For selected KPIs (Revenue, EBIT, Net Income, FCF, EBITDA) and Ratios (Margins, ROE, D/E), the following variations are calculated:

*   `{kpi}_lag1`: Annualized growth rate over the last 1 quarter.
*   `{kpi}_lag4`: Annualized growth rate over the last 4 quarters (YoY).
*   `{kpi}_lag1_lag1`: Acceleration (change in 1-quarter growth rate vs previous quarter).
*   `{kpi}_lag4_lag1`: Acceleration (change in YoY growth rate vs previous quarter).

## 4. Audit Findings & Recommendations

### 🟢 Resolved Issues
1.  **Missing Method**: The code calls `TechnicalIndicators.augmenting_ratios`.
    *   *Status*: The method has been fully implemented in `src/alpharank/features/indicators.py`. It calculates the "days since last negative" for specified KPIs, effectively measuring positive streaks or recovery time.

### 🟡 Observations
1.  **Balance Sheet Smoothing**: Balance sheet items (Assets, Equity, Debt) are smoothed using a 4-quarter average.
    *   *Context*: Standard practice often uses the latest value or a 2-point average (Start/End). Using a 4-point average makes the ratios less sensitive to recent changes in capital structure. This is a design choice but worth noting.
2.  **Growth Annualization**: The `TechnicalIndicators.increase` function annualizes growth rates (e.g., `(1+r)^4 - 1` for quarterly growth).
    *   *Context*: This allows comparing growth rates across different time horizons (QoQ vs YoY) on the same scale.
3.  **Forward Filling**: Fundamental data is forward-filled to daily/monthly frequency.
    *   *Context*: This assumes the last known fundamental data remains valid until a new report is filed. This is standard practice in backtesting to avoid look-ahead bias (using `filing_date`).

## 5. Usage Example
To generate the full dataset:

```python
from alpharank.data.processing import FundamentalProcessor

final_df = FundamentalProcessor.calculate_all_ratios(
    balance_sheet=df_balance,
    cash_flow=df_cash,
    income_statement=df_income,
    earnings=df_earnings,
    monthly_return=df_prices
)
```
