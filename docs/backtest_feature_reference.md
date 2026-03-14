# Backtest Feature Reference

This document is the source of truth for the `scripts/run_backtest.py` / `alpharank.backtest` data model.

It describes:

- raw inputs and file mapping
- monthly price construction
- technical features
- fundamental features
- target construction
- timing semantics (`decision_month` vs `holding_month`)
- missing-value handling and feature filtering

## Scope

Main code paths:

- `src/alpharank/backtest/config.py`
- `src/alpharank/backtest/data_loading.py`
- `src/alpharank/backtest/features.py`
- `src/alpharank/backtest/fundamentals.py`
- `src/alpharank/backtest/datasets.py`
- `src/alpharank/backtest/pipeline.py`
- `src/alpharank/backtest/portfolio.py`

## Raw Inputs

The backtest loads:

- `US_Finalprice.parquet`: equity daily prices
- `US_Income_statement.parquet`
- `US_Balance_sheet.parquet`
- `US_Cash_flow.parquet`
- `US_Earnings.parquet`
- `SP500_Constituents.csv`
- `SP500Price.parquet`

Price column resolution:

- first existing column among `adjusted_close`, `close`, `adj_close`

Ticker membership filter:

- constituents are normalized to `*.US`
- the monthly modeling frame is inner-joined on `(ticker, year_month)` against the monthly constituent list

## Time Semantics

The backtest uses two distinct month labels:

- `decision_month`: month on which the signal is built
- `holding_month`: next month on which the realized return is measured

Compatibility note:

- `year_month` is kept in several outputs and is equal to `decision_month`

Interpretation:

- if `decision_month = 2010-05-01`
- then the portfolio decision is taken using information available at the end of May 2010
- and the realized test return is measured over June 2010, stored in `holding_month = 2010-06-01`

As-of dates:

- `decision_asof_date`: last daily date used inside the decision month
- `holding_asof_date`: last daily date used inside the holding month for the stock
- `benchmark_holding_asof_date`: last daily date used inside the holding month for the benchmark

Completeness flag:

- `holding_period_complete = 1`
- iff both `holding_asof_date` and `benchmark_holding_asof_date` are within the last 7 calendar days of the holding month
- this is a pragmatic completeness heuristic, not an exchange-calendar proof

## Monthly Prices

For each ticker `i` and month `t`:

- `P_{i,t}` = last available close in month `t`
- `monthly_return_{i,t} = P_{i,t} / P_{i,t-1} - 1`

Implementation:

- daily prices are grouped by `(ticker, year_month)`
- the last date and last close are kept
- `pct_change()` over ticker produces `monthly_return`

Benchmark:

- same logic on SP500 monthly closes
- `index_monthly_return_t = I_t / I_{t-1} - 1`

## Strict 1M Forward Target

For each ticker `i` and decision month `t`:

- let `next_observed_month_i(t)` be the next available month for ticker `i`
- the target is kept only if `next_observed_month_i(t) = t + 1 month`

This avoids accidental multi-month horizons when a ticker has gaps in monthly history.

Constructed fields:

- `holding_month_{i,t} = t + 1 month` only if the next observed month is exactly one month later, else `null`
- `future_return_{i,t} = monthly_return_{i,t+1}` only in that strict 1M case, else `null`
- `benchmark_future_return_t = index_monthly_return_{t+1}` joined on `holding_month`

Derived target variables:

- `future_excess_return_{i,t} = future_return_{i,t} - benchmark_future_return_t`
- `future_relative_return_{i,t} = (1 + future_return_{i,t}) / (1 + benchmark_future_return_t) - 1`

Binary classification label used by the model:

- `target_label_{i,t} = 1[ future_excess_return_{i,t} > outperformance_threshold ]`

Current entrypoint preset in `scripts/run_backtest.py`:

- `outperformance_threshold = 0.15`

## Feature Configuration

The modern backtest path is now explicitly parameterized.

`TechnicalFeatureConfig` controls:

- `roc_windows`
- `ema_pairs`
- `price_to_ema_spans`
- `rsi_windows`
- `rsi_ratio_pairs`
- `bollinger_windows`
- `stochastic_windows`
- `range_windows`
- `volatility_windows`
- `volatility_ratio_pairs`

`FundamentalFeatureConfig` controls:

- `quarterly_growth_lags`

Current preset in `scripts/run_backtest.py`:

- `roc_windows = (1, 3, 6, 12)`
- `ema_pairs = ((2, 6), (3, 6), (3, 12), (6, 12), (6, 18), (12, 24))`
- `price_to_ema_spans = (3, 6, 12, 24)`
- `rsi_windows = (3, 6, 12, 24)`
- `rsi_ratio_pairs = ((3, 12), (6, 24))`
- `bollinger_windows = (6, 12)`
- `stochastic_windows = ((6, 3), (12, 3))`
- `range_windows = (6, 12)`
- `volatility_windows = (3, 6, 12)`
- `volatility_ratio_pairs = ((3, 12), (6, 12))`
- `quarterly_growth_lags = (1, 4, 12)`

These selected specs are persisted in each run `metadata.json`.

## Technical Features

All technical features are computed on monthly data and grouped by ticker.

Base series:

- `r_t = monthly_return_t`
- `P_t = last_close_t`

The old lag/mean/momentum placeholders (`ret_lag_*`, `ret_mean_*`, `mom_*`) are no longer the core technical feature set.

The current design intentionally uses indicator families and relative transforms.

### ROC / relative price change

For each `w` in `roc_windows`:

- `price_roc_{w}m = P_t / P_{t-w} - 1`

### EMA structure

For each EMA span `s`:

- `ema_s = EWM(P_t, span=s, adjust=False)`

For each pair `(s, l)` in `ema_pairs`:

- `ema_ratio_{s}_{l} = ema_s / ema_l`

For each `s` in `price_to_ema_spans`:

- `price_to_ema_s = P_t / ema_s - 1`

### RSI family

Define:

- `gain_t = max(r_t, 0)`
- `loss_t = max(-r_t, 0)`

For each `w` in `rsi_windows`:

- `avg_gain_w = mean(gain_t, ..., gain_{t-w+1})`
- `avg_loss_w = mean(loss_t, ..., loss_{t-w+1})`
- `RS_w = avg_gain_w / avg_loss_w`
- `rsi_{w}m = 100 - 100 / (1 + RS_w)`

For each pair `(s, l)` in `rsi_ratio_pairs`:

- `rsi_ratio_{s}_{l} = rsi_s / rsi_l`

### Bollinger relative position

For each `w` in `bollinger_windows`:

- `sma_w = mean(P_t, ..., P_{t-w+1})`
- `std_w = std(P_t, ..., P_{t-w+1})`
- `upper_w = sma_w + 2 * std_w`
- `lower_w = sma_w - 2 * std_w`
- `bollinger_percent_b_{w}m = (P_t - lower_w) / (upper_w - lower_w)`
- `bollinger_bandwidth_{w}m = (upper_w - lower_w) / sma_w`

### Stochastic oscillator

For each pair `(n, d)` in `stochastic_windows`:

- `low_n = min(P_t, ..., P_{t-n+1})`
- `high_n = max(P_t, ..., P_{t-n+1})`
- `%K_{n} = 100 * (P_t - low_n) / (high_n - low_n)`
- `stoch_d_{n}_{d} = mean(%K_n over last d months)`

### Range location

For each `w` in `range_windows`:

- `rolling_high_{w}m = max(P_t, ..., P_{t-w+1})`
- `rolling_low_{w}m = min(P_t, ..., P_{t-w+1})`
- `dist_to_{w}m_high = P_t / rolling_high_{w}m - 1`
- `dist_to_{w}m_low = P_t / rolling_low_{w}m - 1`
- `range_position_{w}m = (P_t - rolling_low_{w}m) / (rolling_high_{w}m - rolling_low_{w}m)`

### Volatility regime

For each `w` in `volatility_windows`:

- `volatility_{w}m = std(r_t, ..., r_{t-w+1})`

For each pair `(s, l)` in `volatility_ratio_pairs`:

- `volatility_ratio_{s}_{l} = volatility_s / volatility_l`

Division rule for all technical ratios:

- denominator must be non-null and `abs(denominator) > 1e-12`
- otherwise the ratio is set to `null`

## Fundamental Data Preparation

Quarterly statements are normalized to:

- `ticker`
- `date`
- `filing_date`
- `quarter_end`
- `report_date = coalesce(filing_date, quarter_end)`

Within each `(ticker, quarter_end)` group:

- the last `filing_date` is retained
- the last value per selected raw column is retained

Monthly alignment:

- monthly price rows are `join_asof(..., strategy="backward")`
- therefore a monthly row only sees the latest report already available as of its `date`

This is the key anti-lookahead rule for fundamentals.

## Raw Fundamental Inputs Used

Income statement:

- `totalRevenue -> total_revenue`
- `netIncome -> net_income`
- `ebitda -> ebitda`
- `ebit -> ebit`
- `grossProfit -> gross_profit`

Balance sheet:

- `commonStockSharesOutstanding -> shares_outstanding`
- `totalStockholderEquity -> equity`
- `netDebt -> net_debt`
- `totalAssets -> total_assets`
- `cashAndShortTermInvestments -> cash_short_term`

Cash flow:

- `freeCashFlow -> free_cashflow`

Earnings:

- `epsActual -> eps_actual`

## TTM / Rolling Quarterly Features

TTM sums:

- `total_revenue_ttm = sum(last 4 quarterly total_revenue)`
- `net_income_ttm = sum(last 4 quarterly net_income)`
- `ebitda_ttm = sum(last 4 quarterly ebitda)`
- `ebit_ttm = sum(last 4 quarterly ebit)`
- `gross_profit_ttm = sum(last 4 quarterly gross_profit)`
- `free_cashflow_ttm = sum(last 4 quarterly free_cashflow)`
- `eps_actual_ttm = sum(last 4 quarterly eps_actual)`

4-quarter averages:

- `shares_outstanding_avg4q = mean(last 4 quarterly shares_outstanding)`
- `equity_avg4q = mean(last 4 quarterly equity)`
- `net_debt_avg4q = mean(last 4 quarterly net_debt)`
- `total_assets_avg4q = mean(last 4 quarterly total_assets)`
- `cash_short_term_avg4q = mean(last 4 quarterly cash_short_term)`

## Derived Fundamental Ratios

Market-value layer:

- `market_cap_t = last_close_t * shares_outstanding_avg4q_t`
- `enterprise_value_t = market_cap_t + net_debt_avg4q_t`

Important:

- `market_cap` and `enterprise_value` are intermediate construction variables
- they are used to build valuation ratios
- they are **not** exposed as model features

Profitability / quality:

- `gross_margin_ttm = gross_profit_ttm / total_revenue_ttm`
- `ebit_margin_ttm = ebit_ttm / total_revenue_ttm`
- `ebitda_margin_ttm = ebitda_ttm / total_revenue_ttm`
- `net_margin_ttm = net_income_ttm / total_revenue_ttm`
- `fcf_margin_ttm = free_cashflow_ttm / total_revenue_ttm`
- `roe_ttm = net_income_ttm / equity_avg4q`
- `roa_ttm = net_income_ttm / total_assets_avg4q`
- `gross_profit_to_assets = gross_profit_ttm / total_assets_avg4q`
- `ebit_to_assets = ebit_ttm / total_assets_avg4q`
- `fcf_to_assets = free_cashflow_ttm / total_assets_avg4q`
- `asset_turnover_ttm = total_revenue_ttm / total_assets_avg4q`
- `accrual_ratio = (net_income_ttm - free_cashflow_ttm) / total_assets_avg4q`
- `fcf_to_net_income = free_cashflow_ttm / net_income_ttm`

Balance sheet / capital structure:

- `debt_to_equity = net_debt_avg4q / equity_avg4q`
- `net_debt_to_assets = net_debt_avg4q / total_assets_avg4q`
- `net_debt_to_ebitda = net_debt_avg4q / ebitda_ttm`
- `equity_to_assets = equity_avg4q / total_assets_avg4q`
- `cash_to_assets = cash_short_term_avg4q / total_assets_avg4q`

Valuation yields / inverted multiples:

- `earnings_yield = eps_actual_ttm / last_close_t`
- `sales_yield = total_revenue_ttm / market_cap_t`
- `book_to_price = equity_avg4q / market_cap_t`
- `fcf_yield = free_cashflow_ttm / market_cap_t`
- `ebitda_to_ev = ebitda_ttm / enterprise_value_t`

Growth:

- for each lag `L` in `quarterly_growth_lags`:
  - `x_ttm_growth_{L}q = x_ttm_t / x_ttm_{t-L} - 1`
- `share_dilution_{L}q = shares_outstanding_avg4q_t / shares_outstanding_avg4q_{t-L} - 1`

Division rule:

- all ratios use a guarded division
- denominator must be non-null and `abs(denominator) > 1e-12`
- otherwise the result is `null`

## Fundamental Growth Features Used By The Model

Raw dollar TTM levels are **not** selected as model features.

The model keeps only:

- valuation yields / quality / margin ratios
- growth features derived from TTM series
- dilution features derived from shares outstanding growth

For each TTM series `x_ttm`, the following quarterly growths are constructed on the quarterly reporting timeline before monthly `asof` joining:

- `x_ttm_growth_1q = x_ttm_t / x_ttm_{t-1q} - 1`
- `x_ttm_growth_4q = x_ttm_t / x_ttm_{t-4q} - 1`
- `x_ttm_growth_12q = x_ttm_t / x_ttm_{t-12q} - 1`

Applied to:

- `total_revenue_ttm`
- `net_income_ttm`
- `ebitda_ttm`
- `ebit_ttm`
- `gross_profit_ttm`
- `free_cashflow_ttm`
- `eps_actual_ttm`

This means the selected fundamental growth features are:

- `total_revenue_ttm_growth_1q`
- `total_revenue_ttm_growth_4q`
- `total_revenue_ttm_growth_12q`
- `net_income_ttm_growth_1q`
- `net_income_ttm_growth_4q`
- `net_income_ttm_growth_12q`
- `ebitda_ttm_growth_1q`
- `ebitda_ttm_growth_4q`
- `ebitda_ttm_growth_12q`
- `ebit_ttm_growth_1q`
- `ebit_ttm_growth_4q`
- `ebit_ttm_growth_12q`
- `gross_profit_ttm_growth_1q`
- `gross_profit_ttm_growth_4q`
- `gross_profit_ttm_growth_12q`
- `free_cashflow_ttm_growth_1q`
- `free_cashflow_ttm_growth_4q`
- `free_cashflow_ttm_growth_12q`
- `eps_actual_ttm_growth_1q`
- `eps_actual_ttm_growth_4q`
- `eps_actual_ttm_growth_12q`
- `share_dilution_1q`
- `share_dilution_4q`
- `share_dilution_12q`

The following raw dollar features are intentionally excluded from the model:

- `market_cap`
- `enterprise_value`
- `total_revenue_ttm`
- `net_income_ttm`
- `ebitda_ttm`
- `free_cashflow_ttm`
- `shares_outstanding_avg4q`
- `equity_avg4q`
- `net_debt_avg4q`
- `total_assets_avg4q`

## Feature Filtering and Imputation

Excluded from model features:

- identity / timing columns:
  - `ticker`
  - `year_month`
  - `decision_month`
  - `holding_month`
  - `decision_asof_date`
  - `holding_asof_date`
  - `benchmark_holding_asof_date`
  - `holding_period_complete`
- target / realized columns:
  - `monthly_return`
  - `future_return`
  - `benchmark_future_return`
  - `future_excess_return`
  - `future_relative_return`

Sparse-feature filtering:

- for each candidate feature `x`
- `missing_ratio(x) = mean(is_null(x))`
- keep feature iff `missing_ratio(x) <= missing_feature_threshold`

Imputation for kept features:

- first: monthly cross-sectional median, `median(x | year_month=t)`
- second: full-sample median of the feature
- third: `0.0`

Non-finite handling:

- if a numeric feature is `+/-inf` or `NaN`, it is converted to `null` before imputation

## Portfolio Construction

The model predicts a score `prediction` for each scored row.

Monthly selection:

- rank by `prediction` descending within `decision_month`
- keep top `N = top_n`

Portfolio return aggregation:

- monthly backtest performance is aggregated on `holding_month`
- `portfolio_return = mean(future_return of selected names)`
- `benchmark_return = mean(benchmark_future_return of selected names)`
- `active_return = portfolio_return - benchmark_return`
- `hit_rate = mean(target_label of selected names)`

## Known Limitations

- no transaction costs
- no slippage
- no liquidity or capacity model
- equal-weight portfolio
- `holding_period_complete` is a heuristic, not a full exchange-calendar implementation
- `eps_growth_qoq` is computed on the TTM EPS series shifted by one observation, not on raw single-quarter EPS

## Recommended Reading Order

If you need to audit a run:

1. `outputs/.../fold_index.parquet`
2. `outputs/.../debug_predictions_long.parquet`
3. `outputs/.../debug_predictions_full.parquet`
4. `outputs/.../backtest_audit_report.html`
