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

- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/data_loading.py`
- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/features.py`
- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/fundamentals.py`
- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/datasets.py`
- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/pipeline.py`
- `/Users/nicolas.rusinger/AlphaRank/src/alpharank/backtest/portfolio.py`

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

Default in `scripts/run_backtest.py`:

- `outperformance_threshold = 0.0`

## Technical Features

All technical features are computed on monthly data and grouped by ticker.

Base series:

- `r_t = monthly_return_t`
- `P_t = last_close_t`

Features:

- `ret_lag_1_t = r_{t-1}`
- `ret_lag_2_t = r_{t-2}`
- `ret_mean_3m_t = mean(r_t, r_{t-1}, r_{t-2})`
- `ret_mean_6m_t = mean(r_t, ..., r_{t-5})`
- `ret_vol_3m_t = std(r_t, r_{t-1}, r_{t-2})`
- `ret_vol_6m_t = std(r_t, ..., r_{t-5})`
- `mom_3m_t = sum(r_t, r_{t-1}, r_{t-2})`
- `mom_6m_t = sum(r_t, ..., r_{t-5})`

EMAs:

- `ema_3_t = EWM(P_t, span=3, adjust=False)`
- `ema_12_t = EWM(P_t, span=12, adjust=False)`
- `ema_ratio_3_12_t = ema_3_t / ema_12_t`

Range / position:

- `rolling_high_12m_t = max(P_t, ..., P_{t-11})`
- `rolling_low_12m_t = min(P_t, ..., P_{t-11})`
- `dist_to_12m_high_t = P_t / rolling_high_12m_t - 1`
- `dist_to_12m_low_t = P_t / rolling_low_12m_t - 1`

RSI-like feature:

- `gain_t = max(r_t, 0)`
- `loss_t = max(-r_t, 0)`
- `avg_gain_6m_t = mean(gain_t, ..., gain_{t-5})`
- `avg_loss_6m_t = mean(loss_t, ..., loss_{t-5})`
- `RS_t = avg_gain_6m_t / (avg_loss_6m_t + 1e-12)`
- `rsi_6m_t = 100 - 100 / (1 + RS_t)`

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
- `enterprise_value_t = market_cap_t + net_debt_avg4q_t - cash_short_term_avg4q_t`

Important:

- `market_cap` and `enterprise_value` are intermediate construction variables
- they are used to build valuation ratios
- they are **not** exposed as model features

Margins and returns:

- `net_margin_ttm = net_income_ttm / total_revenue_ttm`
- `ebitda_margin_ttm = ebitda_ttm / total_revenue_ttm`
- `gross_margin_ttm = gross_profit_ttm / total_revenue_ttm`
- `roe_ttm = net_income_ttm / equity_avg4q`
- `roa_ttm = net_income_ttm / total_assets_avg4q`
- `debt_to_equity = net_debt_avg4q / equity_avg4q`
- `fcf_margin_ttm = free_cashflow_ttm / total_revenue_ttm`

Valuation:

- `pe_ttm = last_close_t / eps_actual_ttm`
- `price_to_sales = market_cap_t / total_revenue_ttm`
- `price_to_book = market_cap_t / equity_avg4q`
- `ev_to_ebitda = enterprise_value_t / ebitda_ttm`

Growth:

- `revenue_growth_yoy_t = total_revenue_ttm_t / total_revenue_ttm_{t-4} - 1`
- `net_income_growth_yoy_t = net_income_ttm_t / net_income_ttm_{t-4} - 1`
- `eps_growth_qoq_t = eps_actual_ttm_t / eps_actual_ttm_{t-1} - 1`

Division rule:

- all ratios use a guarded division
- denominator must be non-null and `abs(denominator) > 1e-12`
- otherwise the result is `null`

## Fundamental Growth Features Used By The Model

Raw dollar TTM levels are **not** selected as model features.

The model keeps only:

- valuation / quality / margin ratios
- growth features derived from TTM series

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

The following raw dollar features are intentionally excluded from the model:

- `market_cap`
- `enterprise_value`
- `total_revenue_ttm`
- `net_income_ttm`
- `ebitda_ttm`
- `free_cashflow_ttm`

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
