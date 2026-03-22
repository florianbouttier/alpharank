# Open-Source Cadrage Status 2025

## Scope

Reference EODHD-style datasets:

- `data/US_Finalprice.parquet`
- `data/US_Income_statement.parquet`
- `data/US_Balance_sheet.parquet`
- `data/US_Cash_flow.parquet`
- `data/US_Earnings.parquet`

Current best full-open audit run:

- `data/open_source/sp500_2025_2025_consolidated_v4/`
- HTML report: `data/open_source/sp500_2025_2025_consolidated_v4/report.html`
- Ticker deep dives: `data/open_source/sp500_2025_2025_consolidated_v4/tickers/`
- KPI deep dives: `data/open_source/sp500_2025_2025_consolidated_v4/kpis/`

Historical price transition run:

- `data/open_source/price_transition_20050101/`
- HTML report: `data/open_source/price_transition_20050101/report.html`

## Executive Summary

Current status:

- Price output can already match the old EODHD file format exactly.
- Financial outputs do not yet match the old EODHD file format.
- The current full-open financial layer is still a normalized long-format consolidation with lineage, not a drop-in replacement for the old wide parquets.
- The best current 2025 reference is `sp500_2025_2025_consolidated_v4`.

Practical conclusion:

- `price`: ready for transition format-wise
- `income_statement`: not parity
- `balance_sheet`: not parity
- `cash_flow`: not parity
- `earnings`: not parity

## Format Compatibility

### Price

Old file:

- `data/US_Finalprice.parquet`
- rows: `6,195,454`
- columns: `8`

Open replacement:

- `data/open_source/price_transition_20050101/US_Finalprice.parquet`
- rows: `3,492,460`
- columns: `8`

Schema status:

- exact schema match
- columns: `ticker`, `date`, `adjusted_close`, `close`, `open`, `high`, `low`, `volume`

Interpretation:

- format parity: `yes`
- coverage parity: `no`

### Income Statement

Old file:

- `data/US_Income_statement.parquet`
- rows: `90,753`
- columns: `35`

Open candidate:

- `data/open_source/sp500_2025_2025_consolidated_v4/financials_open_source_consolidated/income_statement.parquet`
- rows: `8,882`
- columns: `18`

Schema status:

- not compatible
- old format is wide KPI columns like `totalRevenue`, `grossProfit`, `operatingIncome`, `netIncome`
- new format is long with `metric`, `value`, `source`, `selected_source`, `selected_form`, `selected_fiscal_period`, lineage fields

### Balance Sheet

Old file:

- `data/US_Balance_sheet.parquet`
- rows: `90,644`
- columns: `65`

Open candidate:

- `data/open_source/sp500_2025_2025_consolidated_v4/financials_open_source_consolidated/balance_sheet.parquet`
- rows: `8,834`
- columns: `18`

Schema status:

- not compatible
- old format is wide KPI columns like `cashAndEquivalents`, `totalAssets`, `totalLiab`, `totalStockholderEquity`, `commonStockSharesOutstanding`
- new format is long + lineage

### Cash Flow

Old file:

- `data/US_Cash_flow.parquet`
- rows: `84,716`
- columns: `33`

Open candidate:

- `data/open_source/sp500_2025_2025_consolidated_v4/financials_open_source_consolidated/cash_flow.parquet`
- rows: `6,204`
- columns: `18`

Schema status:

- not compatible
- old format is wide KPI columns like `totalCashFromOperatingActivities`, `capitalExpenditures`, `freeCashFlow`
- new format is long + lineage

### Earnings

Old file:

- `data/US_Earnings.parquet`
- rows: `71,019`
- columns: `9`

Open candidate:

- `data/open_source/sp500_2025_2025_consolidated_v4/earnings_yfinance.parquet`
- rows: `0`
- columns: `8`

Schema status:

- not compatible
- old fields missing: `beforeAfterMarket`, `currency`, `date`, `epsDifference`
- new fields added: `earningsDatetime`, `period_end`, `source`

Interpretation:

- current open earnings dataset is not usable as a replacement yet

## 2025 SP500 Full-Open Audit Summary

Universe:

- `518` tickers
- missing from both Yahoo and SEC mapping: `ANSS`, `BF.B`, `BRK.B`, `DFS`, `HES`, `IPG`, `JNPR`, `K`, `WBA`

### Price

Source:

- `yfinance`

Results:

- matched rows: `126,818`
- error rows: `1,623`
- coverage gap rows: `1,546`
- error rate over matched rows: `1.28%`

### Open Consolidated vs EODHD

Threshold used in the audit reports: `0.5%`

| Statement | Matched | Error Rows | Coverage Gap Rows | Error Rate |
| --- | ---: | ---: | ---: | ---: |
| balance_sheet | 7,096 | 413 | 1,898 | 5.82% |
| cash_flow | 5,982 | 781 | 381 | 13.06% |
| income_statement | 7,764 | 1,665 | 1,542 | 21.45% |
| shares | 1,966 | 1,365 | 829 | 69.43% |

Interpretation:

- `balance_sheet` is the closest of the financial statements
- `income_statement` is still far from parity
- `cash_flow` is materially off
- `shares` is the worst category by far

## KPI Hotspots

Open consolidated only:

| Statement | KPI | Matched | Error Rows | Coverage Gap Rows | Error Rate |
| --- | --- | ---: | ---: | ---: | ---: |
| balance_sheet | total_liabilities | 2,034 | 253 | 332 | 12.44% |
| balance_sheet | cash_and_equivalents | 994 | 62 | 901 | 6.24% |
| balance_sheet | stockholders_equity | 2,034 | 56 | 334 | 2.75% |
| balance_sheet | total_assets | 2,034 | 42 | 331 | 2.06% |
| cash_flow | free_cash_flow | 2,032 | 383 | 83 | 18.85% |
| cash_flow | capital_expenditures | 1,921 | 311 | 196 | 16.19% |
| cash_flow | operating_cash_flow | 2,029 | 87 | 102 | 4.29% |
| income_statement | operating_income | 1,872 | 602 | 473 | 32.16% |
| income_statement | gross_profit | 1,828 | 578 | 437 | 31.62% |
| income_statement | revenue | 2,032 | 377 | 309 | 18.55% |
| income_statement | net_income | 2,032 | 108 | 323 | 5.31% |
| shares | outstanding_shares | 1,966 | 1,365 | 829 | 69.43% |

Interpretation:

- best KPI today: `net_income`
- acceptable but not clean: `total_assets`, `stockholders_equity`
- weak: `revenue`, `capital_expenditures`, `free_cash_flow`, `cash_and_equivalents`, `total_liabilities`
- very weak: `gross_profit`, `operating_income`, `outstanding_shares`

## Source Contribution in the Consolidation

### Balance Sheet

- `sec_companyfacts`: `7,181` selected rows
- `sec_filing`: `635`
- `simfin`: `102`
- `yfinance`: `916`

### Cash Flow

- `sec_companyfacts`: `1,267` selected rows
- `sec_filing`: `14`
- `simfin`: `130`
- `yfinance`: `4,793`

### Income Statement

- `sec_companyfacts`: `6,150` selected rows
- `sec_filing`: `109`
- `simfin`: `116`
- `yfinance`: `2,507`

### Shares

- `sec_companyfacts`: `1,857` selected rows
- `sec_filing`: `668`
- `yfinance`: `189`

Interpretation:

- the consolidation is still heavily dependent on `yfinance` for `cash_flow`
- `sec_filing` helps for coverage but does not solve parity alone
- `simfin` helps a little, but not enough to close the gap

## Historical Price Transition Since 2005

Reference output:

- `data/open_source/price_transition_20050101/US_Finalprice.parquet`

Coverage:

- tickers present in the reference universe: `828`
- Yahoo-covered tickers: `655`
- missing tickers: `173`
- coverage gap rows: `1,005,161`

Interpretation:

- price format parity is solved
- historical coverage parity is not solved
- the missing tail is mostly delisted / renamed names that Yahoo no longer serves cleanly

## Where We Are

The open stack is good enough to:

- generate a clean audit report with deep dives per ticker and per KPI
- produce a drop-in replacement for `US_Finalprice.parquet`
- compare open sources against EODHD with explicit lineage

The open stack is not yet good enough to:

- replace `US_Income_statement.parquet`
- replace `US_Balance_sheet.parquet`
- replace `US_Cash_flow.parquet`
- replace `US_Earnings.parquet`
- preserve historical deleted ticker coverage at EODHD level

## Recommended Next Step

If the goal is a true drop-in replacement, the next deliverable should be:

1. Export EODHD-shaped wide parquets from `financials_open_source_consolidated`.
2. Keep only the KPI subset we can justify today.
3. Generate a second audit pass on those wide replacement parquets.
4. Treat `earnings` and `shares` as separate recovery projects because they are currently the least reliable.

That would let us answer the real question cleanly:

- "Can the open stack replace the old data files without breaking the backtests?"
