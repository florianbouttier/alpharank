# Common Portfolio And Backtest Engine

Last updated: 2026-08-09

This document is the source of truth for the code shared after signal
generation by the Legacy and boosting methodologies.

## Scope And Separation Of Responsibilities

Signal generation remains methodology-specific:

- Legacy ranks relative-price EMA ratios, applies fundamental and sector
  filters, tunes annually, and combines four Optuna tracks;
- boosting fits leakage-aware models and emits out-of-sample scores,
  probabilities, and optional risk forecasts.

Once a methodology has emitted a finalized monthly basket, both paths use
`src/alpharank/portfolio/` for holdings validation, allocation primitives,
monthly simulation, performance statistics, calendar alignment, and standard
audit artifacts.

The shared package is deliberately independent from XGBoost, Optuna, SHAP, and
the Legacy strategy classes.

## Package Layout

```text
src/alpharank/portfolio/
├── contracts.py       # canonical holdings and monthly schemas
├── allocation.py      # top-N, equal/risk weights, turnover
├── simulation.py      # gross/net returns, costs, benchmark comparison
├── performance.py     # CAGR, Sharpe, drawdown, annual returns
├── comparison.py      # explicit common-calendar alignment
├── artifacts.py       # standard parquet/csv/json audit outputs
└── adapters/
    ├── legacy.py      # finalized Legacy baskets -> common holdings
    └── boosting.py    # OOS boosting scores -> common holdings
```

Compatibility modules such as `alpharank.backtest.portfolio`,
`alpharank.backtest.kpis`, `alpharank.strategy.analytics`, and
`alpharank.multihorizon.trading` remain public, but their shared calculations
now delegate to this package.

## Canonical Holdings Contract

Every row represents one security selected at decision month `t` for the next
holding month `t+1`.

Required columns:

| Column | Meaning |
| --- | --- |
| `strategy` | Stable strategy identifier |
| `decision_month` | Month whose end-of-month information produced the signal |
| `holding_month` | Exactly `decision_month + 1 month` |
| `ticker` | Security identifier |
| `target_weight` | Long-only target weight; sums to one per strategy/month |
| `realized_return` | Security return realized during the holding month |
| `benchmark_return` | S&P 500 return over the same holding month |

Optional but recommended fields are `score`, `selection_rank`, `sector`, model
votes, calibrated probability, and predicted risk.

The validator rejects duplicate ticker/month rows, negative or non-finite
weights, weights that do not sum to one, and any holding month that is not
exactly one month after the decision month.

## Canonical Monthly Contract

The common simulator writes one row per strategy and holding month:

- `gross_return`: weighted realized stock return before trading costs;
- `turnover`: `0.5 * sum(abs(current_weight - previous_weight))`; the first
  invested month has turnover `1.0`;
- `transaction_cost`: `turnover * transaction_cost_bps / 10_000`;
- `net_return`: `gross_return - transaction_cost`;
- `benchmark_return`: S&P 500 total return on the same month;
- `active_return`: `net_return - benchmark_return`;
- `relative_return`: `(1 + net_return) / (1 + benchmark_return) - 1`;
- `n_positions` and concentration diagnostics.

Legacy production uses zero transaction cost to reproduce the published
historical convention. Alpha risk research currently uses 10 bps times
turnover and keeps gross and net returns separately.

If a Legacy holding has no realized return, the historical convention is
preserved: the missing name is excluded from that month's performance and the
remaining available weights are renormalized. The target holdings themselves
are not rewritten. A month with no available return is rejected.

## Performance Convention

Comparisons between Alpha, Legacy, and SPY use one convention:

- CAGR compounds monthly returns using `months / 12` as elapsed years;
- annualized volatility is monthly sample standard deviation times `sqrt(12)`;
- Sharpe is `(CAGR - risk_free_rate) / annualized_volatility`;
- max drawdown is computed on the compounded wealth curve;
- worst year uses complete January-to-December years only;
- partial boundary years remain visible in annual tables but cannot become the
  reported worst full calendar year.

The older arithmetic Sharpe remains available only through the explicit
`sharpe_convention="arithmetic"` compatibility option. New Alpha/Legacy/SPY
comparisons must use the Legacy convention.

## Standard Artifacts

`write_common_portfolio_artifacts()` produces:

```text
<prefix>_holdings.parquet
<prefix>_monthly.parquet
<prefix>_monthly.csv
<prefix>_annual.csv
<prefix>_performance.csv
<prefix>_calendar.json
```

Legacy runs use the `legacy_common` prefix. Generic boosting runs use
`boosting_common`. Multi-horizon allocation scripts use `portfolio_common`.
Reports should consume these files instead of reimplementing financial
formulas inside an individual experiment script.

## Frozen Parity Validation

The validation entrypoint is:

```bash
./.venv/bin/python scripts/validate_common_portfolio_engine.py \
  --legacy-detailed outputs/2026-07-27/runs/20260727_221253/legacy_detailed_returns_polars.parquet \
  --legacy-aggregated outputs/2026-07-27/runs/20260727_221253/legacy_aggregated_returns_polars.parquet \
  --alpha-holdings outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_holdings.parquet \
  --alpha-monthly outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_monthly.csv \
  --output outputs/common_portfolio_engine_validation_20260809.json
```

Validated result on 2026-08-09 with tolerance `1e-12`:

| Reference | Rows/months | Maximum absolute error |
| --- | ---: | ---: |
| Legacy `Combined_Equal` | 197 complete months | `1.87e-16` |
| Legacy `Combined_Frequency` | 197 complete months | `2.08e-16` |
| Eight Alpha/risk allocations | 1,376 strategy-months | `2.22e-16` turnover; `1.67e-16` return |

The Legacy comparison ends at June 2026 because later rows in that frozen run
do not have a complete benchmark holding period. Partial future rows remain in
the production audit package but are not accepted by the completed-month
comparison contract.

## No-Lookahead Rules

The shared engine does not decide which data a model may see; that remains the
responsibility of each signal adapter and training pipeline. It enforces the
downstream temporal contract only.

Every caller must additionally prove:

- features are available by the decision cutoff;
- S&P 500 membership is point-in-time;
- target maturity and horizon purging are respected during training;
- a filtered period slices already saved OOS returns and does not retrain;
- Legacy and boosting use an identical holding-month calendar when compared.

## Tests

Core tests are in `tests/test_portfolio_engine.py`. They cover timing,
lookahead rejection, missing-return handling, adapter equivalence, turnover,
calendar alignment, and complete-year performance. Existing backtest and
multi-horizon tests exercise the compatibility wrappers.
