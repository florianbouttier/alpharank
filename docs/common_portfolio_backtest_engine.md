# Common Portfolio And Backtest Engine

Last updated: 2026-08-11

This document is the source of truth for the code shared after signal
generation by the Legacy and boosting methodologies.

The complete signal-generation pseudocode, shared liquidity thresholds,
fundamental-data usage, and end-to-end architecture are documented in
[`legacy_boosting_methodology.md`](./legacy_boosting_methodology.md).

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
├── attribution.py     # exact security/month/year CAGR attribution
├── comparison.py      # explicit common-calendar alignment
├── lineage.py         # input-snapshot hash comparison and fail-closed gate
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

## Benchmark Contract

Performance comparisons use one explicit benchmark convention:

- identifier: `spy_total_return_adjusted_close`;
- label: `SPY total return`;
- source column: `adjusted_close`;
- distributions: included through the adjusted-price series.

`SP500` in older `legacy_monthly_returns_*.parquet` files is a price-return
series calculated from `close`. It is retained only as part of the frozen
Legacy signal/replay history. It must not be used as the standard performance
benchmark and must be described as `SPY price return` when audited.

`src/alpharank/portfolio/benchmark.py` is the only shared constructor for
monthly benchmark returns. New common artifacts record benchmark id, label,
price column and distribution treatment in `<prefix>_calendar.json`.

For a standalone Legacy performance query, use the latest validated Legacy
run and rebuild its common replay with:

```bash
./.venv/bin/python scripts/build_legacy_common_replay.py \
  --run-dir outputs/2026-07-27/runs/20260727_221253 \
  --output-dir outputs/common_portfolio_replays/legacy_20260727_221253_spy_total_return
```

For a Legacy/Boosting comparison, use the Legacy replay attached to the exact
same snapshot as Boosting, even if a newer standalone Legacy replay exists.
The UI and report must show both snapshot ids instead of using an unqualified
`Legacy` label.

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

`advanced_performance_statistics()` extends the same canonical base with
Sortino, Calmar, information ratio, beta, alpha, correlation, benchmark hit
rate, historical VaR/CVaR, Omega and upside/downside capture. The interactive
research center precomputes every contiguous start/end window with this
function; its browser code no longer owns a separate implementation of the
headline performance statistics.

## Exact CAGR Attribution

CAGR is nonlinear and cannot be split by adding ordinary percentage-point
contributions. All security, year, and month waterfalls therefore use
`src/alpharank/portfolio/attribution.py` and the following exact contract.

For month `t`, the simulator's simple-return contribution is allocated to each
available security as `effective_weight * realized_return`; unavailable returns
follow the simulator's existing weight-renormalization rule. Transaction costs
are a separate negative component. Those simple contributions must sum to the
month's `net_return` within `1e-12`.

Each simple contribution is then multiplied by
`log1p(net_return_t) / net_return_t`. The resulting components add exactly to
`log1p(net_return_t)`. Over `N` selected months, the annualized additive value
shown by the waterfall is:

```text
component_log_annualized = (12 / N) * sum(component_log_contribution_t)
CAGR = exp(sum(component_log_annualized)) - 1
```

The chart's `Effet composé` bridge is the nonlinear difference between the sum
of annualized log contributions and the final CAGR. The exhaustive table also
shows each component's marginal CAGR impact. For a one-month action drill-down,
the terminal metric is that month's return rather than a misleading annualized
rate. New reports must preserve full precision for these audit rows; rounding
individual components before reconciliation is prohibited.

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

## Frozen Parity And Lineage Validation

Two independent conditions now exist:

- `engine_parity_passed`: each frozen strategy is reproduced by the common
  simulator;
- `comparison_eligible`: every input hash consumed by boosting is identical to
  the corresponding Legacy snapshot hash, and both methods declare the same
  full-trajectory data-quality ticker exclusions and monthly price-eligibility
  policy.

Overall `passed=true` requires both. Distinct snapshots may be replayed only
with `--allow-distinct-snapshots`; this leaves `comparison_eligible=false` and
must never be interpreted as a strategy-performance comparison.

The validation entrypoint is:

```bash
./.venv/bin/python scripts/validate_common_portfolio_engine.py \
  --legacy-detailed outputs/2026-07-27/runs/20260727_221253/legacy_detailed_returns_polars.parquet \
  --legacy-aggregated outputs/2026-07-27/runs/20260727_221253/legacy_aggregated_returns_polars.parquet \
  --alpha-holdings outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_holdings.parquet \
  --alpha-monthly outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/allocation_monthly.csv \
  --legacy-data-manifest outputs/2026-07-27/runs/20260727_221253/data_input_manifest.json \
  --alpha-data-manifest outputs/multihorizon_boosting/legacy_ema_risk_overlay_ticker_quarantine_v6_20260726/manifest.json \
  --allow-distinct-snapshots \
  --output outputs/common_portfolio_engine_validation_20260809.json
```

The 2026-08-09 reference proves mechanical parity only. Legacy used the
2026-07-27 snapshot while Alpha used the 2026-07-13 snapshot, so all seven
Alpha-consumed input hashes differ and `comparison_eligible=false`.

Mechanical result with tolerance `1e-12`:

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
- Legacy and boosting input manifests pass `require_matching_data_contexts()`;
  a shared calendar or simulator is not sufficient when source hashes differ.

### Target maturity versus portfolio maturity

The multi-horizon runner exposes an explicit `--score-only-end-month YYYY-MM`
for the recent causal tail. This does not relax the H6 learning contract:

- train, validation, AUC, NDCG, calibration, and all other model-quality
  metrics use only rows whose H6 target is evaluable;
- a fitted outer-fold model may score later test decisions without an H6 label;
- those later decisions enter a portfolio replay only when their one-month
  stock and benchmark returns are complete;
- predictions record `target_status=evaluable`,
  `ticker_target_unavailable`, or `horizon_pending`;
- SHAP may explain all scored test rows, but its report must identify the H6
  status instead of implying that pending rows are model-quality observations.

The comparable calendar is the intersection of realized one-month Boosting,
Legacy, and SPY returns. `scripts/build_common_legacy_boosting_replay.py`
enforces matching input hashes and rejects a Legacy/Boosting pair that does not
declare the exact same `input_snapshot/` and full-trajectory ticker exclusions.

## Retained Same-Raw-Snapshot Reference — 2026-08-11

The retained replay below passes the original raw-input hash gate:

```text
open-source run: 20260811_001503
retained snapshot: data/open_source/history/output/open_source_output_20260811_014746
Legacy run: outputs/2026-08-11/runs/20260811_035522
Boosting run: outputs/multihorizon_boosting/legacy_ema_latest_common_score_tail_20260811_001503_standard
common replay: outputs/common_portfolio_replays/legacy_boosting_20260811_001503_035522_standard
holding calendar: 2011-08 through 2026-07, 180 months
benchmark: spy_total_return_adjusted_close
```

All seven required hashes match. However, Legacy declared only
`SII.US/CBE.US/TIE.US`, while Boosting declared the ten-ticker
`historical_ticker_exclusions_v1` registry. The artifact predates the new
preprocessing-parity gate and is not an eligible pure strategy comparison.
Boosting uses 10 bps times turnover; Legacy retains its historical zero-cost
convention.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 27.8676% | 38.3422% | 0.6747 | -35.7550% |
| Boosting Top 10 | 23.5175% | 32.3422% | 0.6653 | -32.3580% |
| Legacy | 19.2049% | 22.3273% | 0.7706 | -23.3821% |
| SPY total return | 14.3779% | 14.3043% | 0.8653 | -23.9272% |

The higher Boosting CAGR is not a promotion decision: it comes with materially
higher volatility, lower Sharpe than Legacy and SPY, and a deeper drawdown.
These retained metrics remain useful for diagnosing the old artifact, but must
not support a Legacy-versus-Boosting conclusion. Regenerate both methods with
matching exclusions before interpreting the performance gap.

The public comparison also requires the versioned Boosting profile
`legacy_ema_latest_common_v1`. Launch it with
`--latest-common-comparison-profile`; the common replay builder validates the
actual manifest values and fails closed on drift. This gate was added after an
intermediate 2026-08-11 run accidentally used the permissive defaults for the
three price-tradability thresholds and produced materially different returns.

`data_revision_audit.json` beside the common replay is generated by
`scripts/audit_open_source_snapshot_revisions.py`. Against the preceding clean
snapshot, the current standard replay has identical ticker/month membership,
maximum Legacy numerical drift `2.22e-16`, and zero changed Boosting prediction
rows. The input audit still records four mutable Yahoo earnings revisions from
2005-2007; same-snapshot comparability does not turn those vendor values into a
true historical point-in-time feed.

## Tests

Core tests are in `tests/test_portfolio_engine.py` and
`tests/test_portfolio_attribution.py`. They cover timing,
lookahead rejection, missing-return handling, adapter equivalence, turnover,
calendar alignment, and complete-year performance. Existing backtest and
multi-horizon tests exercise the compatibility wrappers.

## Current Eligible Reference — 2026-08-12

This reference supersedes the ineligible 2026-08-11 comparison above:

```text
open-source run: 20260811_001503
retained snapshot: data/open_source/history/output/open_source_output_20260811_014746
Legacy run: outputs/2026-08-12/runs/20260812_171646
Boosting run: outputs/multihorizon_boosting/legacy_ema_latest_common_shared_eligibility_final_20260812
common replay: outputs/common_portfolio_replays/legacy_boosting_shared_eligibility_final_20260812
holding calendar: 2011-08 through 2026-07, 180 months
benchmark: SPY total return from adjusted_close
transaction costs: 10 bps times turnover for Boosting and Legacy in this comparison
```

The replay passes all three blocking controls: seven required raw-input hashes,
the ten-ticker `historical_ticker_exclusions_v1` registry, and
`monthly_price_eligibility_v1` (`>=10` observations, median daily dollar
volume `>=USD 1m`, OHLC violation rate `<=5%`). Legacy applies that monthly
gate before its EMA Top 30 ranking; Boosting applies the same gate before its
monthly ranking.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 10 | 24.2818% | 32.8269% | 0.6788 | -40.3548% |
| Boosting Top 5 | 23.6497% | 36.1436% | 0.5990 | -29.6798% |
| Legacy | 17.0257% | 22.2918% | 0.6740 | -24.6856% |
| SPY total return | 14.3779% | 14.3043% | 0.8653 | -23.9272% |

The standalone Legacy production artifact retains its historical zero-cost
convention. The cross-method replay resimulates its saved holdings with the
same 10 bps times turnover as Boosting, so the comparison cost policy is
homogeneous. Boosting has the highest CAGR but not the best risk-adjusted result: SPY has
the highest Sharpe, and Top 10 has the deepest drawdown. This is an eligible
historical comparison, not a production promotion decision.

The full start-year and calendar-year tables are generated by
`scripts/experiments/render_start_year_performance.py`. Boosting has no OOS
portfolio before August 2011; rows requested for 2010 or 2011 are marked
partial rather than backfilled.
