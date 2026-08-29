# Common Portfolio And Backtest Engine

Last updated: 2026-08-29

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

## Completed-Month Contract

The raw immutable snapshot may contain the current partial month. That month is
freshness evidence, not a realized period. Legacy truncates price inputs to the
last completed decision month before signal construction. Boosting keeps the
same completed decision month available for scoring, but nulls every target
whose end month is later than the cutoff. The common adapter then admits only
rows with both a stock and benchmark one-month return.

The common replay must therefore end at the same completed holding month for
Legacy, Boosting, and SPY. A partial current-month return is a hard failure, not
an extra observation.

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

New simulations additionally require `feature_max_asof_at`,
`signal_cutoff_at`, `execution_at`, `first_return_observation_at` and
`holding_return_end_at`. Their strict order is feature availability at or before
the signal, then signal before execution, then the first return observation
strictly after execution, with the return ending inside `holding_month`.
Terminal returns resolved from an event must identify an event effective inside
that same holding interval. `causal_timing_policy="legacy_month_only"` is an
explicit reproduction escape hatch for the immutable `v1-audited-biased`
baseline; it is not admissible for promotion of a corrected methodology.

The reviewed terminal registry has two separate jobs. A
`terminal_consideration` record values a position that was validly opened
before an event; a `post_terminal_entry_block` record only rejects a new order
after primary trading ended and deliberately contains no shareholder value.
The v2 registry is a hash-bound delta over v1 and adds RX, TSS, TWTR and ABMD,
the four impossible fills exposed when the Legacy SEC/PE eligibility gate was
removed. These corporate-action records protect execution realism; they are
not SEC fundamental features and never enter a model score.

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

### Convention de clôture approuvée le 2026-08-20

La convention économique AlphaRank est l'achat simulé à la clôture de référence
et le rendement mensuel de clôture ajustée à clôture ajustée. La variante
`next_session_open_v1`, ajoutée pendant l'audit `LEG-003`, reste un test de
sensibilité utile mais ne remplace pas la série canonique.

Le runtime identifie cette convention par
`reference_close_adjusted_close_v1`. `scripts/run_legacy.py` la fige dans
`run_config.canonical_execution_policy` et déclare séparément
`next_session_open_v1` sous `mandatory_execution_sensitivities`. Le rapport
versionné `execution_return_bridge.parquet`, accompagné de
`execution_return_policy.json`, ne peut être écrit que si les deux séries
partagent exactement titres, mois, poids et barème de coûts. Les replays `v2`
déjà scellés continuent d'accepter explicitement l'ancienne politique.

Le diagnostic du snapshot `9a2058c9…425ad`, sur les 180 mois réalisés d'août
2011 à juillet 2026 et avec 10 points de base multipliés par le turnover pour
les trois stratégies investissables, mesure un écart de CAGR clôture moins
prochaine ouverture de `+2,87` points pour Boosting Top 5, `+2,08` points pour
Boosting Top 10 et `+1,08` point pour Legacy. Il s'agit de l'effet composé de
petits écarts répétés à la frontière des mois, pas d'un rendement ajouté en une
seule journée.

Ces chiffres sont diagnostiques tant que `LIVE-022` n'a pas séparé l'ancien et
le nouveau titre portant le symbole SNDK. Le rapport exploratoire correspondant
est conservé sous
`outputs/production_refresh_20260820/execution_close_runtime_v2/`.
Son manifeste rapproche 540 lignes sur trois stratégies et 180 mois, avec
`sensitivity_is_canonical=false`; il reste non publiable jusqu'au replay SNDK.

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

`advanced_performance_statistics()` étend la même base canonique avec Sortino,
Calmar, information ratio, beta, alpha, corrélation, taux de surperformance du
benchmark, VaR/CVaR historique, Omega et capture haussière/baissière. Il expose
également :

- `annualized_excess_return`, défini comme le CAGR de la stratégie moins le CAGR
  du benchmark sur le même calendrier ;
- `tracking_error`, écart-type échantillon des rendements mensuels actifs
  multiplié par `sqrt(12)` ;
- `skewness` et `excess_kurtosis`, moments centrés de population des rendements
  mensuels, la kurtosis étant exprimée relativement à la loi normale.

Sans benchmark, les deux premières métriques restent explicitement `NaN`. Les
consommateurs utilisent cette fonction commune et ne réimplémentent aucun KPI.

`portfolio_period_statistics()` ajoute aux mêmes métriques les mesures
opérationnelles de la fenêtre : turnover, coûts, nombre de positions et
concentrations maximales. `subperiod_portfolio_metric_grid()` applique cette
fonction à chaque couple inclusif de mois sur le calendrier commun. Le rapport
HTML standard sélectionne uniquement ces valeurs pré-calculées ; sa définition,
ses filtres et ses matrices par année de départ sont décrits dans
[`performance_reporting_standard.md`](performance_reporting_standard.md).

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

### Optional causal-trend universe

The common replay can add the `METH-006` research universe without altering its
native strategies:

```bash
./.venv/bin/python scripts/build_common_legacy_boosting_replay.py \
  --legacy-run-dir <legacy-run> \
  --boosting-run-dir <boosting-run> \
  --output-dir <new-output-root> \
  --native-only \
  --include-causal-trend-universe
```

The builder validates every fold OOS replay against its recorded SHA-256,
requires an exact one-to-one match with all prediction keys, then applies
`causal_majority_relative_ema_v1` before score ranking. It emits
`causal_trend_eligibility.parquet`, monthly coverage CSV, trend-filtered
realized holdings and trend-filtered live holdings. A missing raw EMA pair makes
that ticker-month ineligible instead of falling back to a smaller vote.

Because the policy was chosen after reviewing recent SHAP behaviour, a replay
containing it records `methodology_status=post_hoc_research_diagnostic` and
`publication_eligible=false`. The native strategies remain mechanically
comparable inside the same run, but the trend variant cannot become canonical
from this retrospective result alone. `METH-007` owns the clean full-history
run and its dated economic evidence.

## Current No-SEC Reference — 2026-08-28

```text
fresh composed package: outputs/data_refresh_replay_20260827/composed_history/alpharank_input_20260827_122648_5bfbc1d3cb04
prices and benchmark acquired through: 2026-08-26
Legacy run: outputs/no_sec_fresh_replay_20260828/legacy/2026-08-28/runs/20260828_184601
Boosting run: outputs/no_sec_fresh_replay_20260828/boosting
common replay: outputs/no_sec_fresh_replay_20260828/common_replay
fundamental eligibility: no_sec_fundamentals_v1
holding calendar: 2011-08 through 2026-07, 180 realized months
benchmark: SPY total return from adjusted_close
transaction costs: 10 bps times turnover for Boosting and Legacy
```

This is the reference produced by `REPLAY-005`. Legacy selection uses prices,
historical membership and `monthly_price_eligibility_v1`, with no PE, market
capitalization or other SEC value. Boosting has no direct fundamental feature;
it is nevertheless retrained because its point-in-time EMA catalogue comes
from the winners of that Legacy run. The common replay passes the seven input
hashes, the ten-ticker exclusion registry, the public Boosting profile and the
shared price gate. It blocks eight post-terminal prediction rows and selects
zero approved-censored return.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 19.9416% | 35.9188% | 0.4995 | -37.2782% |
| Boosting Top 10 | 19.9271% | 31.4932% | 0.5692 | -31.3583% |
| Boosting Top 15 | 18.8992% | 29.7037% | 0.5689 | -32.7137% |
| Boosting Top 20 | 17.8061% | 28.6770% | 0.5512 | -31.8439% |
| Legacy | 19.6430% | 26.5670% | 0.6641 | -26.4931% |
| SPY total return | 14.3975% | 14.3014% | 0.8669 | -23.9272% |

The standalone Legacy frequency series starts in February 2010 and has a
18.7106% CAGR over 198 months. The common table above is the only valid direct
Legacy/Boosting comparison because it uses their exact shared 180-month
calendar and the same 10 bps cost convention. Both the common manifest and its
performance table are retained under the paths above; the output manifest SHA
is `a7cbd381…4fef4` and the performance-table SHA is `ee039bc1…d6e34`.

The production `latest.json` pointer was still bound to the 20 August snapshot
when the first no-SEC smoke was launched. That run is retained as an audit
artifact but is not called “fresh” or used for this promotion; freshness here
means the explicitly identified composed package acquired through 26 August,
not whichever package happened to be the last promoted pointer.

## Causal-Trend Allocation Diagnostic — 2026-08-29

`METH-007` ran the policy implemented at commit `3b9bf72` against the exact
Legacy and Boosting inputs of the no-SEC reference above. The retained research
replay is:

```text
outputs/no_sec_fresh_replay_20260828/common_replay_causal_trend_top5_top10
```

It uses the same 180 realized months, 10 bps times turnover, SPY total-return
benchmark, eight terminal-entry blocks and zero selected censored return. All
88,950 model prediction keys have one eligibility row: 40,135 pass before
maturity and terminal gates, 39,912 completed rows remain eligible, and 217 of
502 July 2026 live candidates pass. Native holdings are exactly identical to
the reference; re-simulation differs only by floating-point residue up to
`6.94e-18` on monthly net returns.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 19.9416% | 35.9188% | 0.4995 | -37.2782% | 51.5429% |
| Boosting Top 5, causal trend | 23.6320% | 31.7346% | 0.6817 | -27.3678% | 49.1366% |
| Boosting Top 10 | 19.9271% | 31.4932% | 0.5692 | -31.3583% | 47.2495% |
| Boosting Top 10, causal trend | 21.9512% | 26.1454% | 0.7631 | -24.7904% | 46.0514% |
| Legacy | 19.6430% | 26.5670% | 0.6641 | -26.4931% | 44.3899% |
| SPY total return | 14.3975% | 14.3014% | 0.8669 | -23.9272% | 0.0000% |

The point estimates improve both return and risk, but temporal confirmation is
not decisive. Each trend variant beats its native comparator in only 8 of 14
full calendar years. A paired moving-block bootstrap with 50,000 samples,
12-month blocks and seed 42 gives a 95% annualized mean-return interval of
`[-17.61%, 24.71%]` for Top 5 and `[-9.85%, 11.06%]` for Top 10; the respective
Sharpe-difference intervals are `[-0.263, 0.552]` and `[-0.111, 0.430]`. Every
interval crosses zero, so the variant remains post-hoc research and is not
promoted despite the better full-sample table.

The July 2026 decision produces this equal-weight August Top 10:

| Rank | Ticker | Boosting score | Positive trend pairs |
| ---: | --- | ---: | ---: |
| 1 | DELL.US | 0.349872 | 49 / 49 |
| 2 | SNDK.US | 0.337033 | 42 / 49 |
| 3 | MU.US | 0.318474 | 49 / 49 |
| 4 | WDC.US | 0.314044 | 47 / 49 |
| 5 | STX.US | 0.311204 | 49 / 49 |
| 6 | INTC.US | 0.299029 | 42 / 49 |
| 7 | MRVL.US | 0.295930 | 42 / 49 |
| 8 | AMD.US | 0.287544 | 49 / 49 |
| 9 | LRCX.US | 0.281418 | 42 / 49 |
| 10 | FLEX.US | 0.259821 | 42 / 49 |

Only DELL overlaps the native Top 10. The largest individual July SHAP
contribution for all ten names is the cross-sectional z-score of relative EMA
pair `100/311`, between `+0.2449` and `+0.3117`; this is a current-snapshot
explanation, not evidence of SHAP stability through time and not an input to
the trend filter.

The initial Top 5/10/15/20 attempt is retained as a failed run at
`outputs/no_sec_fresh_replay_20260828/common_replay_causal_trend`. Top 15 and
Top 20 would select `SATS.US` for May 2026 despite the missing return under that
snapshot key, so the common engine stopped. The economic observations were
subsequently found in the same snapshot under the provider key `ECHO.US`.
SATS changed ticker only on 24 June 2026, announced 22 June; excluding it at
the April decision using that later event would be a leak.

`DATA-029` derives an additive, line-audited SATS extension from ECHO daily
returns after validating the common 24 April anchor. `REPLAY-006` then rebuilds
every upstream feature and target on immutable composition
`bb1f90a907bbb25e34a32f632abce2f6e4982c1005daa80e778c0f335d968375`.
The score is exactly unchanged at `0.1067675874`, SATS remains rank 14 on the
April trend-filtered decision, and its April-to-May return becomes evaluable at
`+4.9131095%`. The retained failed run is not modified.

The new common replay at
`outputs/sats_echo_replay_20260829/common_replay_causal_trend` completes all
Top 5/10/15/20 strategies. Legacy holdings and performance are identical;
common Boosting scores and probabilities have zero changes; Top 5/10 results
are identical to the prior successful replay. Top 15/20 add this evidence:

| Strategy | CAGR | Volatility | Sharpe | Max drawdown | Mean turnover |
| --- | ---: | ---: | ---: | ---: | ---: |
| Boosting Top 15, causal trend | 21.1093% | 23.9306% | 0.7985 | -35.3611% | 48.2660% |
| Boosting Top 20, causal trend | 18.3883% | 21.7840% | 0.7523 | -33.8463% | 47.7861% |

For the May 2026 holding month, SATS contributes `+0.327541%` gross to Top 15
and `+0.245655%` to Top 20. The portfolios return `+18.3627%` and `+14.6553%`
net respectively, versus `+5.2626%` for SPY. Both manifests pass matching data
lineage, the common profile and price policy; eight terminal entries are
blocked and zero selected censored return remains. The trend replay stays
`publication_eligible=false` because it is post-hoc research, not because of a
missing SATS price.

The autonomous report is
`outputs/sats_echo_replay_20260829/ticker_transition_replay_report.html`; its
hash and the full run protocol are frozen in
`docs/research/sats_echo_replay_20260829.md`.

Key SHA-256 evidence:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `34da25d01c94a8ba01bd41fc99c434aefdfc68e6b4f170b43e01f06a9128cb81` |
| `comparison_common_performance.csv` | `ad89959aee95f62332be58ed122e70b9fe54f4e236c21eaa111fe31efddc7225` |
| `comparison_common_holdings.parquet` | `8e8c6fb304127da91a56f67a5f2431fb4078572f888fa3a4c5642705e037f1b2` |
| `causal_trend_eligibility.parquet` | `f20bc325607c3a3c1383a210f5e8099d9765649cd695d773cf3dcc2c73222d39` |
| `boosting_live_score_holdings.csv` | `56ca017b6b2db051ac646d356a6de16a9b4b421fe0282708715a71cfaf517142` |

The manifest records `status=research_common_strategy_replay`,
`methodology_status=post_hoc_research_diagnostic`,
`comparison_eligible=true` and `publication_eligible=false`.

## Retained SEC/PE Reference — 2026-08-16

```text
ingestion: 20260816_103942
composed snapshot: data/model_inputs/history/alpharank_input_20260816_120458_2a01288bab06
Legacy run: outputs/production_refresh_20260816/legacy_runs_v3/2026-08-16/runs/20260816_142810
Boosting run: outputs/production_refresh_20260816/boosting_latest_common_v3
common replay: outputs/production_refresh_20260816/common_replay_v4_sec_universe
holding calendar: 2011-08 through 2026-07, 180 realized months
benchmark: SPY total return from adjusted_close
transaction costs: 10 bps times turnover for Boosting and Legacy
```

The replay passes the seven required input hashes, the ten-ticker
`historical_ticker_exclusions_v1` registry, and the shared
`monthly_price_eligibility_v1` policy. The last raw price date is 2026-08-14,
but August is incomplete and excluded before feature construction. The July
decision is retained as score-only for the August target; it has no realized
August return and therefore cannot enter performance.

| Strategy | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 28.1562% | 41.4778% | 0.6306 | -51.9717% |
| Boosting Top 10 | 26.5717% | 33.4204% | 0.7352 | -25.3178% |
| Boosting Top 15 | 24.3051% | 30.7396% | 0.7256 | -25.5978% |
| Boosting Top 20 | 20.6066% | 28.6445% | 0.6496 | -31.9839% |
| Legacy | 18.9965% | 23.1727% | 0.7335 | -25.7052% |
| SPY total return | 14.3975% | 14.3014% | 0.8669 | -23.9272% |

These figures use the common cost and calendar convention. Standalone Legacy
production retains its historical zero-cost convention. Boosting remains R&D;
its higher CAGR is paired with much higher volatility, and Top 5 has a 51.97%
maximum drawdown.

### SEC/PE-universe sensitivity

`common_replay_v4_sec_universe` reconstructs Legacy's point-in-time valuation
eligibility for every one of the 88,948 Boosting test predictions. The registry
contains 75,824 eligible ticker-months, 5,581 observations with PE at or below
zero, 2,434 with PE at or above 100, 2,086 without a point-in-time PE despite
some SEC rows, and 3,023 with no SEC source row. The matched variants filter the
Boosting predictions to the same `0 < PE < 100` universe before ranking.

| Strategy on Legacy PE universe | CAGR | Volatility | Sharpe | Max drawdown |
| --- | ---: | ---: | ---: | ---: |
| Boosting Top 5 | 29.7055% | 29.2400% | 0.9475 | -25.7806% |
| Boosting Top 10 | 26.0726% | 26.1535% | 0.9204 | -21.8392% |
| Boosting Top 15 | 20.8325% | 23.8169% | 0.7907 | -26.1402% |
| Boosting Top 20 | 19.9549% | 22.2000% | 0.8088 | -25.8106% |

Top 10 loses only 0.50 percentage point of CAGR after imposing Legacy's SEC/PE
eligibility, while volatility falls by 7.27 points and maximum drawdown improves
by 3.48 points. This rejects the narrow hypothesis that Top 10's historical
advantage is mainly caused by selecting names unavailable to Legacy because of
SEC coverage. It does not by itself prove absence of universe, model-selection,
or data-snooping bias.

A 50,000-draw circular block bootstrap with 12-month blocks is stored in
`paired_block_bootstrap_top10_matched.csv`. Matched Top 10 minus Legacy has an
annualized arithmetic-return difference of +6.49 points, but its 95% interval
[-3.18, +15.92] crosses zero. Against SPY the difference is +12.09 points with
a 95% interval [+5.54, +19.56]. The same-data result is therefore strong enough
to continue validation, but not sufficient to replace Legacy.

The retained machine-readable evidence is stored with
`common_replay_v4_sec_universe`: performance, monthly returns, the valuation
registry and `paired_block_bootstrap_top10_matched.csv`. It is a research
diagnostic, not a promotion or an application dashboard contract.

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

Core tests are in `tests/unit/test_portfolio_engine.py` and
`tests/unit/portfolio/test_portfolio_attribution.py`. They cover timing,
lookahead rejection, missing-return handling, adapter equivalence, turnover,
calendar alignment, and complete-year performance. Existing backtest and
multi-horizon tests exercise the compatibility wrappers.

## Previous Eligible Reference — 2026-08-12

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
