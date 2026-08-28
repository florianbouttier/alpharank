# Legacy And Boosting Methodologies

Last updated: 2026-08-17

This document is the source of truth for signal generation in the two methods
compared by the common replay. Portfolio simulation, benchmarks,
performance metrics, lineage validation, and CAGR attribution are documented
separately in [`common_portfolio_backtest_engine.md`](./common_portfolio_backtest_engine.md).

## Current Validated Comparison

The current replay is built from one immutable composed input snapshot:

- source ingestion: `20260816_103942`, with prices and SEC filings through
  `2026-08-14`;
- composed snapshot:
  `data/model_inputs/history/alpharank_input_20260816_120458_2a01288bab06`;
- Legacy run:
  `outputs/production_refresh_20260816/legacy_runs_v3/2026-08-16/runs/20260816_142810`;
- Boosting run:
  `outputs/production_refresh_20260816/boosting_latest_common_v3`;
- common replay:
  `outputs/production_refresh_20260816/common_replay_v4_sec_universe`;
- input lineage: all seven required hashes, the ten-ticker quarantine, and the
  monthly price-eligibility policy match;
- decision calendar with realized one-month returns: 2011-07 through 2026-06;
- holding calendar: 2011-08 through completed 2026-07;
- score-only decision: 2026-07, producing the August 2026 target without using
  partial August prices or unevaluable future labels;
- benchmark: SPY total return from `adjusted_close`;
- timing: decision at month `t`, holding and return at month `t+1`.

The composed package uses strict SEC-only fundamentals. Prices use one fresh
Yahoo vintage for 502 refreshable active tickers and preserve 338 validated
inactive/terminal histories from the frozen EODHD-seeded lineage. Its routine
price revision gate passes with zero removed historical keys, zero return
availability changes, zero historical daily-return changes above 1 bp, and
zero adjustment-transition findings. The SEC package is a reviewed one-time
point-in-time migration: raw Companyfacts retain filing versions and model
exports select the earliest available filing.

The two methods may intentionally use different signal logic, but they may not
silently use different data-quality quarantines or monthly tradability gates.
A common snapshot proves raw-data comparability, not methodological identity or
preprocessing parity.

## Data Usage Matrix

| Input | Legacy `Combined_Frequency` | Boosting `legacy_ema_latest_common_v1` |
| --- | --- | --- |
| Daily stock prices | Yes | Yes |
| Daily S&P 500 prices | Yes | Yes |
| Historical S&P 500 membership | Yes | Yes |
| Full-trajectory data-quality quarantine | Required; versioned registry is now the default | Required; must match Legacy exactly |
| Volume and OHLC quality | Yes, shared point-in-time monthly gate | Yes, identical shared gate |
| Income, balance, cash flow, earnings | Yes | Loaded and hash-checked, but not fed to the current model |
| Sector | Yes, position cap | No |
| Legacy historical winners | Native output | Yes, only to define the causal EMA feature catalogue available in each fold |

For any new official comparison, the fundamental files in this matrix must be
SEC/GAAP only. EODHD is retained for historical prices, not fundamental values.
Both methods must consume the same composed snapshot and hashes even though the
current Boosting profile ultimately drops fundamental features.

### Does The Current Boosting Model Use Fundamentals?

No. The shared research-frame builder computes broad technical and fundamental
features, and the manifest records the fundamental input hashes. The public
profile then uses `feature_mode=legacy_winners_pit_ema_only`, which removes all
non-relative-EMA features before fold preprocessing and model fitting.

Each retained EMA pair contributes five columns:

1. raw short-EMA / long-EMA ratio on stock price divided by S&P 500 price;
2. monthly cross-sectional percentile rank;
3. monthly cross-sectional z-score;
4. top-quartile flag;
5. bottom-quartile flag.

The 16 outer folds contain 10 to 185 features and zero fundamental features.
The last fold has 37 causal EMA pairs, hence 185 model inputs. Other research
modes such as `broad` or `legacy_winners_pit_ema_plus` may include fundamental
features, but they are not the current public comparison.

## Shared Liquidity And Price-Quality Gate

The policy `monthly_price_eligibility_v1` applies all three conditions to each
ticker and decision month in both Legacy and Boosting:

| Condition | Threshold | Exact calculation |
| --- | ---: | --- |
| Price observations | at least 10 | non-null daily `close` observations in the month |
| Median daily dollar volume | at least USD 1,000,000 | median of daily `close * volume` |
| OHLC violation rate | at most 5% | invalid OHLC days divided by observed days |

A daily OHLC row is invalid if any of these conditions is true:

- `high < max(open, close, low)`;
- `low > min(open, close, high)`;
- `high < low`;
- `open <= 0` or `close <= 0`;
- a required OHLC value is null.

Failure removes only that ticker-month from both investable universes. A ticker
can become eligible again in another month. Legacy applies the gate before its
monthly EMA Top 30 so the next eligible security can enter the ranking; it is
not removed after selection. Separately, the versioned
historical ticker quarantine removes the complete trajectory of known broken
identifiers before feature construction. This list must match Legacy exactly
for a common comparison.

Observed effect on the current snapshot:

| Decision month | S&P 500 rows | Eligible | Excluded | Explanation |
| --- | ---: | ---: | ---: | --- |
| 2026-05 | 503 | 499 | 4 | permanent quarantine removes `SW.US`; `BK.US` and `SATS.US` have no monthly price rows; `CTRA.US` has only 5 observations |
| 2026-06 | 504 | 502 | 2 | permanent quarantine removes `SW.US`; `SATS.US` has no monthly price rows |
| 2026-07 | 503 | 502 | 1 | permanent quarantine removes `SW.US`; every remaining member passes the monthly gate |

Legacy's published policy is `no_sec_fundamentals_v1`. It intersects observed
monthly prices, historical membership and the shared price-eligibility gate;
it does not read or compute a SEC value. The former
`legacy_pe_market_cap_v1` policy, which additionally required available market
capitalization and `0 < PE < 100`, remains available only for an explicitly
labelled compatibility replay. Boosting's public feature profile has no
fundamental feature, but its exact EMA pairs are relearned from the Legacy run
produced under the same policy; SEC therefore has no direct model feature and
the Legacy policy still has an indirect effect on its learned EMA catalogue.

The single code owner is `src/alpharank/data/price_eligibility.py`. Every run
records the policy id and three thresholds in its manifest. Legacy also writes
`monthly_price_eligibility.parquet`. A Legacy/Boosting replay fails closed if
raw hashes, full-trajectory exclusions, or the monthly policy differ.

## Complete Legacy Pseudocode

Canonical entrypoint: `scripts/run_legacy.py`.

```text
INPUT one immutable composed package:
      approved EODHD/open prices + SEC-only fundamentals + constituents

1. Create timestamped run directory.
2. Copy every required source into run/input_snapshot.
3. Hash source files, copied files, critical code, and run configuration.
4. Refuse an unclean open-source package lineage.
5. Load prices, SPY and historical constituents. The immutable package still
   contains the four fundamental files required by its schema, but the default
   selection policy does not read their values.
6. Remove the declared versioned historical ticker quarantine from all relevant
   inputs. A new production run uses historical_ticker_exclusions_v1 by default.
7. Build monthly_price_eligibility_v1 from rows observed in each decision month.
   Intersect historical S&P membership with eligible ticker-months before EMA
   ranking and retain the complete eligibility ledger.

8. Build market series:
   stock_monthly_return[ticker, month]
       = product(1 + adjusted_close_daily_return) - 1
   relative_close[ticker, day]
       = stock_adjusted_close / SP500_close

9. Apply the manifest's fundamental eligibility policy:

   `no_sec_fundamentals_v1`, the production default, intersects observed
   ticker-month prices with historical S&P membership and performs no
   fundamental calculation.

   `legacy_pe_market_cap_v1`, retained for compatibility, builds point-in-time
   fundamental eligibility:
   a. consolidate statements using filing/report dates;
   b. compute rolling net income, shares, revenue, equity, debt, cash, EBITDA;
   c. forward-fill only after the value's available date;
   d. compute market_cap and valuation ratios;
   e. retain historical S&P 500 members with market_cap present and 0 < PE < 100;
   f. shift this eligibility by one month before applying it to holdings.

   A compatibility PE replay performs steps a-f only when that policy is
   requested explicitly.

10. Run four independent annual expanding Optuna tracks:
   track 11 = alpha 2, seed 42
   track 12 = alpha 2, seed 41
   track 21 = alpha 1, seed 42
   track 22 = alpha 1, seed 41
   common settings = EMA, trailing half-life 120 months, mean aggregation

11. For each January split S from first_date onward, in each track:
    train_prices = rows with decision month < S

    Optuna samples 30 candidates in production:
        n_long in [50, 400]
        n_short in [1, 100]
        n_asset in [5, 30]
        n_max_per_sector in [1, 2]

    For each candidate:
        relative_ema_short = EMA(relative_close, n_short)
        relative_ema_long  = EMA(relative_close, n_long)
        mtr = relative_ema_short / relative_ema_long
        take last daily mtr for each ticker and decision month
        keep only point-in-time S&P 500 members
        preselect the 30 highest mtr values
        move signal month t to holding month t+1
        intersect with shifted fundamental eligibility
        cap each sector at n_max_per_sector
        retain highest n_asset values
        portfolio_return = equal-weight realized return
        relative_factor = portfolio_gross_factor / SP500_price_return_factor
        monthly_score = log(1 + alpha * (relative_factor - 1))
        objective = decreasing weighted mean of monthly_score,
                    with 120-month half-life

    Re-score the ten best sampled candidates plus stable anchor candidates.
    Resolve an exact tie deterministically by asset count, sector cap,
    long EMA, then short EMA.
    Refit the selected parameters on all available prices.
    Keep only holdings after S until the next annual split replaces them.

12. Combine the four finalized tracks by month:
    Combined_Equal:
        union all selected tickers and assign equal weights
    Combined_Frequency (production Legacy):
        union all selected tickers
        vote_count = number of tracks selecting the ticker
        target_weight = vote_count / sum(vote_count)

13. Convert finalized baskets to the common holdings contract:
    decision_month = holding_month - 1 month
    Legacy historical transaction cost = 0 bps
    simulate returns with the shared portfolio engine
    compare with SPY total return from adjusted_close

14. Write detailed holdings, monthly returns, reports, manifest, checkpoints,
    immutable input snapshot, and CLI log.
```

The fundamental screen is a filter, not a predictive regression input. Legacy's
ranking score itself is the relative EMA ratio; fundamentals decide whether a
ranked stock remains investable.

## Complete Boosting Pseudocode

Canonical research entrypoint:
`scripts/experiments/run_multihorizon_boosting.py --latest-common-comparison-profile`.

```text
INPUT the exact Legacy run/input_snapshot and Legacy detailed holdings
PROFILE legacy_ema_latest_common_v1

1. Validate profile constants and hash every input.
2. Load daily prices, SPY, historical constituents, and fundamentals.
3. Apply the shared historical ticker quarantine.
4. Build the monthly liquidity/OHLC eligibility table using only data in month t.
5. Build a broad monthly research frame:
   monthly prices + technicals + fundamentals + relative EMA features
   INNER JOIN historical S&P 500 membership at decision month t
   FILTER the point-in-time price eligibility at t

6. Build each relative EMA feature from daily data available through month t:
   relative_close = stock_price / SPY_price
   ratio(short,long) = EMA(relative_close, short) / EMA(relative_close, long)
   take the last ratio in decision month t
   add rank, z-score, top-quartile, and bottom-quartile cross-sectional forms

7. Build the six-month learning target:
   stock_future_6m = close[t+6] / close[t] - 1
   spy_future_6m   = spy[t+6] / spy[t] - 1
   future_excess_6m = (1 + stock_future_6m) / (1 + spy_future_6m) - 1
   target = 1 when the stock is in the top 10% of future_excess_6m at t,
            otherwise 0

   classify every H6 row before training as evaluable, terminal-event resolved,
   horizon pending, benchmark unavailable, ticker target unavailable, or
   terminal event unresolved. A mature missing benchmark/ticker/terminal target
   fails the fold before fitting; it is never removed by a generic null filter.
   Write the complete train/validation/test census to
   fold_target_censoring.csv for every outer fold.

8. Create strict expanding outer walk-forward windows:
   minimum train = 62 months
   validation = 6 months
   test = 12 months
   step = 12 months
   purge H-1 months between label-bearing intervals for H=6
   include the final partial test window
   hold one fitted model fixed over each test block

9. For each outer fold:
   train_cutoff = last decision month in outer train
   read only Legacy EMA winners observable by train_cutoff
   candidate_features = five forms of each causally available winning EMA pair
   discard every broad technical, regime, and fundamental feature

   inner CPCV exists for tuning, but the public profile has n_trials=0;
   therefore use the fixed XGBoost parameters below without parameter search.

   fit preprocessing on outer train only:
       remove feature if train missing ratio > 35%
       record train global median
   transform train, validation, and test:
       fill with same-month cross-sectional median
       then train global median
       then zero

   fit XGBoost classifier through mlcraft:
       eta=0.04, max_depth=5, min_child_weight=20
       subsample=0.8, colsample_bytree=0.75
       lambda=5.0, alpha=0.2
       seed=42+fold, up to 100 rounds
       early stopping after 25 validation rounds

   if validation contains both classes:
       fit isotonic calibration on validation raw probabilities only

   predict train, validation, and untouched outer test rows
   compute test metrics only where the six-month target is mature
   retain score-only tail decisions through 2026-06 when the t+1 return exists,
   even if the six-month target is still pending

   compute SHAP for every outer-test prediction; no row sampling

10. For each test decision month and N in {5, 10, 20}:
    rank raw XGBoost scores descending with deterministic ticker tie-break
    retain isotonic probabilities for calibration diagnostics only
    select top N
    assign equal target weights
    hold during t+1

11. Convert predictions to the common holdings contract, then join realized
    returns without changing the selected names. If a selected return is absent,
    resolve it only from a sourced terminal event in the holding month:
    cash consideration for a cash merger; cash plus exchange ratio times the
    successor month-end price for a stock merger; explicit recovery, including
    zero, for bankruptcy or delisting; or a one-for-one/declared-ratio successor
    price for a ticker change. Every event and successor price must identify the
    same immutable price vintage. Keep an unresolved row missing so simulation
    fails closed; never promote the next surviving ticker.
12. Simulate gross return, turnover, 10 bps * turnover cost, and net return.
13. Fail comparison unless all seven input hashes, both full-trajectory
    ticker-exclusion sets, and both monthly price-eligibility policies match.
14. Resimulate both Legacy and Boosting holdings with the same comparison cost
    policy (`10 bps * turnover` in the current reference); keep standalone
    Legacy production metrics under their documented historical convention.
15. Write predictions, fold boundaries, feature manifests, train/validation/test
    metrics, exhaustive SHAP, holdings, monthly returns, performance, and lineage.
```

## Code Architecture

```text
data/open_source/output or retained history snapshot
                    |
                    v
scripts/run_legacy.py ---------------------- immutable Legacy run package
        |                                             |
        v                                             |
src/alpharank/data/price_eligibility.py               |
        |                                             |
        v                                             |
src/alpharank/data/processing.py                      |
        |                                             |
        v                                             |
src/alpharank/strategy/legacy.py                      |
        |                                             |
        +---- detailed finalized Legacy baskets ------+
                                                      |
                                                      v
scripts/experiments/run_multihorizon_boosting.py <----+
        |
        v
src/alpharank/multihorizon/
    config.py          versioned public profile
    data.py            research frame and targets; consumes shared gate
    legacy_ema.py      causal Legacy-winner EMA feature catalogue
    splits.py          outer walk-forward and purged CPCV
    preprocessing.py   train-only sparse filter and imputation
    modeling.py        mlcraft/XGBoost and calibration
    pipeline.py        fold orchestration, predictions, SHAP, artifacts
        |
        +---- out-of-sample Boosting scores
        |
        v
src/alpharank/portfolio/
    adapters/          methodology outputs -> common holdings
    contracts.py       decision t / holding t+1 validation
    allocation.py      top-N, weights, turnover helpers
    terminal_returns.py terminal-event total shareholder returns
    simulation.py      gross/net return and transaction costs
    benchmark.py       SPY adjusted-close total return
    performance.py     shared performance statistics
    attribution.py     exact CAGR decomposition
    lineage.py         same-snapshot hash gate
    artifacts.py       standard audit package
        |
        v
scripts/build_common_legacy_boosting_replay.py
        |
        +---- auditable Parquet / CSV / JSON artifacts
```

## Ownership And Change Rules

| Concern | Single owner |
| --- | --- |
| Monthly production entrypoint | `scripts/run_legacy.py` |
| Legacy signal and annual Optuna logic | `src/alpharank/strategy/legacy.py` |
| Current Boosting profile | `src/alpharank/multihorizon/config.py` |
| Shared monthly liquidity/OHLC gate | `src/alpharank/data/price_eligibility.py` |
| Boosting data and target construction | `src/alpharank/multihorizon/data.py` |
| Walk-forward and purge | `src/alpharank/multihorizon/splits.py` |
| XGBoost fitting | `src/alpharank/multihorizon/modeling.py` |
| Shared holdings and simulation | `src/alpharank/portfolio/` |
| Common Legacy/Boosting lineage gate | `src/alpharank/replay/common_strategy.py` |

Portfolio owns interactive monitoring and allocation views. AlphaRank publishes
only the underlying immutable calculations, manifests and audit artifacts.

Do not implement another local CAGR, Sharpe, drawdown, benchmark, portfolio
return, or lineage comparison. Signal changes belong to the method-specific
modules; everything after finalized holdings belongs to
`src/alpharank/portfolio/`.

## Validated Reference Run

The current same-data reference uses the composed package
`alpharank_input_20260827_122648_5bfbc1d3cb04`, whose prices and benchmark were
acquired through 26 August 2026. Legacy run `20260828_184601` declares
`no_sec_fundamentals_v1`; Boosting uses only the exact point-in-time Legacy
winner EMA catalogue. The common replay under
`outputs/no_sec_fresh_replay_20260828/common_replay` passes matching input
hashes, ticker exclusions, monthly price eligibility and terminal-entry gates.
Its complete realized holding calendar is August 2011 through July 2026;
Boosting has no OOS portfolio before August 2011.

On that exact calendar, with 10 bps times turnover for both strategies and no
simulated cost for SPY, native-universe CAGR is 19.9416% for Boosting Top 5,
19.9271% for Top 10, 18.8992% for Top 15, 17.8061% for Top 20, 19.6430% for
Legacy, and 14.3975% for SPY total return. The manifest records
`comparison_eligible=true`, `publication_eligible=true`, eight post-terminal
prediction rows blocked and zero selected censored return. Boosting remains a
research result, not a promotion into the canonical monthly production
workflow.

The retained 16 August PE-universe replay remains a diagnostic for SEC coverage
asymmetry. `src/alpharank/strategy/legacy_valuation.py` can reconstruct its
point-in-time `0 < PE < 100` universe when explicitly requested; the matched
Top 10 CAGR was 26.0726%, with 26.1535% volatility, 0.9204 Sharpe, and
-21.8392% maximum drawdown. Those figures do not describe the current no-SEC
default and are not a substitute for chronological out-of-sample validation.

Boosting retains 88,950 test predictions and exactly 88,950 SHAP rows over 181
decision months. SHAP is exhaustive, not sampled. There are 16 outer folds;
the H6 target is mature through decision January 2026, while February through
July are score-only H6 rows. July's one-month return is deliberately null
because August is incomplete.

### Completed-month boundary

Both methods share the same completed-month contract. Legacy truncates daily
price inputs before feature construction. Boosting may still score the final
completed decision month, but every target whose return window ends after that
month is forced to null. Such rows remain available for portfolio selection
and exhaustive SHAP; they are excluded from model metrics and realized
performance. This prevents a partial current month from becoming either a
one-month return or a mature six-month label.
