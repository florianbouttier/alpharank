# Exact Legacy EMA boosting study — 2026-07-25

## Conclusion

The best autonomous result in this screening is:

`legacy_winners_pit_ema_only / classification / 6 months / top 5`

On its native out-of-sample period, November 2013 through October 2025
(144 monthly decisions):

| portfolio | net total return | CAGR | Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| EMA-only boosting, top 5 | +2,628.5% | 31.7% | 0.969 | -30.3% |
| S&P 500 | +364.7% | 13.7% | 0.960 | -23.9% |
| Legacy | +511.8% | 16.3% | 0.873 | -23.4% |

This is the first clean boosting result in this track that beats Legacy on
both total return and Sharpe over a long native test period. It still takes
more concentration and drawdown risk than Legacy. The result is a research
candidate, not a production proof, because this same walk-forward history was
used to compare many methods, horizons and top-N values.

The initial run incorrectly used isotonic-calibrated probabilities to rank
stocks. Calibration plateaus created ties which were implicitly broken by
ticker order. Those outputs are retained under
`outputs/multihorizon_boosting/invalid_calibrated_tie_20260725`. All figures in
this note come from the corrected rerun: raw booster probability for ranking,
isotonic probability only for Brier, log-loss and calibration error.

## What was tested

Frozen data package:

`outputs/2026-07-19/runs/20260719_194418`

Horizons:

`1, 3, 6, 12, 24, 36 months`

Objectives:

- classification of the future top 10% excess-return stocks;
- regression of future stock excess return versus the S&P 500;
- monthly grouped ranking of future excess-return ranks;
- teacher classification of the actual Legacy basket.

Feature modes:

1. `broad`: former 45-pair rounded EMA grid plus the full point-in-time
   technical, risk, regime and fundamental set.
2. `legacy_winners_pit_ema_only`: only exact EMA pairs that had already appeared
   as winners in one of the four Legacy Optuna paths by the end of the outer
   training fold.
3. `legacy_winners_pit_ema_plus`: the same point-in-time winner pairs plus
   non-relative-EMA context.
4. `legacy_active_oracle`: the four EMA pairs chosen by Legacy for the current
   month. This is diagnostic and not autonomous.

The frozen Legacy schedule contains 32 distinct exact winning pairs. In the
6-month walk-forward, the autonomous point-in-time feature dictionary grows
from 2 known pairs / 10 EMA variables in the first fold to 24 pairs / 120 EMA
variables in the last fold. Future winners are never injected into earlier
folds.

Outer protocol:

- expanding walk-forward;
- 72 training months;
- 24 validation months;
- 12 test months;
- horizon-aware purges between training, validation and test;
- one fixed booster over each annual test block;
- fold-only sparse filtering and medians;
- no hyperparameter search in this screening (`n_trials=0`);
- 10 bps multiplied by monthly turnover in trading backtests.

Test coverage:

| horizon | first test | last test | months | folds |
|---:|---:|---:|---:|---:|
| 1 | 2013-01 | 2025-12 | 156 | 13 |
| 3 | 2013-05 | 2025-04 | 144 | 12 |
| 6 | 2013-11 | 2025-10 | 144 | 12 |
| 12 | 2014-11 | 2024-10 | 120 | 10 |
| 24 | 2016-11 | 2023-10 | 84 | 7 |
| 36 | 2018-11 | 2022-10 | 48 | 4 |

The active oracle starts only in 2010. It has no valid 36-month test window
under the unchanged protocol; the runner records this as `unavailable.json`
instead of shortening training.

## Model quality on the test sets

### Future top-decile classification

| mode | best horizon | ROC-AUC | PR-AUC | PR lift / prevalence |
|---|---:|---:|---:|---:|
| broad | 3m | 0.639 | 0.171 | 1.69x |
| exact EMA only | 6m | 0.611 | 0.164 | 1.62x |
| exact EMA + context | 3m | 0.641 | 0.172 | 1.70x |
| active oracle | 6m | 0.580 | 0.152 | 1.50x |

The broad and EMA-plus models discriminate future winners better globally.
EMA-only at 6 months is weaker in aggregate AUC but much stronger in the
extreme top 5 portfolio. AUC therefore does not select the trading winner by
itself.

### Regression

All excess-return regressions have negative out-of-sample R2 and normalized
RMSE above 1. They do not beat a constant forecast for return magnitude.

The strongest ordinal information is at 24 months:

- broad: Spearman IC `0.096`, R2 `-0.038`;
- EMA plus: Spearman IC `0.092`, R2 `-0.034`;
- EMA only: Spearman IC `0.047`, R2 `-0.054`.

Regression remains useful as a ranking diagnostic, not as a calibrated point
forecast.

### Native ranking

Ranking does not show one stable winning horizon. Exact EMA-only reaches
Legacy overlap `47.2%` at 36 months and EMA-plus `48.2%`, but this is based on
only four outer folds. The broad 24-month ranker is the more defensible
long-horizon diagnostic: IC `0.048`, NDCG@10 lift `+0.016`, Legacy overlap
`40.4%` for top 20.

### Reproducing the actual Legacy basket

The direct teacher objective is clearly the right statistical problem when the
goal is to copy Legacy choices:

| mode | ROC-AUC | PR-AUC | PR lift | overlap top 5 | top 10 | top 20 |
|---|---:|---:|---:|---:|---:|---:|
| broad | 0.958 | 0.325 | 19.1x | 28.6% | 49.6% | 73.7% |
| exact EMA only | **0.975** | **0.342** | **20.1x** | 29.9% | 51.4% | **77.4%** |
| exact EMA + context | 0.965 | 0.325 | 19.1x | **31.8%** | 51.3% | 73.7% |
| active oracle | 0.978 | 0.375 | 22.1x | 28.8% | **52.9%** | 80.9% |

The oracle confirms that the current Legacy pair identity contains additional
copy information, but it cannot be used as an autonomous replacement.

## Trading comparison

### Corrected autonomous champion

`classification / 6m / exact EMA only`:

| top N | net total return | CAGR | Sharpe | max drawdown |
|---:|---:|---:|---:|---:|
| 5 | +2,628.5% | 31.7% | 0.969 | -30.3% |
| 10 | +1,135.1% | 23.3% | 0.835 | -36.4% |
| 20 | +791.4% | 20.0% | 0.766 | -38.2% |

The edge is concentrated in the first five names. Only top 5 beats the
same-period Legacy Sharpe of `0.873`.

On the period shared by all four feature modes for this exact configuration,
November 2018 through October 2025:

| feature mode | net total return | CAGR | Sharpe | max drawdown |
|---|---:|---:|---:|---:|
| broad | +194.2% | 16.7% | 0.525 | -52.4% |
| exact EMA only | **+815.7%** | **37.2%** | **1.003** | -30.3% |
| exact EMA + context | +248.9% | 19.6% | 0.619 | -42.5% |
| active oracle | +541.2% | 30.4% | 0.858 | -31.9% |
| Legacy | +132.6% | 12.8% | 0.667 | **-23.4%** |

Adding the full context dilutes the useful exact-EMA signal. Even the active
oracle ranks worse than the accumulated prior-winner dictionary here.

Yearly robustness for the EMA-only top 5:

- beats Legacy in 9 of 13 calendar-year fragments;
- 2016 is unusually strong at `+128.3%`;
- excluding 2016, CAGR is still `25.3%` and Sharpe `0.814`, versus Legacy
  CAGR `14.7%` and Sharpe `0.784`;
- from 2018 onward, CAGR is `30.0%`, Sharpe `0.876`, drawdown `-30.3%`,
  versus Legacy CAGR `13.1%`, Sharpe `0.677`, drawdown `-23.4%`.

### Teacher portfolio

The strongest Legacy-copy trading result is EMA-plus teacher top 5 over
January 2018 through December 2025:

- net total return `+685.1%`;
- CAGR `29.4%`;
- Sharpe `1.057`;
- max drawdown `-20.0%`;
- same-period Legacy: `+208.6%`, CAGR `15.1%`, Sharpe `0.751`,
  drawdown `-23.4%`.

This is useful for probabilistic rationalization of Legacy decisions, but the
target is the Legacy basket itself, not future outperformance.

## SHAP

For the autonomous 6-month EMA-only classifier, the strongest aggregate SHAP
signals are:

| feature | mean absolute SHAP | descriptive direction |
|---|---:|---|
| `relative_ema_ratio_95_72_z_month` | 0.069 | higher decreases score |
| `relative_ema_ratio_100_326_z_month` | 0.064 | higher increases score |
| `relative_ema_ratio_95_72_rank_month` | 0.039 | higher increases score |
| `relative_ema_ratio_92_183_z_month` | 0.039 | higher decreases score |
| `relative_ema_ratio_27_106_z_month` | 0.038 | higher increases score |
| `relative_ema_ratio_7_333` | 0.034 | higher decreases score |
| `relative_ema_ratio_100_326` | 0.032 | higher decreases score |

The pair `95/72` is not a conventional short-below-long EMA. It is present
because it actually won in Legacy and must not be silently normalized away.
Opposite directions between raw, rank and z-score versions are possible in a
tree model with correlated variables and interactions. SHAP directions are
descriptive, not causal or monotonic constraints.

SHAP used 20 test rows per outer fold. The aggregate is sufficient to identify
signal families, but not to claim precise economic thresholds.

## Bias and leakage audit

Controls implemented:

- frozen input snapshot and hashed Legacy inputs;
- future-return labels shifted strictly after the decision month;
- horizon maturity and purges in outer walk-forward;
- preprocessing fitted inside each fold;
- exact EMA winners selected only through the outer training cutoff;
- no Legacy label, basket weight or vote count in economic model features;
- raw score used for ranking and calibrated probability used only for
  probability-quality metrics;
- invalid tied-score runs retained and clearly separated.

Remaining risks:

1. **Research selection bias.** The same walk-forward history was used to choose
   objective, horizon, feature mode and top N. The champion needs a new
   untouched meta-holdout or prospective paper period.
2. **Legacy-assisted feature discovery.** The EMA dictionary comes from prior
   Legacy winners. It is point-in-time inside each fold, but the research idea
   itself was selected after inspecting Legacy.
3. **Concentration.** The advantage is largest at top 5 and weakens sharply at
   top 10/20.
4. **Execution simplification.** The backtest uses equal weights, monthly close
   returns and 10 bps times turnover; it does not model market impact, taxes,
   borrow constraints or intramonth execution.
5. **Data revisions and membership.** Results rely on the retained July 2026
   snapshot and its historical constituent/fundamental reconstruction. A
   second independent data snapshot should reproduce the result.
6. **Multiple comparisons.** Six horizons, three economic objectives, four
   feature modes and three top-N values were inspected. The best line should
   not be treated as a single pre-registered test.
7. **Long horizons.** 24 months has seven folds and 36 months only four; the
   latter remains exploratory.

## Reproduction and artifacts

Runner:

`scripts/experiments/run_multihorizon_boosting.py`

Comparison:

`scripts/experiments/compare_multihorizon_feature_modes.py`

Corrected runs:

- `outputs/multihorizon_boosting/screening_clean_20260725`
- `outputs/multihorizon_boosting/legacy_winners_pit_ema_only_20260725`
- `outputs/multihorizon_boosting/legacy_winners_pit_ema_plus_20260725`
- `outputs/multihorizon_boosting/legacy_active_oracle_20260725`

Consolidated machine-readable outputs:

- `outputs/multihorizon_boosting/exact_legacy_ema_comparison_20260725/model_metrics_all_modes.csv`
- `outputs/multihorizon_boosting/exact_legacy_ema_comparison_20260725/trading_native_all_modes.csv`
- `outputs/multihorizon_boosting/exact_legacy_ema_comparison_20260725/trading_common_across_modes.csv`

Each run retains its manifest, test coverage, fold metrics, fold feature
manifest, monthly trading series, aggregate backtests, SHAP importance,
direction and plots. Large row-level prediction and SHAP sample parquets were
removed only after those aggregates were produced because the workstation had
less than 700 MB free.

## Decision

1. Keep `legacy_winners_pit_ema_only / classification / 6m` as the autonomous
   challenger.
2. Keep `legacy_winners_pit_ema_only / teacher / 1m` as the cleanest
   probabilistic Legacy-copy model.
3. Use EMA-plus teacher only when the objective is trading a probabilistic
   imitation of Legacy, not forecasting future outperformance.
4. Reject return regression as a calibrated magnitude forecast for now.
5. Do not promote 36-month results.
6. Next validation must freeze the 6-month/top-5 specification before looking
   at a new holdout. Add probability calibration, expected volatility and
   downside-risk heads without changing the stock-ranking score until the
   challenger has survived that holdout.
