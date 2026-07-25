# Protocol

## Model and horizon matrix

Economic targets are evaluated at 1, 3, 6, 12, 24 and 36 months:

| family | target | score semantics |
|---|---|---|
| classification | top future-excess-return decile | calibrated probability |
| regression | clipped cumulative future excess return | expected excess return |
| ranking | monthly relevance from future excess-return percentile | relative ranking score |
| teacher diagnostic | Legacy `Combined_Frequency` selection | calibrated probability of Legacy selection |

The 24- and 36-month targets are explicitly exploratory. They have fewer
independent regimes, more overlapping labels and a greater sensitivity to
survivorship and data revisions. They may generate hypotheses but cannot alone
justify a production decision.

Future volatility and future downside-deviation targets are stored in the
research frame for the next multi-head allocation step.

## Features

- Ex-ante grid of 43 daily relative-price EMA pairs. The grid is declared
  before the experiment and is not harvested from the final Legacy results.
- For every EMA ratio: raw value, monthly percentile, monthly z-score, top and
  bottom quartile flags.
- Monthly momentum, EMA, RSI, Bollinger, stochastic, range and volatility
  features.
- Point-in-time fundamental features joined by filing date.
- SPY regime, breadth and cross-sectional dispersion.
- Historical constituent membership.

Ticker identity, future targets and Legacy labels/weights are excluded from the
economic feature matrix.

## Temporal validation

The outer estimate is an expanding walk-forward. A model is frozen for each
12-month test block. For horizon `h`:

- the latest validation decision must be at least `h` months before the first
  test decision, so its label is observable;
- training decisions whose label interval overlaps the validation interval are
  purged;
- feature sparse filtering, cross-sectional null fill and fallback medians are
  fitted using training rows only;
- the test block never participates in feature choice, calibration, early
  stopping or tuning.

The default minimum expanding history is 72 months. This is the longest common
choice that still leaves several truly out-of-sample windows for the 36-month
exploratory target in the available 2005-2026 panel.

Optional tuning uses combinatorial purged cross-validation only inside the
pre-test train/validation history. Every inner fold repeats train-only feature
selection and imputation. The final screening starts with fixed conservative
parameters; tuning is reserved for shortlisted candidates to limit
multiple-testing bias.

## Metrics

Every economic model is compared on the same outputs:

- mean monthly Spearman information coefficient;
- mean monthly NDCG;
- average horizon excess return of top 5, 10 and 20;
- realized one-month excess return of those same baskets;
- overlap and Jaccard similarity with the Legacy basket;
- ROC AUC, average precision and Brier score for probability models;
- fold dispersion and calendar coverage.

The primary economic choice must use out-of-sample return and risk. Legacy
overlap is a representation diagnostic, not the optimization objective.

## SHAP

TreeSHAP is sampled separately from every outer test fold. Each
method/horizon directory contains:

- `shap_samples.parquet`: row-level out-of-sample SHAP values;
- `shap_importance.csv`: mean absolute SHAP by feature;
- `shap_importance.png`: compact global plot.

Interpretation is split into:

1. stable variables shared across folds;
2. horizon-specific variables;
3. variables explaining economic predictions versus variables explaining the
   Legacy teacher;
4. sign/shape checks on the row-level samples before any causal language.

SHAP explains the fitted model, not the market, and correlated EMA features can
share or exchange importance.

## Bias and leakage checklist

- Frozen input snapshot and exact matching Legacy output.
- Historical S&P 500 membership, not current constituents.
- Filing-date joins for fundamentals.
- Exact calendar-gap requirement for every future target.
- No global imputation or feature filtering before folds.
- No future test block in Optuna, calibration or early stopping.
- No final Legacy-selected EMA pairs used to define the economic feature grid.
- Direct Legacy labels restricted to months actually present in the Legacy
  output.
- Fixed hypothesis matrix declared before reading results.
- 24/36-month conclusions downgraded for low effective sample size.
- Snapshot lineage caveat retained: this is a coherent research replay, not a
  claim that the underlying open-source package is production-clean.
