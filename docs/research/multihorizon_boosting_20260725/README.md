# Multi-horizon boosting versus Legacy

Status: screening and shortlisted CPCV completed 2026-07-25.

This directory is the detailed audit trail requested for the multi-horizon
study. The durable conclusion is also recorded in
`docs/boosting_signal_copy_model_catalog.md`; this folder keeps the protocol,
commands, run identifiers, diagnostics, and interpretation together.

## Question

Which pure boosting formulation and forward horizon best:

1. forecasts each stock's future excess return versus the S&P 500;
2. constructs a useful top-N portfolio;
3. recovers the names selected by Legacy without using those names as an
   economic target;
4. produces probabilities and risk forecasts that can later rationalize
   allocation decisions?

The direct Legacy classifier is kept as a teacher/oracle diagnostic. It is not
an admissible final economic signal.

## What Legacy actually does

Legacy is deterministic conditional on a frozen data snapshot, code version,
Optuna trial count and seeds, but it is not a fixed hand-written EMA rule.

- The investable universe is point-in-time S&P 500 membership intersected with
  a fundamental filter: positive non-null P/E below 100 and non-null market cap.
- Prices are converted to stock/S&P 500 relative prices.
- For each candidate, Legacy computes daily exponential moving averages and
  ranks the short/long EMA ratio cross-sectionally at month end.
- Optuna searches `n_long` from 50 to 400, `n_short` from 1 to 100,
  `n_asset`, and a sector cap of one or two names. The selected portfolio also
  obeys the sector cap.
- The search is rerun on expanding history at year-end. Its objective is based
  on realized portfolio return relative to the index over a trailing 120-month
  selection window.
- Four paths are fitted: alpha 2 with seeds 42 and 41, then alpha 1 with seeds
  42 and 41.
- `Combined_Frequency` is the union of the four baskets. A stock's final weight
  is proportional to the number of paths that selected it and normalized within
  the month.

This explains why Legacy can look simple while being difficult to learn: the
observable rule is an ensemble of changing cross-sectional EMA rankings,
universe filters, sector constraints, yearly parameter selection and frequency
weighting.

## Files

- `protocol.md`: exact targets, folds, leakage controls, metrics and SHAP plan.
- `experiment_log.md`: commands, run IDs, results and next decisions.
- Runtime artifacts: `outputs/multihorizon_boosting/<run_id>/`.

## Source code

- `src/alpharank/multihorizon/`: research package.
- `scripts/experiments/run_multihorizon_boosting.py`: reproducible CLI.
- sibling `mlcraft`: generic grouped XGBoost ranking support.
