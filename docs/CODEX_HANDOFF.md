# Codex Handoff

Last updated: 2026-03-14
Branch at write time: `update_probalisor`

This file is the practical handoff for a new Codex session on this repository. It summarizes the active architecture, the decisions already made with the user, the sensitive parts of the codebase, and the recent history that matters for continuation.

## 1. Current priorities

The repository has two real working tracks:

1. `scripts/run_legacy.py`
2. `scripts/run_backtest.py`

The user currently cares about both, but with different intent:

### Legacy

- goal: speed up the old pipeline
- direction chosen: migrate the runtime path to `polars` as much as possible
- constraint: keep `pandas` only when absolutely necessary, mainly visualization/report rendering
- key user expectation: no silent regression, parity and auditability matter more than elegance

### Backtest / boosting

- goal: learn stock outperformance versus benchmark, not absolute return
- benchmark logic now matters explicitly in the target
- the user wants auditability first:
  - fold-by-fold visibility
  - full SHAP visibility
  - clean debug exports
  - explicit timing semantics

## 2. Current architecture

- `scripts/run_legacy.py`: legacy pipeline entrypoint
- `scripts/run_backtest.py`: boosting / walk-forward backtest entrypoint
- `src/alpharank/backtest/`: modular backtest pipeline
- `src/alpharank/data/`: shared data transforms
- `src/alpharank/strategy/`: legacy strategy path
- `src/alpharank/visualization/`: reporting / plotting helpers
- `src/_old/`: archived code, not for new work

Canonical reference for backtest formulas and feature construction:

- [`docs/backtest_feature_reference.md`](./backtest_feature_reference.md)

Do not reconstruct feature formulas from memory when this document exists. Update it when behavior changes.

## 3. Legacy pipeline state

### Decisions already made

- `run_legacy` is intended to be `polars`-first.
- `pandas` should remain only at visualization/report boundaries if needed.
- user explicitly wants pandas removed from the critical path wherever possible.

### Important files

- `scripts/run_legacy.py`
- `src/alpharank/data/processing.py`
- `src/alpharank/utils/returns.py`
- `src/alpharank/features/indicators.py`
- `src/alpharank/strategy/legacy.py`
- `src/alpharank/visualization/plotting.py`

### Legacy-specific notes

- `StrategyLearner.fiting` was identified by the user as a major bottleneck and should stay under scrutiny for runtime.
- benchmark work for legacy exists under:
  - `benchmarks/legacy-benchmark-repo/`
- benchmark data/logs/results are intentionally ignored from git.

## 4. Backtest pipeline state

### 4.1 Target definition

The backtest is now designed to predict benchmark outperformance.

Current target logic:

- `future_return`: next-month stock return
- `benchmark_future_return`: next-month benchmark return
- `future_relative_return = (1 + future_return) / (1 + benchmark_future_return)`
- `future_excess_return = future_relative_return - 1`
- `target_label = future_excess_return > outperformance_threshold`

This replaced the old absolute-return target logic.

Main files:

- `src/alpharank/backtest/datasets.py`
- `src/alpharank/backtest/pipeline.py`
- `scripts/run_backtest.py`

### 4.2 Timing semantics

This was a major source of confusion and has been clarified.

- `decision_month`: month at which the decision is formed using information available at that point
- `holding_month`: next month, during which the simulated position is held

Interpretation:

- a row with `decision_month = 2010-05-01` means the model decides at end of May 2010
- the realized return that validates the decision is in June 2010

Important:

- exports/reports should prefer `decision_month` and `holding_month`
- do not rely on `year_month` alone if semantics matter

### 4.3 One-month horizon enforcement

A real issue existed: some targets were effectively using the next available observation, not necessarily the next calendar month.

This has been fixed:

- only strict one-month holding transitions are retained
- gaps larger than one month are excluded from target construction

If results ever look suspiciously optimistic again, re-check this first.

### 4.4 Fundamental feature policy

The user does not want raw absolute dollar-level accounting values in the model.

Do not use features like:

- raw `net_income_ttm`
- raw `total_revenue_ttm`
- raw `ebitda_ttm`
- raw `free_cashflow_ttm`
- raw `market_cap`
- raw `enterprise_value`

The model should use:

- ratios
- growth rates
- relative quantities

Current backtest feature policy:

- technical features should come from explicit indicator families, not ad hoc lags
- the modern path is configured through:
  - `TechnicalFeatureConfig`
  - `FundamentalFeatureConfig`
- the active `scripts/run_backtest.py` preset emphasizes:
  - ROC windows
  - EMA ratios
  - price-to-EMA distances
  - RSI levels and RSI ratios
  - Bollinger relative position / bandwidth
  - stochastic oscillator
  - range location
  - volatility levels and volatility ratios
- fundamental features should remain ratio-first:
  - margins
  - returns on capital / assets
  - balance-sheet structure ratios
  - inverted valuation multiples / yields
  - dilution
  - TTM growth for revenue / earnings / EBITDA / EBIT / gross profit / FCF / EPS
- do not reintroduce raw size proxies or dollar-level statement features into the model
- preserve the monthly `join_asof(..., strategy="backward")` rule to avoid lookahead bias

Main file:

- `src/alpharank/backtest/fundamentals.py`

### 4.5 SHAP reporting policy

The user wants exhaustive SHAP visibility, fold by fold.

Current expectation for the PDF:

- fold 1 full block, then fold 2 full block, etc.
- for each fold:
  - beeswarm
  - second-order SHAP matrix / heatmap with diagonal kept
  - all 1D dependence plots sorted by decreasing mean `|SHAP|`, with color driven by interaction feature
  - top interaction dependence plots only, ranked by mean `|interaction SHAP|`

Current parameterization:

- `shap_top_features`: controls the breadth of the fold/global SHAP views
- `shap_top_interactions`: controls how many top interaction pair plots are rendered per fold
- default `shap_top_interactions = 5`

Main file:

- `src/alpharank/backtest/explainability.py`

### 4.6 Notebook-first orchestration

`scripts/run_backtest.py` should now be treated as a notebook orchestration helper, not just a terminal entrypoint.

Preferred workflow:

1. phase 1 learning only
2. inspect predictions / fold KPIs
3. phase 2 backtest from the learning artifacts you decided to keep

The script exposes these helpers:

- `default_config(**overrides)`: reproducible config factory for notebook use
- `run_learning(config=None)`: runs only phase 1 and persists intermediate outputs under the run directory
- `load_learning(run_dir)`: reloads the persisted phase-1 artifacts
- `learning_kpis(...)`: compact fold-level modeling KPI view
- `list_folds(...)`: fold windows / skip reasons / row counts
- `load_fold_predictions(run_dir, fold)`: reload a specific fold scoring table
- `run_backtest(config=None, learning=..., run_dir=...)`: runs only phase 2 from an in-memory or reloaded learning run
- `backtest_fold_kpis(...)`: compact fold-level trading KPI view
- `load_fold_monthly_returns(run_dir, fold)`: reload per-fold portfolio returns after phase 2

Important:

- `run_learning(...)` writes top-level intermediate files immediately:
  - `model_frame.parquet`
  - `predictions.parquet`
  - `fold_metrics.parquet`
  - `fold_index.parquet`
  - `best_params.parquet`
  - `learning_metadata.json`
- each fold still keeps its own folder with:
  - `fold_##/predictions.parquet`
  - `fold_##/optuna_trials.csv`
  - `fold_##/best_params.json`
  - SHAP / Optuna / calibration assets

This separation exists because the user wants explicit control over:

- the classification KPIs used to judge the model
- the prediction tables inspected before any trading backtest
- the exact transition from "predict outperformance" to "simulate the strategy"

### 4.7 Backtest audit exports

The user wanted a clean table to inspect what happened line by line.

Current exported debug artifacts include:

- `fold_index.parquet`
- `debug_predictions_long.parquet`
- `debug_predictions_full.parquet`

Purpose:

- `debug_predictions_long`: only rows actually scored by a fold
- `debug_predictions_full`: full model frame plus scoring columns when available
- `fold_index`: fold metadata, split sizes, skip reasons, positive rates

Main files:

- `src/alpharank/backtest/pipeline.py`
- `src/alpharank/backtest/reporting.py`

### 4.8 Dedicated audit report

A separate HTML audit report exists for deep backtest inspection.

Expected content includes:

- portfolio vs benchmark over time
- active return
- prediction vs realized excess return scatter
- purchased names
- monthly selections
- best/worst periods
- best/worst positions
- folds summary

Main file:

- `src/alpharank/backtest/reporting.py`

## 5. Data source caveat

There was confusion between:

- root `data/*.parquet`
- nested `data/US/*.parquet`

At one point, `run_backtest.py` was reading older data from `data/US/` while `run_legacy.py` used fresher root-level files. This explained why one path seemed to stop earlier in time than the other.

When debugging date coverage, always verify:

1. which loader is used
2. which path wins
3. max available date in the actual file consumed

Relevant file:

- `src/alpharank/backtest/data_loading.py`

## 6. Git and repo hygiene

### 6.1 History rewrite already happened

The repo history was rewritten to remove oversized tracked data blobs that blocked GitHub pushes.

Removed from history:

- large files under `data/`
- `.env`
- `.DS_Store`
- `experiments/optuna_report.html`

This means:

- commit hashes before the rewrite are obsolete
- if another clone exists elsewhere, it may need a clean resync

### 6.2 Current `.gitignore` policy

The repo now ignores:

- `outputs/`
- `debug/`
- parquet/csv/feather/arrow/ipc/h5/hdf5/pickle artifacts
- dataset snapshots under `data/**`
- benchmark artifacts under `benchmarks/**/data`, `logs`, `results`

Tracked under `data/` should remain code only, e.g.:

- `data/US/df_data.py`

## 7. Testing and environment

### Environment

`python3` on the host may not have `pytest` or project deps. Prefer the repo virtualenv when validating:

- `.venv/bin/python`

### Typical test commands

```bash
.venv/bin/python -m pytest -q tests
```

or targeted:

```bash
.venv/bin/python -m pytest -q tests/test_backtest_features.py tests/test_backtest_fundamentals.py
```

## 8. Recent commits worth reading

Recent useful history on `update_probalisor` after the history rewrite:

- `c773539` `chore: stop tracking local data artifacts`
- `79e8d76` `feat: export exhaustive 2d shap interactions by fold`
- `0c051cf` `feat: keep only ratio and growth fundamental features`
- `20d2a61` `docs: add backtest feature and formula reference`
- `cc46da3` `fix: align lift curves with ranked bucket calibration`
- `147c04e` `feat: add per-fold validation and test lift curves`
- `e488a02` `fix: clarify backtest timing and enforce 1m holding horizon`
- `d08b82f` `feat: add dedicated backtest audit report`
- `8e96763` `feat: export detailed backtest debug prediction tables`
- `8200acb` `feat: add exhaustive per-fold shap dependence plots`
- `098e807` `fix: restore retained optuna charts in training report`
- `5c9afe6` `fix: target benchmark outperformance in backtest`

## 9. Working rules that matter with this user

- Prefer small targeted commits.
- Commit regularly.
- Do not hide regressions behind refactors.
- If performance does not improve, explain why concretely.
- If a result is optimistic, provide audit surfaces instead of hand-waving.
- The user values directness over polish.

### 4.9 Reload caveat

Reloading a learning run from disk with `load_learning(run_dir)` is enough to:

- inspect fold predictions
- inspect modeling KPIs
- rerun the portfolio backtest from saved predictions

But it does not reload in-memory SHAP explanation objects. Consequence:

- fold-level SHAP assets already written on disk remain available
- the consolidated global SHAP PDF is only guaranteed when phase 2 is run from the original in-memory `LearningArtifacts`

## 10. Current local state at handoff time

At the time this file was written:

- branch: `update_probalisor`
- upstream: `origin/update_probalisor`
- the tree may or may not be dirty depending on the exact checkpoint; always read `git status` first
- the most recent backtest-oriented changes to understand before continuing are in:
  - `scripts/run_backtest.py`
  - `src/alpharank/backtest/pipeline.py`
  - `docs/backtest_feature_reference.md`
  - targeted tests under `tests/test_backtest_*.py`

Do not overwrite local state casually. Read the working tree first if continuing from this exact checkout.
