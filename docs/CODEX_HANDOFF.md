# Codex Handoff

Last updated: 2026-06-27
Branch at write time: `data-backfill-fixes`

This file is the practical handoff for a new Codex session on this repository. It summarizes the active architecture, the decisions already made with the user, the sensitive parts of the codebase, and the recent history that matters for continuation.

## 0. Documentation Map

This file is the central cross-track note. Do not create scattered experiment
notes when one of these canonical documents already fits:

- Boosting / Legacy-copy R&D:
  [`docs/boosting_signal_copy_model_catalog.md`](./boosting_signal_copy_model_catalog.md)
- Monthly portfolio production:
  [`docs/monthly_portfolio_runbook.md`](./monthly_portfolio_runbook.md)
- SEC/open-source data status:
  [`docs/sec_open_source_status.md`](./sec_open_source_status.md)
- SEC data robustness and replay incident:
  [`docs/sec_data_robustness_plan.md`](./sec_data_robustness_plan.md)
- SEC/open-source audit appendices:
  [`docs/audit_donnees_financieres_2025.md`](./audit_donnees_financieres_2025.md),
  [`docs/rapport_couverture_sp500_revenue_netincome.md`](./rapport_couverture_sp500_revenue_netincome.md),
  [`docs/rapport_trous_ticker_par_ticker.md`](./rapport_trous_ticker_par_ticker.md)
- Backtest feature formulas:
  [`docs/backtest_feature_reference.md`](./backtest_feature_reference.md)
- Open-source ingestion architecture:
  [`docs/open_source_ingestion_architecture.md`](./open_source_ingestion_architecture.md)

When work changes a method, run procedure, data lineage rule, or R&D conclusion,
update the relevant canonical doc in the same commit. New notes should be rare
and linked here if they become durable references.

## 0.1 June 2026 Summary

### Boosting / Legacy-copy R&D

Goal clarified with the user:

```text
Find model selections that recover Legacy trades, while learning from future
excess return where possible.
```

The primary metric is now:

```text
number of stocks common between model and Legacy for the month
/
number of stocks chosen by Legacy for that month
```

Important methods tested:

- CPCV / walk-forward XGBoost via the local `mlcraft` library.
- Direct classifiers on `future_excess_return > 0%` and `> 5%`.
- Direct regression on clipped future excess return.
- Ranking objectives.
- Residual/init-score models using a base model then residual boosting.
- EMA-rich feature sets.
- Seven signal-copy variants, including `distill_legacy`, pairwise ranking,
  regression, monotone EMA constraints, two-stage EMA/full model, EMA-gated full
  model, and weighted classifiers.
- Atomic Legacy decomposition of the Optuna/EMA blocks.
- Generalized EMA experts: parametric EMA rules selected by trailing future
  excess return, without using the final Legacy selection as a feature.

Current conclusion:

- Direct future-return models with the original feature set did not recover
  Legacy well enough, especially once Optuna stopped optimizing on Legacy.
- Atomic Legacy features prove the representation problem: once the exact
  atomic Legacy signals are visible, recovery can exceed 98%, but that is not a
  generalizable final model.
- 2026-06-27 diagnostic found that simple tradable EMA rank scores recover more
  than 50% of Legacy on 2015+, but this is only an explanatory baseline, not a
  solution. The user explicitly corrected the goal: do not copy Legacy; estimate
  future relative return per stock, then build a controlled portfolio.
- Calibration run `outputs/return_forecast_calibration_20260627_161954` shows
  that the clean EMA boosting regressions still do not quantify average future
  relative return well enough: top deciles are not robustly better than bottom
  deciles and top-K lift vs universe is weak.
- First allocation candidate that beats Legacy on important metrics:
  `outputs/portfolio_boosting_blend_backtest_20260627_171645`.
  It uses an mlcraft XGBoost classifier predicting top-10% future excess return,
  then a score blend `rank(prediction_boosting) + 0.10 * technical_z_mean`.
  The `60% hybrid top5 / 40% SPY` curve beats `Combined_Frequency` in total
  return, CAGR, and Sharpe (`+1,184%`, `25.5%`, `0.95` vs `+1,118%`, `24.9%`,
  `0.93`) but still has worse max drawdown (`-28.5%` vs `-23.5%`).
- New direction: reduce drawdown on this boosting-based candidate through a
  dynamic risk overlay or portfolio constraints. Treat Legacy recomposition as
  secondary diagnostics only.

Detailed source of truth:
[`docs/boosting_signal_copy_model_catalog.md`](./boosting_signal_copy_model_catalog.md).

### Monthly Portfolio / Replayability

The monthly `ptf du mois` remains the Legacy workflow, not the boosting R&D
workflow.

Current production rule:

- run through `scripts/run_legacy.py`;
- compute from a timestamped `input_snapshot/`, not from a mutable data folder;
- keep `data_input_manifest.json`, file hashes, source run ids, code hashes, and
  legacy parquet outputs;
- prefer `--open-source-run-id` for point-in-time replays.

Detailed source of truth:
[`docs/monthly_portfolio_runbook.md`](./monthly_portfolio_runbook.md).

### SEC / Open-Source Data

The SEC-only fundamentals track was separated from the broader mixed
open-source package. The active SEC-quality work focuses on reducing missing
coverage for `epsActual`, `revenue`, and `net_income`, while keeping lineage
visible.

Current notable result:

- the metric-level hybrid package combines the best observed EPS path with the
  better `revenue` / `net_income` path;
- latest documented candidate: `outputs/sec_kpi_hybrid_output_latest/`;
- worst-year missingness documented at: `epsActual 0.86%`, `revenue 1.90%`,
  `net_income 1.90%`.

Detailed sources of truth:

- [`docs/sec_open_source_status.md`](./sec_open_source_status.md)
- [`docs/sec_data_robustness_plan.md`](./sec_data_robustness_plan.md)

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
- `run_legacy.py` now writes reusable parquet audit artifacts in addition to HTML:
  - `legacy_comparison_curves_polars.parquet`
  - `legacy_aggregated_returns_polars.parquet`
  - `legacy_detailed_returns_polars.parquet`
  - `legacy_monthly_returns_polars.parquet`
  - `legacy_cumulative_returns_polars.parquet`
  - `legacy_drawdowns_polars.parquet`
  - `legacy_annual_returns_polars.parquet`
  - `legacy_metrics_polars.parquet`
- the same family is also checkpointed under `outputs/checkpoints/` for reload without rerunning the whole legacy pipeline

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

- prefer `main(...)` parameters and notebook helpers over `argparse` for the user-facing workflow scripts
- the user does not want CLI flag orchestration for the main flows; if a script is meant to be edited and launched locally, keep the parameters explicit in code
- `run_learning(...)` writes top-level intermediate files immediately:
  - `model_frame.parquet`
  - `predictions.parquet`
  - `fold_metrics.parquet`
  - `fold_index.parquet`
  - `best_params.parquet`
  - `data_input_manifest.json`
  - `learning_metadata.json`
- each fold still keeps its own folder with:
  - `fold_##/predictions.parquet`
  - `fold_##/optuna_trials.csv`
  - `fold_##/best_params.json`
  - SHAP / Optuna / calibration assets

Prediction/data lineage is now explicit at the run-folder level:

- each `xgboost_timefold_backtest_YYYYMMDD_HHMMSS/` directory should be treated as a prediction snapshot
- `data_input_manifest.json` records the exact input files used for that run, including canonical paths, hashes, timestamps, and frame summaries
- when a data snapshot exists under `data/latest_snapshot.json`, the run manifest also stores:
  - `source_snapshot_id`
  - `source_snapshot_manifest_path`
  - `source_snapshot_dir`
  - `source_snapshot_match` (`full_match`, `partial_match`, or `no_path_match`)
- `learning_metadata.json` and final `metadata.json` both expose a compact `data_lineage` block so you can tell immediately which data snapshot produced the predictions
- notebook/helper access point: `load_data_input_manifest(run_dir)` in `scripts/run_backtest.py`

This separation exists because the user wants explicit control over:

- the classification KPIs used to judge the model
- the prediction tables inspected before any trading backtest
- the exact transition from "predict outperformance" to "simulate the strategy"

### 4.7 Dedicated application-backtest layer

There is now a distinct application/backtest layer for testing trading rules without changing the learning pipeline itself.

Main files:

- `src/alpharank/backtest/application.py`
- `scripts/run_backtest_application.py`

Intended use:

- load predictions already produced by the boosting pipeline
- apply a dedicated trading rule (`top_n` or `prediction > threshold`)
- optionally filter names whose last available decision-month price is older than `x` months
- compare one or several resulting backtest curves, including legacy curves, with the legacy comparison report machinery
- `scripts/run_backtest_application.py` is meant to be edited via its top-level `SCENARIO_SPECS` block and then run directly, not driven through CLI flags

Do not fold these application experiments back into the learning config. The user wants them decoupled on purpose.

### 4.8 Backtest audit exports

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

### 4.9 Dedicated audit report

A separate HTML backtest report exists for deep backtest inspection, distinct from the learning/training report.

Expected content includes:

- global backtest KPIs
- per-fold test-period KPIs (`Portfolio` / `Benchmark` / `Active`)
- fold-by-fold test-period tables
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
- The user does not want `argparse` in the main workflow scripts; prefer editable `main(...)` arguments.

### 4.10 Reload caveat

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
