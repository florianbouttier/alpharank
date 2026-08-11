# AlphaRank Agent Instructions

These instructions are mandatory for future agent sessions in this repository.
Keep this file current whenever the monthly workflow, data lineage, or run
procedure changes.

## Autonomy

- Run commands proactively when they help complete the task.
- Prefer executing checks, tests, linters, and local scripts instead of just suggesting them.
- Do not ask for approval unless blocked by the runtime.
- When the user delegates an R&D track and cannot be present continuously, keep moving autonomously: inspect the code/data, choose the next defensible experiment, run it, document the result, and continue to the next bottleneck.
- Favor transparent execution over waiting: state the current assumption, make the smallest reversible implementation that tests it, and record why the choice was made.
- If a result invalidates the current path, pivot to the next highest-signal diagnostic instead of stopping at a failed experiment.

## Git Safety

- Never run `git push`.
- Never run `git push --force`.
- Avoid deleting files unless deletion is clearly part of the requested change.
- Prefer deleting code inside tracked files over deleting whole files.
- Before deleting a tracked file, create a commit that preserves the prior state.
- Do not rewrite git history unless explicitly requested.
- Do not create a git commit unless the user explicitly asks for one.
- If the user explicitly asks for committed autonomous work on a multi-step track, treat that as permission to create focused commits for completed, verified chunks until the user changes direction.

## File Safety

- Be conservative with non-code deletions.
- Do not delete user files, data files, env files, secrets, or docs unless explicitly requested.
- If a file seems risky to remove, keep it and explain why.

## Working Style

- Prefer minimal diffs.
- Prefer root-cause fixes over superficial patches.
- Run the narrowest relevant tests after changes.
- Summarize what changed, what was run, and remaining risks.
- When a task exposes stale or ambiguous process documentation, update the relevant docs in the same task.
- Keep documentation centralized. Do not create scattered one-off notes when a canonical note exists.
- `docs/CODEX_HANDOFF.md` is the central cross-track handoff and must point to the specialized docs that matter.
- For boosting/Legacy-copy R&D, update `docs/boosting_signal_copy_model_catalog.md`; do not create parallel experiment notes unless they are linked from the handoff and intentionally promoted.
- For Legacy/Boosting signal semantics, liquidity gates, feature usage, full
  pseudocode, or code ownership, update `docs/legacy_boosting_methodology.md`.
- For SEC/open-source data status, update `docs/sec_open_source_status.md` and `docs/sec_data_robustness_plan.md` as applicable.
- For monthly portfolio production, update `docs/monthly_portfolio_runbook.md`.
- For portfolio simulation, performance metrics, or cross-method comparison,
  update `docs/common_portfolio_backtest_engine.md` and use
  `src/alpharank/portfolio/`; do not add another local CAGR, Sharpe, drawdown,
  turnover, or annual-return implementation.
- A monthly SHAP view must state its row count and sampling status. Never
  present a fold-level SHAP sample as exhaustive monthly coverage; exhaustive
  month views must match the test-prediction count for that month.
- Keep model-target maturity and portfolio-return maturity as separate
  calendars. H6 quality metrics may use only rows with an evaluable H6 target;
  a causal score-only tail may extend the one-month portfolio replay through
  the last decision month with a complete t+1 return. Reports must distinguish
  `evaluable`, `ticker_target_unavailable`, and `horizon_pending` rows.
- Every performance answer must identify the run/snapshot and benchmark
  convention. Standalone Legacy defaults to the latest validated replay and
  `SPY total return` from `adjusted_close`; the historical `SP500` model in
  Legacy artifacts is price return from `close` and is not a standard
  performance benchmark. Cross-method comparisons require the same snapshot.
- CAGR attribution must use `src/alpharank/portfolio/attribution.py`: ordinary
  percentage-point CAGR contributions are not additive. Report annualized log
  contributions, isolate transaction costs, show the compound-effect bridge,
  and require monthly plus final CAGR reconciliation within `1e-12`.
- Every delegated research update should record, in the relevant central doc, the hypothesis, data used, command/run id, primary metric, result, and next decision.
- Prefer project-visible memory in `AGENTS.md`, `AGENT.md`, and relevant docs over relying on chat history.

## Monthly Portfolio Workflow

- The canonical monthly `ptf du mois` workflow is the legacy pipeline in `scripts/run_legacy.py`.
- Do not use the XGBoost/backtest R&D workflow for monthly production unless explicitly requested.
- The canonical runbook is `docs/monthly_portfolio_runbook.md`; read it before running or replaying a monthly portfolio.
- Current open-source production data is `data/open_source/output`.
- Launch monthly runs through `./.venv/bin/python scripts/run_legacy.py ...` so the CLI captures a timestamped log.
- For point-in-time replay, prefer `open_source_run_id` over the mutable `data/open_source/output` path.
- Every monthly run must leave a durable audit trail:
  - `outputs/YYYY-MM-DD/latest_legacy_run.json`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/legacy_detailed_returns_polars.parquet`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/portfolio_report_*_<YYYY-MM>.html`
  - a log under `logs/legacy_runs/` when the command is launched manually
- The legacy runner must compute from the timestamped run `input_snapshot/`; `data/open_source/output` is only a source to copy from.
- A manifest without `input_snapshot_dir`, `run_config.source_input_sha256`, and `code_context.critical_file_sha256` is not a complete replay package.
- For open-source production data, a manifest with `open_source_run_id_match=false`, `open_source_output_manifest_run_id_match=false`, or `open_source_output_matches_published_snapshot=false` is not a clean monthly package and must not be used as production truth without data-package investigation.
- If rerunning a historical month with newer data changes the portfolio, treat it as data revision/look-ahead risk until proven otherwise.
- Legacy and boosting may generate different signals, but finalized holdings
  must pass through the shared portfolio contract for new backtests and reports.
  Any Legacy/boosting performance comparison must also prove matching input
  data hashes and full-trajectory ticker exclusions through
  `src/alpharank/portfolio/lineage.py`. A shared simulator
  replay on distinct snapshots is mechanical validation only and must be marked
  `comparison_eligible=false`; it is not evidence that the strategies are
  comparable.
  Public latest-common Boosting comparisons must use
  `--latest-common-comparison-profile`; the common replay builder validates the
  versioned profile and fails closed on configuration drift. Run
  `scripts/audit_open_source_snapshot_revisions.py` between successive clean
  snapshots and publish its downstream Legacy/Boosting impact with the report.
  A change to Legacy aggregation is not production-safe unless
  `scripts/validate_common_portfolio_engine.py` reproduces the frozen Legacy
  and Alpha references within `1e-12`.

## Open-Source Data Lineage

- Open-source replacement work should preserve lineage fields for every consolidated value.
- Published open-source outputs must be historized under `data/open_source/history/output/` after the final package is written, so the retained snapshot contains the exact legacy files, lineage files, and manifest for the published run.
- The active package lineage entrypoint is `data/open_source/output/lineage/manifest.json`.
- The ingestion run entrypoint is `data/open_source/official/manifests/latest_run.json`.
- Full ingestion, current-constituent refresh, and ticker restore must share
  `data/open_source/official/locks/nightly.lock.json`; never run publishers
  concurrently.
- A clean monthly open-source run must expose matching `open_source_output_run_id`, `open_source_output_lineage_run_id`, `open_source_output_snapshot_run_id`, and `open_source_ingestion_run_id` in the legacy run manifest, plus `open_source_output_matches_published_snapshot=true` when an ingestion manifest advertises a published output snapshot.

## Additional Project Context

`AGENT.md` contains the broader AlphaRank guide and legacy project notes. Keep it
in sync with this file when agent-facing process rules change.
