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

- Do not run `git push` unless the user explicitly asks for it. An explicit
  request authorizes a normal push of the reviewed current branch after tests
  pass and the staged scope has been summarized.
- Never run `git push --force` unless the user explicitly requests that exact
  operation and the remote branch/history impact has been reviewed first.
- Avoid deleting files unless deletion is clearly part of the requested change.
- Prefer deleting code inside tracked files over deleting whole files.
- Before deleting a tracked file, create a commit that preserves the prior state.
- Do not rewrite git history unless explicitly requested.
- Do not create a git commit unless the user explicitly asks for one.
- If the user explicitly asks for committed autonomous work on a multi-step track, treat that as permission to create focused commits for completed, verified chunks until the user changes direction.
- Never stage generated snapshots, raw data, secrets, caches, or large output
  artifacts merely because a code push was requested. Commit only reviewed
  source, configuration, tests, and documentation unless the user explicitly
  requests a versioned data artifact.

## File Safety

- Be conservative with non-code deletions.
- Do not delete user files, data files, env files, secrets, or docs unless explicitly requested.
- If a file seems risky to remove, keep it and explain why.

## Working Style

- Prefer minimal diffs.
- Prefer root-cause fixes over superficial patches.
- Run the narrowest relevant tests after changes.
- Summarize what changed, what was run, and remaining risks.
- Before committing or pushing a production-data, Legacy, or Boosting change,
  run the verification suite documented in README section `Production Data
  Contract And Anti-Leakage Controls` and the strict Legacy replay validator
  when a new monthly run is part of the change. Never describe a change as
  leakage-safe from unit tests alone: report the snapshot, completed-month
  cutoff, filing-date policy, universe policy, and remaining coverage risks.
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
- Cross-method performance comparisons must resimulate every investable
  strategy with the same transaction-cost policy; a standalone production
  convention may be shown separately but cannot be mixed into the comparison.
- The permanent performance KPI contract is mandatory in user handoffs and
  public reports. Always show the run/snapshot/composition id, code commit,
  methodology status, requested and effective period, decision/holding-month
  convention, benchmark, transaction-cost policy and gross-versus-net basis.
  For every investable strategy and SPY, report total return, CAGR, annualized
  volatility, Legacy-convention Sharpe, Sortino, Calmar, maximum drawdown with
  peak/trough/recovery dates and duration, positive-month rate, best and worst
  realized month, worst complete calendar year, all calendar-year returns and
  CAGR from January 1 of every year from 2010 through the current year. Against
  SPY also report annualized excess return, alpha, beta, correlation,
  information ratio, tracking error, benchmark hit rate and up/down capture.
  Report VaR 95%, CVaR 95%, skewness and excess kurtosis; turnover, transaction
  costs, average/maximum position count, average/maximum single-name weight,
  concentration/HHI, cash exposure, return-coverage or missing-return counts,
  and logged terminal-event contribution. Keep model-quality metrics separate:
  OOS rows/folds, IC/Spearman, NDCG@5/10/20, ROC AUC, PR AUC and lift, Brier,
  log-loss, calibration error and SHAP row count/sampling status when relevant.
  If a KPI is unavailable, print `unavailable` with the reason; never silently
  omit it or substitute a different convention.
- Every delegated research update should record, in the relevant central doc, the hypothesis, data used, command/run id, primary metric, result, and next decision.
- Prefer project-visible memory in `AGENTS.md`, `AGENT.md`, and relevant docs over relying on chat history.

## Monthly Portfolio Workflow

- The canonical monthly `ptf du mois` workflow is the legacy pipeline in `scripts/run_legacy.py`.
- Do not use the XGBoost/backtest R&D workflow for monthly production unless explicitly requested.
- The canonical runbook is `docs/monthly_portfolio_runbook.md`; read it before running or replaying a monthly portfolio.
- `data/open_source/output` remains a mixed-source research/replay package and
  is not a valid new production input. Build production inputs with
  `scripts/open_source/build_composed_model_snapshot.py`, which combines a
  guarded EODHD/open price package with a strict SEC-only package and preserves
  both lineages and hashes in one immutable folder. Canonical composed history
  is `data/model_inputs/history/`; resolve the current package only through
  `data/model_inputs/manifests/latest.json`.
- Official model fundamentals are SEC/GAAP only: `sec_companyfacts`,
  filing-level SEC XBRL, and explicitly labelled values derived only from SEC
  facts. Never use EODHD, Yahoo, SimFin, or StockAnalysis fundamental values in
  an official monthly snapshot. EODHD may provide a ticker-to-CIK bridge but
  never a final fundamental value.
- Raw SEC Companyfacts uniqueness includes `filing_date`; never collapse later
  restatements onto an earlier filing version. Model exports select the earliest
  filing version available for each ticker/statement/metric/period/source while
  raw storage retains every version.
- Routine price refreshes roll forward the last validated price lineage: keep
  inactive histories byte-stable and replace refreshable active tickers with
  one full Yahoo vintage. A carried-forward terminal ticker requires a sourced
  removal event in the constituent registry and must be listed in the manifest.
- The durable price seed is the complete preceding validated published lineage,
  not EODHD alone. A ticker first acquired from Yahoo must remain byte-stable
  after it leaves the active universe or becomes unavailable upstream, even if
  it never existed in EODHD. Resolve the preceding lineage from
  `data/model_inputs/manifests/latest.json` and retain
  `persistent_price_history_registry.parquet`; routine deletion of any prior
  inactive ticker/date is forbidden.
- Launch monthly runs through `./.venv/bin/python scripts/run_legacy.py ...` so the CLI captures a timestamped log.
- For current production, pass the immutable composed `snapshot_dir`. Keep
  `open_source_run_id` only for explicit replays of the older open-source
  package contract.
- Every monthly run must leave a durable audit trail:
  - `outputs/YYYY-MM-DD/latest_legacy_run.json`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/legacy_detailed_returns_polars.parquet`
  - `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/portfolio_report_*_<YYYY-MM>.html`
  - a log under `logs/legacy_runs/` when the command is launched manually
- If the newest observed prices belong to the still-open calendar month,
  retain them in the raw snapshot for audit but exclude that whole month before
  Legacy feature construction. Record the cutoff as
  `run_config.decision_data_completed_through_month`. Boosting may score that
  completed decision month, but every label ending later must remain null.
- The legacy runner must compute from the timestamped run `input_snapshot/`; `data/open_source/output` is only a source to copy from.
- Snapshot storage should use copy-on-write clones where the filesystem supports them, with a physical-copy fallback and `input_snapshot/storage_manifest.json`; never replace immutable snapshot semantics with symlinks.
- A manifest without `input_snapshot_dir`, `run_config.source_input_sha256`, and `code_context.critical_file_sha256` is not a complete replay package.
- For open-source production data, a manifest with `open_source_run_id_match=false`, `open_source_output_manifest_run_id_match=false`, or `open_source_output_matches_published_snapshot=false` is not a clean monthly package and must not be used as production truth without data-package investigation.
- If rerunning a historical month with newer data changes the portfolio, treat it as data revision/look-ahead risk until proven otherwise.
- Legacy and boosting may generate different signals, but finalized holdings
  must pass through the shared portfolio contract for new backtests and reports.
  Any Legacy/boosting performance comparison must also prove matching input
  data hashes and full-trajectory ticker exclusions through
  `src/alpharank/portfolio/lineage.py`, plus the same shared monthly price
  eligibility policy from `src/alpharank/data/price_eligibility.py`. A shared simulator
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
- The stock-price contract is hybrid even though fundamentals are open source:
  `data/eodhd/output/US_Finalprice.parquet` is the immutable paid historical
  seed for former constituents/delisted names, and open providers extend or
  refresh downloadable names. Never infer full historical coverage from Yahoo
  or from active-universe freshness. The active output still lacks 110
  normalized EODHD-only historical tickers / 419,656 rows and contains no EODHD
  source. The deterministic reviewed candidate is
  `outputs/hybrid_price_candidate_20260815_final/`; do not call the active package
  complete until that candidate is promoted with SEC-only fundamentals.
- A price ingestion change must preserve frozen EODHD-only rows, record
  `eodhd_frozen_history` lineage, normalize ticker aliases explicitly, and audit
  the EODHD-to-open transition plus split continuity. "Retained inactive" means
  only already-seeded rows and is not evidence that the full EODHD archive was
  seeded.
- Full ingestion must use `src/alpharank/data/prices/`: one complete Yahoo
  vintage for all active tickers, the immutable EODHD seed for inactive
  history, and same-vintage daily returns from immutable run deltas for recent
  inactive continuations. Reject tails after a gap over 10 calendar days as
  possible symbol reuse.
- Price publication must retain `price_composition.json`,
  `price_revision_guard.json`, daily-return revisions, transition findings, and
  historical key removals. Routine runs must keep both historical-price
  overrides false; migration review records return revisions and key removals
  separately.
- The frozen EODHD file is immutable source evidence, not an assertion that
  every historical adjustment is forever correct. Newly verified corporate
  actions must be applied through a versioned correction overlay: splits adjust
  pre-event OHLC inversely and volume directly; dividends leave raw OHLCV
  untouched and recompute only adjusted/total-return values. Record event date,
  ratio or cash amount, evidence source, retrieval time, affected keys, and
  before/after hashes. Never rewrite the frozen EODHD archive or an immutable
  published snapshot in place.
- The mixed-source financial consolidation under `data/open_source/output` may
  remain useful for R&D and discrepancy analysis, but it is not an official
  fundamental source. As verified on 2026-08-14, retained T1
  `open_source_output_20260811_014746` selected 19,243 non-SEC consolidated
  fundamental values (14,754 Yahoo and 4,489 SimFin); monthly Legacy runs from
  that package are replay evidence, not SEC-only production truth.
- A new production snapshot must come from a `full_ingestion` source-refresh
  contract. The production policy refetches the complete available Yahoo price
  history for the latest active universe and SPY, complete SEC companyfacts history, mutable SEC submissions,
  StockAnalysis history, and SimFin bulk files before rebuilding targets.
  Recent filing-level fallback years may remain bounded because accession-level
  XBRL documents are immutable; they are fetched on demand only for active
  issuers with no recognized companyfacts row for the year and are not retained
  as a persistent payload cache.
- `_cache/` is disposable transport state, not a replay source. Never use cache
  timestamps as data freshness evidence. Read `source_refresh_contract` and
  `data_freshness` in the published manifest instead. Distinguish fiscal period
  end from SEC filing date when reporting freshness.
- Full ingestion must execute the historical revision guard before publication,
  retain `historical_revision_guard.json`, and fail on changes older than 730
  days unless a human-reviewed explicit override is recorded. A missing report
  makes the run non-production even if download and publication completed.
- Never use an output snapshot or normalized raw/target store carrying a
  quarantine marker as production truth. Preserve it for audit and restore the
  last clean output by verified hashes.
- A snapshot whose `source_refresh_contract.snapshot_scope` is
  `price_history_repair` or `reference_refresh` is diagnostic and must not be
  used as monthly production truth. `validate_legacy_replay_package.py` rejects
  such a package when the scope is present.
- Published open-source outputs must be historized under `data/open_source/history/output/` after the final package is written, so the retained snapshot contains the exact legacy files, lineage files, and manifest for the published run.
- Output snapshots and monthly input snapshots must use byte-identical APFS
  copy-on-write clones where available. Use
  `scripts/open_source/compact_output_history.py` to deduplicate retained exact
  duplicates; never archive snapshots into a format that prevents direct replay.
- The active package lineage entrypoint is `data/open_source/output/lineage/manifest.json`.
- The ingestion run entrypoint is `data/open_source/official/manifests/latest_run.json`.
- Full ingestion, current-constituent refresh, and ticker restore must share
  `data/open_source/official/manifests/nightly.lock.json`; never run publishers
  concurrently.
- A clean monthly open-source run must expose matching `open_source_output_run_id`, `open_source_output_lineage_run_id`, `open_source_output_snapshot_run_id`, and `open_source_ingestion_run_id` in the legacy run manifest, plus `open_source_output_matches_published_snapshot=true` when an ingestion manifest advertises a published output snapshot.

## Additional Project Context

`AGENT.md` contains the broader AlphaRank guide and legacy project notes. Keep it
in sync with this file when agent-facing process rules change.
