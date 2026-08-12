# AlphaRank Agent Guide

Codex sessions should read `AGENTS.md` first. This file keeps the broader
AlphaRank project context and must remain in sync when agent-facing run
instructions change.

Documentation rule: keep `docs/CODEX_HANDOFF.md` as the central cross-track
handoff. Specialized docs should be linked from that handoff instead of being
created as unconnected one-off notes.

Before changing or explaining either signal methodology, read
`docs/legacy_boosting_methodology.md`. It is the canonical pseudocode and code
architecture reference for Legacy and the public latest-common Boosting
profile, including its liquidity gate and fundamental-data usage.

This repository is organized around two active workflows:

1. `legacy` (current production baseline)
2. `boosting` (experimentation pipeline based on XGBoost + walk-forward backtests)

## Active Library Layout

- `src/alpharank/backtest/`: modular boosting implementation (data loading, features, CV folds, Optuna tuning, SHAP, reporting)
- `src/alpharank/multihorizon/`: current leakage-aware multi-horizon boosting, risk, SHAP, and live scoring implementation
- Monthly SHAP reporting must disclose row counts and sampling. A month labelled exhaustive must contain one explanation for every test prediction in that month.
- Model quality and portfolio performance have distinct maturity calendars. H6 metrics exclude immature targets, while a frozen out-of-sample model may score later decisions for a one-month replay when t+1 returns are complete; expose evaluable, ticker-unavailable, and horizon-pending counts separately.
- Performance answers must name the snapshot and benchmark convention. Use `SPY total return` from `adjusted_close`; old Legacy `SP500` rows are price returns from `close` and cannot be reported as the standard benchmark. Standalone Legacy uses the latest validated replay, while Legacy/Boosting comparisons require one shared snapshot.
- CAGR decompositions must use `src/alpharank/portfolio/attribution.py`, keep transaction costs separate, and reconcile monthly returns plus final CAGR within `1e-12`; do not add ordinary percentage-point CAGR contributions.
- Cross-method performance comparisons must resimulate every investable strategy with the same transaction-cost policy; show any standalone production convention separately.
- `src/alpharank/portfolio/`: methodology-neutral holdings contract, allocation primitives, simulator, metrics, comparison, and artifacts shared after signal generation
- `src/alpharank/data/`: shared data processing/services used by legacy and utilities
- `src/alpharank/strategy/`: legacy strategy implementation
- `src/alpharank/visualization/`: legacy visual/reporting helpers

Legacy and boosting signal generation must remain separate, but new portfolio
return, turnover, cost, benchmark-alignment, and performance code must use
`src/alpharank/portfolio/`. The canonical contract and parity gate are in
`docs/common_portfolio_backtest_engine.md`.

Every Legacy/boosting performance comparison must also pass the shared data
lineage gate in `src/alpharank/portfolio/lineage.py`, including identical raw
input hashes, full-trajectory ticker exclusions, and the shared monthly price
eligibility policy. Replaying each strategy
through the same simulator on different snapshots validates only the simulator;
it must be marked `comparison_eligible=false` and must not be presented as a
data-constant strategy comparison.
The public latest-common Boosting comparison must be launched with
`--latest-common-comparison-profile`; the common replay validates this
versioned profile and rejects configuration drift. Compare successive clean
snapshots with `scripts/audit_open_source_snapshot_revisions.py` and retain the
data plus downstream replay impact beside the common replay.

## Archived Code

Old modules not part of the active library were moved to:

- `src/_old/`

This archive is kept for reference only and should not be used for new development.

## Script Entry Points

- `scripts/run_legacy.py`: legacy pipeline
- `scripts/run_backtest.py`: boosting experimentation pipeline

Other scripts may exist for utility/debug tasks but are not core project entry points.

## Monthly Portfolio Production

The monthly `ptf du mois` is produced by `scripts/run_legacy.py`, not by the
XGBoost/backtest R&D workflow unless the user explicitly asks for that.

Before running, replaying, or explaining a monthly portfolio, read:

- `docs/monthly_portfolio_runbook.md`

Current production source:

- `data/open_source/output`

Launch monthly runs through `./.venv/bin/python scripts/run_legacy.py ...` so
the CLI captures a timestamped log under `logs/legacy_runs/`.

For point-in-time replay, prefer `open_source_run_id` over the mutable output
folder. Every monthly run must leave a durable audit trail through:

- `outputs/YYYY-MM-DD/latest_legacy_run.json`
- `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json`
- `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/`
- `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/legacy_detailed_returns_polars.parquet`
- `outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/portfolio_report_*_<YYYY-MM>.html`
- `logs/legacy_runs/` when launching the command manually

The legacy runner must compute from the timestamped run `input_snapshot/`.
`data/open_source/output` is only a source to copy from. A manifest without
`input_snapshot_dir`, `run_config.source_input_sha256`, and
`code_context.critical_file_sha256` is not a complete replay package.
On filesystems that support it, snapshots use byte-identical copy-on-write
clones, with a physical-copy fallback and `input_snapshot/storage_manifest.json`.
Symlinks are forbidden because they do not preserve immutable replay semantics.
For open-source production data, a manifest with `open_source_run_id_match=false`,
`open_source_output_manifest_run_id_match=false`, or
`open_source_output_matches_published_snapshot=false` is not a clean monthly
package and must not be used as production truth without data-package
investigation.

Full ingestion, current-constituent refresh, and ticker restore are all data
publishers and must acquire the shared
`data/open_source/official/locks/nightly.lock.json` before mutating state.

If a historical month is recalculated with a newer data package, treat any
portfolio drift as data revision/look-ahead risk until the manifest proves the
same point-in-time data snapshot was used.

## Backtest Feature Reference

For any work on `scripts/run_backtest.py` or `src/alpharank/backtest/*`, the feature/data construction source of truth is:

- `docs/backtest_feature_reference.md`

This file documents:

- raw input mapping
- timing semantics (`decision_month`, `holding_month`)
- target construction
- technical feature formulas
- fundamental feature formulas
- sparse-feature filtering and imputation
- portfolio aggregation logic

Do not infer these formulas from memory when the document is available; use the document as the canonical reference and update it when the code changes.

For cross-session continuity and recent repo history/decisions, also use:

- `docs/CODEX_HANDOFF.md`

Boosting workflow is modularized in the library with explicit phases:

- `run_learning_phase(config)` for fold training/validation/test modeling outputs
- `run_backtest_phase(config, learning_artifacts)` for portfolio construction/backtest
- `run_boosting_backtest(config)` orchestration helper that runs both phases and writes final report/artifacts

## Packaging / Imports

The project uses a `src/` layout and is intended to be installed in editable mode:

```bash
pip install -e .
```

After installation, imports should always use `alpharank.*`.
Do not add `sys.path.append(...)` hacks.

## Conda Setup

Dedicated environment setup is standardized with:

- `environment.yml`
- `scripts/setup_conda_env.sh`

Use:

```bash
bash scripts/setup_conda_env.sh alpharank
conda activate alpharank
```

## Development Notes

- Keep new code inside `src/alpharank/`.
- Prefer extending `alpharank.legacy` or `alpharank.boosting` APIs for discoverability.
- If a component is deprecated, move it to `src/_old/` instead of deleting immediately.
- Reports/artifacts generated by training should go under `outputs/`.
- Do not create a git commit unless the user explicitly asks for one.
- When comparing `data/eodhd/output/` versus `data/open_source/output/`, do not stop at aggregate performance deltas. Continue autonomously until the first failing stage is isolated with evidence:
  - input alignment and cutoff dates
  - price coverage
  - price return equivalence, not only raw adjusted-close level comparisons
  - share-count / market-cap construction
  - `Sector` mapping
  - `US_Earnings` semantics and source mix (`period_end`, `reportDate`, EPS meaning, estimate/surprise coverage)
  - Optuna/model-parameter drift between the two runs
  - latest model-stage vote differences and final portfolio overlap
- For legacy replacement work, a claim like `can't replace EODHD yet` is not sufficient by itself. The output must identify which dataset or semantic mismatch is driving the gap and point to the concrete files/parquets used to prove it.
- For source-replacement audits, always publish explicit acceptance gates with thresholds and pass/fail status. At minimum cover:
  - selection/coverage equivalence
  - price return equivalence
  - 2025 statement/earnings error rates
  - latest market-source earnings coverage
  - final backtest overlap/performance gates

## Open-Source Data Layout

The open-source replacement data model must stay clean and discoverable.

Use only these top-level folders under `data/open_source/`:

- `_cache/`: fetch caches
- `official/`: canonical ingestion outputs
- `output/`: user-facing exact-name package for backtests and manual inspection
- `audit/`: discrepancy reports and audit artifacts
- `archive/`: preserved exploratory or legacy open-source runs

Rules:

- Never create ad hoc run folders directly under `data/open_source/`.
- The internal canonical lineage lives under `data/open_source/official/target/`.
- The user-facing exact-name package lives under `data/open_source/output/`.
- The user-facing lineage package lives under `data/open_source/output/lineage/`.
- The official exported lineage must include the selected financial, earnings, and general-reference files under `data/open_source/output/lineage/`.
- `US_Earnings.parquet` and `US_General.parquet` must be published from official consolidations, not ad hoc one-off artifacts.
- Published open-source outputs must be historized under `data/open_source/history/output/` after the final package is written, so the retained snapshot contains the exact legacy files, lineage files, and manifest for the published run.
- The legacy reference mirror must live under `data/eodhd/output/` with the same exact filenames as `data/open_source/output/`.
- If exploratory outputs must be kept, move them under `data/open_source/archive/` instead of leaving them at the root.
- When documentation mentions the open-source store, prefer the words `official`, `target`, `output`, `audit`, and `archive` over ambiguous names like `live` or `clean`.
- When changing ingestion semantics, consolidation priority, natural keys, quarter normalization, or lineage schema, update the corresponding docs in the same task. At minimum keep `README.md`, `docs/open_source_ingestion_architecture.md`, and any package-specific contract document current.
- `data/sec/output/` is the official SEC-only fundamentals package. Keep its source policy simple: SEC only, with `companyfacts` then `filing-level` extraction, and document any exception explicitly before shipping it.
