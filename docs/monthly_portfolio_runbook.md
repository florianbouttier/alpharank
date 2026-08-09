# Monthly Portfolio Runbook

This is the canonical runbook for the monthly `ptf du mois`.

## Production Workflow

The monthly portfolio is produced by the legacy workflow in `scripts/run_legacy.py`.
Do not use the XGBoost time-fold backtest for the monthly portfolio unless the
request explicitly asks for the boosting/R&D workflow.

Before running the workflow, read `AGENTS.md` and `AGENT.md`. If the procedure
or expected artifacts differ from those files, update the documentation in the
same task.

The current open-source data source is:

```text
data/open_source/output
```

Run the full monthly workflow from the repository root:

```bash
./.venv/bin/python scripts/run_legacy.py \
  --n-trials 30 \
  --n-jobs 1 \
  --first-date 2010-01 \
  --data-dir data/open_source/output \
  --output-dir outputs \
  --checkpoints-dir outputs/checkpoints_open_source_YYYYMMDD
```

Use a date-stamped checkpoint directory so a new run does not overwrite another
audit trail.

When launching manually, capture stdout/stderr under `logs/legacy_runs/` with a
timestamped filename and reference the output folder in the final handoff. The
minimum durable evidence for a monthly run is the command log plus
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json` and
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/`. The legacy runner
computes from this local input snapshot, not from the mutable source directory.

For a point-in-time replay, prefer the immutable open-source run identifier
instead of the mutable `data/open_source/output` directory:

```bash
./.venv/bin/python scripts/run_legacy.py \
  --n-trials 30 \
  --n-jobs 1 \
  --first-date 2010-01 \
  --open-source-run-id YYYYMMDD_HHMMSS \
  --output-dir outputs \
  --checkpoints-dir outputs/checkpoints_open_source_YYYYMMDD
```

`open_source_run_id` resolves first against
`data/open_source/history/output/open_source_output_*/`, then against the
currently published output package. The resolved package is copied into
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/` before the model loads
any data.

## Expected Outputs

The run writes into a timestamped run directory:

```text
outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/
```

The date directory also contains `latest_legacy_run.json`, a pointer to the most
recent run instance for that date.

Key files:

- `data_input_manifest.json`: files, row counts, hashes, and max dates used by the run.
- `input_snapshot/`: canonical copy of every model input file used by the run, including copied lineage metadata when available.
- `legacy_detailed_returns_polars.parquet`: ticker-level monthly portfolios.
- `legacy_aggregated_returns_polars.parquet`: aggregated model returns.
- `legacy_metrics_polars.parquet`: model comparison metrics.
- `legacy_common_holdings.parquet`: canonical completed-month holdings for
  `Combined_Equal` and `Combined_Frequency`.
- `legacy_common_monthly.parquet`: canonical gross/net/benchmark return ledger
  produced by the shared portfolio simulator.
- `legacy_common_annual.csv`, `legacy_common_performance.csv`, and
  `legacy_common_calendar.json`: shared Legacy/Alpha/SPY reporting convention.
- `portfolio_report_frequency_polars_<YYYY-MM>.html`: frequency-weighted portfolio snapshot.
- `portfolio_report_equal_polars_<YYYY-MM>.html`: equal-weighted portfolio snapshot.

The two production portfolio views are:

- `Combined_Frequency`: consensus-weighted view, using model frequency as weights.
- `Combined_Equal`: equal-weighted view of the selected tickers.

The Legacy signal, annual Optuna search, and consensus votes remain owned by
`StrategyLearner`. Finalized baskets are now adapted to the common portfolio
engine documented in `docs/common_portfolio_backtest_engine.md`. This does not
change the production strategy: the frozen 2026-07-27 replay reproduces both
combined return series below `2.1e-16` absolute monthly error. Run
`scripts/validate_common_portfolio_engine.py` after any change to portfolio
aggregation, weighting, return handling, or performance metrics.

## Recovering A Prior Monthly Run

1. Check the legacy logs:

```bash
ls -lt logs/legacy_runs/run_legacy_*.log
```

2. Match the run date to the output folder under `outputs/YYYY-MM-DD/`.
3. Inspect `outputs/YYYY-MM-DD/latest_legacy_run.json` to find the run instance,
   then inspect its `data_input_manifest.json` for `input_snapshot_dir`, source
   files, hashes, code context, runtime context, and snapshot dates.
4. Validate the retained replay package:

```bash
./.venv/bin/python scripts/validate_legacy_replay_package.py outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json
```

5. Extract the requested month from `legacy_detailed_returns_polars.parquet`, filtering on:
   - `portfolio_model == "Combined_Frequency"` or `portfolio_model == "Combined_Equal"`
   - `year_month == <requested month>`

Example:

```bash
./.venv/bin/python -c "import polars as pl; df=pl.read_parquet('outputs/2026-04-02/runs/YYYYMMDD_HHMMSS/legacy_detailed_returns_polars.parquet'); print(df.filter((pl.col('portfolio_model')=='Combined_Frequency') & (pl.col('year_month').dt.strftime('%Y-%m')=='2026-04')).sort(['weight_normalized','ticker'], descending=[True,False]))"
```

## Audit Notes

For open-source runs, the legacy manifest captures the exact
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/*` files, hashes, row
counts, and max dates. It also records the original source directory and source
file hashes under `run_config.source_input_files` and
`run_config.source_input_sha256`. The copied snapshot, not the mutable source
path, is the replay source of truth.

The manifest propagates the upstream open-source context from:

- `data/open_source/output/lineage/manifest.json`
- `data/open_source/official/manifests/latest_run.json`

Required audit fields in
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json`:

- `input_snapshot_dir`
- `source_data_dir`
- `run_config.source_input_sha256`
- `code_context.git_head`
- `code_context.git_dirty`
- `code_context.critical_file_sha256`
- `open_source_output_run_id`
- `open_source_output_lineage_run_id`
- `open_source_output_snapshot_run_id`
- `open_source_output_manifest_run_id_match`
- `open_source_ingestion_published_output_snapshot`
- `open_source_output_matches_published_snapshot`
- `open_source_output_published_snapshot_differing_files`
- `open_source_ingestion_run_id`
- `open_source_run_id_match`
- `open_source_price_window`
- `open_source_financial_years_refreshed`
- `open_source_ticker_count`

If `open_source_run_id_match`, `open_source_output_manifest_run_id_match`, or
`open_source_output_matches_published_snapshot` is false, do not treat the run
as a clean point-in-time monthly portfolio without investigating the data
package first. A clean open-source package must have matching
`snapshot_manifest.json`, `lineage/manifest.json`, ingestion run manifest ids,
and file hashes matching the ingestion manifest's published output snapshot.

If the `input_snapshot/` directory is missing, the run is not fully replayable.
Treat historical outputs without that directory as audit evidence only, not as a
complete reproducibility package.
