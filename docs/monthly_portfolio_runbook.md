# Monthly Portfolio Runbook

This is the canonical runbook for the monthly `ptf du mois`.

## Production Workflow

The monthly portfolio is produced by the stable command
`scripts/run_legacy.py`; its testable implementation lives in
`src/alpharank/production/legacy_pipeline.py` and both files are hashed in the
run provenance.
Do not use the XGBoost time-fold backtest for the monthly portfolio unless the
request explicitly asks for the boosting/R&D workflow.

Before running the workflow, read `AGENTS.md`. If the procedure
or expected artifacts differ from those files, update the documentation in the
same task.

## Required Input Composition

The production composition contract is implemented. New official runs resolve
`data/model_inputs/manifests/latest.json`, freeze that pointer once, and pass the
resolved immutable directory explicitly to `scripts/run_legacy.py`.

The validated 2026-08-16 reference resolves
`data/model_inputs/history/alpharank_input_20260816_120458_2a01288bab06`.
Every future official snapshot must continue to combine:

- approved hybrid prices: frozen EODHD history for delisted/former constituents
  plus audited open-source extensions and corporate-action corrections;
- SEC-only fundamentals from an immutable `data/sec/output` snapshot;
- the historical constituent calendar;
- explicit price-lineage, SEC-lineage, source hashes, and one composition id.

`data/open_source/output` is a mixed-source R&D/replay package. It must not be
used for a new production recommendation: retained T1 selected 19,243 non-SEC
fundamental values, and its price layer is still missing the frozen EODHD seed.
The old T1/T2 Legacy runs remain valid evidence of what the old package
produced, but they are not SEC-only production runs. Never reconstruct a model
input by manually copying unmanifested files.

For a new month, extend the membership calendar, run the full ingestion, build
the guarded price package, build the SEC-only package, then compose and promote
a new immutable snapshot. The following ingestion commands alone do not create
a production model input:

```bash
./.venv/bin/python scripts/open_source/refresh_sp500_constituents.py \
  --target-month YYYY-MM-01

./.venv/bin/python scripts/open_source/nightly_ingestion.py
```

Despite its historical name, the second command applies the multi-source full
refresh contract: complete available Yahoo price history for the latest active universe and SPY, full SEC companyfacts
payloads including historical revisions, fresh SEC submissions, and refreshed
fallback bulk/history sources. HTTP payloads are not replay data;
`official/raw`, run deltas, and immutable snapshots are retained.

### What happens to a nightly download

The pipeline deliberately separates receiving data from making it usable by the
model:

1. it assigns one run id before contacting a provider;
2. it archives the Yahoo observation in the differential RAW chain, including
   rows whose adjusted price is empty, and stores only a compact archive pointer
   in the run folder;
3. it lists every previously validated ticker/date pair that is missing, retries
   those gaps, and saves both the initial and remaining lists;
4. if a required old provider price is still missing, DEF retains the exact
   validated prefix for that ticker and records it; the run is rejected only
   when no valid exact prior key can resolve the gap;
5. only a successful run rebuilds the cleaned tables from the retained raw
   history and creates a versioned publication;
6. price and SEC-only packages are then composed into a new immutable model
   input, whose files are checked by hash before the `latest.json` pointer moves;
7. Legacy copies that immutable model input into its own `input_snapshot/` and
   calculates from the copy.

A rejected Yahoo attempt is audit evidence, not a model input. Its folder keeps
`prices_yfinance_attempted_summary.json`, the initial and remaining gap
Parquets, `price_validated_key_coverage.json`, and the pointer to the immutable
RAW delta archive. The stable nightly status file carries the same run id even
when the run fails. This makes it possible to explain a provider failure without
filling a missing price, inventing a delisting, duplicating the full observation
or altering the last validated publication.

If every declared acquisition completed but a price gate was reviewed only
afterward, do not run the network ingestion again. First build the strict SEC
candidate from the retained raw store, then reassess the immutable acquisition
run into a new price package:

```bash
./.venv/bin/python scripts/open_source/build_acquired_price_package.py \
  --acquisition-run-dir data/open_source/official/runs/<run_id> \
  --sec-package-dir <validated-sec-package> \
  --eodhd-seed data/eodhd/output/US_Finalprice.parquet \
  --output-dir <new-price-candidate> \
  --expected-through YYYY-MM-DD
```

This command performs no provider request. It refuses an incomplete acquisition,
binds prices and SPY to the same run id, recomputes both price gates against the
current reviewed-move registry, validates full data freshness, and records the
original acquisition and review hashes in the new manifest. If the acquisition
resolved provider revisions through a validated-history return extension, the
command must replay that same reconciliation and hash-bind its report and
Parquet audit; it may never fall back to the provider's rewritten history. The
original run folder and its initially failed gate remain unchanged audit evidence.

When a complete provider observation rewrites old adjusted prices, the run
audits those changes in `price_daily_return_revisions.parquet` and
`price_revision_diagnostic.json`. The canonical candidate keeps all previously
validated `ticker,date` rows unchanged and writes only the post-anchor dates in
`price_return_extension_audit.parquet`, using the provider's new daily returns.
This resolves a provider restatement without silently approving it; a sourced
correction overlay remains the only way to replace published history.
The revision Parquet stores only differences above the versioned 1 bp material
threshold, plus return-availability changes; smaller provider-level changes are
not duplicated outside the differential RAW archive.

The durable target layout is `RAW -> STG -> DEF -> MART`. The Yahoo RAW archive
stores a full logical observation for every ingestion but writes only inserted
or changed row content. Missing and restored ticker/date keys are events, and
an identical ingestion adds only a manifest with zero new price rows. Network
download still occurs because Yahoo does not provide a reliable change feed;
deduplication happens before durable storage, not before the request.

DEF resolves the exact `ticker,date` key after STG normalization. A missing or
null current Yahoo value may reuse the last validated value for that same key;
the selected row keeps its original ingestion id and the run writes
`def_price_selection_audit.parquet`. It is forbidden to carry a quote from one
date to another. Provider completeness remains reported separately, and the
resolved table still has to pass every price-revision and continuity gate.

Fallback financial rows produced by this command are for R&D/audit only. The
composed model snapshot must replace every fundamental file with its SEC-only
counterpart and validate allowed lineage sources.

Before Legacy, require `source_refresh_contract.snapshot_scope=full_ingestion`
and inspect `data_freshness`. Also require a retained
`official/runs/<run_id>/historical_revision_guard.json`, the same guard embedded
in `source_refresh_contract`, and no quarantine marker. Historical revisions
older than 730 days block by default; the override requires explicit review and
is never part of the normal monthly command. Do not confuse
`financials.max_fiscal_period_end` with source freshness: a June accounting
period can be normal in August, while an old `max_sec_filing_date` must fail the
freshness gate.

The historical command below reproduces the mixed-source workflow only. It is
retained for replay/diagnosis and must not be labelled current production:

```bash
./.venv/bin/python scripts/run_legacy.py \
  --n-trials 30 \
  --n-jobs 1 \
  --first-date 2010-01 \
  --data-dir data/open_source/output \
  --output-dir outputs \
  --checkpoints-dir outputs/checkpoints_open_source_YYYYMMDD
```

Use a date-stamped checkpoint directory so a diagnostic replay does not
overwrite another audit trail. A future production command must point
`--data-dir` to the immutable composed package, not
`data/open_source/output`.

The versioned `historical_ticker_exclusions_v1` data-quality quarantine is
enabled by default. The manifest must record its path, id, hash, and complete
excluded-ticker list. `--no-ticker-exclusion-registry` exists only to reproduce
an older historical convention and makes that run ineligible for a common
Legacy/Boosting comparison that uses the registry.

Legacy and Boosting also share `monthly_price_eligibility_v1`, implemented only
in `src/alpharank/data/price_eligibility.py`. For each decision month, a ticker
requires at least 10 daily prices, USD 1,000,000 median daily `close * volume`,
and at most 5% invalid OHLC rows. Legacy intersects this table with historical
S&P membership before ranking EMA signals. The run retains
`monthly_price_eligibility.parquet` and records the policy id plus thresholds in
`data_input_manifest.json`.

Current `--checkpoints-dir` artifacts are diagnostic snapshots, not a supported
resume contract. A failed process must be restarted from its immutable input
snapshot; never splice downstream files from an earlier process into a new run.
Use `--no-checkpoints` when disk space is constrained; it skips only optional
diagnostics and preserves the complete canonical replay package.

When launching manually, capture stdout/stderr under `logs/legacy_runs/` with a
timestamped filename and reference the output folder in the final handoff. The
minimum durable evidence for a monthly run is the command log plus
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/data_input_manifest.json` and
`outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot/`. The legacy runner
computes from this local input snapshot, not from the mutable source directory.

For a point-in-time replay of the historical mixed-source workflow, prefer the
immutable open-source run identifier instead of the mutable
`data/open_source/output` directory:

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
- `input_snapshot/`: canonical copy of every model input file used by the run, including copied lineage metadata when available. On APFS/macOS the runner uses byte-identical copy-on-write clones; other filesystems use physical copies. `storage_manifest.json` records the effective mode, and neither mode uses symlinks.
- `monthly_price_eligibility.parquet`: exhaustive ticker-month observations,
  dollar volume, OHLC violation rate, and eligibility decision.
- `legacy_detailed_returns_polars.parquet`: ticker-level monthly portfolios.
- `legacy_aggregated_returns_polars.parquet`: aggregated model returns.
- `legacy_metrics_polars.parquet`: model comparison metrics.
- `legacy_common_holdings.parquet`: canonical completed-month holdings for
  `Combined_Equal` and `Combined_Frequency`.
- `legacy_common_monthly.parquet`: canonical gross/net/benchmark return ledger
  produced by the shared portfolio simulator.
- `legacy_common_annual.csv`, `legacy_common_performance.csv`, and
  `legacy_common_calendar.json`: shared Legacy/Alpha/SPY reporting convention.
- `legacy_execution_policy.json`: convention runtime canonique
  `reference_close_adjusted_close_v1` et scénarios de sensibilité obligatoires.
- `execution_return_bridge.parquet` et `execution_return_policy.json` lorsqu'un
  replay comparatif est produit : rapprochement clôture/prochaine ouverture sur
  les mêmes positions, mois, poids et paramètres de coûts.
- `portfolio_report_frequency_polars_<YYYY-MM>.html`: frequency-weighted portfolio snapshot.
- `portfolio_report_equal_polars_<YYYY-MM>.html`: equal-weighted portfolio snapshot.

The two production portfolio views are:

- `Combined_Frequency`: consensus-weighted view, using model frequency as weights.
- Standard performance benchmark: `SPY total return` built from
  `SP500Price.parquet.adjusted_close`. The historical `SP500` row in raw Legacy
  monthly artifacts uses `close` and is signal/replay compatibility data, not
  the benchmark for a performance answer.
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
- `open_source_sec_companyfacts_years_refreshed`
- `open_source_source_refresh_scope`
- `open_source_source_refresh_contract`
- `open_source_data_freshness`
- `open_source_ticker_count`

If `open_source_run_id_match`, `open_source_output_manifest_run_id_match`, or
`open_source_output_matches_published_snapshot` is false, do not treat the run
as a clean point-in-time monthly portfolio without investigating the data
package first. A clean open-source package must have matching
`snapshot_manifest.json`, `lineage/manifest.json`, ingestion run manifest ids,
and file hashes matching the ingestion manifest's published output snapshot.
If `open_source_source_refresh_scope` is present and is not `full_ingestion`,
the package is diagnostic and not production-clean.

If the `input_snapshot/` directory is missing, the run is not fully replayable.
Treat historical outputs without that directory as audit evidence only, not as a
complete reproducibility package.

## Derived Data Replay Gate

Le contrat obligatoire de comparaison après refresh est
[`data_refresh_replay_contract.md`](data_refresh_replay_contract.md). Un nouveau
snapshot ne peut pas être promu sur la seule base de contrôles de fraîcheur :
Legacy et Boosting sont rejoués au cutoff historique commun. Le résultat doit
être soit une identité des portefeuilles, soit une attribution exhaustive des
écarts à des révisions data sourcées. Toute différence restante de code,
configuration, runtime ou origine inconnue bloque la promotion.

The input snapshot is necessary but not sufficient. The first derived monthly
selection checkpoint, `polars_stocks_selections.parquet`, must also be
recomputable bit-for-bit from the retained `input_snapshot/` and the recorded
code context. If two recomputations from the same snapshot produce different
rows, ratios, or ticker/month membership, treat every downstream portfolio from
that run as non-production until the preprocessing bug is fixed and the month is
replayed.

The fundamental preprocessing must obey point-in-time semantics:

- statement rows without a usable as-of publication date must not be
  forward-filled into monthly ratios;
- when several statement rows map to the same `ticker` and as-of date, choose
  the row deterministically using `quarter_end`, filing dates, and stable
  tie-break columns before joining to monthly prices;
- monthly ratio aggregation must select the last available observation by
  explicit date ordering, not by implicit dataframe row order.

## Current Constituents And Live Price Refresh

Legacy joins the price/feature rows to the exact ticker membership recorded for
each calendar month in `SP500_Constituents.csv`. Fresh daily prices alone do not
extend the production calendar: the constituent snapshots must also reach the
decision month.

For the 2026 refresh, official index and company announcements are recorded in
the versioned registry
`configs/data_quality/sp500_constituent_changes_2026.json`. Rebuild the monthly
snapshots and retain the generated audit:

```bash
./.venv/bin/python scripts/open_source/refresh_sp500_constituents.py \
  --target-month YYYY-MM-01
```

The targeted price command remains available for diagnosis and repair:

```bash
./.venv/bin/python scripts/open_source/refresh_current_constituent_prices.py
```

It publishes a package marked
`snapshot_scope=current_constituent_price_refresh`; it is not the final monthly
production package. Follow it with the full ingestion described above.

The constituent reconstruction must run before the full ingestion. The price
refresh reads the latest reconstructed monthly snapshot at startup; reversing
that order would download symbols that an official event has already removed
and omit their replacements. For August 2026, the sourced events therefore
remove EA and AVB, add FERG and RDDT, and change EQR to VMRK before Yahoo is
queried. AVB and VMRK remain distinct securities: the reviewed shareholder
event values an AVB position as 2.793 VMRK shares and never joins the two price
series as a one-day AVB return.

The resulting full ingestion is production-clean only when:

- every current constituent has a non-null adjusted price through the same
  recent trading session;
- `data/open_source/official/manifests/latest_run.json`,
  `data/open_source/output/lineage/manifest.json`, and the retained
  `history/output/open_source_output_*` snapshot expose the same run id;
- `scripts/validate_legacy_replay_package.py` accepts the Legacy run manifest.

Price publication is fail-closed. The refresh builds the prospective merged
series in memory and runs `src/alpharank/data/open_source/price_quality.py`
before changing raw or published parquet files. Any recent absolute
adjusted-close move of 40% or more blocks the run for investigation; do not
raise the threshold merely to publish.

When a provider has partially rewritten a split or another corporate action,
restore only the affected ticker from a retained clean snapshot and keep the
replaced rows in quarantine:

```bash
./.venv/bin/python scripts/open_source/restore_price_tickers_from_snapshot.py \
  --snapshot-dir data/open_source/history/output/open_source_output_<clean> \
  --tickers TICKER \
  --reason "audited provider adjustment recovery"
```

Then rerun the normal current-constituent refresh. The recovery command does
not publish; only a subsequent full refresh that passes the price gate may
become production truth. The 2026-08-11 reference follows this procedure:
MNST was restored from `open_source_output_20260810_015044`, and clean run
`20260811_003849` was retained as
`open_source_output_20260811_004415`. Its price gate passed, current-member
coverage is 501 tickers through 2026-08-10 and two through 2026-08-07, and the
Legacy replay package `outputs/2026-08-11/runs/20260811_030547` validates.

This recovery package was superseded later the same night by the full ingestion
run `20260811_001503`, retained as
`open_source_output_20260811_014746`. Its post-publication audit passes: 503
current members covered, 502 through 2026-08-10 and one through 2026-08-07, with
no current-member adjusted-close move of 40% or more. The current Legacy replay
is `outputs/2026-08-11/runs/20260811_035522` and validates against that retained
snapshot. Refresh and restore commands must acquire
`data/open_source/official/manifests/nightly.lock.json`; never run a second publisher
while the full ingestion holds it.

If the latest price date is inside an unfinished month, use the preceding
calendar month as the decision month. A partial July dataset can be displayed
as freshness evidence, but it must not silently become the July decision input
for an August portfolio.

`scripts/run_legacy.py` enforces this before feature construction: the raw
snapshot and manifest retain the newest partial prices for audit, while model
inputs are truncated to `run_config.decision_data_completed_through_month`.
Consequently, a run performed during August produces the final August target
from the completed July decision and must not publish a September target based
on partial August prices.

## Retained 2026-08-16 data refresh

The full ingestion run is `20260816_103942`. Prices and SEC filing dates both
reach 2026-08-14. New model runs must resolve
`data/model_inputs/manifests/latest.json`, currently composed snapshot
`data/model_inputs/history/alpharank_input_20260816_120458_2a01288bab06`,
or the byte-identical `input_snapshot/` retained by its Legacy run.

The Legacy CLI does not infer this pointer: resolve `snapshot_dir` from the
JSON pointer and pass it explicitly with `--data-dir`. A run pointed at
`data/open_source/output` is not production under this contract.

The completed canonical Legacy run is
`outputs/production_refresh_20260816/legacy_runs_v3/2026-08-16/runs/20260816_142810`.
Its strict replay validator passes. It excludes partial August prices before
feature construction, uses the completed July decision, and publishes the
August 2026 `Combined_Frequency` target. Its realized performance ledger stops
at holding month July 2026; this is expected because August is not complete.
The aligned Boosting replay is
`outputs/production_refresh_20260816/boosting_latest_common_v3`, and the
same-snapshot common comparison is
`outputs/production_refresh_20260816/common_replay_v4_sec_universe`.

The price package passed without a historical revision override: zero removed
old keys, zero old return-availability changes, zero old daily-return changes
above 1 bp, and zero adjustment-transition findings. EA is carried forward
from the previous validated package because Yahoo no longer serves its full
history after delisting; the exception is allowed only because the versioned
constituent registry records its official 2026-08-05 removal.

For every later refresh, the complete preceding validated price lineage is the
durable base. This includes tickers first ingested from Yahoo that never existed
in EODHD. When such a ticker leaves the active universe, its published rows are
copied byte-for-byte and registered as `inactive_open_source_only`; they are not
redownloaded, discarded, or reconstructed from the current universe. Resolve
the base from `data/model_inputs/manifests/latest.json` rather than selecting an
older EODHD-only seed by hand.

The SEC package is a reviewed one-time point-in-time migration. Raw
Companyfacts now retain each `filing_date`; model exports select the earliest
filing version. The manifest records the migration note and exhaustive
historical revision guard. Do not regenerate this migration with an unlabelled
`--allow-historical-revisions` flag: the CLI requires
`--revision-review-note`.

## Retained 2026-08-20 target valued at the 2026-08-19 close

The retained full ingestion is `20260820_011146`; the composed immutable input
has composition id `9a2058c9…425ad`. The aligned artifacts are Legacy
`20260820_055245`, Boosting `boosting_latest_common_live021_final_r2`, and common
replay `common_replay_live021_final`.

The snapshot contains stock and SPY prices through the 2026-08-19 session. The
monthly ranking still stops at the completed July month, as recorded by
`run_config.decision_data_completed_through_month=2026-07-01`. Therefore:

- the signal cutoff is the 2026-07-31 close;
- the August target is the latest valid monthly target;
- the 2026-08-19 close can value a requested order on that date, but the first
  nineteen August days must not rerank the strategy;
- buying on August 19 is a late entry into the August holding window, not a new
  mid-month AlphaRank signal.

The exact retained target, before considering any current holdings or cash, is:

- Boosting Top 5, 20% each: `CHTR`, `BSX`, `ZTS`, `LULU`, `DELL`;
- Boosting Top 10, 10% each: `CHTR`, `BSX`, `ZTS`, `LULU`, `DELL`, `TTD`,
  `FISV`, `ACN`, `INTU`, `IT`;
- Legacy `Combined_Frequency`: `AMAT` 20%, `DELL` 20%, `MRVL` 20%, `MU` 20%,
  `FLEX` 10%, `HPE` 10%.

All 15 distinct selected tickers have one exact retained close on 2026-08-19.
The current SNDK identity defect affects historical Boosting analysis but SNDK
is absent from all three current baskets. FISV is the valid 2026 trading symbol
for Fiserv after its return to Nasdaq in November 2025, corroborated by the
[Fiserv announcement](https://investors.fiserv.com/news-releases/news-release-details/fiserv-announces-transfer-stock-exchange-listing-nasdaq) and 2026 SEC filings.

The machine-readable Boosting target is retained in
`outputs/production_refresh_20260820/common_replay_live021_final/boosting_live_score_holdings.parquet`.
AlphaRank stops at the immutable target, scores, weights and price evidence.
Portfolio owns any interactive allocation view and the reconciliation with
current holdings or cash; no trade is sent from AlphaRank.

## Refresh reliability replay on 2026-08-24 and 2026-08-25

The bootstrap refresh `20260824_214818` is quarantined and must never be used as
a model input. Yahoo introduced 44 historical daily-return revisions above 1
basis point across 30 tickers. The price gate stopped before SEC and fallback
fundamental acquisition, before snapshot publication, and before either model.
The canonical pointer therefore still resolves composition `9a2058c9…425ad`.

Legacy run `20260825_000250` and the aligned Boosting/common replays were then
recomputed from that unchanged canonical snapshot. At the common `2026-07-01`
cutoff, all 7,994 Legacy positions, 88,948 Boosting predictions and 6,395 common
positions match the retained baseline, with zero material value difference.

A second full attempt, `20260825_001501`, downloaded 2,731,732 Yahoo rows for
502 active tickers through the 2026-08-24 session. It was also quarantined
before fundamentals and publication: 45 historical daily-return revisions over
1 basis point remained across 30 tickers, including ten pre-cutoff `AVB.US`
revisions and inconsistent half/double `MNST.US` price levels. Legacy run
`20260825_092319`, the aligned Boosting run and the common replay were therefore
recomputed from the still-canonical composition `9a2058c9…425ad`; all five
portfolio comparison stages remained identical through the common cutoff.

The complete execution record, source-by-source statuses, commands, drift keys
and report hashes are in
[`research/data_refresh_replay_20260824_20260825.md`](research/data_refresh_replay_20260824_20260825.md).
Future refreshes must use
[`data_refresh_replay_contract.md`](data_refresh_replay_contract.md) and may not
reuse the blocked candidate or override its revision gate without a separately
reviewed migration.

## Boosting Production Candidate

Boosting remains an explicit R&D production candidate; it does not replace the
canonical Legacy monthly portfolio. Once the full Legacy replay has produced
`legacy_detailed_returns_polars.parquet`, score the latest completed month from
that same immutable `input_snapshot/`:

```bash
./.venv/bin/python scripts/experiments/run_live_alpha_portfolio.py \
  --data-dir outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/input_snapshot \
  --legacy-detailed outputs/YYYY-MM-DD/runs/YYYYMMDD_HHMMSS/legacy_detailed_returns_polars.parquet \
  --decision-month YYYY-MM-01
```

This command freezes the selected research specification: XGBoost
classification at six months, exact Legacy-winning relative EMA variables,
strictly mature labels, a chronological calibration window, Top 5 as the
retained portfolio and Top 10 as a non-promoted diagnostic. Its HTML report
shows both Alpha and Legacy for the same holding month. The reported live
validation block is used for early stopping and calibration, so it is not a new
sealed test.
