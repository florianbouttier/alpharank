# Open-Source Data Layout

This folder has one official purpose: hold the canonical open-source replacement data model.

## Fundamental boundary

The financial consolidation in this tree is multi-source research data. It is
not the official model-fundamental package. Official monthly fundamentals must
come from `data/sec/output` and may contain only SEC companyfacts, filing-level
SEC extraction, or explicitly labelled values derived solely from SEC facts.
EODHD/Yahoo/SimFin/StockAnalysis fundamental values are forbidden in an
official snapshot; an EODHD ticker-to-CIK bridge is identity metadata only.

The retained T1 package from 2026-08-11 contains 19,243 selected non-SEC
fundamental values, so it remains reproducible replay evidence but is not
SEC-only production truth. Current production inputs are composed under
`data/model_inputs/history/` and selected through
`data/model_inputs/manifests/latest.json`.

## Price-history warning

The price contract is not open-source-only. The frozen archive at
`data/eodhd/output/US_Finalprice.parquet` is the initial historical seed for
delisted and former constituents. It is not the long-term persistence boundary.
After the first validated publication, the complete preceding published price
lineage becomes the seed of the next refresh.

Therefore, a ticker first downloaded from Yahoo remains byte-stable after it
leaves the active universe or becomes unavailable upstream, even when it never
existed in EODHD. The routine roll-forward resolves the previous lineage from
`data/model_inputs/manifests/latest.json`, rejects any removed prior inactive
ticker/date, and writes
`lineage/persistent_price_history_registry.parquet` with the source and
lifecycle state of every retained ticker.

The 2026-08-16 canonical package has 3,709,695 rows / 840 tickers: 502 active
tickers from one Yahoo vintage and 338 retained inactive/terminal histories.
The real persistence replay resolves the preceding snapshot automatically,
preserves every row, and identifies `EA.US` as a non-EODHD terminal history.
Future Yahoo-only delistings are classified `inactive_open_source_only` without
requiring a manual exception once they leave the active constituent universe.

The frozen EODHD archive itself is never edited. If new evidence establishes a
missing or incorrect split/dividend, publish a versioned correction overlay and
a new immutable derived snapshot. Splits may restate pre-event OHLC and inverse
volume; dividends leave raw OHLCV unchanged and alter only adjusted/total-return
values. The overlay must retain event evidence and before/after hashes.

## What matters

- `official/`: the canonical open-source store
- `output/`: the exact-name user-facing package
- `audit/`: HTML and parquet comparison reports versus EODHD
- `archive/`: old probes, experiments, and one-off runs kept only for reference
- `_cache/`: disposable transport cache; it may be empty and is never required
  to replay a published snapshot

## Where to look first

If you want the exact files for backtests/manual use, start here:

- `output/`
- `output/lineage/`

If you want the internal canonical store, start here:

- `official/raw/`
- `official/target/general_reference.parquet`
- `official/target/general_reference_lineage.parquet`
- `official/target/earnings_open_source_consolidated.parquet`
- `official/target/earnings_open_source_lineage.parquet`
- `official/target/financials_open_source_consolidated.parquet`
- `official/target/financials_open_source_lineage.parquet`
- `official/target/financials_open_source_source_summary.parquet`
- `official/target/legacy_compatible/`
- `official/manifests/latest_run.json`

## Reading order

1. `official/manifests/latest_run.json`
2. `output/`
3. `output/lineage/`
4. `official/target/financials_open_source_consolidated.parquet`
5. `audit/` only if you want discrepancy analysis
6. `archive/` only if you are debugging old exploratory work

In the manifest, read both:

- `source_refresh_contract`: what was actually downloaded for this run
- `data_freshness`: observed price, filing, fiscal-period, earnings, and
  membership dates

Only `snapshot_scope=full_ingestion` is eligible for a production monthly run.
`official/raw/` is the retained normalized source history. Deleting `_cache/`
does not delete that history or any published replay package.

A successful production manifest also proves current network coverage for all
active constituents and SPY, complete SEC mapping, and successful active-universe
SEC submissions/companyfacts refreshes. A failed or interrupted run is rolled
back transactionally and cannot replace the published snapshot.

It must also contain a prepublication `historical_revision_guard` and the run
must retain `official/runs/<run_id>/historical_revision_guard.json`. Historical
changes older than 730 days block publication by default. Any
`QUARANTINED.json`, `quarantined_run_*.json`, or `raw_store_quarantine.json`
marker makes the corresponding data non-production until a later guarded full
ingestion succeeds.

Price publication additionally requires `price_composition.json`,
`price_revision_guard.json`, exhaustive daily-return and removed-key parquet
artifacts, complete inactive EODHD-key coverage, and one full Yahoo vintage for
every active ticker. Historical return revisions above 1 bp and historical key
removals have separate migration-only overrides; both are disabled in routine
production.

## Rule

Do not create new ad hoc top-level run folders directly under `data/open_source/`.

Use only:

- `official/` for canonical ingestion outputs
- `output/` for the exact-name published package
- `audit/` for audit outputs
- `archive/` for preserved but non-canonical historical runs
