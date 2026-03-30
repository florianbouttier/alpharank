# Open-Source Data Layout

This folder has one official purpose: hold the canonical open-source replacement data model.

## What matters

- `official/`: the canonical open-source store
- `output/`: the exact-name user-facing package
- `audit/`: HTML and parquet comparison reports versus EODHD
- `archive/`: old probes, experiments, and one-off runs kept only for reference
- `_cache/`: source caches for SEC and SimFin

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

## Rule

Do not create new ad hoc top-level run folders directly under `data/open_source/`.

Use only:

- `official/` for canonical ingestion outputs
- `output/` for the exact-name published package
- `audit/` for audit outputs
- `archive/` for preserved but non-canonical historical runs
