# Open-Source Data Layout

This folder has one official purpose: hold the canonical open-source replacement data model.

## What matters

- `official/`: the canonical open-source store
- `audit/`: HTML and parquet comparison reports versus EODHD
- `archive/`: old probes, experiments, and one-off runs kept only for reference
- `_cache/`: source caches for SEC and SimFin

## Where to look first

If you want the official new data model, start here:

- `official/raw/`
- `official/target/financials_open_source_consolidated.parquet`
- `official/target/financials_open_source_lineage.parquet`
- `official/target/financials_open_source_source_summary.parquet`
- `official/target/legacy_compatible/`
- `official/manifests/latest_run.json`

## Reading order

1. `official/manifests/latest_run.json`
2. `official/target/financials_open_source_consolidated.parquet`
3. `official/target/financials_open_source_lineage.parquet`
4. `audit/` only if you want discrepancy analysis
5. `archive/` only if you are debugging old exploratory work

## Rule

Do not create new ad hoc top-level run folders directly under `data/open_source/`.

Use only:

- `official/` for canonical ingestion outputs
- `audit/` for audit outputs
- `archive/` for preserved but non-canonical historical runs
