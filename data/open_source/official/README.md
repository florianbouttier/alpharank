# Official Open-Source Store

This is the canonical open-source data store.

## Canonical folders

- `raw/`: source-normalized append/upsert store
- `target/`: canonical selected outputs of the new data model
- `target/legacy_compatible/`: EODHD-shaped exports for backtests
- `manifests/`: latest successful run pointer
- `runs/`: immutable per-run deltas and manifests

## Canonical lineage files

- `target/general_reference.parquet`
- `target/general_reference_lineage.parquet`
- `target/earnings_open_source_consolidated.parquet`
- `target/earnings_open_source_lineage.parquet`
- `target/earnings_open_source_long.parquet`
- `target/financials_open_source_consolidated.parquet`
- `target/financials_open_source_lineage.parquet`
- `target/financials_open_source_source_summary.parquet`

## Contract

- Raw data is never deleted by the ingestion pipeline.
- Corrections happen by upsert on the natural key.
- `target/` is rebuilt from the full `raw/` store.
