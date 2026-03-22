# Official Target Layer

This folder contains the canonical selected outputs of the open-source data model.

Expected files:

- `prices_open_source.parquet`
- `benchmark_prices_open_source.parquet`
- `earnings_open_source.parquet`
- `earnings_open_source_long.parquet`
- `financials_open_source_consolidated.parquet`
- `financials_open_source_lineage.parquet`
- `financials_open_source_source_summary.parquet`
- `legacy_compatible/`

If one of these files is missing, the latest ingestion likely did not complete successfully enough to rebuild the full target layer.

The canonical financial lineage file is:

- `financials_open_source_lineage.parquet`
