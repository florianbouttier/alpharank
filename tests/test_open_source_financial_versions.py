from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.ingestion_reference import _upsert_financial_dataset
from alpharank.data.open_source.storage import OpenSourceLivePaths


def test_sec_raw_upsert_preserves_distinct_filing_versions(tmp_path: Path) -> None:
    paths = OpenSourceLivePaths(tmp_path / "official")
    paths.ensure()
    first = pl.DataFrame(
        {
            "ticker": ["AAA.US"],
            "statement": ["shares"],
            "metric": ["outstanding_shares"],
            "date": ["2020-12-31"],
            "filing_date": ["2021-02-01"],
            "value": [100.0],
            "source": ["sec_companyfacts"],
            "ingested_at": ["2026-08-16T00:00:00+00:00"],
        }
    )
    restatement = first.with_columns(
        pl.lit("2022-02-01").alias("filing_date"),
        pl.lit(110.0).alias("value"),
        pl.lit("2026-08-16T01:00:00+00:00").alias("ingested_at"),
    )

    _upsert_financial_dataset(
        paths=paths,
        run_id="run_a",
        file_name="financials_sec_companyfacts.parquet",
        deltas=[first],
    )
    merged = _upsert_financial_dataset(
        paths=paths,
        run_id="run_b",
        file_name="financials_sec_companyfacts.parquet",
        deltas=[restatement],
    )

    assert merged.select("filing_date", "value").sort("filing_date").rows() == [
        ("2021-02-01", 100.0),
        ("2022-02-01", 110.0),
    ]
