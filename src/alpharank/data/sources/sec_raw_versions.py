"""Point-in-time preservation helpers for normalized SEC financial facts."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

import polars as pl

from alpharank.data.publishing.snapshot_storage import copy_snapshot_file


SEC_FINANCIAL_VERSION_KEY = (
    "ticker",
    "statement",
    "metric",
    "date",
    "filing_date",
    "source",
)


def rebuild_full_companyfacts_versions(
    *,
    retained: pl.DataFrame,
    full_refresh: pl.DataFrame,
) -> pl.DataFrame:
    """Replace refreshed tickers while retaining every distinct filing version."""

    if full_refresh.is_empty():
        raise RuntimeError("A full Companyfacts version rebuild requires a non-empty refresh")
    refreshed_tickers = full_refresh.get_column("ticker").unique().to_list()
    unrefreshed = retained.filter(~pl.col("ticker").is_in(refreshed_tickers))
    combined = pl.concat([unrefreshed, full_refresh], how="diagonal_relaxed")
    result = (
        combined.sort([*SEC_FINANCIAL_VERSION_KEY, "ingested_at"])
        .unique(subset=list(SEC_FINANCIAL_VERSION_KEY), keep="last", maintain_order=True)
        .sort(list(SEC_FINANCIAL_VERSION_KEY))
    )
    unique_count = result.select(
        pl.struct(SEC_FINANCIAL_VERSION_KEY).n_unique()
    ).item()
    if unique_count != result.height:
        raise RuntimeError("SEC Companyfacts version key is not unique after rebuild")
    return result


def build_sec_raw_version_candidate(
    *,
    retained_raw_dir: Path,
    run_raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Create a replayable raw SEC package with Companyfacts filing versions."""

    retained_raw_dir = retained_raw_dir.resolve()
    run_raw_dir = run_raw_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    required = (
        "financials_sec_companyfacts.parquet",
        "financials_sec_filing.parquet",
        "earnings_sec_calendar.parquet",
        "earnings_sec_actuals.parquet",
        "general_reference_lineage.parquet",
    )
    for name in required:
        source = retained_raw_dir / name
        if not source.exists():
            raise FileNotFoundError(source)
        copy_snapshot_file(source, output_dir / name)

    full_refresh_path = run_raw_dir / "financials_sec_companyfacts.parquet"
    if not full_refresh_path.exists():
        raise FileNotFoundError(full_refresh_path)
    retained = pl.read_parquet(retained_raw_dir / "financials_sec_companyfacts.parquet")
    full_refresh = pl.read_parquet(full_refresh_path)
    rebuilt = rebuild_full_companyfacts_versions(
        retained=retained,
        full_refresh=full_refresh,
    )
    rebuilt.write_parquet(output_dir / "financials_sec_companyfacts.parquet")
    refreshed_tickers = full_refresh.get_column("ticker").n_unique()
    multi_version_groups = (
        rebuilt.group_by(["ticker", "statement", "metric", "date", "source"])
        .agg(pl.col("filing_date").n_unique().alias("filing_versions"))
        .filter(pl.col("filing_versions") > 1)
        .height
    )
    return {
        "retained_rows": retained.height,
        "full_refresh_rows": full_refresh.height,
        "output_rows": rebuilt.height,
        "refreshed_tickers": refreshed_tickers,
        "multi_filing_version_groups": multi_version_groups,
        "version_key": list(SEC_FINANCIAL_VERSION_KEY),
    }
