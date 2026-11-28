"""Point-in-time preservation helpers for normalized SEC financial facts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from alpharank.data.ingestion.raw_archive_frames import file_sha256
from alpharank.data.ingestion.storage import merge_upsert_frames

SEC_FINANCIAL_VERSION_KEY = (
    "ticker",
    "statement",
    "metric",
    "date",
    "filing_date",
    "source",
)

SEC_RAW_DELTA_KEYS = {
    "financials_sec_filing.parquet": SEC_FINANCIAL_VERSION_KEY,
    "earnings_sec_calendar.parquet": (
        "ticker",
        "period_end",
        "reportDate",
        "accession_number",
        "source",
    ),
    "earnings_sec_actuals.parquet": (
        "ticker",
        "period_end",
        "reportDate",
        "source",
    ),
    "general_reference_lineage.parquet": ("ticker", "source"),
}


def rebuild_full_companyfacts_versions(
    *,
    retained: pl.DataFrame,
    full_refresh: pl.DataFrame,
) -> pl.DataFrame:
    """Upsert observed filing versions without treating provider omissions as deletions."""

    if full_refresh.is_empty():
        raise RuntimeError("A full Companyfacts version rebuild requires a non-empty refresh")
    return merge_upsert_frames(
        retained,
        full_refresh,
        key_cols=SEC_FINANCIAL_VERSION_KEY,
        order_cols=("ingested_at",),
    )


def _companyfacts_observations(retained: pl.DataFrame, refresh: pl.DataFrame) -> pl.DataFrame:
    keys = list(SEC_FINANCIAL_VERSION_KEY)
    refreshed_tickers = refresh.get_column("ticker").unique().to_list()
    prior = _companyfacts_version_values(
        retained.filter(pl.col("ticker").is_in(refreshed_tickers)), "retained_value"
    )
    observed = _companyfacts_version_values(refresh, "observed_value").with_columns(
        pl.lit(True).alias("_seen")
    )
    return (
        prior.join(observed, on=keys, how="left", validate="1:1", nulls_equal=True)
        .with_columns(
            pl.when(pl.col("_seen").is_null())
            .then(pl.lit("absent_retained_version"))
            .when(~pl.col("retained_value").eq_missing(pl.col("observed_value")))
            .then(pl.lit("observed_value_revision"))
            .alias("observation")
        )
        .filter(pl.col("observation").is_not_null())
        .drop("_seen")
        .sort(keys)
    )


def _companyfacts_version_values(frame: pl.DataFrame, value_column: str) -> pl.DataFrame:
    keys = list(SEC_FINANCIAL_VERSION_KEY)
    return (
        frame.select(*keys, "value", "ingested_at")
        .sort("ingested_at")
        .unique(subset=keys, keep="last", maintain_order=True)
        .select(*keys, pl.col("value").alias(value_column))
    )


def _known_issuer_ids(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    fields = [pl.col(name).cast(pl.String) for name in ("sec_cik", "cik") if name in frame.columns]
    if not fields:
        return pl.DataFrame(schema={"ticker": pl.String, "issuer_cik": pl.String})
    return (
        frame.select("ticker", pl.coalesce(fields).str.strip_chars_start("0").alias("issuer_cik"))
        .filter(pl.col("issuer_cik").str.contains(r"^[0-9]+$"))
        .unique()
    )


def _reject_known_issuer_conflicts(retained_dir: Path, run_dir: Path) -> None:
    name = "general_reference_lineage.parquet"
    retained = _known_issuer_ids(retained_dir / name)
    observed = _known_issuer_ids(run_dir / name)
    observed_existing_symbols = observed.join(retained.select("ticker"), on="ticker", how="semi")
    conflicting = observed_existing_symbols.join(retained, on=["ticker", "issuer_cik"], how="anti")
    if not conflicting.is_empty():
        examples = conflicting.get_column("ticker").unique().sort().head(20).to_list()
        raise RuntimeError(
            f"SEC cumulative parent and acquisition disagree on issuer CIK: {examples}"
        )


def _merge_raw_delta(
    *,
    retained_path: Path,
    delta_path: Path,
    output_path: Path,
    version_key: tuple[str, ...],
) -> dict[str, object]:
    retained = pl.read_parquet(retained_path)
    delta = pl.read_parquet(delta_path)
    merged = merge_upsert_frames(
        retained,
        delta,
        key_cols=version_key,
        order_cols=("ingested_at",),
    )
    merged.write_parquet(output_path)
    return {
        "retained_rows": retained.height,
        "run_rows": delta.height,
        "output_rows": merged.height,
        "output_tickers": merged.get_column("ticker").n_unique(),
        "version_key": list(version_key),
    }


def build_sec_raw_version_candidate(
    *,
    retained_raw_dir: Path,
    run_raw_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    """Roll a retained raw SEC package forward with one acquisition run."""

    retained_raw_dir = retained_raw_dir.resolve()
    run_raw_dir = run_raw_dir.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    required = (
        "financials_sec_companyfacts.parquet",
        *SEC_RAW_DELTA_KEYS,
    )
    retained_sources = _input_records(retained_raw_dir, required)
    run_sources = _input_records(run_raw_dir, required)
    _reject_known_issuer_conflicts(retained_raw_dir, run_raw_dir)
    full_refresh_path = run_raw_dir / "financials_sec_companyfacts.parquet"
    retained = pl.read_parquet(retained_raw_dir / "financials_sec_companyfacts.parquet")
    full_refresh = pl.read_parquet(full_refresh_path)
    rebuilt = rebuild_full_companyfacts_versions(
        retained=retained,
        full_refresh=full_refresh,
    )
    observations = _companyfacts_observations(retained, full_refresh)
    output_dir.mkdir(parents=True, exist_ok=False)
    observations_path = output_dir / "companyfacts_version_observations.parquet"
    observations.write_parquet(observations_path)
    rebuilt.write_parquet(output_dir / "financials_sec_companyfacts.parquet")
    dataset_reports = {
        name: _merge_raw_delta(
            retained_path=retained_raw_dir / name,
            delta_path=run_raw_dir / name,
            output_path=output_dir / name,
            version_key=version_key,
        )
        for name, version_key in SEC_RAW_DELTA_KEYS.items()
    }
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
        "retained_raw_sources": retained_sources,
        "run_raw_sources": run_sources,
        "cumulative_raw_outputs": _input_records(output_dir, required),
        "companyfacts_version_observations": {
            **_file_record(observations_path),
            "counts": dict(observations.group_by("observation").len().iter_rows()),
        },
        "datasets": dataset_reports,
    }


def _input_records(directory: Path, names: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {name: _file_record(directory / name) for name in names}


def _file_record(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": file_sha256(path)}
