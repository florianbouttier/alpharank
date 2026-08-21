"""Shared schemas and frame normalization for open-source ingestion stages."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import polars as pl

from alpharank.data.sources.general_reference import (
    empty_general_reference_lineage_frame,
)
from alpharank.data.ingestion.refresh_policy import SourceRefreshPolicy
from alpharank.data.quality.revision_guard import audit_historical_revisions
from alpharank.data.ingestion.storage import OpenSourceLivePaths, write_json

RAW_PRICE_SCHEMA = {
    "date": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "adjusted_close": pl.Float64,
    "ticker": pl.String,
    "source": pl.String,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}


def _audit_and_validate_historical_revisions(
    *,
    paths: OpenSourceLivePaths,
    run_id: str,
    legacy_paths: Mapping[str, Path],
    expected_through: str,
    source_refresh_policy: SourceRefreshPolicy,
    source_refresh_contract: dict[str, object],
) -> dict[str, object]:
    report = audit_historical_revisions(
        previous_output_dir=paths.output_dir,
        candidate_paths={path.name: path for path in legacy_paths.values()},
        expected_through=expected_through,
        guard_days=source_refresh_policy.historical_revision_guard_days,
    )
    review_note = (source_refresh_policy.historical_revision_review_note or "").strip()
    report["override_enabled"] = source_refresh_policy.allow_historical_revisions
    report["revision_review_note"] = review_note or None
    report["approval_recorded"] = bool(
        source_refresh_policy.allow_historical_revisions and review_note
    )
    source_refresh_contract["historical_revision_guard"] = report
    write_json(paths.run_dir(run_id) / "historical_revision_guard.json", report)
    if (
        report["historical_revisions_detected"]
        and not source_refresh_policy.allow_historical_revisions
    ):
        raise RuntimeError(
            "Historical fundamental revisions require explicit review; "
            f"blocked_datasets={report['blocked_datasets']}. "
            "No package was published."
        )
    if report["historical_revisions_detected"] and not review_note:
        raise RuntimeError(
            "Historical fundamental revision approval requires a non-empty "
            "review note. No package was published."
        )
    return report

RAW_FINANCIAL_SCHEMA = {
    "ticker": pl.String,
    "statement": pl.String,
    "metric": pl.String,
    "date": pl.String,
    "filing_date": pl.String,
    "value": pl.Float64,
    "source": pl.String,
    "source_label": pl.String,
    "accession_number": pl.String,
    "form": pl.String,
    "fiscal_period": pl.String,
    "fiscal_year": pl.Int64,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}

RAW_EARNINGS_SCHEMA = {
    "ticker": pl.String,
    "period_end": pl.String,
    "reportDate": pl.String,
    "earningsDatetime": pl.String,
    "epsEstimate": pl.Float64,
    "epsActual": pl.Float64,
    "surprisePercent": pl.Float64,
    "source": pl.String,
    "source_label": pl.String,
    "calendar_source": pl.String,
    "actual_source": pl.String,
    "estimate_source": pl.String,
    "accession_number": pl.String,
    "form": pl.String,
    "fiscal_period": pl.String,
    "fiscal_year": pl.Int64,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}

RAW_GENERAL_SCHEMA = {
    "ticker": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "cik": pl.String,
    "source": pl.String,
    "Sector": pl.String,
    "industry": pl.String,
    "sector_source": pl.String,
    "sector_raw_value": pl.String,
    "sic": pl.String,
    "sic_description": pl.String,
    "mapping_rule": pl.String,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}


def _with_price_ingestion_metadata(
    frame: pl.DataFrame,
    *,
    dataset: str,
    run_id: str,
    ingested_at: str,
    source: str = "yfinance",
) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_price_frame()
    metadata = {
        "source": source,
        "dataset": dataset,
        "ingestion_run_id": run_id,
        "ingested_at": ingested_at,
    }
    expressions = [
        pl.col(column) if column in frame.columns else pl.lit(value).alias(column)
        for column, value in metadata.items()
    ]
    return frame.with_columns(expressions).select(list(RAW_PRICE_SCHEMA))


def _with_financial_ingestion_metadata(frame: pl.DataFrame, *, dataset: str, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_FINANCIAL_SCHEMA)
    expressions: list[pl.Expr] = [
        pl.lit(dataset).alias("dataset"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ]
    for column, dtype in RAW_FINANCIAL_SCHEMA.items():
        if column in frame.columns or column in {"dataset", "ingestion_run_id", "ingested_at"}:
            continue
        expressions.append(pl.lit(None).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(RAW_FINANCIAL_SCHEMA))


def _with_earnings_ingestion_metadata(frame: pl.DataFrame, *, dataset: str, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_EARNINGS_SCHEMA)
    expressions: list[pl.Expr] = [
        pl.lit(dataset).alias("dataset"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ]
    for column, dtype in RAW_EARNINGS_SCHEMA.items():
        if column in frame.columns or column in {"dataset", "ingestion_run_id", "ingested_at"}:
            continue
        expressions.append(pl.lit(None).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(RAW_EARNINGS_SCHEMA))


def _with_general_ingestion_metadata(frame: pl.DataFrame, *, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_GENERAL_SCHEMA)
    return frame.with_columns(
        [
            pl.col("cik").cast(pl.Utf8, strict=False),
            pl.lit("general_reference").alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(RAW_GENERAL_SCHEMA))


def _with_general_lineage_ingestion_metadata(frame: pl.DataFrame, *, run_id: str, ingested_at: str) -> pl.DataFrame:
    schema = {column: pl.String for column in empty_general_reference_lineage_frame().columns}
    schema.update({"dataset": pl.String, "ingestion_run_id": pl.String, "ingested_at": pl.String})
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    return frame.with_columns(
        [
            pl.lit("general_reference_lineage").alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(schema))


def _concat_or_empty(frames: Sequence[pl.DataFrame], *, empty: pl.DataFrame | None = None) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return empty if empty is not None else _empty_raw_financial_base()
    return pl.concat(non_empty, how="vertical")


def _empty_raw_financial_base() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )


def _empty_raw_earnings_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RAW_EARNINGS_SCHEMA)


def _empty_raw_price_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RAW_PRICE_SCHEMA)


def _empty_sec_profile_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "cik": pl.String,
            "sic": pl.String,
            "sic_description": pl.String,
        }
    )


def _clean_financial_columns() -> list[str]:
    return ["ticker", "statement", "metric", "date", "filing_date", "value", "source", "source_label", "form", "fiscal_period", "fiscal_year"]


def _filter_financial_year(frame: pl.DataFrame, *, year: int) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_financial_base()
    return frame.filter(pl.col("date").str.starts_with(str(year)))


def _filter_financial_years(frame: pl.DataFrame, *, years: Sequence[int]) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_financial_base()
    year_values = tuple(str(year) for year in years)
    return frame.filter(pl.col("date").str.slice(0, 4).is_in(year_values))
