from __future__ import annotations

from typing import Iterable

import polars as pl

from alpharank.data.open_source.consolidation import FinancialSourceInput, consolidate_financial_sources
from alpharank.data.open_source.earnings import consolidate_earnings
from alpharank.data.open_source.general_reference import (
    GENERAL_REFERENCE_LINEAGE_COLUMNS,
    build_general_reference,
    empty_general_reference_frame,
    empty_general_reference_lineage_frame,
)


def build_sec_only_general_reference(
    *,
    sec_mapping: pl.DataFrame,
    sec_profiles: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if sec_mapping.is_empty():
        empty = empty_general_reference_frame()
        return empty, empty_general_reference_lineage_frame()

    general_reference, lineage = build_general_reference(
        tickers=sec_mapping.get_column("ticker").cast(pl.Utf8).to_list(),
        sec_mapping=sec_mapping,
        yahoo_metadata=_empty_yahoo_metadata_frame(),
        sec_profiles=sec_profiles,
    )
    general_reference, lineage = _canonicalize_general_outputs(general_reference, lineage)
    if lineage.is_empty():
        return general_reference, lineage
    sanitized_lineage = lineage.with_columns(
        [
            pl.lit(None).cast(pl.Utf8).alias("yahoo_name"),
            pl.lit(None).cast(pl.Utf8).alias("yahoo_exchange"),
            pl.lit(None).cast(pl.Utf8).alias("yahoo_sector"),
            pl.lit(None).cast(pl.Utf8).alias("yahoo_industry"),
            pl.lit("sec_mapping").alias("selected_name_source"),
            pl.lit("sec_mapping").alias("selected_exchange_source"),
            pl.lit("sec_only_general").alias("source"),
        ]
    ).select(list(GENERAL_REFERENCE_LINEAGE_COLUMNS))
    return general_reference, sanitized_lineage


def build_sec_only_general_reference_from_raw_lineage(raw_lineage: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    if raw_lineage.is_empty():
        empty = empty_general_reference_frame()
        return empty, empty_general_reference_lineage_frame()

    latest = raw_lineage.sort(_general_lineage_sort_columns(raw_lineage)).unique(
        subset=["ticker"], keep="last", maintain_order=True
    )
    sec_mapping = (
        latest.select(
            [
                pl.col("ticker").str.replace(r"\.US$", "").alias("ticker"),
                pl.col("sec_name").cast(pl.Utf8).alias("name"),
                pl.col("sec_exchange").cast(pl.Utf8).alias("exchange"),
                pl.col("sec_cik").cast(pl.Int64, strict=False).alias("cik"),
            ]
        )
        .filter(pl.col("ticker").is_not_null() & pl.col("name").is_not_null() & pl.col("cik").is_not_null())
        .unique(subset=["ticker"], keep="last", maintain_order=True)
        .sort("ticker")
    )
    sec_profiles = (
        latest.select(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("sec_cik").cast(pl.Utf8).alias("cik"),
                pl.col("sec_sic").cast(pl.Utf8).alias("sic"),
                pl.col("sec_sic_description").cast(pl.Utf8).alias("sic_description"),
            ]
        )
        .unique(subset=["ticker"], keep="last", maintain_order=True)
        .sort("ticker")
    )
    return build_sec_only_general_reference(sec_mapping=sec_mapping, sec_profiles=sec_profiles)


def build_sec_only_financials(
    *,
    sec_companyfacts: pl.DataFrame,
    sec_filing: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return consolidate_financial_sources(
        [
            FinancialSourceInput(source_name="sec_companyfacts", frame=sec_companyfacts, priority=1),
            FinancialSourceInput(source_name="sec_filing", frame=sec_filing, priority=2),
        ]
    )


def build_sec_only_earnings(
    *,
    sec_calendar: pl.DataFrame,
    sec_actuals: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    consolidated, lineage, long_frame = consolidate_earnings(
        sec_calendar=_select_sec_calendar_columns(sec_calendar),
        yahoo_earnings=_empty_yahoo_earnings_frame(),
        sec_actuals=_select_sec_actual_columns(sec_actuals),
    )
    consolidated = consolidated.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("epsEstimate"),
            pl.lit(None).cast(pl.Float64).alias("surprisePercent"),
            pl.lit(None).cast(pl.Utf8).alias("estimate_source"),
            pl.lit(None).cast(pl.Utf8).alias("surprise_source"),
        ]
    )
    lineage = lineage.with_columns(
        [
            pl.lit(None).cast(pl.Float64).alias("yahoo_epsActual"),
            pl.lit(None).cast(pl.Float64).alias("yahoo_epsEstimate"),
            pl.lit(None).cast(pl.Float64).alias("yahoo_surprisePercent"),
            pl.lit(None).cast(pl.Utf8).alias("yahoo_reportDate"),
            pl.lit(None).cast(pl.Utf8).alias("yahoo_earningsDatetime"),
            pl.lit(None).cast(pl.Int64).alias("yahoo_match_diff_days"),
            pl.lit(None).cast(pl.Float64).alias("selected_epsEstimate"),
            pl.lit(None).cast(pl.Float64).alias("selected_surprisePercent"),
        ]
    )
    return consolidated, lineage, long_frame


def _empty_yahoo_metadata_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "name": pl.String,
            "exchange": pl.String,
            "sector_raw_value": pl.String,
            "industry": pl.String,
        }
    )


def _empty_yahoo_earnings_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "epsEstimate": pl.Float64,
            "epsActual": pl.Float64,
            "surprisePercent": pl.Float64,
            "source": pl.String,
        }
    )


def _select_sec_calendar_columns(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.select(
        [
            "ticker",
            "period_end",
            "reportDate",
            "earningsDatetime",
            "accession_number",
            "form",
            "fiscal_period",
            "fiscal_year",
            "source",
            "source_label",
        ]
    )


def _select_sec_actual_columns(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.select(
        [
            "ticker",
            "period_end",
            "reportDate",
            "epsActual",
            "source",
            "source_label",
            "form",
            "fiscal_period",
            "fiscal_year",
        ]
    )


def _general_lineage_sort_columns(frame: pl.DataFrame) -> list[str]:
    columns: list[str] = ["ticker"]
    if "ingested_at" in frame.columns:
        columns.append("ingested_at")
    return columns


def _canonicalize_general_outputs(
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lineage = general_reference_lineage
    if not lineage.is_empty():
        sort_cols = _general_lineage_sort_columns(lineage)
        lineage = lineage.sort(sort_cols).unique(subset=["ticker"], keep="last", maintain_order=True).sort("ticker")
        return lineage.select(general_reference.columns), lineage

    general = general_reference
    if general.is_empty():
        return general, lineage
    sort_cols = _general_lineage_sort_columns(general)
    general = general.sort(sort_cols).unique(subset=["ticker"], keep="last", maintain_order=True).sort("ticker")
    return general, lineage
