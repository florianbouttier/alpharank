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
    non_share_companyfacts = sec_companyfacts.filter(pl.col("metric") != "outstanding_shares")
    non_share_filing = sec_filing.filter(pl.col("metric") != "outstanding_shares")
    share_companyfacts = _sanitize_share_quality(sec_companyfacts.filter(pl.col("metric") == "outstanding_shares"))
    share_filing = _sanitize_share_quality(sec_filing.filter(pl.col("metric") == "outstanding_shares"))

    consolidated_parts: list[pl.DataFrame] = []
    lineage_parts: list[pl.DataFrame] = []

    non_share_consolidated, non_share_lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput(source_name="sec_companyfacts", frame=non_share_companyfacts, priority=1),
            FinancialSourceInput(source_name="sec_filing", frame=non_share_filing, priority=2),
        ]
    )
    if not non_share_consolidated.is_empty():
        consolidated_parts.append(non_share_consolidated)
        lineage_parts.append(non_share_lineage)

    share_consolidated, share_lineage, _ = consolidate_financial_sources(
        [
            FinancialSourceInput(source_name="sec_filing", frame=share_filing, priority=1),
            FinancialSourceInput(source_name="sec_companyfacts", frame=share_companyfacts, priority=2),
        ]
    )
    if not share_consolidated.is_empty():
        consolidated_parts.append(share_consolidated)
        lineage_parts.append(share_lineage)

    consolidated = (
        pl.concat(consolidated_parts, how="diagonal_relaxed").sort(["ticker", "statement", "metric", "date"])
        if consolidated_parts
        else pl.DataFrame(schema={"ticker": pl.String})
    )
    lineage = (
        pl.concat(lineage_parts, how="diagonal_relaxed").sort(["ticker", "statement", "metric", "date"])
        if lineage_parts
        else pl.DataFrame(schema={"ticker": pl.String})
    )
    consolidated = _canonicalize_financial_quarter_fields(consolidated)
    lineage = _canonicalize_financial_quarter_fields(lineage)
    consolidated, lineage = _finalize_share_quarters(consolidated, lineage)
    source_summary = _summarize_financial_sources(consolidated)
    return consolidated, lineage, source_summary


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
    return (
        _canonicalize_earnings_quarter_fields(consolidated),
        _canonicalize_earnings_quarter_fields(lineage),
        _canonicalize_earnings_quarter_fields(long_frame),
    )


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


def _canonicalize_financial_quarter_fields(frame: pl.DataFrame) -> pl.DataFrame:
    return _canonicalize_quarter_fields(
        frame,
        ticker_col="ticker",
        date_col="date",
        year_col="selected_fiscal_year",
        period_col="selected_fiscal_period",
    )


def _canonicalize_earnings_quarter_fields(frame: pl.DataFrame) -> pl.DataFrame:
    return _canonicalize_quarter_fields(
        frame,
        ticker_col="ticker",
        date_col="period_end",
        year_col="fiscal_year",
        period_col="fiscal_period",
    )


def _canonicalize_quarter_fields(
    frame: pl.DataFrame,
    *,
    ticker_col: str,
    date_col: str,
    year_col: str,
    period_col: str,
) -> pl.DataFrame:
    if frame.is_empty() or date_col not in frame.columns or year_col not in frame.columns or period_col not in frame.columns:
        return frame

    quarterly_dates = (
        frame.filter(
            pl.col(date_col).is_not_null()
            & pl.col(year_col).is_not_null()
            & pl.col(period_col).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().ne("FY")
        )
        .select(
            [
                pl.col(ticker_col),
                pl.col(date_col),
                pl.col(year_col).cast(pl.Int64, strict=False).alias(year_col),
            ]
        )
        .unique()
        .with_columns(pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_quarter_dt"))
        .filter(pl.col("_quarter_dt").is_not_null())
        .sort([ticker_col, year_col, "_quarter_dt"])
        .with_columns(
            [
                pl.col("_quarter_dt").rank("ordinal").over([ticker_col, year_col]).cast(pl.Int64).alias("_quarter_rank"),
                pl.len().over([ticker_col, year_col]).cast(pl.Int64).alias("_quarter_count"),
            ]
        )
        .with_columns(
            pl.when((pl.col("_quarter_count") >= 1) & (pl.col("_quarter_count") <= 4))
            .then(pl.concat_str([pl.lit("Q"), pl.col("_quarter_rank").cast(pl.Utf8)]))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("_canonical_fiscal_period")
        )
        .select([ticker_col, date_col, year_col, "_canonical_fiscal_period"])
    )
    if quarterly_dates.is_empty():
        return frame

    return (
        frame.join(
            quarterly_dates,
            on=[ticker_col, date_col, year_col],
            how="left",
        )
        .with_columns(
            pl.coalesce([pl.col("_canonical_fiscal_period"), pl.col(period_col).cast(pl.Utf8, strict=False)]).alias(period_col)
        )
        .drop("_canonical_fiscal_period")
    )


def _finalize_share_quarters(
    consolidated: pl.DataFrame,
    lineage: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if consolidated.is_empty():
        return consolidated, lineage

    share_period_keys = ["ticker", "selected_fiscal_year", "selected_fiscal_period"]
    consolidated_shares = consolidated.filter(pl.col("metric") == "outstanding_shares")
    if consolidated_shares.is_empty():
        return consolidated, lineage

    selected_share_rows = (
        consolidated_shares.filter(
            pl.col("selected_fiscal_year").is_not_null()
            & pl.col("selected_fiscal_period").is_not_null()
            & pl.col("value").is_not_null()
        )
        .with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("_date_dt"))
        .sort(
            [
                "ticker",
                "selected_fiscal_year",
                "selected_fiscal_period",
                "_date_dt",
                "source_priority",
                "filing_date",
            ],
            descending=[False, False, False, False, False, True],
        )
        .unique(subset=share_period_keys, keep="first", maintain_order=True)
        .select(share_period_keys + ["date"])
    )
    if selected_share_rows.is_empty():
        return consolidated, lineage

    filtered_shares = consolidated_shares.join(
        selected_share_rows,
        on=share_period_keys + ["date"],
        how="inner",
    )
    filtered_lineage_shares = lineage.filter(pl.col("metric") == "outstanding_shares").join(
        selected_share_rows,
        on=share_period_keys + ["date"],
        how="inner",
    )

    consolidated_out = pl.concat(
        [consolidated.filter(pl.col("metric") != "outstanding_shares"), filtered_shares],
        how="diagonal_relaxed",
    ).sort(["ticker", "statement", "metric", "date"])
    lineage_out = pl.concat(
        [lineage.filter(pl.col("metric") != "outstanding_shares"), filtered_lineage_shares],
        how="diagonal_relaxed",
    ).sort(["ticker", "statement", "metric", "date"])
    return consolidated_out, lineage_out


def _sanitize_share_quality(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame

    rows = frame.sort(["ticker", "date", "filing_date"]).to_dicts()
    by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_ticker.setdefault(str(row["ticker"]), []).append(row)

    cleaned_rows: list[dict[str, object]] = []
    for ticker_rows in by_ticker.values():
        values = [_safe_float(row.get("value")) for row in ticker_rows]
        for idx, row in enumerate(ticker_rows):
            value = values[idx]
            if value is None or value <= 0 or value > 1.0e11:
                continue
            anchor = _neighbor_share_anchor(values, idx)
            if anchor is not None:
                ratio = max(value, anchor) / min(value, anchor)
                if ratio >= 20.0:
                    continue
            cleaned_rows.append(row)

    if not cleaned_rows:
        return frame.head(0)
    return pl.DataFrame(cleaned_rows, schema=frame.schema).sort(["ticker", "date", "filing_date"])


def _neighbor_share_anchor(values: list[float | None], index: int) -> float | None:
    neighbors: list[float] = []
    if index > 0 and values[index - 1] is not None and values[index - 1] > 0:
        neighbors.append(values[index - 1])
    if index + 1 < len(values) and values[index + 1] is not None and values[index + 1] > 0:
        neighbors.append(values[index + 1])
    if len(neighbors) < 2:
        return None
    return sorted(neighbors)[len(neighbors) // 2]


def _safe_float(value: object) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    if numeric <= 0:
        return None
    return numeric


def _summarize_financial_sources(consolidated: pl.DataFrame) -> pl.DataFrame:
    if consolidated.is_empty():
        return pl.DataFrame(
            schema={
                "statement": pl.String,
                "selected_source": pl.String,
                "selected_rows": pl.Int64,
                "fallback_rows": pl.Int64,
                "ticker_count": pl.Int64,
                "metric_count": pl.Int64,
                "fallback_rate_pct": pl.Float64,
            }
        )
    return (
        consolidated.group_by(["statement", "selected_source"])
        .agg(
            [
                pl.len().alias("selected_rows"),
                pl.col("fallback_used").sum().cast(pl.Int64).alias("fallback_rows"),
                pl.col("ticker").n_unique().alias("ticker_count"),
                pl.col("metric").n_unique().alias("metric_count"),
            ]
        )
        .with_columns(
            pl.when(pl.col("selected_rows") > 0)
            .then((pl.col("fallback_rows") / pl.col("selected_rows")) * 100.0)
            .otherwise(0.0)
            .alias("fallback_rate_pct")
        )
        .sort(["statement", "selected_rows", "selected_source"], descending=[False, True, False])
    )
