from __future__ import annotations

from collections import Counter
from datetime import date as dt_date
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
    for metric_name in ("revenue", "net_income"):
        consolidated, lineage = _finalize_metric_quarters(consolidated, lineage, metric=metric_name)
    consolidated, lineage = _finalize_share_quarters(consolidated, lineage)
    source_summary = _summarize_financial_sources(consolidated)
    return consolidated, lineage, source_summary


def build_sec_only_earnings(
    *,
    sec_calendar: pl.DataFrame,
    sec_actuals: pl.DataFrame,
    sec_financials: pl.DataFrame | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if sec_financials is not None and not sec_financials.is_empty():
        derived_actuals = _build_sec_derived_eps_actuals(sec_financials)
        if not derived_actuals.is_empty():
            sec_actuals = pl.concat([sec_actuals, derived_actuals], how="diagonal_relaxed")
    sec_calendar = _augment_sec_calendar_with_actuals(
        sec_calendar=_select_sec_calendar_columns(sec_calendar),
        sec_actuals=_select_sec_actual_columns(sec_actuals),
    )
    consolidated, lineage, long_frame = consolidate_earnings(
        sec_calendar=sec_calendar,
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


def _build_sec_derived_eps_actuals(sec_financials: pl.DataFrame) -> pl.DataFrame:
    if sec_financials.is_empty():
        return pl.DataFrame(schema=_select_sec_actual_columns(empty_earnings_actuals_frame()).schema)

    quarterly_financials = sec_financials.filter(
        pl.col("selected_fiscal_year").is_not_null()
        & pl.col("selected_fiscal_period").is_in(["Q1", "Q2", "Q3", "Q4"])
    )
    if quarterly_financials.is_empty():
        return pl.DataFrame(schema=_select_sec_actual_columns(empty_earnings_actuals_frame()).schema)

    net_income = quarterly_financials.filter(pl.col("metric") == "net_income").select(
        [
            "ticker",
            pl.col("date").alias("period_end"),
            pl.col("filing_date").alias("net_income_reportDate"),
            pl.col("value").alias("net_income_value"),
            pl.col("selected_fiscal_year").alias("fiscal_year"),
            pl.col("selected_fiscal_period").alias("fiscal_period"),
            pl.col("selected_source").alias("net_income_source"),
            pl.col("selected_source_label").alias("net_income_source_label"),
            pl.col("selected_form").alias("form"),
        ]
    )
    shares = quarterly_financials.filter(pl.col("metric") == "outstanding_shares").select(
        [
            "ticker",
            pl.col("date").alias("share_period_end"),
            pl.col("filing_date").alias("share_reportDate"),
            pl.col("value").alias("share_value"),
            pl.col("selected_fiscal_year").alias("fiscal_year"),
            pl.col("selected_fiscal_period").alias("fiscal_period"),
            pl.col("selected_source").alias("share_source"),
            pl.col("selected_source_label").alias("share_source_label"),
        ]
    )
    weighted_shares = quarterly_financials.filter(pl.col("metric") == "weighted_average_diluted_shares").select(
        [
            "ticker",
            pl.col("date").alias("share_period_end"),
            pl.col("filing_date").alias("share_reportDate"),
            pl.col("value").alias("share_value"),
            pl.col("selected_fiscal_year").alias("fiscal_year"),
            pl.col("selected_fiscal_period").alias("fiscal_period"),
            pl.col("selected_source").alias("share_source"),
            pl.col("selected_source_label").alias("share_source_label"),
        ]
    )
    share_sources: list[pl.DataFrame] = []
    if not shares.is_empty():
        share_sources.append(shares)
    if not weighted_shares.is_empty():
        share_sources.append(weighted_shares)
    if net_income.is_empty() or not share_sources:
        return pl.DataFrame(schema=_select_sec_actual_columns(empty_earnings_actuals_frame()).schema)
    shares = (
        pl.concat(share_sources, how="diagonal_relaxed")
        .with_columns(
            pl.when(pl.col("share_source") == "sec_companyfacts")
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_share_priority")
        )
        .sort(
            ["ticker", "fiscal_year", "fiscal_period", "_share_priority", "share_reportDate", "share_period_end"],
            descending=[False, False, False, False, True, True],
        )
        .unique(subset=["ticker", "fiscal_year", "fiscal_period"], keep="first", maintain_order=True)
        .drop("_share_priority")
    )

    derived = (
        net_income.join(shares, on=["ticker", "fiscal_year", "fiscal_period"], how="inner")
        .with_columns(
            [
                pl.when(pl.col("share_value") > 0)
                .then(pl.col("net_income_value") / pl.col("share_value"))
                .otherwise(pl.lit(None).cast(pl.Float64))
                .alias("epsActual"),
                pl.coalesce([pl.col("net_income_reportDate"), pl.col("share_reportDate")]).alias("reportDate"),
                pl.coalesce([pl.col("period_end"), pl.col("share_period_end")]).alias("period_end"),
            ]
        )
        .filter(
            pl.col("epsActual").is_not_null()
            & pl.col("epsActual").is_finite()
            & (pl.col("epsActual").abs() <= 1_000.0)
            & pl.col("reportDate").is_not_null()
            & pl.col("period_end").is_not_null()
        )
        .select(
            [
                "ticker",
                "period_end",
                "reportDate",
                "epsActual",
                pl.lit("sec_derived_eps").alias("source"),
                pl.concat_str(
                    [
                        pl.lit("derived_from_net_income_and_shares"),
                        pl.col("net_income_source"),
                        pl.col("share_source"),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("source_label"),
                "form",
                "fiscal_period",
                "fiscal_year",
            ]
        )
        .sort(["ticker", "period_end", "reportDate"])
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
    )
    return _select_sec_actual_columns(derived)


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


def _augment_sec_calendar_with_actuals(*, sec_calendar: pl.DataFrame, sec_actuals: pl.DataFrame) -> pl.DataFrame:
    if sec_calendar.is_empty() or sec_actuals.is_empty():
        return sec_calendar

    missing_periods = (
        sec_actuals.select(["ticker", "period_end", "reportDate", "form", "fiscal_period", "fiscal_year"])
        .join(sec_calendar.select(["ticker", "period_end"]).unique(), on=["ticker", "period_end"], how="anti")
        .with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("earningsDatetime"),
                pl.lit(None).cast(pl.Utf8).alias("accession_number"),
                pl.lit("sec_actuals_backfill").alias("source"),
                pl.lit("period_from_sec_actuals").alias("source_label"),
            ]
        )
        .select(sec_calendar.columns)
    )
    if missing_periods.is_empty():
        return sec_calendar

    return (
        pl.concat([sec_calendar, missing_periods], how="diagonal_relaxed")
        .sort(["ticker", "period_end", "reportDate", "source"])
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
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
            & pl.col(period_col).cast(pl.Utf8, strict=False).fill_null("").str.strip_chars().ne("FY")
        )
        .select(
            [
                pl.col(ticker_col),
                pl.col(date_col),
                pl.col(year_col).cast(pl.Int64, strict=False).alias(year_col),
                pl.col(period_col).cast(pl.Utf8, strict=False).alias(period_col),
            ]
        )
        .unique()
        .with_columns(pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_quarter_dt"))
        .filter(pl.col("_quarter_dt").is_not_null())
        .sort([ticker_col, "_quarter_dt"])
    )
    if quarterly_dates.is_empty():
        return frame

    canonical_rows = _build_canonical_quarter_rows(
        quarterly_dates=quarterly_dates,
        ticker_col=ticker_col,
        date_col=date_col,
        year_col=year_col,
        period_col=period_col,
    )
    if canonical_rows.is_empty():
        return frame

    return (
        frame.join(
            canonical_rows,
            on=[ticker_col, date_col],
            how="left",
        )
        .with_columns(
            [
                pl.coalesce([pl.col("_canonical_fiscal_period"), pl.col(period_col).cast(pl.Utf8, strict=False)]).alias(period_col),
                pl.coalesce([pl.col("_canonical_fiscal_year"), pl.col(year_col).cast(pl.Int64, strict=False)]).alias(year_col),
            ]
        )
        .drop(["_canonical_fiscal_period", "_canonical_fiscal_year"])
    )


def _build_canonical_quarter_rows(
    *,
    quarterly_dates: pl.DataFrame,
    ticker_col: str,
    date_col: str,
    year_col: str,
    period_col: str,
) -> pl.DataFrame:
    rows = quarterly_dates.select([ticker_col, date_col, "_quarter_dt", year_col, period_col]).to_dicts()
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row[ticker_col]), []).append(row)

    canonical_rows: list[dict[str, object]] = []
    for ticker, ticker_rows in grouped.items():
        ordered = sorted(ticker_rows, key=lambda row: row["_quarter_dt"])
        canonical_rows.extend(
            _canonicalize_ticker_quarters(
                ticker=ticker,
                ticker_rows=ordered,
                ticker_col=ticker_col,
                date_col=date_col,
                year_col=year_col,
                period_col=period_col,
            )
        )
    return pl.DataFrame(canonical_rows)


def _canonicalize_ticker_quarters(
    *,
    ticker: str,
    ticker_rows: list[dict[str, object]],
    ticker_col: str,
    date_col: str,
    year_col: str,
    period_col: str,
) -> list[dict[str, object]]:
    if not ticker_rows:
        return []

    anchor_index = None
    anchor_year = None
    anchor_quarter = None
    for index in range(len(ticker_rows) - 1, -1, -1):
        row = ticker_rows[index]
        source_period = _quarter_number(row.get(period_col))
        source_year = _safe_int(row.get(year_col))
        if source_period is not None and source_year is not None:
            anchor_index = index
            anchor_year = source_year
            anchor_quarter = source_period
            break

    if anchor_index is None:
        anchor_index = len(ticker_rows) - 1
        anchor_date = ticker_rows[anchor_index]["_quarter_dt"]
        assert isinstance(anchor_date, dt_date)
        anchor_year = anchor_date.year
        anchor_quarter = ((anchor_date.month - 1) // 3) + 1

    canonical: list[tuple[int, int] | None] = [None] * len(ticker_rows)
    canonical[anchor_index] = (anchor_year, anchor_quarter)

    for index in range(anchor_index + 1, len(ticker_rows)):
        prev = canonical[index - 1]
        prev_date = ticker_rows[index - 1]["_quarter_dt"]
        current_date = ticker_rows[index]["_quarter_dt"]
        if prev is None or not isinstance(prev_date, dt_date) or not isinstance(current_date, dt_date):
            continue
        steps = _quarter_step_count(prev_date, current_date)
        canonical[index] = _shift_quarter(prev[0], prev[1], steps)

    for index in range(anchor_index - 1, -1, -1):
        nxt = canonical[index + 1]
        next_date = ticker_rows[index + 1]["_quarter_dt"]
        current_date = ticker_rows[index]["_quarter_dt"]
        if nxt is None or not isinstance(next_date, dt_date) or not isinstance(current_date, dt_date):
            continue
        steps = _quarter_step_count(current_date, next_date)
        canonical[index] = _shift_quarter(nxt[0], nxt[1], -steps)

    result: list[dict[str, object]] = []
    q4_month = _preferred_q4_month(ticker_rows=ticker_rows, canonical=canonical, period_col=period_col)
    fiscal_year_mode = _infer_fiscal_year_mode(
        ticker_rows=ticker_rows,
        canonical=canonical,
        year_col=year_col,
        q4_month=q4_month,
    )
    for row, canonical_value in zip(ticker_rows, canonical, strict=True):
        fallback_year, canonical_quarter = canonical_value if canonical_value is not None else (None, None)
        source_quarter = _quarter_number(row.get(period_col))
        effective_quarter = source_quarter or canonical_quarter
        quarter_dt = row["_quarter_dt"]
        canonical_year = fallback_year
        if effective_quarter is not None and q4_month is not None and isinstance(quarter_dt, dt_date):
            canonical_year = _infer_fiscal_year_from_date(
                quarter_dt=quarter_dt,
                q4_month=q4_month,
                mode=fiscal_year_mode,
            )
        result.append(
            {
                ticker_col: ticker,
                date_col: row[date_col],
                "_canonical_fiscal_year": canonical_year,
                "_canonical_fiscal_period": f"Q{effective_quarter}" if effective_quarter is not None else None,
            }
        )
    return result


def _quarter_number(value: object) -> int | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if text in {"Q1", "Q2", "Q3", "Q4"}:
        return int(text[1])
    return None


def _safe_int(value: object) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _quarter_step_count(start: dt_date, end: dt_date) -> int:
    days = max((end - start).days, 0)
    if days < 45:
        return 0
    return min(8, int((days + 45) // 91))


def _shift_quarter(year: int, quarter: int, step_count: int) -> tuple[int, int]:
    absolute = (year * 4) + (quarter - 1) + step_count
    shifted_year, shifted_zero_based_quarter = divmod(absolute, 4)
    return shifted_year, shifted_zero_based_quarter + 1


def _modal_q4_month(*, ticker_rows: list[dict[str, object]], canonical: list[tuple[int, int] | None]) -> int | None:
    months = [
        row["_quarter_dt"].month
        for row, canonical_value in zip(ticker_rows, canonical, strict=True)
        if canonical_value is not None
        and canonical_value[1] == 4
        and isinstance(row["_quarter_dt"], dt_date)
    ]
    if not months:
        return None
    return Counter(months).most_common(1)[0][0]


def _preferred_q4_month(
    *,
    ticker_rows: list[dict[str, object]],
    canonical: list[tuple[int, int] | None],
    period_col: str,
) -> int | None:
    source_months = [
        row["_quarter_dt"].month
        for row in ticker_rows
        if _quarter_number(row.get(period_col)) == 4 and isinstance(row.get("_quarter_dt"), dt_date)
    ]
    if source_months:
        return Counter(source_months).most_common(1)[0][0]
    return _modal_q4_month(ticker_rows=ticker_rows, canonical=canonical)


def _infer_fiscal_year_mode(
    *,
    ticker_rows: list[dict[str, object]],
    canonical: list[tuple[int, int] | None],
    year_col: str,
    q4_month: int | None,
) -> str:
    if q4_month is None or q4_month not in {1, 2}:
        return "end_year"

    candidate_modes = ("end_year", "bridge_year")
    best_mode = "end_year"
    best_score: tuple[int, int] | None = None
    for mode in candidate_modes:
        mismatches = 0
        comparisons = 0
        for row, canonical_value in zip(ticker_rows, canonical, strict=True):
            source_year = _safe_int(row.get(year_col))
            quarter_dt = row.get("_quarter_dt")
            if source_year is None or not isinstance(quarter_dt, dt_date) or canonical_value is None:
                continue
            inferred_year = _infer_fiscal_year_from_date(quarter_dt=quarter_dt, q4_month=q4_month, mode=mode)
            comparisons += 1
            if inferred_year != source_year:
                mismatches += 1
        score = (mismatches, -comparisons)
        if best_score is None or score < best_score:
            best_mode = mode
            best_score = score
    return best_mode


def _infer_fiscal_year_from_date(*, quarter_dt: dt_date, q4_month: int, mode: str) -> int:
    if q4_month == 12:
        return quarter_dt.year
    if mode == "bridge_year" and q4_month in {1, 2}:
        return quarter_dt.year if quarter_dt.month > q4_month else quarter_dt.year - 1
    return quarter_dt.year + 1 if quarter_dt.month > q4_month else quarter_dt.year


def _finalize_metric_quarters(
    consolidated: pl.DataFrame,
    lineage: pl.DataFrame,
    *,
    metric: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if consolidated.is_empty():
        return consolidated, lineage

    period_keys = ["ticker", "selected_fiscal_year", "selected_fiscal_period", "metric"]
    metric_rows = consolidated.filter(pl.col("metric") == metric)
    if metric_rows.is_empty():
        return consolidated, lineage

    selected_rows = (
        metric_rows.filter(
            pl.col("selected_fiscal_year").is_not_null()
            & pl.col("selected_fiscal_period").is_not_null()
            & pl.col("value").is_not_null()
        )
        .with_columns(pl.col("date").str.strptime(pl.Date, strict=False).alias("_date_dt"))
        .sort(
            [
                "ticker",
                "metric",
                "selected_fiscal_year",
                "selected_fiscal_period",
                "source_priority",
                "filing_date",
                "_date_dt",
            ],
            descending=[False, False, False, False, False, True, True],
        )
        .unique(subset=period_keys, keep="first", maintain_order=True)
        .select(period_keys + ["date"])
    )
    if selected_rows.is_empty():
        return consolidated, lineage

    filtered_metric = metric_rows.join(selected_rows, on=period_keys + ["date"], how="inner")
    filtered_lineage_metric = lineage.filter(pl.col("metric") == metric).join(
        selected_rows,
        on=period_keys + ["date"],
        how="inner",
    )
    filtered_metric = filtered_metric.unique(subset=period_keys + ["date"], keep="first", maintain_order=True)
    filtered_lineage_metric = filtered_lineage_metric.unique(
        subset=period_keys + ["date"],
        keep="first",
        maintain_order=True,
    )

    consolidated_out = pl.concat(
        [consolidated.filter(pl.col("metric") != metric), filtered_metric],
        how="diagonal_relaxed",
    ).sort(["ticker", "statement", "metric", "date"])
    lineage_out = pl.concat(
        [lineage.filter(pl.col("metric") != metric), filtered_lineage_metric],
        how="diagonal_relaxed",
    ).sort(["ticker", "statement", "metric", "date"])
    return consolidated_out, lineage_out


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
    filtered_shares = filtered_shares.unique(
        subset=share_period_keys + ["date"],
        keep="first",
        maintain_order=True,
    )
    filtered_lineage_shares = filtered_lineage_shares.unique(
        subset=share_period_keys + ["date"],
        keep="first",
        maintain_order=True,
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
