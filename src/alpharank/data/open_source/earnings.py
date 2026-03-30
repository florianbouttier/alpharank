from __future__ import annotations

from typing import Iterable

import polars as pl

from alpharank.data.open_source.sec import _select_best_facts


SEC_EPS_TAGS: tuple[str, ...] = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "EarningsPerShareBasic",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
    "IncomeLossFromContinuingOperationsPerBasicShare",
)


def empty_earnings_calendar_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
            "source": pl.String,
            "source_label": pl.String,
        }
    )


def empty_earnings_actuals_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "epsActual": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )


def empty_earnings_consolidated_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "epsActual": pl.Float64,
            "epsEstimate": pl.Float64,
            "surprisePercent": pl.Float64,
            "selected_source": pl.String,
            "candidate_sources": pl.String,
            "calendar_source": pl.String,
            "actual_source": pl.String,
            "estimate_source": pl.String,
            "surprise_source": pl.String,
            "source_label": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
        }
    )


def empty_earnings_lineage_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
            "candidate_sources": pl.String,
            "calendar_source": pl.String,
            "actual_source": pl.String,
            "estimate_source": pl.String,
            "surprise_source": pl.String,
            "selected_source": pl.String,
            "source_label": pl.String,
            "yahoo_reportDate": pl.String,
            "yahoo_earningsDatetime": pl.String,
            "yahoo_match_diff_days": pl.Int64,
            "yahoo_epsActual": pl.Float64,
            "yahoo_epsEstimate": pl.Float64,
            "yahoo_surprisePercent": pl.Float64,
            "sec_epsActual": pl.Float64,
            "selected_epsActual": pl.Float64,
            "selected_epsEstimate": pl.Float64,
            "selected_surprisePercent": pl.Float64,
        }
    )


def build_sec_companyfacts_earnings_actuals(*, ticker: str, facts_payload: dict[str, object]) -> pl.DataFrame:
    selected = _select_best_facts("income_statement", ("us-gaap",), SEC_EPS_TAGS, facts_payload.get("facts", {}))  # type: ignore[arg-type]
    rows: list[dict[str, object]] = []
    for fact in selected:
        value = fact.get("val")
        end = fact.get("end")
        filed = fact.get("filed")
        if value is None or end is None or filed is None:
            continue
        rows.append(
            {
                "ticker": f"{ticker}.US",
                "period_end": str(end),
                "reportDate": str(filed),
                "epsActual": float(value),
                "source": "sec_companyfacts",
                "source_label": str(fact.get("tag") or "sec_eps_actual"),
                "form": str(fact.get("form") or ""),
                "fiscal_period": str(fact.get("fp") or ""),
                "fiscal_year": int(fact.get("fy")) if fact.get("fy") is not None else None,
            }
        )
    if not rows:
        return empty_earnings_actuals_frame()
    return (
        pl.DataFrame(rows)
        .sort(["ticker", "period_end", "reportDate", "source_label"])
        .unique(subset=["ticker", "period_end"], keep="last", maintain_order=True)
        .sort(["ticker", "period_end"])
    )


def consolidate_earnings(
    *,
    sec_calendar: pl.DataFrame,
    yahoo_earnings: pl.DataFrame,
    sec_actuals: pl.DataFrame,
    match_tolerance_days: int = 21,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    if sec_calendar.is_empty():
        empty = empty_earnings_consolidated_frame()
        return empty, empty_earnings_lineage_frame(), empty_earnings_long_frame()

    calendar = (
        sec_calendar.sort(["ticker", "period_end", "reportDate", "accession_number"])
        .unique(subset=["ticker", "period_end"], keep="last", maintain_order=True)
        .sort(["ticker", "period_end"])
    )
    yahoo_matches = _match_yahoo_to_sec_calendar(sec_calendar=calendar, yahoo_earnings=yahoo_earnings, tolerance_days=match_tolerance_days)
    sec_actual = (
        sec_actuals.rename(
            {
                "reportDate": "sec_reportDate",
                "epsActual": "sec_epsActual",
                "source": "sec_source",
                "source_label": "sec_source_label",
                "form": "sec_form",
                "fiscal_period": "sec_fiscal_period",
                "fiscal_year": "sec_fiscal_year",
            }
        )
        .sort(["ticker", "period_end", "sec_reportDate"])
        .unique(subset=["ticker", "period_end"], keep="last", maintain_order=True)
        .sort(["ticker", "period_end"])
    )

    joined = calendar.join(yahoo_matches, on=["ticker", "period_end"], how="left", coalesce=True)
    joined = joined.join(sec_actual, on=["ticker", "period_end"], how="left", coalesce=True, suffix="_sec")

    consolidated = (
        joined.with_columns(
            [
                pl.coalesce([pl.col("yahoo_earningsDatetime"), pl.col("earningsDatetime")]).alias("earningsDatetime"),
                pl.coalesce([pl.col("yahoo_epsActual"), pl.col("sec_epsActual")]).alias("epsActual"),
                pl.col("yahoo_epsEstimate").alias("epsEstimate"),
                pl.col("yahoo_surprisePercent").alias("surprisePercent"),
                pl.lit("sec_submissions").alias("calendar_source"),
                pl.when(pl.col("yahoo_epsActual").is_not_null())
                .then(pl.lit("yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.lit("sec_companyfacts"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("actual_source"),
                pl.when(pl.col("yahoo_epsEstimate").is_not_null())
                .then(pl.lit("yfinance"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("estimate_source"),
                pl.when(pl.col("yahoo_surprisePercent").is_not_null())
                .then(pl.lit("yfinance"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("surprise_source"),
            ]
        )
        .with_columns(
            [
                pl.concat_str(
                    [
                        pl.lit("sec_submissions"),
                        pl.when(pl.col("yahoo_reportDate").is_not_null()).then(pl.lit("yfinance")).otherwise(pl.lit(None).cast(pl.Utf8)),
                        pl.when(pl.col("sec_epsActual").is_not_null()).then(pl.lit("sec_companyfacts")).otherwise(pl.lit(None).cast(pl.Utf8)),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("candidate_sources"),
                pl.when(pl.col("yahoo_reportDate").is_not_null())
                .then(pl.lit("sec_submissions+yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.lit("sec_submissions+sec_companyfacts"))
                .otherwise(pl.lit("sec_submissions"))
                .alias("selected_source"),
                pl.concat_str(
                    [
                        pl.lit("calendar=sec_submissions"),
                        pl.when(pl.col("actual_source").is_not_null())
                        .then(pl.concat_str([pl.lit("actual="), pl.col("actual_source")], separator=""))
                        .otherwise(pl.lit(None).cast(pl.Utf8)),
                        pl.when(pl.col("estimate_source").is_not_null())
                        .then(pl.concat_str([pl.lit("estimate="), pl.col("estimate_source")], separator=""))
                        .otherwise(pl.lit(None).cast(pl.Utf8)),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("source_label"),
            ]
        )
        .select(empty_earnings_consolidated_frame().columns)
        .sort(["ticker", "period_end"])
    )

    lineage = (
        joined.with_columns(
            [
                pl.coalesce([pl.col("yahoo_earningsDatetime"), pl.col("earningsDatetime")]).alias("earningsDatetime"),
                pl.coalesce([pl.col("yahoo_epsActual"), pl.col("sec_epsActual")]).alias("selected_epsActual"),
                pl.col("yahoo_epsEstimate").alias("selected_epsEstimate"),
                pl.col("yahoo_surprisePercent").alias("selected_surprisePercent"),
                pl.lit("sec_submissions").alias("calendar_source"),
                pl.when(pl.col("yahoo_epsActual").is_not_null())
                .then(pl.lit("yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.lit("sec_companyfacts"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("actual_source"),
                pl.when(pl.col("yahoo_epsEstimate").is_not_null())
                .then(pl.lit("yfinance"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("estimate_source"),
                pl.when(pl.col("yahoo_surprisePercent").is_not_null())
                .then(pl.lit("yfinance"))
                .otherwise(pl.lit(None).cast(pl.Utf8))
                .alias("surprise_source"),
                pl.concat_str(
                    [
                        pl.lit("sec_submissions"),
                        pl.when(pl.col("yahoo_reportDate").is_not_null()).then(pl.lit("yfinance")).otherwise(pl.lit(None).cast(pl.Utf8)),
                        pl.when(pl.col("sec_epsActual").is_not_null()).then(pl.lit("sec_companyfacts")).otherwise(pl.lit(None).cast(pl.Utf8)),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("candidate_sources"),
                pl.when(pl.col("yahoo_reportDate").is_not_null())
                .then(pl.lit("sec_submissions+yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.lit("sec_submissions+sec_companyfacts"))
                .otherwise(pl.lit("sec_submissions"))
                .alias("selected_source"),
            ]
        )
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("calendar=sec_submissions"),
                    pl.when(pl.col("actual_source").is_not_null())
                    .then(pl.concat_str([pl.lit("actual="), pl.col("actual_source")], separator=""))
                    .otherwise(pl.lit(None).cast(pl.Utf8)),
                    pl.when(pl.col("estimate_source").is_not_null())
                    .then(pl.concat_str([pl.lit("estimate="), pl.col("estimate_source")], separator=""))
                    .otherwise(pl.lit(None).cast(pl.Utf8)),
                ],
                separator=" | ",
                ignore_nulls=True,
            ).alias("source_label")
        )
        .select(
            [
                "ticker",
                "period_end",
                "reportDate",
                "earningsDatetime",
                "accession_number",
                "form",
                "fiscal_period",
                "fiscal_year",
                "candidate_sources",
                "calendar_source",
                "actual_source",
                "estimate_source",
                "surprise_source",
                "selected_source",
                "source_label",
                "yahoo_reportDate",
                "yahoo_earningsDatetime",
                "yahoo_match_diff_days",
                "yahoo_epsActual",
                "yahoo_epsEstimate",
                "yahoo_surprisePercent",
                "sec_epsActual",
                "selected_epsActual",
                "selected_epsEstimate",
                "selected_surprisePercent",
            ]
        )
        .sort(["ticker", "period_end"])
    )

    long_frame = earnings_to_long_frame(consolidated)
    return consolidated, lineage, long_frame


def earnings_to_long_frame(consolidated: pl.DataFrame) -> pl.DataFrame:
    if consolidated.is_empty():
        return empty_earnings_long_frame()

    metric_specs = (
        ("eps_actual", "epsActual", "actual_source"),
        ("eps_estimate", "epsEstimate", "estimate_source"),
        ("surprise_percent", "surprisePercent", "surprise_source"),
    )
    frames: list[pl.DataFrame] = []
    for metric_name, value_column, source_column in metric_specs:
        frames.append(
            consolidated.select(
                [
                    pl.col("ticker"),
                    pl.lit("earnings").alias("statement"),
                    pl.lit(metric_name).alias("metric"),
                    pl.col("period_end").alias("date"),
                    pl.col("reportDate").alias("filing_date"),
                    pl.col(value_column).cast(pl.Float64, strict=False).alias("value"),
                    pl.lit("open_source_earnings").alias("source"),
                    pl.coalesce([pl.col(source_column), pl.lit("unknown")]).alias("source_label"),
                ]
            ).filter(pl.col("value").is_not_null())
        )
    return pl.concat(frames, how="vertical").sort(["ticker", "metric", "date"])


def empty_earnings_long_frame() -> pl.DataFrame:
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
        }
    )


def _match_yahoo_to_sec_calendar(*, sec_calendar: pl.DataFrame, yahoo_earnings: pl.DataFrame, tolerance_days: int) -> pl.DataFrame:
    if sec_calendar.is_empty() or yahoo_earnings.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "period_end": pl.String,
                "yahoo_reportDate": pl.String,
                "yahoo_earningsDatetime": pl.String,
                "yahoo_epsEstimate": pl.Float64,
                "yahoo_epsActual": pl.Float64,
                "yahoo_surprisePercent": pl.Float64,
                "yahoo_match_diff_days": pl.Int64,
            }
        )

    calendar = sec_calendar.select(["ticker", "period_end", "reportDate"]).with_row_index("calendar_row_id").with_columns(
        pl.col("reportDate").cast(pl.Date, strict=False).alias("report_date_dt")
    )
    yahoo = (
        yahoo_earnings.select(["ticker", "reportDate", "earningsDatetime", "epsEstimate", "epsActual", "surprisePercent"])
        .rename(
            {
                "reportDate": "yahoo_reportDate",
                "earningsDatetime": "yahoo_earningsDatetime",
                "epsEstimate": "yahoo_epsEstimate",
                "epsActual": "yahoo_epsActual",
                "surprisePercent": "yahoo_surprisePercent",
            }
        )
        .with_row_index("yahoo_row_id")
        .with_columns(pl.col("yahoo_reportDate").cast(pl.Date, strict=False).alias("yahoo_report_date_dt"))
    )
    candidates = (
        calendar.join(yahoo, on="ticker", how="inner")
        .with_columns((pl.col("yahoo_report_date_dt") - pl.col("report_date_dt")).dt.total_days().alias("date_diff_days"))
        .with_columns(pl.col("date_diff_days").abs().alias("abs_date_diff_days"))
        .filter(pl.col("abs_date_diff_days") <= tolerance_days)
        .sort(["calendar_row_id", "abs_date_diff_days", "yahoo_row_id"])
    )

    matched_rows: list[dict[str, object]] = []
    used_yahoo_rows: set[int] = set()
    for row in candidates.iter_rows(named=True):
        yahoo_row_id = int(row["yahoo_row_id"])
        if yahoo_row_id in used_yahoo_rows:
            continue
        used_yahoo_rows.add(yahoo_row_id)
        matched_rows.append(
            {
                "ticker": row["ticker"],
                "period_end": row["period_end"],
                "yahoo_reportDate": row["yahoo_reportDate"],
                "yahoo_earningsDatetime": row["yahoo_earningsDatetime"],
                "yahoo_epsEstimate": row["yahoo_epsEstimate"],
                "yahoo_epsActual": row["yahoo_epsActual"],
                "yahoo_surprisePercent": row["yahoo_surprisePercent"],
                "yahoo_match_diff_days": row["date_diff_days"],
            }
        )
    return (
        pl.DataFrame(matched_rows)
        .sort(["ticker", "period_end"])
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
        if matched_rows
        else pl.DataFrame(
            schema={
                "ticker": pl.String,
                "period_end": pl.String,
                "yahoo_reportDate": pl.String,
                "yahoo_earningsDatetime": pl.String,
                "yahoo_epsEstimate": pl.Float64,
                "yahoo_epsActual": pl.Float64,
                "yahoo_surprisePercent": pl.Float64,
                "yahoo_match_diff_days": pl.Int64,
            }
        )
    )
