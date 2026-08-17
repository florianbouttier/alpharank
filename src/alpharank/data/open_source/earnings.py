from __future__ import annotations

from typing import Iterable

import polars as pl

from alpharank.data.open_source.sec import _clean_fact, _extract_unit_records, _select_best_facts


SEC_EPS_TAGS: tuple[str, ...] = (
    "EarningsPerShareDiluted",
    "EarningsPerShareBasicAndDiluted",
    "EarningsPerShareBasic",
    "IncomeLossFromContinuingOperationsPerDilutedShare",
    "IncomeLossFromContinuingOperationsPerBasicShare",
    "IncomeLossFromContinuingOperationsPerBasicAndDilutedShare",
    "IncomeLossFromContinuingOperationsBasicAndDilutedNetOfTaxPerShare",
    "NetIncomeLossAvailableToCommonStockholdersBasicAndDilutedPerShare",
    "NetIncomeLossAvailableToCommonStockholdersPerShareDiluted",
    "NetIncomeLossAvailableToCommonStockholdersPerShareBasicAndDiluted",
    "NetIncomeLossPerOutstandingLimitedPartnershipUnit",
    "IncomeLossFromContinuingOperationsPerOutstandingLimitedPartnershipUnitBasicNetOfTax",
    "NetIncomeLossPerOutstandingLimitedPartnershipUnitBasicNetOfTax",
    "NetIncomeLossPerOutstandingLimitedPartnershipUnitDiluted",
    "IncomeLossFromContinuingOperationsPerOutstandingLimitedPartnershipUnitDilutedNetOfTax",
    "NetIncomeLossNetOfTaxPerOutstandingLimitedPartnershipUnitDiluted",
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
            "sec_reportDate": pl.String,
            "earningsDatetime": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
            "calendar_duplicate_count": pl.UInt32,
            "calendar_candidate_accessions": pl.List(pl.String),
            "calendar_resolution_rule": pl.String,
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


def resolve_earnings_calendar_duplicates(
    sec_calendar: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Enforce the canonical calendar key and retain every duplicate decision."""

    if sec_calendar.is_empty():
        audit = pl.DataFrame(
            schema={
                "ticker": pl.String,
                "period_end": pl.String,
                "calendar_duplicate_count": pl.UInt32,
                "calendar_candidate_accessions": pl.List(pl.String),
                "calendar_resolution_rule": pl.String,
            }
        )
        return sec_calendar, audit
    required = {"ticker", "period_end", "reportDate", "accession_number"}
    missing = sorted(required - set(sec_calendar.columns))
    if missing:
        raise ValueError("SEC earnings calendar is missing: " + ", ".join(missing))

    rule = "valid_post_period_report_then_shortest_lag_then_accession"
    prioritized = sec_calendar.with_columns(
        pl.col("period_end").cast(pl.Date, strict=False).alias("_period_end_dt"),
        pl.col("reportDate").cast(pl.Date, strict=False).alias("_report_date_dt"),
    ).with_columns(
        pl.when(
            pl.col("_period_end_dt").is_not_null()
            & pl.col("_report_date_dt").is_not_null()
            & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
        )
        .then(pl.lit(0))
        .otherwise(pl.lit(1))
        .alias("_timing_penalty"),
        pl.when(
            pl.col("_period_end_dt").is_not_null()
            & pl.col("_report_date_dt").is_not_null()
            & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
        )
        .then((pl.col("_report_date_dt") - pl.col("_period_end_dt")).dt.total_days())
        .otherwise(pl.lit(99999))
        .alias("_lag_days"),
    )
    audit_all = prioritized.group_by(["ticker", "period_end"]).agg(
        pl.len().cast(pl.UInt32).alias("calendar_duplicate_count"),
        pl.col("accession_number")
        .drop_nulls()
        .unique()
        .sort()
        .alias("calendar_candidate_accessions"),
        pl.lit(rule).alias("calendar_resolution_rule"),
    )
    selected = (
        prioritized.sort(
            [
                "ticker",
                "period_end",
                "_timing_penalty",
                "_lag_days",
                "reportDate",
                "accession_number",
            ]
        )
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
        .drop("_period_end_dt", "_report_date_dt", "_timing_penalty", "_lag_days")
        .join(audit_all, on=["ticker", "period_end"], how="left")
        .sort(["ticker", "period_end"])
    )
    return selected, audit_all.filter(pl.col("calendar_duplicate_count") > 1).sort(
        ["ticker", "period_end"]
    )


def build_sec_companyfacts_earnings_actuals(*, ticker: str, facts_payload: dict[str, object]) -> pl.DataFrame:
    selected = _select_best_eps_facts(("us-gaap",), SEC_EPS_TAGS, facts_payload.get("facts", {}))  # type: ignore[arg-type]
    return _build_sec_earnings_actuals_frame(
        ticker=ticker,
        selected=selected,
        source="sec_companyfacts",
    )


def build_sec_filing_earnings_actuals(*, ticker: str, facts_payload: dict[str, object]) -> pl.DataFrame:
    selected = _select_best_eps_facts(("us-gaap", "ifrs-full"), SEC_EPS_TAGS, facts_payload)
    return _build_sec_earnings_actuals_frame(
        ticker=ticker,
        selected=selected,
        source="sec_filing",
    )


def _select_best_eps_facts(
    fact_roots: Iterable[str],
    tags: Iterable[str],
    facts_payload: dict[str, object],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for tag_priority, tag in enumerate(tuple(tags)):
        for fact_root in fact_roots:
            unit_payload = (
                facts_payload.get(fact_root, {}).get(tag, {}).get("units", {})  # type: ignore[union-attr]
            )
            records = _extract_unit_records(unit_payload, preferred_units=("USD/shares", "pure"))
            cleaned = [_clean_fact("income_statement", tag, record, tag_priority=tag_priority) for record in records]
            candidates.extend(record for record in cleaned if record is not None)

    quarterly_candidates = [
        record
        for record in candidates
        if record.get("fp") in {"Q1", "Q2", "Q3", "Q4"}
        and record.get("duration_days") is not None
        and 60 <= int(record["duration_days"]) <= 130
    ]
    if not quarterly_candidates:
        return []

    chosen_by_end: dict[str, dict[str, object]] = {}
    for record in sorted(
        quarterly_candidates,
        key=lambda item: (
            str(item.get("end") or ""),
            int(item.get("tag_priority") or 0),
            int(bool(item.get("has_dimensions"))),
            str(item.get("filed") or ""),
        ),
    ):
        end = str(record["end"])
        current = chosen_by_end.get(end)
        if current is None:
            chosen_by_end[end] = record
            continue
        current_key = (
            int(current.get("tag_priority") or 0),
            int(bool(current.get("has_dimensions"))),
            str(current.get("filed") or ""),
        )
        record_key = (
            int(record.get("tag_priority") or 0),
            int(bool(record.get("has_dimensions"))),
            str(record.get("filed") or ""),
        )
        if record_key < current_key:
            chosen_by_end[end] = record
    return list(chosen_by_end.values())


def align_sec_actuals_to_calendar(*, sec_calendar: pl.DataFrame, sec_actuals: pl.DataFrame) -> pl.DataFrame:
    if sec_calendar.is_empty() or sec_actuals.is_empty():
        return sec_actuals

    aligned_once = _align_sec_actuals_to_calendar_by_report_date(sec_calendar=sec_calendar, sec_actuals=sec_actuals)
    return _align_sec_actuals_to_calendar_by_fiscal_labels(sec_calendar=sec_calendar, sec_actuals=aligned_once)


def _calendar_rows_missing_actuals(*, sec_calendar: pl.DataFrame, sec_actuals: pl.DataFrame) -> pl.DataFrame:
    calendar = sec_calendar
    if "fiscal_period" not in calendar.columns:
        calendar = calendar.with_columns(pl.lit(None).cast(pl.Utf8).alias("fiscal_period"))
    if "fiscal_year" not in calendar.columns:
        calendar = calendar.with_columns(pl.lit(None).cast(pl.Int64).alias("fiscal_year"))
    exact = sec_actuals.select(["ticker", "period_end"]).with_columns(pl.lit(True).alias("has_exact"))
    return (
        calendar.select(["ticker", "period_end", "reportDate", "fiscal_period", "fiscal_year"])
        .join(exact, on=["ticker", "period_end"], how="left")
        .filter(pl.col("has_exact").fill_null(False).not_())
    )


def _apply_aligned_sec_actual_rows(*, sec_actuals: pl.DataFrame, aligned: pl.DataFrame) -> pl.DataFrame:
    if aligned.is_empty():
        return sec_actuals

    consumed_originals = aligned.select(["ticker", "_aligned_from_period_end", "_aligned_from_reportDate"]).rename(
        {
            "_aligned_from_period_end": "period_end",
            "_aligned_from_reportDate": "reportDate",
        }
    )
    aligned = aligned.drop(["_aligned_from_period_end", "_aligned_from_reportDate"]).select(sec_actuals.columns)
    remaining_originals = sec_actuals.join(
        consumed_originals.with_columns(pl.lit(True).alias("_consumed_for_alignment")),
        on=["ticker", "period_end", "reportDate"],
        how="left",
    ).filter(pl.col("_consumed_for_alignment").fill_null(False).not_()).drop("_consumed_for_alignment")

    return _select_best_sec_actual_rows(pl.concat([remaining_originals, aligned], how="vertical_relaxed"))


def _align_sec_actuals_to_calendar_by_report_date(*, sec_calendar: pl.DataFrame, sec_actuals: pl.DataFrame) -> pl.DataFrame:
    missing_calendar = _calendar_rows_missing_actuals(sec_calendar=sec_calendar, sec_actuals=sec_actuals)
    if missing_calendar.is_empty():
        return sec_actuals

    candidates = (
        missing_calendar.rename({"period_end": "calendar_period_end", "reportDate": "calendar_reportDate"})
        .join(
            sec_actuals.rename({"period_end": "actual_period_end", "reportDate": "actual_reportDate"}),
            left_on=["ticker", "calendar_reportDate"],
            right_on=["ticker", "actual_reportDate"],
            how="inner",
        )
        .with_columns(
            [
                pl.col("calendar_period_end").str.strptime(pl.Date, strict=False).alias("calendar_period_end_dt"),
                pl.col("actual_period_end").str.strptime(pl.Date, strict=False).alias("actual_period_end_dt"),
            ]
        )
        .with_columns(
            [
                (pl.col("calendar_period_end_dt") - pl.col("actual_period_end_dt")).dt.total_days().abs().alias("period_gap_days"),
                pl.when(pl.col("actual_period_end_dt") <= pl.col("calendar_period_end_dt"))
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("after_calendar_penalty"),
            ]
        )
        .sort(["ticker", "calendar_period_end", "after_calendar_penalty", "period_gap_days", "actual_period_end"])
        .unique(subset=["ticker", "calendar_period_end"], keep="first", maintain_order=True)
    )
    if candidates.is_empty():
        return sec_actuals

    aligned = candidates.select(
        [
            pl.col("actual_period_end").alias("_aligned_from_period_end"),
            pl.col("calendar_reportDate").alias("_aligned_from_reportDate"),
            "ticker",
            pl.col("calendar_period_end").alias("period_end"),
            pl.col("calendar_reportDate").alias("reportDate"),
            "epsActual",
            "source",
            pl.concat_str([pl.col("source_label"), pl.lit("aligned_reportDate")], separator=" | ").alias("source_label"),
            "form",
            "fiscal_period",
            "fiscal_year",
        ]
    ).sort(["ticker", "period_end", "reportDate"])

    return _apply_aligned_sec_actual_rows(sec_actuals=sec_actuals, aligned=aligned)


def _align_sec_actuals_to_calendar_by_fiscal_labels(*, sec_calendar: pl.DataFrame, sec_actuals: pl.DataFrame) -> pl.DataFrame:
    missing_calendar = _calendar_rows_missing_actuals(sec_calendar=sec_calendar, sec_actuals=sec_actuals)
    if missing_calendar.is_empty():
        return sec_actuals

    candidates = (
        missing_calendar.rename(
            {
                "period_end": "calendar_period_end",
                "reportDate": "calendar_reportDate",
                "fiscal_period": "calendar_fiscal_period",
                "fiscal_year": "calendar_fiscal_year",
            }
        )
        .join(
            sec_actuals.rename(
                {
                    "period_end": "actual_period_end",
                    "reportDate": "actual_reportDate",
                    "fiscal_period": "actual_fiscal_period",
                    "fiscal_year": "actual_fiscal_year",
                }
            ),
            left_on=["ticker", "calendar_fiscal_period", "calendar_fiscal_year"],
            right_on=["ticker", "actual_fiscal_period", "actual_fiscal_year"],
            how="inner",
        )
        .with_columns(
            [
                pl.col("calendar_period_end").str.strptime(pl.Date, strict=False).alias("calendar_period_end_dt"),
                pl.col("actual_period_end").str.strptime(pl.Date, strict=False).alias("actual_period_end_dt"),
                pl.col("calendar_reportDate").str.strptime(pl.Date, strict=False).alias("calendar_report_date_dt"),
                pl.col("actual_reportDate").str.strptime(pl.Date, strict=False).alias("actual_report_date_dt"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("calendar_period_end_dt").is_not_null() & pl.col("actual_period_end_dt").is_not_null())
                .then((pl.col("calendar_period_end_dt") - pl.col("actual_period_end_dt")).dt.total_days().abs())
                .otherwise(pl.lit(99999))
                .alias("period_gap_days"),
                pl.when(pl.col("calendar_report_date_dt").is_not_null() & pl.col("actual_report_date_dt").is_not_null())
                .then((pl.col("calendar_report_date_dt") - pl.col("actual_report_date_dt")).dt.total_days().abs())
                .otherwise(pl.lit(99999))
                .alias("report_gap_days"),
            ]
        )
        .sort(["ticker", "calendar_period_end", "report_gap_days", "period_gap_days", "actual_reportDate", "actual_period_end"])
        .unique(subset=["ticker", "calendar_period_end"], keep="first", maintain_order=True)
    )
    if candidates.is_empty():
        return sec_actuals

    aligned = candidates.select(
        [
            pl.col("actual_period_end").alias("_aligned_from_period_end"),
            pl.col("actual_reportDate").alias("_aligned_from_reportDate"),
            "ticker",
            pl.col("calendar_period_end").alias("period_end"),
            pl.col("calendar_reportDate").alias("reportDate"),
            "epsActual",
            "source",
            pl.concat_str([pl.col("source_label"), pl.lit("aligned_fiscal_period")], separator=" | ").alias("source_label"),
            pl.col("calendar_fiscal_period").alias("fiscal_period"),
            pl.col("calendar_fiscal_year").alias("fiscal_year"),
            "form",
        ]
    ).sort(["ticker", "period_end", "reportDate"])

    return _apply_aligned_sec_actual_rows(sec_actuals=sec_actuals, aligned=aligned)


def _build_sec_earnings_actuals_frame(
    *,
    ticker: str,
    selected: list[dict[str, object]],
    source: str,
) -> pl.DataFrame:
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
                "source": source,
                "source_label": str(fact.get("tag") or "sec_eps_actual"),
                "form": str(fact.get("form") or ""),
                "fiscal_period": str(fact.get("fp") or ""),
                "fiscal_year": int(fact.get("fy")) if fact.get("fy") is not None else None,
            }
        )
    if not rows:
        return empty_earnings_actuals_frame()
    return _select_best_sec_actual_rows(pl.DataFrame(rows))


def _select_best_sec_actual_rows(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    prioritized = frame.with_columns(
        [
            pl.col("period_end").str.strptime(pl.Date, strict=False).alias("_period_end_dt"),
            pl.col("reportDate").str.strptime(pl.Date, strict=False).alias("_report_date_dt"),
            pl.when(pl.col("epsActual").is_not_null()).then(pl.lit(0)).otherwise(pl.lit(1)).alias("_null_penalty"),
        ]
    ).with_columns(
        [
            pl.when(
                pl.col("_report_date_dt").is_not_null()
                & pl.col("_period_end_dt").is_not_null()
                & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
            )
            .then(pl.lit(0))
            .otherwise(pl.lit(1))
            .alias("_timing_penalty"),
            pl.when(
                pl.col("_report_date_dt").is_not_null()
                & pl.col("_period_end_dt").is_not_null()
                & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
            )
            .then((pl.col("_report_date_dt") - pl.col("_period_end_dt")).dt.total_days())
            .otherwise(pl.lit(99999))
            .alias("_lag_days"),
        ]
    )
    return (
        prioritized.sort(
            ["ticker", "period_end", "_null_penalty", "_timing_penalty", "_lag_days", "reportDate", "source_label"]
        )
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
        .drop(["_period_end_dt", "_report_date_dt", "_null_penalty", "_timing_penalty", "_lag_days"])
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

    calendar, _calendar_duplicate_audit = resolve_earnings_calendar_duplicates(
        sec_calendar
    )
    yahoo_matches = _match_yahoo_to_sec_calendar(sec_calendar=calendar, yahoo_earnings=yahoo_earnings, tolerance_days=match_tolerance_days)
    calendar = calendar.rename({"reportDate": "sec_reportDate", "earningsDatetime": "sec_earningsDatetime"})
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
        .with_columns(
            pl.when(pl.col("sec_source") == "sec_companyfacts")
            .then(pl.lit(0))
            .when(pl.col("sec_source") == "sec_filing")
            .then(pl.lit(1))
            .otherwise(pl.lit(9))
            .alias("sec_source_priority")
        )
        .with_columns(
            [
                pl.col("period_end").str.strptime(pl.Date, strict=False).alias("_period_end_dt"),
                pl.col("sec_reportDate").str.strptime(pl.Date, strict=False).alias("_report_date_dt"),
                pl.when(pl.col("sec_epsActual").is_not_null()).then(pl.lit(0)).otherwise(pl.lit(1)).alias("_null_penalty"),
            ]
        )
        .with_columns(
            [
                pl.when(
                    pl.col("_report_date_dt").is_not_null()
                    & pl.col("_period_end_dt").is_not_null()
                    & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
                )
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("_timing_penalty"),
                pl.when(
                    pl.col("_report_date_dt").is_not_null()
                    & pl.col("_period_end_dt").is_not_null()
                    & (pl.col("_report_date_dt") >= pl.col("_period_end_dt"))
                )
                .then((pl.col("_report_date_dt") - pl.col("_period_end_dt")).dt.total_days())
                .otherwise(pl.lit(99999))
                .alias("_lag_days"),
            ]
        )
        .sort(
            [
                "ticker",
                "period_end",
                "_null_penalty",
                "sec_source_priority",
                "_timing_penalty",
                "_lag_days",
                "sec_reportDate",
            ]
        )
        .unique(subset=["ticker", "period_end"], keep="first", maintain_order=True)
        .drop(["_period_end_dt", "_report_date_dt", "_null_penalty", "_timing_penalty", "_lag_days"])
        .sort(["ticker", "period_end"])
    )

    joined = calendar.join(yahoo_matches, on=["ticker", "period_end"], how="left", coalesce=True)
    joined = joined.join(sec_actual, on=["ticker", "period_end"], how="left", coalesce=True, suffix="_sec")

    consolidated = (
        joined.with_columns(
            [
                pl.coalesce([pl.col("yahoo_reportDate"), pl.col("sec_reportDate")]).alias("reportDate"),
                pl.coalesce([pl.col("yahoo_earningsDatetime"), pl.col("sec_earningsDatetime")]).alias("earningsDatetime"),
                pl.coalesce([pl.col("yahoo_epsActual"), pl.col("sec_epsActual")]).alias("epsActual"),
                pl.col("yahoo_epsEstimate").alias("epsEstimate"),
                pl.col("yahoo_surprisePercent").alias("surprisePercent"),
                pl.lit("sec_submissions").alias("calendar_source"),
                pl.when(pl.col("yahoo_epsActual").is_not_null())
                .then(pl.lit("yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.col("sec_source"))
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
                        pl.when(pl.col("sec_epsActual").is_not_null()).then(pl.col("sec_source")).otherwise(pl.lit(None).cast(pl.Utf8)),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("candidate_sources"),
                pl.when(pl.col("yahoo_reportDate").is_not_null())
                .then(pl.lit("sec_submissions+yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.concat_str([pl.lit("sec_submissions+"), pl.col("sec_source")], separator=""))
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
                pl.coalesce([pl.col("yahoo_reportDate"), pl.col("sec_reportDate")]).alias("reportDate"),
                pl.coalesce([pl.col("yahoo_earningsDatetime"), pl.col("sec_earningsDatetime")]).alias("earningsDatetime"),
                pl.coalesce([pl.col("yahoo_epsActual"), pl.col("sec_epsActual")]).alias("selected_epsActual"),
                pl.col("yahoo_epsEstimate").alias("selected_epsEstimate"),
                pl.col("yahoo_surprisePercent").alias("selected_surprisePercent"),
                pl.lit("sec_submissions").alias("calendar_source"),
                pl.when(pl.col("yahoo_epsActual").is_not_null())
                .then(pl.lit("yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.col("sec_source"))
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
                        pl.when(pl.col("sec_epsActual").is_not_null()).then(pl.col("sec_source")).otherwise(pl.lit(None).cast(pl.Utf8)),
                    ],
                    separator=" | ",
                    ignore_nulls=True,
                ).alias("candidate_sources"),
                pl.when(pl.col("yahoo_reportDate").is_not_null())
                .then(pl.lit("sec_submissions+yfinance"))
                .when(pl.col("sec_epsActual").is_not_null())
                .then(pl.concat_str([pl.lit("sec_submissions+"), pl.col("sec_source")], separator=""))
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
                "sec_reportDate",
                "earningsDatetime",
                "accession_number",
                "form",
                "fiscal_period",
                "fiscal_year",
                "calendar_duplicate_count",
                "calendar_candidate_accessions",
                "calendar_resolution_rule",
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
        [
            pl.col("reportDate").cast(pl.Date, strict=False).alias("report_date_dt"),
            pl.col("period_end").cast(pl.Date, strict=False).alias("period_end_dt"),
        ]
    )
    calendar = calendar.with_columns(
        [
            pl.when(pl.col("period_end_dt").is_not_null())
            .then(pl.col("period_end_dt") - pl.duration(days=7))
            .otherwise(pl.col("report_date_dt") - pl.duration(days=tolerance_days))
            .alias("match_window_start"),
            (pl.col("report_date_dt") + pl.duration(days=tolerance_days)).alias("match_window_end"),
        ]
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
        .filter(
            (pl.col("yahoo_report_date_dt") >= pl.col("match_window_start"))
            & (pl.col("yahoo_report_date_dt") <= pl.col("match_window_end"))
        )
        .with_columns(
            [
                (pl.col("yahoo_report_date_dt") - pl.col("report_date_dt")).dt.total_days().alias("date_diff_days"),
                (pl.col("report_date_dt") - pl.col("yahoo_report_date_dt")).dt.total_days().alias("sec_gap_days"),
                pl.when(pl.col("period_end_dt").is_not_null())
                .then((pl.col("yahoo_report_date_dt") - pl.col("period_end_dt")).dt.total_days())
                .otherwise(pl.lit(0))
                .alias("period_gap_days"),
            ]
        )
        .with_columns(
            [
                pl.col("sec_gap_days").abs().alias("abs_sec_gap_days"),
                pl.when(pl.col("yahoo_report_date_dt") <= pl.col("report_date_dt"))
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("after_sec_penalty"),
                pl.when(pl.col("period_gap_days") >= 0)
                .then(pl.lit(0))
                .otherwise(pl.lit(1))
                .alias("before_period_penalty"),
            ]
        )
        .sort(["calendar_row_id", "before_period_penalty", "after_sec_penalty", "abs_sec_gap_days", "yahoo_row_id"])
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
