"""Financial backfill logic for SEC-only package.

This module fills historical gaps in SEC companyfacts data using:
1. Ticker normalization (BRK.B -> BRK-B)
2. EODHD legacy data for CIK-legacy tickers (marked as non-GAAP)
3. Parent company data for spinoffs (remains GAAP/SEC)

IMPORTANT: EODHD backfill is NOT GAAP. It is explicitly tagged as
'eodhd_legacy_backfill' in the lineage so users can filter it out if
they require strict GAAP-only data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


# =============================================================================
# CONFIGURATION TABLES
# =============================================================================

# Tickers where the SEC mapping uses hyphens instead of dots for share classes.
# Normalization: replace '.' with '-' before looking up in SEC mapping.
TICKER_NORMALIZATION_OVERRIDES: dict[str, str] = {
    "BRK.B": "BRK-B",
    "BF.B": "BF-B",
}

# Ticker -> normalized ticker for SEC lookup.
def normalize_sec_ticker(ticker_root: str) -> str:
    """Normalize a ticker for SEC companyfacts lookup.

    The SEC ``company_tickers_exchange.json`` uses hyphens for share
    classes (e.g. ``BRK-B``) while many data vendors use dots
    (``BRK.B``). This function applies the known overrides.
    """
    return TICKER_NORMALIZATION_OVERRIDES.get(ticker_root, ticker_root)


# CIK-legacy tickers: the current SEC CIK only has data from a recent date.
# Before that date we backfill from EODHD legacy (non-GAAP).
# Format: ticker_root -> first_date available in SEC companyfacts.
SEC_START_DATES: dict[str, str] = {
    "APA": "2019-12-31",
    "AVGO": "2017-01-31",
    "BG": "2022-03-31",
    "BLK": "2023-09-30",
    "CI": "2017-03-31",
    "CRH": "2022-12-31",
    "DIS": "2017-09-30",
    "DOW": "2017-12-31",
    "EVRG": "2017-03-31",
    "IR": "2016-06-30",
    "LIN": "2017-03-31",
    "NXPI": "2017-12-31",
    "STE": "2017-03-31",
    "SW": "2023-06-30",
    "TPL": "2019-03-31",
    "VST": "2016-12-31",
    "VTRS": "2018-12-31",
    "XOM": "2008-06-30",
}


# Spinoff tickers: we can backfill pre-spinoff data from the parent SEC entity.
# This data REMAINS GAAP because it comes from the parent company's SEC filings.
# Format: child_ticker -> (parent_ticker, spinoff_effective_date)
SPINOFF_PARENTS: dict[str, tuple[str, str]] = {
    "CARR": ("RTX", "2020-04-03"),
    "FOX": ("NWSA", "2013-06-28"),
    "FOXA": ("NWSA", "2013-06-28"),
    "GEHC": ("GE", "2023-01-04"),
    "GEV": ("GE", "2024-04-02"),
    "KVUE": ("JNJ", "2023-05-04"),
    "MBC": ("DHR", "2023-09-01"),
    "OTIS": ("RTX", "2020-04-03"),
    "SOLV": ("DD", "2023-09-01"),
    "VICI": ("CZR", "2017-10-17"),
}


# =============================================================================
# LINEAGE HELPERS
# =============================================================================

@dataclass(frozen=True)
class BackfillConfig:
    eodhd_enabled: bool = True
    spinoff_enabled: bool = True
    ticker_normalization_enabled: bool = True


# =============================================================================
# EODHD BACKFILL
# =============================================================================

def backfill_from_eodhd(
    *,
    sec_financials: pl.DataFrame,
    eodhd_financials: pl.DataFrame,
    ticker_root: str,
    sec_start_date: str,
    metrics: tuple[str, ...] = ("revenue", "net_income"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Backfill SEC financials with EODHD legacy data before ``sec_start_date``.

    Returns a tuple of (backfilled_frame, lineage_frame).

    The backfilled_frame uses the EODHD value for dates before
    ``sec_start_date`` when the SEC value is missing.

    The lineage_frame records which rows came from EODHD vs SEC.
    """
    if sec_financials.is_empty() and eodhd_financials.is_empty():
        return _empty_backfilled_frame(), _empty_backfill_lineage_frame()

    ticker = f"{ticker_root}.US"
    sec_ticker = sec_financials.filter(pl.col("ticker") == ticker)
    eod_ticker = eodhd_financials.filter(pl.col("ticker") == ticker)

    # We only backfill the specified metrics.
    backfilled_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []

    for metric in metrics:
        sec_metric = sec_ticker.filter(pl.col("metric") == metric)
        eod_metric = eod_ticker.filter(pl.col("metric") == metric)

        if sec_metric.is_empty() and eod_metric.is_empty():
            continue

        # Collect all dates from both sources
        sec_dates = set(sec_metric["date"].to_list()) if not sec_metric.is_empty() else set()
        eod_dates = set(eod_metric["date"].to_list()) if not eod_metric.is_empty() else set()
        all_dates = sorted(sec_dates | eod_dates)

        for date_str in all_dates:
            sec_row = sec_metric.filter(pl.col("date") == date_str)
            eod_row = eod_metric.filter(pl.col("date") == date_str)

            sec_val = sec_row["value"].to_list()[0] if not sec_row.is_empty() else None
            eod_val = eod_row["value"].to_list()[0] if not eod_row.is_empty() else None

            is_before_start = str(date_str) < sec_start_date

            if sec_val is not None:
                # SEC has the value → use it
                backfilled_rows.append({
                    "ticker": ticker,
                    "statement": _statement_for_metric(metric),
                    "metric": metric,
                    "date": date_str,
                    "value": float(sec_val),
                    "source": "sec_companyfacts",
                    "source_label": "sec_companyfacts",
                })
                lineage_rows.append({
                    "ticker": ticker,
                    "metric": metric,
                    "date": date_str,
                    "backfill_source": None,
                    "backfill_reason": None,
                })
            elif is_before_start and eod_val is not None:
                # Backfill from EODHD
                backfilled_rows.append({
                    "ticker": ticker,
                    "statement": _statement_for_metric(metric),
                    "metric": metric,
                    "date": date_str,
                    "value": float(eod_val),
                    "source": "eodhd_legacy_backfill",
                    "source_label": f"eodhd_legacy_backfill_{metric}",
                })
                lineage_rows.append({
                    "ticker": ticker,
                    "metric": metric,
                    "date": date_str,
                    "backfill_source": "eodhd_legacy",
                    "backfill_reason": "cik_legacy",
                })
            elif eod_val is not None:
                # After start date but SEC still missing → also use EODHD as gap filler
                backfilled_rows.append({
                    "ticker": ticker,
                    "statement": _statement_for_metric(metric),
                    "metric": metric,
                    "date": date_str,
                    "value": float(eod_val),
                    "source": "eodhd_legacy_backfill",
                    "source_label": f"eodhd_legacy_backfill_{metric}",
                })
                lineage_rows.append({
                    "ticker": ticker,
                    "metric": metric,
                    "date": date_str,
                    "backfill_source": "eodhd_legacy",
                    "backfill_reason": "sec_gap",
                })

    if not backfilled_rows:
        return _empty_backfilled_frame(), _empty_backfill_lineage_frame()

    backfilled = pl.DataFrame(backfilled_rows, schema=_empty_backfilled_frame().schema)
    lineage = pl.DataFrame(lineage_rows, schema=_empty_backfill_lineage_frame().schema)
    return backfilled, lineage


def _statement_for_metric(metric: str) -> str:
    """Map a metric name to its statement."""
    mapping = {
        "revenue": "income_statement",
        "net_income": "income_statement",
        "gross_profit": "income_statement",
        "operating_income": "income_statement",
        "total_assets": "balance_sheet",
        "total_liabilities": "balance_sheet",
        "stockholders_equity": "balance_sheet",
        "cash_and_equivalents": "balance_sheet",
        "operating_cash_flow": "cash_flow",
        "capital_expenditures": "cash_flow",
        "free_cash_flow": "cash_flow",
        "outstanding_shares": "shares",
    }
    return mapping.get(metric, "income_statement")


def _empty_backfilled_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
        }
    )


def _empty_backfill_lineage_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "backfill_source": pl.String,
            "backfill_reason": pl.String,
        }
    )


# =============================================================================
# SPINOFF BACKFILL
# =============================================================================

def backfill_spinoffs_from_parent(
    *,
    parent_financials: pl.DataFrame,
    child_ticker: str,
    parent_ticker: str,
    spinoff_date: str,
    metrics: tuple[str, ...] = ("revenue", "net_income"),
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Copy parent financials before spinoff date to the child ticker.

    This data remains GAAP because it comes from the parent's SEC filings.

    Returns (child_frame, lineage_frame).
    """
    parent = f"{parent_ticker}.US"
    child = f"{child_ticker}.US"

    parent_data = parent_financials.filter(
        (pl.col("ticker") == parent)
        & (pl.col("metric").is_in(list(metrics)))
        & (pl.col("date") < spinoff_date)
    )

    if parent_data.is_empty():
        return _empty_backfilled_frame(), _empty_backfill_lineage_frame()

    child_data = parent_data.with_columns(
        [
            pl.lit(child).alias("ticker"),
            pl.lit("sec_spinoff_parent").alias("source"),
            pl.concat_str(
                [
                    pl.lit("spinoff_from_"),
                    pl.lit(parent_ticker),
                    pl.lit("_before_"),
                    pl.lit(spinoff_date),
                ]
            ).alias("source_label"),
        ]
    )

    lineage = child_data.select(
        [
            pl.col("ticker"),
            pl.col("metric"),
            pl.col("date"),
            pl.lit("sec_spinoff_parent").alias("backfill_source"),
            pl.concat_str([pl.lit("spinoff_"), pl.lit(parent_ticker)]).alias("backfill_reason"),
        ]
    )

    return child_data, lineage


# =============================================================================
# ORCHESTRATION
# =============================================================================

def apply_financial_backfills(
    *,
    sec_financials: pl.DataFrame,
    sec_companyfacts_raw: pl.DataFrame,
    eodhd_financials: pl.DataFrame,
    config: BackfillConfig | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Apply all backfills (EODHD + spinoffs) to SEC financials.

    Returns (combined_financials, combined_lineage, backfill_audit).

    ``combined_financials`` contains the SEC data plus backfilled rows.
    ``combined_lineage`` tracks which rows were backfilled and why.
    ``backfill_audit`` is a summary table for reporting.
    """
    config = config or BackfillConfig()
    combined = sec_financials.clone()
    all_lineage = _empty_backfill_lineage_frame()
    audit_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # 1. EODHD backfill for CIK-legacy tickers
    # ------------------------------------------------------------------
    if config.eodhd_enabled:
        for ticker_root, sec_start_date in SEC_START_DATES.items():
            backfilled, lineage = backfill_from_eodhd(
                sec_financials=sec_companyfacts_raw,
                eodhd_financials=eodhd_financials,
                ticker_root=ticker_root,
                sec_start_date=sec_start_date,
            )
            if not backfilled.is_empty():
                combined = pl.concat([combined, backfilled], how="diagonal_relaxed")
                all_lineage = pl.concat([all_lineage, lineage], how="diagonal_relaxed")
                audit_rows.append({
                    "ticker": f"{ticker_root}.US",
                    "backfill_type": "eodhd_legacy",
                    "backfill_reason": "cik_legacy",
                    "rows_added": backfilled.height,
                    "date_range": f"{backfilled['date'].min()} -> {backfilled['date'].max()}",
                    "metrics_backfilled": ", ".join(backfilled["metric"].unique().to_list()),
                })

    # ------------------------------------------------------------------
    # 2. Spinoff backfill from parent SEC data
    # ------------------------------------------------------------------
    if config.spinoff_enabled:
        for child_ticker, (parent_ticker, spinoff_date) in SPINOFF_PARENTS.items():
            backfilled, lineage = backfill_spinoffs_from_parent(
                parent_financials=sec_companyfacts_raw,
                child_ticker=child_ticker,
                parent_ticker=parent_ticker,
                spinoff_date=spinoff_date,
            )
            if not backfilled.is_empty():
                combined = pl.concat([combined, backfilled], how="diagonal_relaxed")
                all_lineage = pl.concat([all_lineage, lineage], how="diagonal_relaxed")
                audit_rows.append({
                    "ticker": f"{child_ticker}.US",
                    "backfill_type": "sec_spinoff_parent",
                    "backfill_reason": f"spinoff_from_{parent_ticker}",
                    "rows_added": backfilled.height,
                    "date_range": f"{backfilled['date'].min()} -> {backfilled['date'].max()}",
                    "metrics_backfilled": ", ".join(backfilled["metric"].unique().to_list()),
                })

    # Deduplicate: prefer non-null values over null values.
    # If SEC has a null and backfill has a value, backfill wins.
    combined = (
        combined.with_columns(
            pl.col("value").is_not_null().cast(pl.Int8).alias("_has_value")
        )
        .sort(["ticker", "statement", "metric", "date", "_has_value", "source"], descending=[False, False, False, False, True, False])
        .unique(subset=["ticker", "statement", "metric", "date"], keep="first", maintain_order=True)
        .drop("_has_value")
    )

    audit = pl.DataFrame(
        audit_rows,
        schema={
            "ticker": pl.String,
            "backfill_type": pl.String,
            "backfill_reason": pl.String,
            "rows_added": pl.Int64,
            "date_range": pl.String,
            "metrics_backfilled": pl.String,
        },
    ) if audit_rows else _empty_backfill_audit_frame()

    return combined, all_lineage, audit


def _empty_backfill_audit_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "backfill_type": pl.String,
            "backfill_reason": pl.String,
            "rows_added": pl.Int64,
            "date_range": pl.String,
            "metrics_backfilled": pl.String,
        }
    )
