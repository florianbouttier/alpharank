"""Reference, earnings and fundamental acquisition and consolidation stage."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Sequence

import polars as pl

from alpharank.data.open_source.config import GENERAL_COLUMNS, METRIC_SPECS
from alpharank.data.open_source.earnings import (
    build_sec_companyfacts_earnings_actuals,
    consolidate_earnings,
)
from alpharank.data.open_source.ingestion_frames import (
    _concat_or_empty,
    _empty_raw_earnings_frame,
    _with_earnings_ingestion_metadata,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    append_run_delta,
    upsert_parquet,
)
from alpharank.data.open_source.yahoo import YahooFinanceClient
from alpharank.observability import get_run_logger

LOGGER = get_run_logger(__name__)


def _canonicalize_general_outputs(
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    lineage = general_reference_lineage
    if not lineage.is_empty():
        sort_cols = [column for column in ["ticker", "ingested_at"] if column in lineage.columns]
        if sort_cols:
            lineage = lineage.sort(sort_cols)
        lineage = lineage.unique(subset=["ticker"], keep="last", maintain_order=True).sort("ticker")
        return lineage.select(list(GENERAL_COLUMNS)), lineage

    general = general_reference
    if general.is_empty():
        return general, lineage
    sort_cols = [column for column in ["ticker", "ingested_at"] if column in general.columns]
    if sort_cols:
        general = general.sort(sort_cols)
    general = general.unique(subset=["ticker"], keep="last", maintain_order=True).sort("ticker")
    return general.select(list(GENERAL_COLUMNS)), lineage


def _build_clean_earnings(
    *,
    raw_yahoo_earnings: pl.DataFrame,
    raw_earnings_sec_calendar: pl.DataFrame,
    raw_earnings_sec_actuals: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return consolidate_earnings(
        sec_calendar=raw_earnings_sec_calendar.select(
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
        ),
        yahoo_earnings=raw_yahoo_earnings.select(
            [
                "ticker",
                "period_end",
                "reportDate",
                "earningsDatetime",
                "epsEstimate",
                "epsActual",
                "surprisePercent",
                "source",
            ]
        ),
        sec_actuals=raw_earnings_sec_actuals.select(
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
        ),
    )


def _repair_yahoo_earnings(
    *,
    paths: OpenSourceLivePaths,
    run_id: str,
    ingested_at: str,
    yahoo_client: YahooFinanceClient,
    raw_yahoo_earnings: pl.DataFrame,
    raw_earnings_sec_calendar: pl.DataFrame,
    raw_earnings_sec_actuals: pl.DataFrame,
    candidate_tickers: Sequence[str],
    years: Sequence[int],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, tuple[str, ...]]:
    clean_earnings, clean_earnings_lineage, clean_earnings_long = _build_clean_earnings(
        raw_yahoo_earnings=raw_yahoo_earnings,
        raw_earnings_sec_calendar=raw_earnings_sec_calendar,
        raw_earnings_sec_actuals=raw_earnings_sec_actuals,
    )
    repair_tickers = _identify_yahoo_earnings_repair_tickers(
        clean_earnings_lineage=clean_earnings_lineage,
        raw_yahoo_earnings=raw_yahoo_earnings,
        candidate_tickers=candidate_tickers,
        years=years,
    )
    if not repair_tickers:
        return raw_yahoo_earnings, clean_earnings, clean_earnings_lineage, clean_earnings_long, ()

    repaired = yahoo_client.fetch_earnings_dates(repair_tickers, limit=100)
    if repaired.is_empty():
        return raw_yahoo_earnings, clean_earnings, clean_earnings_lineage, clean_earnings_long, repair_tickers

    repair_delta = _with_earnings_ingestion_metadata(
        repaired,
        dataset="earnings_yfinance_repair",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_yfinance_repair.parquet", repair_delta)
    repaired_raw = upsert_parquet(
        paths.raw_dir / "earnings_yfinance.parquet",
        repair_delta,
        key_cols=["ticker", "reportDate", "source"],
        order_cols=["ingested_at"],
    )
    clean_earnings, clean_earnings_lineage, clean_earnings_long = _build_clean_earnings(
        raw_yahoo_earnings=repaired_raw,
        raw_earnings_sec_calendar=raw_earnings_sec_calendar,
        raw_earnings_sec_actuals=raw_earnings_sec_actuals,
    )
    return repaired_raw, clean_earnings, clean_earnings_lineage, clean_earnings_long, repair_tickers


def _identify_yahoo_earnings_repair_tickers(
    *,
    clean_earnings_lineage: pl.DataFrame,
    raw_yahoo_earnings: pl.DataFrame,
    candidate_tickers: Sequence[str],
    years: Sequence[int],
) -> tuple[str, ...]:
    requested = [f"{ticker}.US" if not str(ticker).endswith(".US") else str(ticker) for ticker in candidate_tickers]
    if not requested:
        return ()

    base = pl.DataFrame({"ticker": requested})
    filtered_lineage = _filter_earnings_years(clean_earnings_lineage, years)
    if filtered_lineage.is_empty():
        latest_lineage = base.with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("yahoo_reportDate"),
                pl.lit(None).cast(pl.Utf8).alias("actual_source"),
                pl.lit(None).cast(pl.Utf8).alias("estimate_source"),
                pl.lit(None).cast(pl.Utf8).alias("surprise_source"),
            ]
        )
    else:
        latest_lineage = (
            filtered_lineage.sort(["ticker", "period_end", "reportDate", "sec_reportDate"])
            .group_by("ticker")
            .tail(1)
            .select(["ticker", "yahoo_reportDate", "actual_source", "estimate_source", "surprise_source"])
        )

    filtered_raw = _filter_earnings_years(raw_yahoo_earnings, years)
    if filtered_raw.is_empty():
        raw_stats = base.with_columns(pl.lit(0).alias("yahoo_recent_count"))
    else:
        raw_stats = filtered_raw.group_by("ticker").agg(pl.len().alias("yahoo_recent_count"))

    candidates = (
        base.join(latest_lineage, on="ticker", how="left", coalesce=True)
        .join(raw_stats, on="ticker", how="left", coalesce=True)
        .with_columns(pl.col("yahoo_recent_count").fill_null(0))
        .filter(
            (pl.col("yahoo_recent_count") == 0)
            | pl.col("yahoo_reportDate").is_null()
            | (pl.col("actual_source") != "yfinance")
            | pl.col("estimate_source").is_null()
            | pl.col("surprise_source").is_null()
        )
        .select("ticker")
        .unique()
        .sort("ticker")
        .to_series()
        .to_list()
    )
    return tuple(ticker[:-3] if ticker.endswith(".US") else ticker for ticker in candidates)


def _filter_earnings_years(frame: pl.DataFrame, years: Sequence[int]) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_earnings_frame()
    prefixes = [str(year) for year in years]
    period_or_report = pl.coalesce(
        [
            pl.col("period_end").cast(pl.Utf8, strict=False),
            pl.col("reportDate").cast(pl.Utf8, strict=False),
        ]
    )
    return frame.filter(
        pl.any_horizontal(
            [period_or_report.str.starts_with(prefix) for prefix in prefixes]
        )
    )


def _upsert_financial_dataset(
    *,
    paths: OpenSourceLivePaths,
    run_id: str,
    file_name: str,
    deltas: Sequence[pl.DataFrame],
) -> pl.DataFrame:
    delta = _concat_or_empty(deltas)
    append_run_delta(paths.run_dir(run_id) / "raw" / file_name, delta)
    return upsert_parquet(
        paths.raw_dir / file_name,
        delta,
        key_cols=[
            "ticker",
            "statement",
            "metric",
            "date",
            "filing_date",
            "source",
        ],
        order_cols=["ingested_at"],
    )


def _fetch_sec_financials(
    sec_client: SecCompanyFactsClient,
    sec_mapping: pl.DataFrame,
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_financials, str(row["ticker"]), str(row["cik"])): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc)})
    return frames, failures


def _fetch_sec_companyfacts_bundle(
    sec_client: SecCompanyFactsClient,
    sec_mapping: pl.DataFrame,
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[pl.DataFrame], list[dict[str, str]]]:
    """Derive all companyfacts outputs while each network payload is resident once."""
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    financial_frames: list[pl.DataFrame] = []
    earnings_frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []

    def fetch_one(ticker: str, cik: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        try:
            financials = sec_client.extract_financials(ticker, cik)
            earnings = _extract_sec_companyfacts_earnings_actuals(sec_client, ticker, cik)
            return financials, earnings
        finally:
            sec_client.discard_company_facts(cik)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_one, str(row["ticker"]), str(row["cik"])): str(row["ticker"])
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                financials, earnings = future.result()
                financial_frames.append(financials)
                earnings_frames.append(earnings)
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append(
                    {
                        "ticker": ticker,
                        "error": str(exc),
                        "dataset": "sec_companyfacts_bundle",
                    }
                )
            if completed % 100 == 0:
                LOGGER.info(
                    "SEC companyfacts acquisition progress",
                    extra={
                        "completed_count": completed,
                        "total_count": len(futures),
                    },
                )
    return financial_frames, earnings_frames, failures


def _fetch_sec_earnings_actuals(
    sec_client: SecCompanyFactsClient,
    sec_mapping: pl.DataFrame,
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_extract_sec_companyfacts_earnings_actuals, sec_client, str(row["ticker"]), str(row["cik"])): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_actuals"})
    return frames, failures


def _fetch_sec_filing_earnings_actuals(
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    *,
    years: Sequence[int],
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_earnings_actuals, str(row["ticker"]), str(row["cik"]), list(years)): str(row["ticker"])
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_actuals_filing"})
            finally:
                sec_client.clear_memory_cache()
            if completed % 100 == 0:
                LOGGER.info(
                    "SEC filing earnings acquisition progress",
                    extra={
                        "completed_count": completed,
                        "total_count": len(futures),
                    },
                )
    return frames, failures


def _fetch_sec_earnings_calendar(
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    *,
    years: Sequence[int],
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_earnings_calendar, str(row["ticker"]), str(row["cik"]), list(years)): str(row["ticker"])
            for row in rows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_calendar"})
            finally:
                sec_client.clear_memory_cache()
            if completed % 100 == 0:
                LOGGER.info(
                    "SEC submissions acquisition progress",
                    extra={
                        "completed_count": completed,
                        "total_count": len(futures),
                    },
                )
    return frames, failures


def _fetch_sec_company_profiles(
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_company_profile, str(row["ticker"]), str(row["cik"])): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "general_reference"})
            finally:
                sec_client.clear_memory_cache()
    return frames, failures


def _fetch_sec_filing_financials(
    sec_client: SecFilingFactsClient,
    sec_mapping: pl.DataFrame,
    *,
    year: int,
    max_workers: int = 1,
) -> tuple[list[pl.DataFrame], list[dict[str, str]]]:
    rows = sec_mapping.select(["ticker", "cik"]).iter_rows(named=True)
    frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(sec_client.extract_financials, str(row["ticker"]), str(row["cik"]), year): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
                failures.append({"ticker": ticker, "error": str(exc)})
            finally:
                sec_client.clear_memory_cache()
    return frames, failures


def _extract_sec_companyfacts_earnings_actuals(
    sec_client: SecCompanyFactsClient,
    ticker: str,
    cik: str,
) -> pl.DataFrame:
    payload = sec_client.fetch_company_facts(cik)
    return build_sec_companyfacts_earnings_actuals(ticker=ticker, facts_payload=payload)


def _identify_sec_filing_fallback_tickers(
    *,
    tickers: tuple[str, ...],
    sec_companyfacts: pl.DataFrame,
) -> tuple[str, ...]:
    requested = {str(ticker).upper().removesuffix(".US") for ticker in tickers}
    covered = (
        {
            str(ticker).upper().removesuffix(".US")
            for ticker in sec_companyfacts.get_column("ticker").unique().to_list()
        }
        if not sec_companyfacts.is_empty()
        else set()
    )
    return tuple(sorted(requested - covered))


def _identify_yfinance_financial_fallback_tickers(
    *,
    tickers: tuple[str, ...],
    sec_companyfacts: pl.DataFrame,
    sec_filing: pl.DataFrame,
) -> tuple[str, ...]:
    required_columns = ["ticker", "statement", "metric", "date"]
    sec_frames = [frame.select(required_columns) for frame in (sec_companyfacts, sec_filing) if not frame.is_empty()]
    sec_combined = pl.concat(sec_frames, how="vertical") if sec_frames else pl.DataFrame(schema={column: pl.Utf8 for column in required_columns})
    return _identify_metric_gap_tickers(
        tickers=tickers,
        financials=sec_combined,
        supported_metrics={
            (spec.statement, spec.metric)
            for spec in METRIC_SPECS
            if spec.statement != "earnings" and spec.yfinance_rows
        },
    )


def _identify_metric_gap_tickers(
    *,
    tickers: tuple[str, ...],
    financials: pl.DataFrame,
    supported_metrics: set[tuple[str, str]],
) -> tuple[str, ...]:
    if not supported_metrics:
        return ()
    expected_metrics = [{"statement": statement, "metric": metric} for statement, metric in sorted(supported_metrics)]
    expectation_grid = pl.DataFrame({"ticker": [f"{ticker}.US" for ticker in tickers]}).join(pl.DataFrame(expected_metrics), how="cross")
    counts = financials.group_by(["ticker", "statement", "metric"]).agg(pl.col("date").n_unique().alias("quarter_count"))
    fallback = (
        expectation_grid.join(counts, on=["ticker", "statement", "metric"], how="left")
        .with_columns(pl.col("quarter_count").fill_null(0))
        .filter(pl.col("quarter_count") < 4)
        .select("ticker")
        .unique()
        .sort("ticker")
        .get_column("ticker")
        .to_list()
    )
    return tuple(ticker.removesuffix(".US") for ticker in fallback)
