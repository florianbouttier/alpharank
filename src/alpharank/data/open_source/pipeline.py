from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_audited_metric_catalog,
    build_coverage_audit,
    build_error_detail_tables,
    build_error_summary_tables,
    build_earnings_alignment,
    build_financial_alignment,
    build_price_alignment,
    load_eodhd_prices,
    load_sp500_tickers_for_year,
    normalize_eodhd_earnings,
    normalize_eodhd_financials,
    summarize_alignment,
    write_detail_reports,
    write_html_report,
)
from alpharank.data.open_source.consolidation import (
    FinancialSourceInput,
    consolidate_financial_sources,
    split_consolidated_by_statement,
)
from alpharank.data.open_source.config import METRIC_SPECS, PILOT_TICKERS
from alpharank.data.open_source.earnings import (
    build_sec_companyfacts_earnings_actuals,
    consolidate_earnings,
    empty_earnings_actuals_frame,
    empty_earnings_calendar_frame,
    empty_earnings_consolidated_frame,
    empty_earnings_lineage_frame,
    empty_earnings_long_frame,
)
from alpharank.data.open_source.general_reference import (
    build_general_reference,
    empty_general_reference_frame,
    empty_general_reference_lineage_frame,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.simfin import SimFinClient
from alpharank.data.open_source.yahoo import YahooFinanceClient


@dataclass(frozen=True)
class OpenSourceCadrageResult:
    output_dir: Path
    tickers: tuple[str, ...]
    sp500_ticker_count: int
    coverage_available_in_yahoo_or_sec: int
    price_rows: int
    sec_rows: int
    sec_filing_rows: int
    simfin_rows: int
    consolidated_rows: int
    lineage_rows: int
    yfinance_financial_ticker_count: int
    yfinance_financial_rows: int
    yfinance_earnings_rows: int
    earnings_alignment_rows: int
    price_alignment_rows: int
    financial_alignment_rows: int


def run_open_source_cadrage(
    *,
    year: int = 2025,
    tickers: Sequence[str] | None = None,
    universe: str = "pilot",
    threshold_pct: float = 0.5,
    reference_data_dir: Path | None = None,
    output_dir: Path | None = None,
    user_agent: str = "Florian Bouttier florianbouttier@example.com",
    simfin_api_key: str | None = None,
) -> OpenSourceCadrageResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or project_root / "data"
    default_folder = _default_audit_folder(year=year, universe=universe, tickers=tickers)
    output_dir = output_dir or project_root / "data" / "open_source" / "audit" / default_folder
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = project_root / "data" / "open_source" / "_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    sp500_tickers = load_sp500_tickers_for_year(reference_data_dir, year)
    if tickers is not None:
        ticker_list = tuple(tickers)
    elif universe == "sp500-2025":
        ticker_list = sp500_tickers
    else:
        ticker_list = tuple(PILOT_TICKERS)
    include_yfinance_financials = True
    include_yfinance_earnings = True
    yahoo_client = YahooFinanceClient(cache_dir=cache_dir / "yfinance")
    simfin_client = SimFinClient(
        api_key=simfin_api_key,
        data_dir=cache_dir / "simfin",
    )
    sec_client = SecCompanyFactsClient(
        user_agent=user_agent,
        cache_dir=cache_dir / "sec_companyfacts",
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=user_agent,
        cache_dir=cache_dir / "sec_filing",
    )

    sec_mapping_all = sec_client.fetch_company_mapping()
    coverage_cache_path = cache_dir / f"ticker_coverage_{year}.parquet"
    if coverage_cache_path.exists():
        yahoo_availability = pl.read_parquet(coverage_cache_path).select(["ticker", "ticker_root", "yahoo_price_available"])
    else:
        yahoo_availability = yahoo_client.audit_price_availability(
            sp500_tickers,
            start_date=f"{year}-12-01",
            end_date=f"{year}-12-15",
        )
        yahoo_availability.write_parquet(coverage_cache_path)
    coverage = build_coverage_audit(
        sp500_tickers=sp500_tickers,
        benchmark_tickers=ticker_list,
        sec_mapping=sec_mapping_all,
        yahoo_availability=yahoo_availability,
    )

    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(ticker_list))
    yahoo_general_metadata = yahoo_client.fetch_company_metadata(ticker_list)
    sec_profiles, sec_profile_failures = _fetch_sec_company_profiles(sec_filing_client, sec_mapping)
    general_reference, general_reference_lineage = build_general_reference(
        tickers=ticker_list,
        sec_mapping=sec_mapping,
        yahoo_metadata=yahoo_general_metadata,
        sec_profiles=_concat_or_empty(sec_profiles, empty=_empty_sec_profile_frame()),
    )
    yahoo_prices = yahoo_client.download_prices(ticker_list, f"{year}-01-01", f"{year + 1}-01-01")
    try:
        yahoo_earnings = yahoo_client.fetch_earnings_dates(ticker_list)
    except Exception:
        yahoo_earnings = pl.DataFrame(
            schema={
                "ticker": pl.String,
                "reportDate": pl.String,
                "earningsDatetime": pl.String,
                "period_end": pl.String,
                "epsEstimate": pl.Float64,
                "epsActual": pl.Float64,
                "surprisePercent": pl.Float64,
                "source": pl.String,
            }
        )
    sec_frames, sec_fetch_failures = _fetch_sec_financials(sec_client, sec_mapping)
    sec_financials = pl.concat(sec_frames, how="vertical") if sec_frames else _empty_financials()
    sec_financials = sec_financials.filter(pl.col("date").str.starts_with(f"{year}"))
    sec_actual_frames, sec_earnings_failures = _fetch_sec_earnings_actuals(sec_client, sec_mapping)
    sec_earnings_actuals = _concat_or_empty(sec_actual_frames, empty=empty_earnings_actuals_frame()).filter(
        pl.col("period_end").str.starts_with(f"{year}")
    )
    sec_calendar_frames, sec_calendar_failures = _fetch_sec_earnings_calendar(sec_filing_client, sec_mapping, year=year)
    sec_earnings_calendar = _concat_or_empty(sec_calendar_frames, empty=empty_earnings_calendar_frame())
    sec_filing_tickers = _identify_sec_filing_fallback_tickers(ticker_list, sec_financials)
    sec_filing_financials = _empty_financials()
    sec_filing_fetch_failures: list[dict[str, str]] = []
    if sec_filing_tickers:
        sec_filing_mapping = sec_mapping.filter(pl.col("ticker").is_in(sec_filing_tickers))
        sec_filing_frames, sec_filing_fetch_failures = _fetch_sec_filing_financials(
            sec_filing_client,
            sec_filing_mapping,
            year=year,
        )
        if sec_filing_frames:
            sec_filing_financials = pl.concat(sec_filing_frames, how="vertical").filter(pl.col("date").str.starts_with(f"{year}"))
    if include_yfinance_financials:
        yfinance_financial_tickers = _identify_yfinance_financial_fallback_tickers(
            tickers=ticker_list,
            sec_companyfacts=sec_financials,
            sec_filing=sec_filing_financials,
        )
        yahoo_financials = (
            yahoo_client.fetch_quarterly_financials(yfinance_financial_tickers).filter(pl.col("date").str.starts_with(f"{year}"))
            if yfinance_financial_tickers
            else _empty_financials().select(["ticker", "statement", "metric", "date", "filing_date", "value", "source", "source_label"])
        )
    else:
        yfinance_financial_tickers = ()
        yahoo_financials = _empty_financials().select(["ticker", "statement", "metric", "date", "filing_date", "value", "source", "source_label"])
    simfin_financials = _empty_financials()
    simfin_fetch_failures: list[dict[str, str]] = []
    if simfin_client.enabled:
        try:
            simfin_financials = simfin_client.fetch_quarterly_financials(ticker_list, year)
            simfin_fetch_failures.extend(simfin_client.last_fetch_failures)
        except Exception as exc:
            print(f"SimFin fetch failed: {exc}")
            simfin_fetch_failures.append({"error": str(exc)})
    consolidated_financials, consolidated_lineage, consolidation_source_summary = consolidate_financial_sources(
        [
            FinancialSourceInput(source_name="sec_companyfacts", frame=sec_financials, priority=1),
            FinancialSourceInput(source_name="sec_filing", frame=sec_filing_financials, priority=2),
            FinancialSourceInput(source_name="simfin", frame=simfin_financials, priority=3),
            FinancialSourceInput(source_name="yfinance", frame=yahoo_financials, priority=4),
        ]
    )
    earnings_consolidated, earnings_lineage, earnings_long = consolidate_earnings(
        sec_calendar=sec_earnings_calendar,
        yahoo_earnings=yahoo_earnings,
        sec_actuals=sec_earnings_actuals,
    )
    consolidated_by_statement = split_consolidated_by_statement(consolidated_financials)
    has_sec_filing_financials = not sec_filing_financials.is_empty()
    has_simfin_financials = not simfin_financials.is_empty()
    has_consolidated_financials = not consolidated_financials.is_empty()
    has_open_source_earnings = not earnings_long.is_empty()

    eodhd_prices = load_eodhd_prices(reference_data_dir, ticker_list, year)
    eodhd_financials = normalize_eodhd_financials(reference_data_dir, ticker_list, year)
    eodhd_earnings = normalize_eodhd_earnings(reference_data_dir, ticker_list, year)
    price_alignment = build_price_alignment(eodhd_prices, yahoo_prices)
    financial_alignment = pl.concat(
        [
            build_financial_alignment(eodhd_financials, sec_financials, "sec_companyfacts"),
            *([build_financial_alignment(eodhd_financials, sec_filing_financials, "sec_filing")] if not sec_filing_financials.is_empty() else []),
            *([build_financial_alignment(eodhd_financials, simfin_financials, "simfin")] if not simfin_financials.is_empty() else []),
            *(
                [build_financial_alignment(eodhd_financials, consolidated_financials, "open_source_consolidated")]
                if not consolidated_financials.is_empty()
                else []
            ),
            *([build_earnings_alignment(eodhd_earnings, earnings_long, open_source="open_source_earnings")] if not earnings_long.is_empty() else []),
            *([build_financial_alignment(eodhd_financials, yahoo_financials, "yfinance")] if include_yfinance_financials else []),
        ],
        how="vertical",
    )
    (
        price_summary,
        statement_summary,
        metric_summary,
        ticker_summary,
        ticker_metric_summary,
        price_ticker_summary,
        price_ticker_metric_summary,
    ) = build_error_summary_tables(
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        threshold_pct=threshold_pct,
    )
    price_error_details, financial_error_details = build_error_detail_tables(
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        threshold_pct=threshold_pct,
    )
    audited_metric_catalog = build_audited_metric_catalog(
        include_yfinance_financials=include_yfinance_financials,
        include_yfinance_earnings=has_open_source_earnings,
        include_sec_filing_financials=has_sec_filing_financials,
        include_simfin_financials=has_simfin_financials,
        include_open_source_consolidated=has_consolidated_financials,
    )

    _write_outputs(
        output_dir=output_dir,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        sec_fetch_failures=[*sec_fetch_failures, *sec_earnings_failures, *sec_profile_failures],
        sec_filing_fetch_failures=[*sec_filing_fetch_failures, *sec_calendar_failures],
        simfin_fetch_failures=simfin_fetch_failures,
        consolidation_source_summary=consolidation_source_summary,
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        yahoo_prices=yahoo_prices,
        yahoo_earnings=yahoo_earnings,
        sec_earnings_calendar=sec_earnings_calendar,
        sec_earnings_actuals=sec_earnings_actuals,
        earnings_consolidated=earnings_consolidated,
        earnings_lineage=earnings_lineage,
        earnings_long=earnings_long,
        yahoo_financials=yahoo_financials,
        sec_financials=sec_financials,
        sec_filing_financials=sec_filing_financials,
        simfin_financials=simfin_financials,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        consolidated_by_statement=consolidated_by_statement,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_metric_summary=ticker_metric_summary,
        price_ticker_summary=price_ticker_summary,
        price_ticker_metric_summary=price_ticker_metric_summary,
        price_error_details=price_error_details,
        financial_error_details=financial_error_details,
        tickers=ticker_list,
        year=year,
        threshold_pct=threshold_pct,
        include_yfinance_financials=include_yfinance_financials,
        include_yfinance_earnings=has_open_source_earnings,
    )

    return OpenSourceCadrageResult(
        output_dir=output_dir,
        tickers=ticker_list,
        sp500_ticker_count=len(sp500_tickers),
        coverage_available_in_yahoo_or_sec=int(coverage.select(pl.col("available_in_yahoo_or_sec").sum()).item()),
        price_rows=yahoo_prices.height,
        sec_rows=sec_financials.height,
        sec_filing_rows=sec_filing_financials.height,
        simfin_rows=simfin_financials.height,
        consolidated_rows=consolidated_financials.height,
        lineage_rows=consolidated_lineage.height,
        yfinance_financial_ticker_count=len(yfinance_financial_tickers),
        yfinance_financial_rows=yahoo_financials.height,
        yfinance_earnings_rows=earnings_consolidated.height,
        earnings_alignment_rows=financial_alignment.filter(pl.col("statement") == "earnings").height,
        price_alignment_rows=price_alignment.height,
        financial_alignment_rows=financial_alignment.height,
    )


def _write_outputs(
    *,
    output_dir: Path,
    coverage: pl.DataFrame,
    audited_metric_catalog: pl.DataFrame,
    sec_fetch_failures: list[dict[str, str]],
    sec_filing_fetch_failures: list[dict[str, str]],
    simfin_fetch_failures: list[dict[str, str]],
    consolidation_source_summary: pl.DataFrame,
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
    yahoo_prices: pl.DataFrame,
    yahoo_earnings: pl.DataFrame,
    sec_earnings_calendar: pl.DataFrame,
    sec_earnings_actuals: pl.DataFrame,
    earnings_consolidated: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
    earnings_long: pl.DataFrame,
    yahoo_financials: pl.DataFrame,
    sec_financials: pl.DataFrame,
    sec_filing_financials: pl.DataFrame,
    simfin_financials: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    consolidated_by_statement: dict[str, pl.DataFrame],
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    price_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    ticker_metric_summary: pl.DataFrame,
    price_ticker_summary: pl.DataFrame,
    price_ticker_metric_summary: pl.DataFrame,
    price_error_details: pl.DataFrame,
    financial_error_details: pl.DataFrame,
    tickers: tuple[str, ...],
    year: int,
    threshold_pct: float,
    include_yfinance_financials: bool,
    include_yfinance_earnings: bool,
) -> None:
    coverage.write_parquet(output_dir / f"ticker_coverage_{year}.parquet")
    audited_metric_catalog.write_parquet(output_dir / "audited_metric_catalog.parquet")
    (output_dir / "sec_fetch_failures.json").write_text(json.dumps(sec_fetch_failures, indent=2), encoding="utf-8")
    (output_dir / "sec_filing_fetch_failures.json").write_text(json.dumps(sec_filing_fetch_failures, indent=2), encoding="utf-8")
    (output_dir / "simfin_fetch_failures.json").write_text(json.dumps(simfin_fetch_failures, indent=2), encoding="utf-8")
    general_reference.write_parquet(output_dir / "general_reference.parquet")
    general_reference_lineage.write_parquet(output_dir / "general_reference_lineage.parquet")
    yahoo_prices.write_parquet(output_dir / "prices_yfinance.parquet")
    yahoo_earnings.write_parquet(output_dir / "earnings_yfinance.parquet")
    sec_earnings_calendar.write_parquet(output_dir / "earnings_sec_calendar.parquet")
    sec_earnings_actuals.write_parquet(output_dir / "earnings_sec_actuals.parquet")
    earnings_consolidated.write_parquet(output_dir / "earnings_open_source_consolidated.parquet")
    earnings_lineage.write_parquet(output_dir / "earnings_open_source_lineage.parquet")
    earnings_long.write_parquet(output_dir / "earnings_open_source_long.parquet")
    yahoo_financials.write_parquet(output_dir / "financials_yfinance.parquet")
    sec_financials.write_parquet(output_dir / "financials_sec_companyfacts.parquet")
    sec_filing_financials.write_parquet(output_dir / "financials_sec_filing.parquet")
    simfin_financials.write_parquet(output_dir / "financials_simfin.parquet")
    consolidated_financials.write_parquet(output_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage.write_parquet(output_dir / "financials_open_source_lineage.parquet")
    consolidation_source_summary.write_parquet(output_dir / "financials_open_source_source_summary.parquet")
    consolidated_dir = output_dir / "financials_open_source_consolidated"
    consolidated_dir.mkdir(parents=True, exist_ok=True)
    for statement, frame in consolidated_by_statement.items():
        frame.write_parquet(consolidated_dir / f"{statement}.parquet")
    price_alignment.write_parquet(output_dir / f"price_alignment_{year}.parquet")
    financial_alignment.write_parquet(output_dir / f"financial_alignment_{year}.parquet")
    price_summary.write_parquet(output_dir / "price_error_summary.parquet")
    statement_summary.write_parquet(output_dir / "statement_error_summary.parquet")
    metric_summary.write_parquet(output_dir / "metric_error_summary.parquet")
    ticker_summary.write_parquet(output_dir / "ticker_error_summary.parquet")
    ticker_metric_summary.write_parquet(output_dir / "ticker_metric_error_summary.parquet")
    price_ticker_summary.write_parquet(output_dir / "price_ticker_error_summary.parquet")
    price_ticker_metric_summary.write_parquet(output_dir / "price_ticker_metric_error_summary.parquet")
    price_error_details.write_parquet(output_dir / "price_error_details.parquet")
    financial_error_details.write_parquet(output_dir / "financial_error_details.parquet")
    summarize_alignment(
        tickers=tickers,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=output_dir / "summary.json",
    )
    write_html_report(
        output_path=output_dir / "report.html",
        year=year,
        threshold_pct=threshold_pct,
        benchmark_tickers=tickers,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        consolidation_source_summary=consolidation_source_summary,
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_metric_summary=ticker_metric_summary,
        price_ticker_summary=price_ticker_summary,
    )
    write_detail_reports(
        output_dir=output_dir,
        year=year,
        threshold_pct=threshold_pct,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        price_error_details=price_error_details,
        financial_error_details=financial_error_details,
        price_summary=price_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_metric_summary=ticker_metric_summary,
        price_ticker_summary=price_ticker_summary,
        price_ticker_metric_summary=price_ticker_metric_summary,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "year": year,
                "tickers": list(tickers),
                "threshold_pct": threshold_pct,
                "files": {
                    "ticker_coverage": f"ticker_coverage_{year}.parquet",
                    "audited_metric_catalog": "audited_metric_catalog.parquet",
                    "sec_fetch_failures": "sec_fetch_failures.json",
                    "sec_filing_fetch_failures": "sec_filing_fetch_failures.json",
                    "simfin_fetch_failures": "simfin_fetch_failures.json",
                    "general_reference": "general_reference.parquet",
                    "general_reference_lineage": "general_reference_lineage.parquet",
                    "prices_yfinance": "prices_yfinance.parquet",
                    "earnings_yfinance": "earnings_yfinance.parquet",
                    "earnings_sec_calendar": "earnings_sec_calendar.parquet",
                    "earnings_sec_actuals": "earnings_sec_actuals.parquet",
                    "earnings_open_source_consolidated": "earnings_open_source_consolidated.parquet",
                    "earnings_open_source_lineage": "earnings_open_source_lineage.parquet",
                    "earnings_open_source_long": "earnings_open_source_long.parquet",
                    "financials_yfinance": "financials_yfinance.parquet",
                    "financials_sec_companyfacts": "financials_sec_companyfacts.parquet",
                    "financials_sec_filing": "financials_sec_filing.parquet",
                    "financials_simfin": "financials_simfin.parquet",
                    "financials_open_source_consolidated": "financials_open_source_consolidated.parquet",
                    "financials_open_source_lineage": "financials_open_source_lineage.parquet",
                    "financials_open_source_source_summary": "financials_open_source_source_summary.parquet",
                    "financials_open_source_consolidated_dir": "financials_open_source_consolidated/",
                    "price_alignment": f"price_alignment_{year}.parquet",
                    "financial_alignment": f"financial_alignment_{year}.parquet",
                    "price_error_summary": "price_error_summary.parquet",
                    "statement_error_summary": "statement_error_summary.parquet",
                    "metric_error_summary": "metric_error_summary.parquet",
                    "ticker_error_summary": "ticker_error_summary.parquet",
                    "ticker_metric_error_summary": "ticker_metric_error_summary.parquet",
                    "price_ticker_error_summary": "price_ticker_error_summary.parquet",
                    "price_ticker_metric_error_summary": "price_ticker_metric_error_summary.parquet",
                    "price_error_details": "price_error_details.parquet",
                    "financial_error_details": "financial_error_details.parquet",
                    "report_html": "report.html",
                    "ticker_report_index_html": "tickers/index.html",
                    "kpi_report_index_html": "kpis/index.html",
                    "summary": "summary.json",
                },
                "source_presence": {
                    "yfinance_financials": include_yfinance_financials,
                    "open_source_earnings": include_yfinance_earnings,
                    "sec_companyfacts": not sec_financials.is_empty(),
                    "sec_filing": not sec_filing_financials.is_empty(),
                    "simfin": not simfin_financials.is_empty(),
                    "open_source_consolidated": not consolidated_financials.is_empty(),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def _default_audit_folder(*, year: int, universe: str, tickers: Sequence[str] | None) -> str:
    if tickers is not None:
        return f"custom_{year}"
    if universe == "sp500-2025":
        return f"sp500_{year}"
    return f"{universe.replace('-', '_')}_{year}"


def _empty_financials() -> pl.DataFrame:
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


def _empty_sec_profile_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "cik": pl.String,
            "sic": pl.String,
            "sic_description": pl.String,
        }
    )


def _concat_or_empty(frames: list[pl.DataFrame], *, empty: pl.DataFrame) -> pl.DataFrame:
    return pl.concat(frames, how="vertical") if frames else empty


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
            except Exception as exc:
                print(f"SEC fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc)})
    return frames, failures


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
            except Exception as exc:
                print(f"SEC earnings actual fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_actuals"})
    return frames, failures


def _fetch_sec_earnings_calendar(
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
            executor.submit(sec_client.extract_earnings_calendar, str(row["ticker"]), str(row["cik"]), [year]): str(row["ticker"])
            for row in rows
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
                print(f"SEC earnings calendar fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_calendar"})
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
            except Exception as exc:
                print(f"SEC company profile fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "general_reference"})
    return frames, failures


def _extract_sec_companyfacts_earnings_actuals(
    sec_client: SecCompanyFactsClient,
    ticker: str,
    cik: str,
) -> pl.DataFrame:
    payload = sec_client.fetch_company_facts(cik)
    return build_sec_companyfacts_earnings_actuals(ticker=ticker, facts_payload=payload)


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
            except Exception as exc:
                print(f"SEC filing fetch failed for {ticker}: {exc}")
                failures.append({"ticker": ticker, "error": str(exc)})
    return frames, failures


def _identify_sec_filing_fallback_tickers(
    tickers: tuple[str, ...],
    sec_companyfacts: pl.DataFrame,
) -> tuple[str, ...]:
    return _identify_metric_gap_tickers(
        tickers=tickers,
        financials=sec_companyfacts,
        supported_metrics={
            (spec.statement, spec.metric)
            for spec in METRIC_SPECS
            if spec.statement != "earnings" and (spec.sec_tags or spec.metric in {"free_cash_flow"})
        },
    )


def _identify_yfinance_financial_fallback_tickers(
    *,
    tickers: tuple[str, ...],
    sec_companyfacts: pl.DataFrame,
    sec_filing: pl.DataFrame,
) -> tuple[str, ...]:
    sec_combined = pl.concat([sec_companyfacts, sec_filing], how="vertical") if not sec_filing.is_empty() else sec_companyfacts
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
