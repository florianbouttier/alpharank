from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_coverage_audit,
    build_error_summary_tables,
    build_financial_alignment,
    build_price_alignment,
    load_eodhd_prices,
    load_sp500_tickers_for_year,
    normalize_eodhd_financials,
    summarize_alignment,
    write_html_report,
)
from alpharank.data.open_source.config import PILOT_TICKERS
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.yahoo import YahooFinanceClient


@dataclass(frozen=True)
class OpenSourceCadrageResult:
    output_dir: Path
    tickers: tuple[str, ...]
    sp500_ticker_count: int
    coverage_available_in_yahoo_or_sec: int
    price_rows: int
    sec_rows: int
    yfinance_financial_rows: int
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
) -> OpenSourceCadrageResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or project_root / "data"
    output_dir = output_dir or project_root / "data" / "open_source" / f"pilot_{year}"
    output_dir.mkdir(parents=True, exist_ok=True)

    sp500_tickers = load_sp500_tickers_for_year(reference_data_dir, year)
    if tickers is not None:
        ticker_list = tuple(tickers)
    elif universe == "sp500-2025":
        ticker_list = sp500_tickers
    else:
        ticker_list = tuple(PILOT_TICKERS)
    yahoo_client = YahooFinanceClient()
    sec_client = SecCompanyFactsClient(user_agent=user_agent)

    sec_mapping_all = sec_client.fetch_company_mapping()
    yahoo_availability = yahoo_client.audit_price_availability(
        sp500_tickers,
        start_date=f"{year}-12-01",
        end_date=f"{year}-12-15",
    )
    coverage = build_coverage_audit(
        sp500_tickers=sp500_tickers,
        benchmark_tickers=ticker_list,
        sec_mapping=sec_mapping_all,
        yahoo_availability=yahoo_availability,
    )

    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(ticker_list))
    general_reference = yahoo_client.fetch_general_reference(ticker_list, sec_mapping)
    yahoo_prices = yahoo_client.download_prices(ticker_list, f"{year}-01-01", f"{year + 1}-01-01")
    yahoo_earnings = yahoo_client.fetch_earnings_dates(ticker_list)
    yahoo_financials = yahoo_client.fetch_quarterly_financials(ticker_list).filter(pl.col("date").str.starts_with(f"{year}"))

    sec_frames = []
    for row in sec_mapping.select(["ticker", "cik"]).iter_rows(named=True):
        sec_frames.append(sec_client.extract_financials(str(row["ticker"]), str(row["cik"])))
    sec_financials = pl.concat(sec_frames, how="vertical") if sec_frames else _empty_financials()
    sec_financials = sec_financials.filter(pl.col("date").str.starts_with(f"{year}"))

    eodhd_prices = load_eodhd_prices(reference_data_dir, ticker_list, year)
    eodhd_financials = normalize_eodhd_financials(reference_data_dir, ticker_list, year)
    price_alignment = build_price_alignment(eodhd_prices, yahoo_prices)
    financial_alignment = pl.concat(
        [
            build_financial_alignment(eodhd_financials, sec_financials, "sec_companyfacts"),
            build_financial_alignment(eodhd_financials, yahoo_financials, "yfinance"),
        ],
        how="vertical",
    )
    price_summary, statement_summary, metric_summary = build_error_summary_tables(
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        threshold_pct=threshold_pct,
    )

    _write_outputs(
        output_dir=output_dir,
        coverage=coverage,
        general_reference=general_reference,
        yahoo_prices=yahoo_prices,
        yahoo_earnings=yahoo_earnings,
        yahoo_financials=yahoo_financials,
        sec_financials=sec_financials,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        tickers=ticker_list,
        year=year,
        threshold_pct=threshold_pct,
    )

    return OpenSourceCadrageResult(
        output_dir=output_dir,
        tickers=ticker_list,
        sp500_ticker_count=len(sp500_tickers),
        coverage_available_in_yahoo_or_sec=int(coverage.select(pl.col("available_in_yahoo_or_sec").sum()).item()),
        price_rows=yahoo_prices.height,
        sec_rows=sec_financials.height,
        yfinance_financial_rows=yahoo_financials.height,
        price_alignment_rows=price_alignment.height,
        financial_alignment_rows=financial_alignment.height,
    )


def _write_outputs(
    *,
    output_dir: Path,
    coverage: pl.DataFrame,
    general_reference: pl.DataFrame,
    yahoo_prices: pl.DataFrame,
    yahoo_earnings: pl.DataFrame,
    yahoo_financials: pl.DataFrame,
    sec_financials: pl.DataFrame,
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    price_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    tickers: tuple[str, ...],
    year: int,
    threshold_pct: float,
) -> None:
    coverage.write_parquet(output_dir / f"ticker_coverage_{year}.parquet")
    general_reference.write_parquet(output_dir / "general_reference.parquet")
    yahoo_prices.write_parquet(output_dir / "prices_yfinance.parquet")
    yahoo_earnings.write_parquet(output_dir / "earnings_yfinance.parquet")
    yahoo_financials.write_parquet(output_dir / "financials_yfinance.parquet")
    sec_financials.write_parquet(output_dir / "financials_sec_companyfacts.parquet")
    price_alignment.write_parquet(output_dir / f"price_alignment_{year}.parquet")
    financial_alignment.write_parquet(output_dir / f"financial_alignment_{year}.parquet")
    price_summary.write_parquet(output_dir / "price_error_summary.parquet")
    statement_summary.write_parquet(output_dir / "statement_error_summary.parquet")
    metric_summary.write_parquet(output_dir / "metric_error_summary.parquet")
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
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "year": year,
                "tickers": list(tickers),
                "threshold_pct": threshold_pct,
                "files": {
                    "ticker_coverage": f"ticker_coverage_{year}.parquet",
                    "general_reference": "general_reference.parquet",
                    "prices_yfinance": "prices_yfinance.parquet",
                    "earnings_yfinance": "earnings_yfinance.parquet",
                    "financials_yfinance": "financials_yfinance.parquet",
                    "financials_sec_companyfacts": "financials_sec_companyfacts.parquet",
                    "price_alignment": f"price_alignment_{year}.parquet",
                    "financial_alignment": f"financial_alignment_{year}.parquet",
                    "price_error_summary": "price_error_summary.parquet",
                    "statement_error_summary": "statement_error_summary.parquet",
                    "metric_error_summary": "metric_error_summary.parquet",
                    "report_html": "report.html",
                    "summary": "summary.json",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )


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
