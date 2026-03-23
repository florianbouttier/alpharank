from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import json
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_error_detail_tables,
    build_error_summary_tables,
    build_price_alignment,
    load_eodhd_prices_between,
    summarize_alignment,
    write_detail_reports,
    write_html_report,
)
from alpharank.data.open_source.yahoo import YahooFinanceClient


@dataclass(frozen=True)
class OpenSourcePriceTransitionResult:
    output_dir: Path
    start_date: str
    end_date: str
    ticker_count: int
    yahoo_price_rows: int
    eodhd_price_rows: int
    price_alignment_rows: int


def run_open_source_price_transition(
    *,
    reference_data_dir: Path,
    output_dir: Path,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    tickers: Sequence[str] | None = None,
    threshold_pct: float = 0.5,
) -> OpenSourcePriceTransitionResult:
    yahoo_client = YahooFinanceClient(cache_dir=reference_data_dir / "open_source" / "_cache" / "yfinance")
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    ticker_list = tuple(tickers) if tickers is not None else _load_reference_tickers(reference_data_dir, start_date=start_date)
    output_dir.mkdir(parents=True, exist_ok=True)

    yahoo_prices = yahoo_client.download_prices(ticker_list, start_date, end_date)
    benchmark_prices = yahoo_client.download_prices(["SPY"], start_date, end_date)
    eodhd_prices = load_eodhd_prices_between(reference_data_dir, ticker_list, start_date=start_date, end_date=end_date)

    coverage = _build_price_coverage(ticker_list, yahoo_prices)
    audited_metric_catalog = pl.DataFrame(
        [
            {
                "source": "yfinance",
                "statement": "price",
                "metric": "adjusted_close",
                "reference_field": "adjusted_close",
                "open_source_field": "adjusted_close",
            }
        ]
    )
    price_alignment = build_price_alignment(eodhd_prices, yahoo_prices)
    financial_alignment = _empty_alignment_frame()
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

    yahoo_prices.write_parquet(output_dir / "US_Finalprice.parquet")
    benchmark_prices.write_parquet(output_dir / "SP500Price.parquet")
    coverage.write_parquet(output_dir / "ticker_coverage.parquet")
    audited_metric_catalog.write_parquet(output_dir / "audited_metric_catalog.parquet")
    price_alignment.write_parquet(output_dir / "price_alignment.parquet")
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
        tickers=ticker_list,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=output_dir / "summary.json",
    )
    write_html_report(
        output_path=output_dir / "report.html",
        year=int(start_date[:4]),
        threshold_pct=threshold_pct,
        benchmark_tickers=ticker_list,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        consolidation_source_summary=_empty_source_summary(),
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_metric_summary=ticker_metric_summary,
        price_ticker_summary=price_ticker_summary,
    )
    write_detail_reports(
        output_dir=output_dir,
        year=int(start_date[:4]),
        threshold_pct=threshold_pct,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        consolidated_financials=_empty_consolidated_frame(),
        consolidated_lineage=_empty_lineage_frame(),
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
                "start_date": start_date,
                "end_date": end_date,
                "tickers": list(ticker_list),
                "threshold_pct": threshold_pct,
                "files": {
                    "final_price": "US_Finalprice.parquet",
                    "sp500_price": "SP500Price.parquet",
                    "ticker_coverage": "ticker_coverage.parquet",
                    "price_alignment": "price_alignment.parquet",
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
                },
                "source_presence": {"yfinance_prices": not yahoo_prices.is_empty()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return OpenSourcePriceTransitionResult(
        output_dir=output_dir,
        start_date=start_date,
        end_date=end_date,
        ticker_count=len(ticker_list),
        yahoo_price_rows=yahoo_prices.height,
        eodhd_price_rows=eodhd_prices.height,
        price_alignment_rows=price_alignment.height,
    )


def _load_reference_tickers(reference_data_dir: Path, *, start_date: str) -> tuple[str, ...]:
    return tuple(
        pl.read_parquet(reference_data_dir / "US_Finalprice.parquet")
        .filter(pl.col("date") >= pl.lit(start_date))
        .select(pl.col("ticker").str.replace(r"\.US$", "").alias("ticker"))
        .unique()
        .sort("ticker")
        .to_series()
        .to_list()
    )


def _build_price_coverage(tickers: Sequence[str], yahoo_prices: pl.DataFrame) -> pl.DataFrame:
    availability = (
        yahoo_prices.select("ticker").unique().with_columns(pl.lit(True).alias("yahoo_price_available"))
        if not yahoo_prices.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "yahoo_price_available": pl.Boolean})
    )
    return (
        pl.DataFrame({"ticker_root": list(tickers)})
        .with_columns(
            [
                (pl.col("ticker_root") + pl.lit(".US")).alias("ticker"),
                pl.lit(True).alias("selected_for_benchmark"),
                pl.lit(False).alias("sec_filing_available"),
            ]
        )
        .join(availability, on="ticker", how="left", coalesce=True)
        .with_columns(
            [
                pl.col("yahoo_price_available").fill_null(False),
            ]
        )
        .with_columns(pl.col("yahoo_price_available").alias("available_in_yahoo_or_sec"))
        .sort("ticker")
    )


def _empty_alignment_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "eodhd_filing_date": pl.String,
            "open_filing_date": pl.String,
            "eodhd_value": pl.Float64,
            "open_value": pl.Float64,
            "eodhd_source_label": pl.String,
            "open_source_label": pl.String,
            "date_diff_days": pl.Int64,
            "source": pl.String,
            "match_status": pl.String,
            "value_diff": pl.Float64,
            "diff_pct": pl.Float64,
        }
    )


def _empty_source_summary() -> pl.DataFrame:
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


def _empty_consolidated_frame() -> pl.DataFrame:
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
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
            "fallback_used": pl.Boolean,
            "candidate_source_count": pl.Int64,
            "candidate_sources": pl.String,
            "candidate_source_labels": pl.String,
        }
    )


def _empty_lineage_frame() -> pl.DataFrame:
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
            "selected_source": pl.String,
            "selected_source_label": pl.String,
            "selected_form": pl.String,
            "selected_fiscal_period": pl.String,
            "selected_fiscal_year": pl.Int64,
            "source_priority": pl.Int64,
        }
    )
