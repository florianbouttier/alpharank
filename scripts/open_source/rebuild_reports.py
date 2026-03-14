#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_audited_metric_catalog,
    build_error_detail_tables,
    build_error_summary_tables,
    summarize_alignment,
    write_detail_reports,
    write_html_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild open-source audit reports from existing parquet outputs.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--threshold-pct", type=float, default=None)
    parser.add_argument("--include-yfinance-financials", action="store_true")
    parser.add_argument("--include-yfinance-earnings", action="store_true")
    parser.add_argument("--include-simfin-financials", action="store_true")
    parser.add_argument("--include-best-effort-financials", action="store_true")
    args = parser.parse_args()

    config_path = args.output_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    year = args.year if args.year is not None else int(config.get("year", 2025))
    threshold_pct = args.threshold_pct if args.threshold_pct is not None else float(config.get("threshold_pct", 0.5))
    tickers = tuple(config.get("tickers", []))

    coverage = pl.read_parquet(args.output_dir / f"ticker_coverage_{year}.parquet")
    price_alignment = pl.read_parquet(args.output_dir / f"price_alignment_{year}.parquet")
    financial_alignment = pl.read_parquet(args.output_dir / f"financial_alignment_{year}.parquet")
    include_yfinance_financials = args.include_yfinance_financials or _has_alignment_source(financial_alignment, "yfinance")
    include_yfinance_earnings = args.include_yfinance_earnings or _has_alignment_source(financial_alignment, "yfinance_earnings")
    include_simfin_financials = args.include_simfin_financials or _has_alignment_source(financial_alignment, "simfin")
    include_best_effort_financials = args.include_best_effort_financials or _has_alignment_source(financial_alignment, "best_effort")

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
        include_yfinance_earnings=include_yfinance_earnings,
        include_simfin_financials=include_simfin_financials,
        include_best_effort_financials=include_best_effort_financials,
    )

    audited_metric_catalog.write_parquet(args.output_dir / "audited_metric_catalog.parquet")
    price_summary.write_parquet(args.output_dir / "price_error_summary.parquet")
    statement_summary.write_parquet(args.output_dir / "statement_error_summary.parquet")
    metric_summary.write_parquet(args.output_dir / "metric_error_summary.parquet")
    ticker_summary.write_parquet(args.output_dir / "ticker_error_summary.parquet")
    ticker_metric_summary.write_parquet(args.output_dir / "ticker_metric_error_summary.parquet")
    price_ticker_summary.write_parquet(args.output_dir / "price_ticker_error_summary.parquet")
    price_ticker_metric_summary.write_parquet(args.output_dir / "price_ticker_metric_error_summary.parquet")
    price_error_details.write_parquet(args.output_dir / "price_error_details.parquet")
    financial_error_details.write_parquet(args.output_dir / "financial_error_details.parquet")

    summarize_alignment(
        tickers=tickers,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=args.output_dir / "summary.json",
    )
    write_html_report(
        output_path=args.output_dir / "report.html",
        year=year,
        threshold_pct=threshold_pct,
        benchmark_tickers=tickers,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        price_summary=price_summary,
        statement_summary=statement_summary,
        metric_summary=metric_summary,
        ticker_summary=ticker_summary,
        ticker_metric_summary=ticker_metric_summary,
        price_ticker_summary=price_ticker_summary,
    )
    write_detail_reports(
        output_dir=args.output_dir,
        year=year,
        threshold_pct=threshold_pct,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
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

    print(f"Reports rebuilt in: {args.output_dir}")


def _has_alignment_source(financial_alignment: pl.DataFrame, source: str) -> bool:
    return financial_alignment.filter(pl.col("source") == source).height > 0


if __name__ == "__main__":
    main()
