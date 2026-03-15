#!/usr/bin/env python3
from __future__ import annotations

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


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def latest_open_source_output_dir(base_dir: str | Path | None = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir else PROJECT_ROOT / "data" / "open_source"
    candidates = [
        path for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("_") and (path / "run_config.json").exists()
    ]
    if not candidates:
        raise FileNotFoundError(f"No open-source output directory with run_config.json found under {root}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main(
    *,
    output_dir: str | Path | None = None,
    year: int | None = None,
    threshold_pct: float | None = None,
    include_yfinance_financials: bool = False,
    include_yfinance_earnings: bool = False,
    include_sec_filing_financials: bool = False,
    include_simfin_financials: bool = False,
    include_open_source_consolidated: bool = False,
) -> None:
    resolved_output_dir = Path(output_dir).expanduser().resolve() if output_dir else latest_open_source_output_dir()
    config_path = resolved_output_dir / "run_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
    year = year if year is not None else int(config.get("year", 2025))
    threshold_pct = threshold_pct if threshold_pct is not None else float(config.get("threshold_pct", 0.5))
    tickers = tuple(config.get("tickers", []))

    coverage = pl.read_parquet(resolved_output_dir / f"ticker_coverage_{year}.parquet")
    price_alignment = pl.read_parquet(resolved_output_dir / f"price_alignment_{year}.parquet")
    financial_alignment = pl.read_parquet(resolved_output_dir / f"financial_alignment_{year}.parquet")
    consolidated_financials = _read_optional_parquet(resolved_output_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage = _read_optional_parquet(resolved_output_dir / "financials_open_source_lineage.parquet")
    consolidation_source_summary = _read_optional_parquet(resolved_output_dir / "financials_open_source_source_summary.parquet")
    include_yfinance_financials = include_yfinance_financials or _has_alignment_source(financial_alignment, "yfinance")
    include_yfinance_earnings = include_yfinance_earnings or _has_alignment_source(financial_alignment, "yfinance_earnings")
    include_sec_filing_financials = include_sec_filing_financials or _has_alignment_source(financial_alignment, "sec_filing")
    include_simfin_financials = include_simfin_financials or _has_alignment_source(financial_alignment, "simfin")
    include_open_source_consolidated = include_open_source_consolidated or _has_alignment_source(
        financial_alignment, "open_source_consolidated"
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
        include_yfinance_earnings=include_yfinance_earnings,
        include_sec_filing_financials=include_sec_filing_financials,
        include_simfin_financials=include_simfin_financials,
        include_open_source_consolidated=include_open_source_consolidated,
    )

    audited_metric_catalog.write_parquet(resolved_output_dir / "audited_metric_catalog.parquet")
    price_summary.write_parquet(resolved_output_dir / "price_error_summary.parquet")
    statement_summary.write_parquet(resolved_output_dir / "statement_error_summary.parquet")
    metric_summary.write_parquet(resolved_output_dir / "metric_error_summary.parquet")
    ticker_summary.write_parquet(resolved_output_dir / "ticker_error_summary.parquet")
    ticker_metric_summary.write_parquet(resolved_output_dir / "ticker_metric_error_summary.parquet")
    price_ticker_summary.write_parquet(resolved_output_dir / "price_ticker_error_summary.parquet")
    price_ticker_metric_summary.write_parquet(resolved_output_dir / "price_ticker_metric_error_summary.parquet")
    price_error_details.write_parquet(resolved_output_dir / "price_error_details.parquet")
    financial_error_details.write_parquet(resolved_output_dir / "financial_error_details.parquet")

    summarize_alignment(
        tickers=tickers,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=resolved_output_dir / "summary.json",
    )
    write_html_report(
        output_path=resolved_output_dir / "report.html",
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
        output_dir=resolved_output_dir,
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

    print(f"Reports rebuilt in: {resolved_output_dir}")


def _has_alignment_source(financial_alignment: pl.DataFrame, source: str) -> bool:
    return financial_alignment.filter(pl.col("source") == source).height > 0


def _read_optional_parquet(path: Path) -> pl.DataFrame:
    if path.exists():
        return pl.read_parquet(path)
    return pl.DataFrame()


if __name__ == "__main__":
    main()
