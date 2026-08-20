#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.data.open_source.earnings import build_sec_companyfacts_earnings_actuals
from alpharank.data.open_source.sec import SecCompanyFactsClient


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Refresh SEC companyfacts raw datasets from the local SEC JSON cache.")
    parser.add_argument("--raw-dir", type=Path, default=project_root / "data" / "open_source" / "official" / "raw")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=project_root / "data" / "open_source" / "_cache" / "sec_companyfacts",
    )
    parser.add_argument(
        "--tickers",
        nargs="*",
        default=None,
        help="Optional list of tickers like AAPL.US. If omitted, refresh all mapped tickers found in raw general lineage.",
    )
    parser.add_argument(
        "--ingestion-run-id",
        type=str,
        default="cache_refresh",
        help="Metadata label written into the refreshed raw rows.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    raw_dir = args.raw_dir.resolve()
    cache_dir = args.cache_dir.resolve()

    general_lineage = pl.read_parquet(raw_dir / "general_reference_lineage.parquet")
    mapping = (
        general_lineage.select(["ticker", "sec_cik"])
        .rename({"sec_cik": "cik"})
        .filter(pl.col("ticker").is_not_null() & pl.col("cik").is_not_null())
        .with_columns(pl.col("cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10))
        .unique(subset=["ticker"], keep="first")
    )
    if args.tickers:
        requested = [ticker.strip().upper() for ticker in args.tickers if ticker.strip()]
        mapping = mapping.filter(pl.col("ticker").str.to_uppercase().is_in(requested))

    client = SecCompanyFactsClient(
        user_agent="alpharank-sec-cache-refresh",
        cache_dir=cache_dir,
    )
    ingested_at = datetime.now(timezone.utc).isoformat()

    financial_frames: list[pl.DataFrame] = []
    earnings_frames: list[pl.DataFrame] = []
    failures: list[dict[str, str]] = []
    for row in mapping.iter_rows(named=True):
        ticker_us = str(row["ticker"])
        ticker_root = ticker_us.removesuffix(".US")
        cik = str(row["cik"])
        try:
            facts_payload = client.fetch_company_facts(cik)
            financials = client.extract_financials(ticker_root, cik)
            if not financials.is_empty():
                financial_frames.append(
                    financials.with_columns(
                        [
                            pl.lit("financials_sec_companyfacts").alias("dataset"),
                            pl.lit(args.ingestion_run_id).alias("ingestion_run_id"),
                            pl.lit(ingested_at).alias("ingested_at"),
                        ]
                    ).select(
                        [
                            "ticker",
                            "statement",
                            "metric",
                            "date",
                            "filing_date",
                            "value",
                            "source",
                            "source_label",
                            "form",
                            "fiscal_period",
                            "fiscal_year",
                            "dataset",
                            "ingestion_run_id",
                            "ingested_at",
                            "accession_number",
                        ]
                    )
                )
            earnings = build_sec_companyfacts_earnings_actuals(ticker=ticker_root, facts_payload=facts_payload)
            if not earnings.is_empty():
                earnings_frames.append(
                    earnings.with_columns(
                        [
                            pl.lit(None).cast(pl.Utf8).alias("earningsDatetime"),
                            pl.lit(None).cast(pl.Float64).alias("epsEstimate"),
                            pl.lit(None).cast(pl.Float64).alias("surprisePercent"),
                            pl.lit(None).cast(pl.Utf8).alias("calendar_source"),
                            pl.col("source").alias("actual_source"),
                            pl.lit(None).cast(pl.Utf8).alias("estimate_source"),
                            pl.lit("earnings_sec_actuals").alias("dataset"),
                            pl.lit(args.ingestion_run_id).alias("ingestion_run_id"),
                            pl.lit(ingested_at).alias("ingested_at"),
                            pl.lit(None).cast(pl.Utf8).alias("accession_number"),
                        ]
                    ).select(
                        [
                            "ticker",
                            "period_end",
                            "reportDate",
                            "earningsDatetime",
                            "epsEstimate",
                            "epsActual",
                            "surprisePercent",
                            "source",
                            "source_label",
                            "calendar_source",
                            "actual_source",
                            "estimate_source",
                            "accession_number",
                            "form",
                            "fiscal_period",
                            "fiscal_year",
                            "dataset",
                            "ingestion_run_id",
                            "ingested_at",
                        ]
                    )
                )
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            failures.append({"ticker": ticker_us, "error": str(exc)})

    refreshed_financials = pl.concat(financial_frames, how="vertical_relaxed") if financial_frames else pl.DataFrame()
    refreshed_earnings = pl.concat(earnings_frames, how="vertical_relaxed") if earnings_frames else pl.DataFrame()

    _upsert_by_ticker(
        path=raw_dir / "financials_sec_companyfacts.parquet",
        tickers=mapping.get_column("ticker").to_list(),
        refreshed=refreshed_financials,
    )
    _upsert_by_ticker(
        path=raw_dir / "earnings_sec_actuals.parquet",
        tickers=mapping.get_column("ticker").to_list(),
        refreshed=refreshed_earnings,
    )

    print(f"Tickers refreshed: {mapping.height}")
    print(f"Financial rows refreshed: {refreshed_financials.height}")
    print(f"Earnings rows refreshed: {refreshed_earnings.height}")
    if failures:
        print(f"Failures: {len(failures)}")
        for failure in failures[:20]:
            print(f"  - {failure['ticker']}: {failure['error']}")


def _upsert_by_ticker(*, path: Path, tickers: list[str], refreshed: pl.DataFrame) -> None:
    existing = pl.read_parquet(path)
    remaining = existing.filter(~pl.col("ticker").is_in(tickers))
    if refreshed.is_empty():
        combined = remaining
    else:
        combined = pl.concat([remaining, refreshed], how="diagonal_relaxed")
    combined.write_parquet(path)


if __name__ == "__main__":
    main()
