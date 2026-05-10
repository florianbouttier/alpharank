#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

import polars as pl

from alpharank.data.open_source.ingestion import (
    _concat_or_empty,
    _empty_raw_financial_base,
    _fetch_sec_company_profiles,
    _fetch_sec_earnings_actuals,
    _fetch_sec_earnings_calendar,
    _fetch_sec_filing_earnings_actuals,
    _fetch_sec_filing_financials,
    _fetch_sec_financials,
    _filter_financial_year,
    _identify_general_reference_refresh_tickers,
    _identify_sec_filing_fallback_tickers,
    _load_reference_tickers,
    _upsert_financial_dataset,
    _with_earnings_ingestion_metadata,
    _with_financial_ingestion_metadata,
    _with_general_ingestion_metadata,
    _with_general_lineage_ingestion_metadata,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.sec_only import build_sec_only_general_reference
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    append_run_delta,
    new_run_id,
    read_json,
    release_json_lock,
    try_acquire_json_lock,
    upsert_parquet,
    utc_now_iso,
    write_run_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backfill SEC-only raw history for general, earnings, and financials.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--start-year", type=int, default=2005)
    parser.add_argument("--end-year", type=int, default=date.today().year)
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker roots without .US suffix.")
    parser.add_argument("--official-dir", type=Path, default=project_root / "data" / "open_source" / "official")
    parser.add_argument("--reference-data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--user-agent", default="Florian Bouttier florianbouttier@example.com")
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--calendar-workers", type=int, default=2)
    parser.add_argument("--actual-workers", type=int, default=2)
    parser.add_argument("--companyfacts-workers", type=int, default=2)
    parser.add_argument("--filing-workers", type=int, default=2)
    parser.add_argument("--filing-earnings-workers", type=int, default=2)
    parser.add_argument("--skip-run-deltas", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    paths = OpenSourceLivePaths(args.official_dir.resolve())
    paths.ensure()

    run_id = new_run_id()
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = paths.manifests_dir / "sec_backfill.lock.json"
    lock_payload = {
        "pid": os.getpid(),
        "run_id": run_id,
        "started_at": utc_now_iso(),
        "script": str(Path(__file__).resolve()),
    }
    acquired, existing_lock = try_acquire_json_lock(lock_path, lock_payload)
    if not acquired:
        raise SystemExit(f"SEC backfill already running: {json.dumps(existing_lock or {}, indent=2)}")

    try:
        _run_backfill(args=args, paths=paths, run_id=run_id)
    finally:
        release_json_lock(lock_path)


def _run_backfill(*, args: argparse.Namespace, paths: OpenSourceLivePaths, run_id: str) -> None:
    ingested_at = utc_now_iso()
    ticker_list = tuple(args.tickers) if args.tickers else _load_reference_tickers(
        args.reference_data_dir.resolve(),
        start_date=f"{args.start_year:04d}-01-01",
    )
    years = tuple(range(args.start_year, args.end_year + 1))

    sec_client = SecCompanyFactsClient(
        user_agent=args.user_agent,
        cache_dir=Path(__file__).resolve().parents[2] / "data" / "open_source" / "_cache" / "sec_companyfacts",
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=args.user_agent,
        cache_dir=Path(__file__).resolve().parents[2] / "data" / "open_source" / "_cache" / "sec_filing",
    )

    print(f"Run id: {run_id}")
    print(f"Tickers requested: {len(ticker_list)}")
    print(f"Years: {args.start_year} -> {args.end_year}")

    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(list(ticker_list))).sort("ticker")
    mapped_tickers = set(sec_mapping.get_column("ticker").cast(pl.Utf8).to_list())
    missing_mapping = sorted(ticker for ticker in ticker_list if ticker not in mapped_tickers)
    print(f"SEC mapping coverage: {sec_mapping.height}/{len(ticker_list)}")
    if missing_mapping:
        print(f"Missing SEC mapping tickers: {len(missing_mapping)}")

    existing_general_reference = (
        pl.read_parquet(paths.raw_dir / "general_reference.parquet")
        if (paths.raw_dir / "general_reference.parquet").exists()
        else pl.DataFrame()
    )
    existing_general_lineage = (
        pl.read_parquet(paths.raw_dir / "general_reference_lineage.parquet")
        if (paths.raw_dir / "general_reference_lineage.parquet").exists()
        else pl.DataFrame()
    )

    general_refresh_tickers = _identify_general_reference_refresh_tickers(
        requested_tickers=ticker_list,
        existing_general_reference=existing_general_reference,
        mode="bootstrap",
    )
    print(f"Refreshing general reference for {len(general_refresh_tickers)} tickers")
    sec_profile_frames, profile_failures = _fetch_sec_company_profiles(
        sec_filing_client,
        sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
        max_workers=args.profile_workers,
    )
    sec_profiles = _concat_or_empty(sec_profile_frames, empty=_empty_sec_profile_frame())
    general_reference_selected, general_reference_lineage_selected = build_sec_only_general_reference(
        sec_mapping=sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
        sec_profiles=sec_profiles,
    )
    general_reference_delta = _with_general_ingestion_metadata(
        general_reference_selected,
        run_id=run_id,
        ingested_at=ingested_at,
    )
    _maybe_append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference.parquet", general_reference_delta, skip_run_deltas=args.skip_run_deltas)
    general_reference = upsert_parquet(
        paths.raw_dir / "general_reference.parquet",
        general_reference_delta,
        key_cols=["ticker", "source"],
        order_cols=["ingested_at"],
    )
    general_reference_lineage_delta = _with_general_lineage_ingestion_metadata(
        general_reference_lineage_selected,
        run_id=run_id,
        ingested_at=ingested_at,
    )
    _maybe_append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference_lineage.parquet", general_reference_lineage_delta, skip_run_deltas=args.skip_run_deltas)
    general_reference_lineage = upsert_parquet(
        paths.raw_dir / "general_reference_lineage.parquet",
        general_reference_lineage_delta,
        key_cols=["ticker", "source"],
        order_cols=["ingested_at"],
    )

    print("Fetching SEC earnings calendar...")
    sec_calendar_frames, sec_calendar_failures = _fetch_sec_earnings_calendar(
        sec_filing_client,
        sec_mapping,
        years=years,
        max_workers=args.calendar_workers,
    )
    sec_calendar_delta = _with_earnings_ingestion_metadata(
        _concat_or_empty(sec_calendar_frames, empty=_empty_raw_earnings_frame()),
        dataset="earnings_sec_calendar",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    _maybe_append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_calendar.parquet", sec_calendar_delta, skip_run_deltas=args.skip_run_deltas)
    raw_earnings_sec_calendar = upsert_parquet(
        paths.raw_dir / "earnings_sec_calendar.parquet",
        sec_calendar_delta,
        key_cols=["ticker", "period_end", "reportDate", "accession_number", "source"],
        order_cols=["ingested_at"],
    )
    print(f"SEC earnings calendar rows: {raw_earnings_sec_calendar.height}")

    print("Fetching SEC earnings actuals...")
    sec_actual_frames, sec_actual_failures = _fetch_sec_earnings_actuals(
        sec_client,
        sec_mapping,
        max_workers=args.actual_workers,
    )
    print("Fetching SEC filing earnings actuals...")
    sec_filing_actual_frames, sec_filing_actual_failures = _fetch_sec_filing_earnings_actuals(
        sec_filing_client,
        sec_mapping,
        years=years,
        max_workers=args.filing_earnings_workers,
    )
    sec_actual_delta = _with_earnings_ingestion_metadata(
        _concat_or_empty(
            [
                _concat_or_empty(sec_actual_frames, empty=_empty_raw_earnings_frame()),
                _concat_or_empty(sec_filing_actual_frames, empty=_empty_raw_earnings_frame()),
            ],
            empty=_empty_raw_earnings_frame(),
        ),
        dataset="earnings_sec_actuals",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    _maybe_append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_actuals.parquet", sec_actual_delta, skip_run_deltas=args.skip_run_deltas)
    raw_earnings_sec_actuals = upsert_parquet(
        paths.raw_dir / "earnings_sec_actuals.parquet",
        sec_actual_delta,
        key_cols=["ticker", "period_end", "reportDate", "source"],
        order_cols=["ingested_at"],
    )
    print(f"SEC earnings actual rows: {raw_earnings_sec_actuals.height}")

    print("Fetching SEC companyfacts financials...")
    sec_frames, sec_failures = _fetch_sec_financials(
        sec_client,
        sec_mapping,
        max_workers=args.companyfacts_workers,
    )
    sec_financials_all = _concat_or_empty(sec_frames, empty=_empty_raw_financial_base())
    sec_companyfacts_delta = _with_financial_ingestion_metadata(
        sec_financials_all,
        dataset="financials_sec_companyfacts",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    raw_sec_financials = _upsert_financial_dataset_local(
        paths=paths,
        run_id=run_id,
        file_name="financials_sec_companyfacts.parquet",
        deltas=[sec_companyfacts_delta],
        skip_run_deltas=args.skip_run_deltas,
    )
    print(
        "SEC companyfacts financial rows: "
        f"{raw_sec_financials.height} "
        f"({raw_sec_financials.select(pl.col('date').min()).item()} -> {raw_sec_financials.select(pl.col('date').max()).item()})"
    )

    filing_deltas: list[pl.DataFrame] = []
    filing_year_summary: list[dict[str, object]] = []
    for year in years:
        sec_year = _filter_financial_year(sec_financials_all, year=year)
        fallback_tickers = _identify_sec_filing_fallback_tickers(
            tickers=ticker_list,
            sec_companyfacts=sec_year,
        )
        print(f"Year {year}: SEC filing fallback tickers {len(fallback_tickers)}")
        filing_year = _empty_raw_financial_base()
        filing_failures: list[dict[str, str]] = []
        if fallback_tickers:
            sec_filing_mapping = sec_mapping.filter(pl.col("ticker").is_in(list(fallback_tickers)))
            filing_frames, filing_failures = _fetch_sec_filing_financials(
                sec_filing_client,
                sec_filing_mapping,
                year=year,
                max_workers=args.filing_workers,
            )
            filing_year = _filter_financial_year(
                _concat_or_empty(filing_frames, empty=_empty_raw_financial_base()),
                year=year,
            )
        filing_delta = _with_financial_ingestion_metadata(
            filing_year,
            dataset="financials_sec_filing",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        filing_deltas.append(filing_delta)
        filing_year_summary.append(
            {
                "year": year,
                "companyfacts_rows": sec_year.height,
                "fallback_ticker_count": len(fallback_tickers),
                "filing_rows": filing_year.height,
                "filing_failures": len(filing_failures),
            }
        )

    raw_sec_filing = _upsert_financial_dataset_local(
        paths=paths,
        run_id=run_id,
        file_name="financials_sec_filing.parquet",
        deltas=filing_deltas,
        skip_run_deltas=args.skip_run_deltas,
    )
    print(
        "SEC filing financial rows: "
        f"{raw_sec_filing.height} "
        f"({raw_sec_filing.select(pl.col('date').min()).item()} -> {raw_sec_filing.select(pl.col('date').max()).item()})"
    )

    failures = {
        "profile_failures": profile_failures,
        "earnings_calendar_failures": sec_calendar_failures,
        "earnings_actual_failures": sec_actual_failures,
        "earnings_filing_actual_failures": sec_filing_actual_failures,
        "companyfacts_failures": sec_failures,
        "missing_sec_mapping": missing_mapping,
    }

    manifest = {
        "run_id": run_id,
        "mode": "sec_core_history_backfill",
        "official_dir": str(paths.base_dir),
        "raw_dir": str(paths.raw_dir),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "ticker_count_requested": len(ticker_list),
        "ticker_count_sec_mapped": sec_mapping.height,
        "general_rows": general_reference.height,
        "general_lineage_rows": general_reference_lineage.height,
        "earnings_sec_calendar_rows": raw_earnings_sec_calendar.height,
        "earnings_sec_actual_rows": raw_earnings_sec_actuals.height,
        "financials_sec_companyfacts_rows": raw_sec_financials.height,
        "financials_sec_filing_rows": raw_sec_filing.height,
        "financials_sec_companyfacts_date_min": raw_sec_financials.select(pl.col("date").min()).item() if not raw_sec_financials.is_empty() else None,
        "financials_sec_companyfacts_date_max": raw_sec_financials.select(pl.col("date").max()).item() if not raw_sec_financials.is_empty() else None,
        "financials_sec_filing_date_min": raw_sec_filing.select(pl.col("date").min()).item() if not raw_sec_filing.is_empty() else None,
        "financials_sec_filing_date_max": raw_sec_filing.select(pl.col("date").max()).item() if not raw_sec_filing.is_empty() else None,
        "filing_year_summary": filing_year_summary,
        "failures": failures,
        "generated_at": utc_now_iso(),
        "skip_run_deltas": args.skip_run_deltas,
    }
    write_run_manifest(paths, run_id, manifest)
    (paths.run_dir(run_id) / "summary.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Backfill manifest: {paths.run_dir(run_id) / 'manifest.json'}")


def _empty_sec_profile_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "cik": pl.String,
            "sic": pl.String,
            "sic_description": pl.String,
        }
    )


def _maybe_append_run_delta(path: Path, frame: pl.DataFrame, *, skip_run_deltas: bool) -> None:
    if not skip_run_deltas:
        append_run_delta(path, frame)


def _upsert_financial_dataset_local(
    *,
    paths: OpenSourceLivePaths,
    run_id: str,
    file_name: str,
    deltas: list[pl.DataFrame],
    skip_run_deltas: bool,
) -> pl.DataFrame:
    delta = _concat_or_empty(deltas)
    _maybe_append_run_delta(paths.run_dir(run_id) / "raw" / file_name, delta, skip_run_deltas=skip_run_deltas)
    return upsert_parquet(
        paths.raw_dir / file_name,
        delta,
        key_cols=["ticker", "statement", "metric", "date", "source"],
        order_cols=["filing_date", "ingested_at"],
    )


def _empty_raw_earnings_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "period_end": pl.String,
            "reportDate": pl.String,
            "earningsDatetime": pl.String,
            "epsEstimate": pl.Float64,
            "epsActual": pl.Float64,
            "surprisePercent": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
            "calendar_source": pl.String,
            "actual_source": pl.String,
            "estimate_source": pl.String,
            "accession_number": pl.String,
            "form": pl.String,
            "fiscal_period": pl.String,
            "fiscal_year": pl.Int64,
            "dataset": pl.String,
            "ingestion_run_id": pl.String,
            "ingested_at": pl.String,
        }
    )


if __name__ == "__main__":
    main()
