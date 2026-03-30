from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Sequence

import polars as pl

from alpharank.data.open_source.benchmark import (
    build_audited_metric_catalog,
    build_coverage_audit,
    build_earnings_alignment,
    build_error_detail_tables,
    build_error_summary_tables,
    build_financial_alignment,
    build_price_alignment,
    load_eodhd_prices,
    load_eodhd_prices_between,
    load_sp500_tickers_for_year,
    normalize_eodhd_earnings,
    normalize_eodhd_financials,
    summarize_alignment,
    write_detail_reports,
    write_html_report,
)
from alpharank.data.open_source.consolidation import FinancialSourceInput, consolidate_financial_sources
from alpharank.data.open_source.config import GENERAL_COLUMNS, METRIC_SPECS
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
from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs
from alpharank.data.open_source.publishing import publish_open_source_output_package
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.simfin import SimFinClient
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    append_run_delta,
    new_run_id,
    upsert_parquet,
    utc_now_iso,
    write_run_manifest,
)
from alpharank.data.open_source.yahoo import YahooFinanceClient


RAW_PRICE_SCHEMA = {
    "date": pl.String,
    "open": pl.Float64,
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64,
    "adjusted_close": pl.Float64,
    "ticker": pl.String,
    "source": pl.String,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}

RAW_FINANCIAL_SCHEMA = {
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
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}

RAW_EARNINGS_SCHEMA = {
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

RAW_GENERAL_SCHEMA = {
    "ticker": pl.String,
    "name": pl.String,
    "exchange": pl.String,
    "cik": pl.String,
    "source": pl.String,
    "Sector": pl.String,
    "industry": pl.String,
    "sector_source": pl.String,
    "sector_raw_value": pl.String,
    "sic": pl.String,
    "sic_description": pl.String,
    "mapping_rule": pl.String,
    "dataset": pl.String,
    "ingestion_run_id": pl.String,
    "ingested_at": pl.String,
}


@dataclass(frozen=True)
class OpenSourceIngestionResult:
    mode: str
    run_id: str
    live_dir: Path
    raw_dir: Path
    target_dir: Path
    clean_dir: Path
    legacy_dir: Path
    output_dir: Path
    output_lineage_dir: Path
    output_snapshot_dir: Path | None
    audit_dirs: tuple[Path, ...]
    ticker_count: int
    price_start_date: str
    price_end_date: str
    refreshed_years: tuple[int, ...]
    price_rows: int
    consolidated_rows: int
    lineage_rows: int


@dataclass(frozen=True)
class OpenSourceReferenceRefreshResult:
    run_id: str
    live_dir: Path
    raw_dir: Path
    target_dir: Path
    output_dir: Path
    output_lineage_dir: Path
    output_snapshot_dir: Path | None
    audit_dirs: tuple[Path, ...]
    ticker_count: int
    refreshed_years: tuple[int, ...]
    general_rows: int
    general_sector_non_null_rows: int
    earnings_rows: int
    earnings_tickers: int


def repair_open_source_price_history(
    *,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    tickers: Sequence[str] | None = None,
    live_dir: Path | None = None,
    reference_data_dir: Path | None = None,
    audit_years: Sequence[int] = (),
    threshold_pct: float = 0.5,
) -> OpenSourceIngestionResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or (project_root / "data")
    open_source_root = project_root / "data" / "open_source"
    paths = OpenSourceLivePaths(
        live_dir or (open_source_root / "official"),
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = new_run_id()
    ingested_at = utc_now_iso()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    if tickers is None:
        current_sp500 = set(load_sp500_tickers_for_year(reference_data_dir, date.today().year))
        existing_price_tickers = set(_load_existing_price_tickers(paths))
        ticker_list = tuple(sorted(current_sp500 | existing_price_tickers))
    else:
        ticker_list = tuple(tickers)

    yahoo_client = YahooFinanceClient(cache_dir=project_root / "data" / "open_source" / "_cache" / "yfinance")
    existing_raw_prices = (
        pl.read_parquet(paths.raw_dir / "prices_yfinance.parquet")
        if (paths.raw_dir / "prices_yfinance.parquet").exists()
        else _empty_raw_price_frame()
    )
    backfill_tickers = _identify_price_history_backfill_tickers(
        requested_tickers=ticker_list,
        existing_prices=existing_raw_prices,
        explicit_start_date=start_date,
        mode="daily",
    )

    price_deltas: list[pl.DataFrame] = []
    if backfill_tickers:
        price_deltas.append(
            _with_price_ingestion_metadata(
                yahoo_client.download_prices(backfill_tickers, start_date, end_date),
                dataset="prices_yfinance_backfill",
                run_id=run_id,
                ingested_at=ingested_at,
            )
        )
    prices_delta = _concat_or_empty(price_deltas, empty=_empty_raw_price_frame())
    if not prices_delta.is_empty():
        prices_delta = (
            prices_delta.sort(["ticker", "date", "source", "dataset", "ingested_at"])
            .unique(subset=["ticker", "date", "source"], keep="last", maintain_order=True)
            .sort(["ticker", "date"])
        )

    benchmark_delta = _with_price_ingestion_metadata(
        yahoo_client.download_prices(["SPY"], start_date, end_date),
        dataset="prices_spy_yfinance_repair",
        run_id=run_id,
        ingested_at=ingested_at,
    )

    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_yfinance.parquet", prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_spy_yfinance.parquet", benchmark_delta)
    raw_prices = upsert_parquet(
        paths.raw_dir / "prices_yfinance.parquet",
        prices_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )
    raw_prices = _canonicalize_price_tickers(raw_prices, ticker_list=ticker_list)
    raw_prices.write_parquet(paths.raw_dir / "prices_yfinance.parquet")
    raw_benchmark_prices = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )

    clean_prices = raw_prices.select(["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]).sort(
        ["ticker", "date"]
    )
    clean_benchmark_prices = raw_benchmark_prices.select(
        ["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]
    ).sort(["ticker", "date"])
    clean_prices.write_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_benchmark_prices.write_parquet(paths.clean_dir / "benchmark_prices_open_source.parquet")

    general_reference = pl.read_parquet(paths.clean_dir / "general_reference.parquet")
    general_reference_lineage = pl.read_parquet(paths.clean_dir / "general_reference_lineage.parquet")
    consolidated_financials = pl.read_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage = pl.read_parquet(paths.clean_dir / "financials_open_source_lineage.parquet")
    source_summary = pl.read_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")
    clean_earnings = pl.read_parquet(paths.clean_dir / "earnings_open_source_consolidated.parquet")
    clean_earnings_lineage = pl.read_parquet(paths.clean_dir / "earnings_open_source_lineage.parquet")
    clean_earnings_long = pl.read_parquet(paths.clean_dir / "earnings_open_source_long.parquet")

    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_consolidated=clean_earnings,
        earnings_lineage=clean_earnings_lineage,
        earnings_long_frame=clean_earnings_long,
        manifest={
            "run_id": run_id,
            "official_dir": str(paths.base_dir),
            "target_dir": str(paths.target_dir),
            "output_dir": str(paths.output_dir),
            "legacy_dir": str(paths.legacy_dir),
            "repair_type": "price_history",
            "price_backfill_ticker_count": len(backfill_tickers),
            "price_backfill_ticker_examples": list(backfill_tickers[:20]),
        },
        history_root=paths.root_dir / "history" / "output",
    )

    audit_dirs: list[Path] = []
    for year in audit_years:
        audit_dirs.append(
            _write_live_audit(
                paths=paths,
                reference_data_dir=reference_data_dir,
                year=year,
                tickers=ticker_list,
                threshold_pct=threshold_pct,
            )
        )

    manifest = {
        "run_id": run_id,
        "mode": "price_history_repair",
        "ingested_at": ingested_at,
        "ticker_count": len(ticker_list),
        "price_backfill_ticker_count": len(backfill_tickers),
        "price_backfill_ticker_examples": list(backfill_tickers[:20]),
        "price_window": {"start_date": start_date, "end_date": end_date},
        "official_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "published_output_snapshot": (
            str(published_output_paths.snapshot_dir.relative_to(paths.root_dir))
            if published_output_paths.snapshot_dir is not None
            else None
        ),
        "audit_dirs": [str(path.relative_to(paths.root_dir)) for path in audit_dirs],
    }
    write_run_manifest(paths, run_id, manifest)

    return OpenSourceIngestionResult(
        mode="price_history_repair",
        run_id=run_id,
        live_dir=paths.base_dir,
        raw_dir=paths.raw_dir,
        target_dir=paths.target_dir,
        clean_dir=paths.clean_dir,
        legacy_dir=paths.legacy_dir,
        output_dir=paths.output_dir,
        output_lineage_dir=paths.output_lineage_dir,
        output_snapshot_dir=published_output_paths.snapshot_dir,
        audit_dirs=tuple(audit_dirs),
        ticker_count=len(ticker_list),
        price_start_date=start_date,
        price_end_date=end_date,
        refreshed_years=(),
        price_rows=clean_prices.height,
        consolidated_rows=consolidated_financials.height,
        lineage_rows=consolidated_lineage.height,
    )


def refresh_open_source_reference_layers(
    *,
    start_year: int = 2005,
    end_year: int | None = None,
    tickers: Sequence[str] | None = None,
    live_dir: Path | None = None,
    reference_data_dir: Path | None = None,
    user_agent: str = "Florian Bouttier florianbouttier@example.com",
    audit_years: Sequence[int] = (),
    threshold_pct: float = 0.5,
) -> OpenSourceReferenceRefreshResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or (project_root / "data")
    open_source_root = project_root / "data" / "open_source"
    paths = OpenSourceLivePaths(
        live_dir or (open_source_root / "official"),
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = new_run_id()
    ingested_at = utc_now_iso()
    final_end_year = end_year or date.today().year
    refreshed_years = tuple(range(start_year, final_end_year + 1))
    ticker_list = tuple(tickers) if tickers is not None else _load_existing_open_source_tickers(paths, reference_data_dir)

    yahoo_client = YahooFinanceClient(cache_dir=project_root / "data" / "open_source" / "_cache" / "yfinance")
    sec_client = SecCompanyFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_companyfacts")
    sec_filing_client = SecFilingFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_filing")

    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(list(ticker_list)))

    existing_general_reference = (
        pl.read_parquet(paths.raw_dir / "general_reference.parquet")
        if (paths.raw_dir / "general_reference.parquet").exists()
        else empty_general_reference_frame()
    )
    existing_general_reference_lineage = (
        pl.read_parquet(paths.raw_dir / "general_reference_lineage.parquet")
        if (paths.raw_dir / "general_reference_lineage.parquet").exists()
        else empty_general_reference_lineage_frame()
    )
    general_refresh_tickers = _identify_general_reference_refresh_tickers(
        requested_tickers=ticker_list,
        existing_general_reference=existing_general_reference,
        mode="daily",
    )
    run_failures: list[dict[str, str]] = []
    if general_refresh_tickers:
        yahoo_general_metadata = yahoo_client.fetch_company_metadata(general_refresh_tickers)
        sec_profile_frames, profile_failures = _fetch_sec_company_profiles(
            sec_filing_client,
            sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
            max_workers=4,
        )
        run_failures.extend(profile_failures)
        sec_profiles = _concat_or_empty(sec_profile_frames, empty=_empty_sec_profile_frame())
        general_reference_selected, general_reference_lineage_selected = build_general_reference(
            tickers=general_refresh_tickers,
            sec_mapping=sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
            yahoo_metadata=yahoo_general_metadata,
            sec_profiles=sec_profiles,
        )
        general_reference_delta = _with_general_ingestion_metadata(
            general_reference_selected,
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference.parquet", general_reference_delta)
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
        append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference_lineage.parquet", general_reference_lineage_delta)
        general_reference_lineage = upsert_parquet(
            paths.raw_dir / "general_reference_lineage.parquet",
            general_reference_lineage_delta,
            key_cols=["ticker", "source"],
            order_cols=["ingested_at"],
        )
    else:
        general_reference = existing_general_reference
        general_reference_lineage = existing_general_reference_lineage
    general_reference, general_reference_lineage = _canonicalize_general_outputs(
        general_reference,
        general_reference_lineage,
    )

    raw_yahoo_earnings = (
        pl.read_parquet(paths.raw_dir / "earnings_yfinance.parquet")
        if (paths.raw_dir / "earnings_yfinance.parquet").exists()
        else _empty_raw_earnings_frame()
    )
    sec_calendar_frames, sec_calendar_failures = _fetch_sec_earnings_calendar(
        sec_filing_client,
        sec_mapping,
        years=refreshed_years,
        max_workers=4,
    )
    run_failures.extend(sec_calendar_failures)
    sec_calendar_delta = _with_earnings_ingestion_metadata(
        _concat_or_empty(sec_calendar_frames, empty=empty_earnings_calendar_frame()),
        dataset="earnings_sec_calendar",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_calendar.parquet", sec_calendar_delta)
    raw_earnings_sec_calendar = upsert_parquet(
        paths.raw_dir / "earnings_sec_calendar.parquet",
        sec_calendar_delta,
        key_cols=["ticker", "period_end", "reportDate", "accession_number", "source"],
        order_cols=["ingested_at"],
    )

    sec_actual_frames, sec_actual_failures = _fetch_sec_earnings_actuals(
        sec_client,
        sec_mapping,
        max_workers=2,
    )
    run_failures.extend(sec_actual_failures)
    sec_actual_delta = _with_earnings_ingestion_metadata(
        _concat_or_empty(sec_actual_frames, empty=empty_earnings_actuals_frame()),
        dataset="earnings_sec_actuals",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_actuals.parquet", sec_actual_delta)
    raw_earnings_sec_actuals = upsert_parquet(
        paths.raw_dir / "earnings_sec_actuals.parquet",
        sec_actual_delta,
        key_cols=["ticker", "period_end", "reportDate", "source"],
        order_cols=["ingested_at"],
    )

    clean_earnings, clean_earnings_lineage, clean_earnings_long = consolidate_earnings(
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
        yahoo_earnings=_filter_earnings_years(raw_yahoo_earnings, refreshed_years).select(
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

    clean_prices = pl.read_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_benchmark_prices = pl.read_parquet(paths.clean_dir / "benchmark_prices_open_source.parquet")
    consolidated_financials = pl.read_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage = pl.read_parquet(paths.clean_dir / "financials_open_source_lineage.parquet")
    source_summary = pl.read_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")

    clean_earnings.write_parquet(paths.clean_dir / "earnings_open_source_consolidated.parquet")
    clean_earnings_lineage.write_parquet(paths.clean_dir / "earnings_open_source_lineage.parquet")
    clean_earnings_long.write_parquet(paths.clean_dir / "earnings_open_source_long.parquet")
    general_reference.write_parquet(paths.clean_dir / "general_reference.parquet")
    general_reference_lineage.write_parquet(paths.clean_dir / "general_reference_lineage.parquet")

    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_consolidated=clean_earnings,
        earnings_lineage=clean_earnings_lineage,
        earnings_long_frame=clean_earnings_long,
        manifest={
            "run_id": run_id,
            "official_dir": str(paths.base_dir),
            "target_dir": str(paths.target_dir),
            "output_dir": str(paths.output_dir),
            "legacy_dir": str(paths.legacy_dir),
            "refresh_type": "reference_layers",
            "refreshed_years": list(refreshed_years),
        },
        history_root=paths.root_dir / "history" / "output",
    )

    audit_dirs = tuple(
        _write_live_audit(
            paths=paths,
            reference_data_dir=reference_data_dir,
            year=year,
            tickers=ticker_list,
            threshold_pct=threshold_pct,
        )
        for year in audit_years
    )

    manifest = {
        "run_id": run_id,
        "mode": "reference_refresh",
        "official_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "output_dir": str(paths.output_dir),
        "legacy_dir": str(paths.legacy_dir),
        "output_snapshot_dir": (
            str(published_output_paths.snapshot_dir.relative_to(paths.root_dir))
            if published_output_paths.snapshot_dir is not None
            else None
        ),
        "ticker_count": len(ticker_list),
        "refreshed_years": list(refreshed_years),
        "general_rows": general_reference.height,
        "general_sector_non_null_rows": general_reference.filter(pl.col("Sector").is_not_null() & (pl.col("Sector") != "")).height,
        "earnings_rows": clean_earnings.height,
        "earnings_tickers": clean_earnings.select(pl.col("ticker").n_unique()).item() if not clean_earnings.is_empty() else 0,
        "failures": run_failures,
        "audit_dirs": [str(path.relative_to(paths.root_dir)) for path in audit_dirs],
    }
    write_run_manifest(paths, run_id, manifest)

    return OpenSourceReferenceRefreshResult(
        run_id=run_id,
        live_dir=paths.base_dir,
        raw_dir=paths.raw_dir,
        target_dir=paths.target_dir,
        output_dir=paths.output_dir,
        output_lineage_dir=paths.output_lineage_dir,
        output_snapshot_dir=published_output_paths.snapshot_dir,
        audit_dirs=audit_dirs,
        ticker_count=len(ticker_list),
        refreshed_years=refreshed_years,
        general_rows=general_reference.height,
        general_sector_non_null_rows=general_reference.filter(pl.col("Sector").is_not_null() & (pl.col("Sector") != "")).height,
        earnings_rows=clean_earnings.height,
        earnings_tickers=clean_earnings.select(pl.col("ticker").n_unique()).item() if not clean_earnings.is_empty() else 0,
    )


def run_open_source_ingestion(
    *,
    mode: str = "daily",
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    tickers: Sequence[str] | None = None,
    live_dir: Path | None = None,
    reference_data_dir: Path | None = None,
    user_agent: str = "Florian Bouttier florianbouttier@example.com",
    simfin_api_key: str | None = None,
    price_lookback_days: int = 7,
    financial_lookback_years: int = 2,
    audit_years: Sequence[int] = (),
    threshold_pct: float = 0.5,
) -> OpenSourceIngestionResult:
    project_root = Path(__file__).resolve().parents[4]
    reference_data_dir = reference_data_dir or (project_root / "data")
    open_source_root = project_root / "data" / "open_source"
    paths = OpenSourceLivePaths(
        live_dir or (open_source_root / "official"),
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = new_run_id()
    ingested_at = utc_now_iso()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    ticker_list = tuple(tickers) if tickers is not None else _load_reference_tickers(reference_data_dir, start_date=start_date)
    price_start = _resolve_price_start(
        mode=mode,
        explicit_start_date=start_date,
        raw_price_path=paths.raw_dir / "prices_yfinance.parquet",
        lookback_days=price_lookback_days,
    )
    existing_raw_prices = (
        pl.read_parquet(paths.raw_dir / "prices_yfinance.parquet")
        if (paths.raw_dir / "prices_yfinance.parquet").exists()
        else _empty_raw_price_frame()
    )
    price_backfill_tickers = _identify_price_history_backfill_tickers(
        requested_tickers=ticker_list,
        existing_prices=existing_raw_prices,
        explicit_start_date=start_date,
        mode=mode,
    )
    refreshed_years = _resolve_refreshed_years(
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        lookback_years=financial_lookback_years,
    )

    yahoo_client = YahooFinanceClient(cache_dir=project_root / "data" / "open_source" / "_cache" / "yfinance")
    sec_client = SecCompanyFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_companyfacts")
    sec_filing_client = SecFilingFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_filing")
    simfin_client = SimFinClient(api_key=simfin_api_key, data_dir=project_root / "data" / "open_source" / "_cache" / "simfin")

    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(list(ticker_list)))
    existing_general_reference = (
        pl.read_parquet(paths.raw_dir / "general_reference.parquet")
        if (paths.raw_dir / "general_reference.parquet").exists()
        else empty_general_reference_frame()
    )
    existing_general_reference_lineage = (
        pl.read_parquet(paths.raw_dir / "general_reference_lineage.parquet")
        if (paths.raw_dir / "general_reference_lineage.parquet").exists()
        else empty_general_reference_lineage_frame()
    )
    general_refresh_tickers = _identify_general_reference_refresh_tickers(
        requested_tickers=ticker_list,
        existing_general_reference=existing_general_reference,
        mode=mode,
    )
    if general_refresh_tickers:
        yahoo_general_metadata = yahoo_client.fetch_company_metadata(general_refresh_tickers)
        sec_profile_frames, _ = _fetch_sec_company_profiles(
            sec_filing_client,
            sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
        )
        general_reference_selected, general_reference_lineage_selected = build_general_reference(
            tickers=general_refresh_tickers,
            sec_mapping=sec_mapping.filter(pl.col("ticker").is_in(list(general_refresh_tickers))),
            yahoo_metadata=yahoo_general_metadata,
            sec_profiles=_concat_or_empty(sec_profile_frames, empty=_empty_sec_profile_frame()),
        )
        general_reference_delta = _with_general_ingestion_metadata(
            general_reference_selected,
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference.parquet", general_reference_delta)
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
        append_run_delta(paths.run_dir(run_id) / "raw" / "general_reference_lineage.parquet", general_reference_lineage_delta)
        general_reference_lineage = upsert_parquet(
            paths.raw_dir / "general_reference_lineage.parquet",
            general_reference_lineage_delta,
            key_cols=["ticker", "source"],
            order_cols=["ingested_at"],
        )
    else:
        general_reference = existing_general_reference
        general_reference_lineage = existing_general_reference_lineage
    general_reference, general_reference_lineage = _canonicalize_general_outputs(
        general_reference,
        general_reference_lineage,
    )

    yahoo_price_deltas = [
        _with_price_ingestion_metadata(
            yahoo_client.download_prices(ticker_list, price_start, end_date),
            dataset="prices_yfinance",
            run_id=run_id,
            ingested_at=ingested_at,
        )
    ]
    if price_backfill_tickers:
        yahoo_price_deltas.append(
            _with_price_ingestion_metadata(
                yahoo_client.download_prices(price_backfill_tickers, start_date, end_date),
                dataset="prices_yfinance_backfill",
                run_id=run_id,
                ingested_at=ingested_at,
            )
        )
    yahoo_prices_delta = _concat_or_empty(yahoo_price_deltas, empty=_empty_raw_price_frame())
    if not yahoo_prices_delta.is_empty():
        yahoo_prices_delta = (
            yahoo_prices_delta.sort(["ticker", "date", "source", "dataset", "ingested_at"])
            .unique(subset=["ticker", "date", "source"], keep="last", maintain_order=True)
            .sort(["ticker", "date"])
        )
    benchmark_prices_delta = _with_price_ingestion_metadata(
        yahoo_client.download_prices(["SPY"], price_start, end_date),
        dataset="prices_spy_yfinance",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_yfinance.parquet", yahoo_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_spy_yfinance.parquet", benchmark_prices_delta)
    raw_prices = upsert_parquet(
        paths.raw_dir / "prices_yfinance.parquet",
        yahoo_prices_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )
    raw_prices = _canonicalize_price_tickers(raw_prices, ticker_list=ticker_list)
    raw_prices.write_parquet(paths.raw_dir / "prices_yfinance.parquet")
    raw_benchmark_prices = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_prices_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )

    earnings_delta = _empty_raw_earnings_frame()
    earnings_sec_calendar_delta = _empty_raw_earnings_frame()
    earnings_sec_actuals_delta = _empty_raw_earnings_frame()
    sec_financial_deltas: list[pl.DataFrame] = []
    sec_filing_deltas: list[pl.DataFrame] = []
    simfin_deltas: list[pl.DataFrame] = []
    yahoo_financial_deltas: list[pl.DataFrame] = []
    run_failures: dict[str, list[dict[str, str]]] = {
        "sec_companyfacts": [],
        "sec_filing": [],
        "simfin": [],
        "yfinance_earnings": [],
    }

    if refreshed_years:
        try:
            earnings_fetched = yahoo_client.fetch_earnings_dates(ticker_list, limit=max(8, len(refreshed_years) * 4))
        except Exception as exc:
            earnings_fetched = _empty_raw_earnings_frame()
            run_failures["yfinance_earnings"].append({"error": str(exc)})
        earnings_delta = _with_earnings_ingestion_metadata(
            earnings_fetched,
            dataset="earnings_yfinance",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_yfinance.parquet", earnings_delta)
        sec_calendar_frames, sec_calendar_failures = _fetch_sec_earnings_calendar(sec_filing_client, sec_mapping, years=refreshed_years)
        run_failures["sec_filing"].extend(sec_calendar_failures)
        earnings_sec_calendar_delta = _with_earnings_ingestion_metadata(
            _concat_or_empty(sec_calendar_frames, empty=empty_earnings_calendar_frame()),
            dataset="earnings_sec_calendar",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_calendar.parquet", earnings_sec_calendar_delta)
        sec_actual_frames, sec_actual_failures = _fetch_sec_earnings_actuals(sec_client, sec_mapping)
        run_failures["sec_companyfacts"].extend(sec_actual_failures)
        earnings_sec_actuals_delta = _with_earnings_ingestion_metadata(
            _filter_earnings_years(_concat_or_empty(sec_actual_frames, empty=empty_earnings_actuals_frame()), refreshed_years),
            dataset="earnings_sec_actuals",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_actuals.parquet", earnings_sec_actuals_delta)

    for year in refreshed_years:
        sec_frames, sec_failures = _fetch_sec_financials(sec_client, sec_mapping)
        run_failures["sec_companyfacts"].extend(sec_failures)
        sec_year = _filter_financial_year(_concat_or_empty(sec_frames), year=year)
        sec_filing_tickers = _identify_sec_filing_fallback_tickers(tickers=ticker_list, sec_companyfacts=sec_year)
        sec_filing_year = _empty_raw_financial_base()
        if sec_filing_tickers:
            sec_filing_mapping = sec_mapping.filter(pl.col("ticker").is_in(list(sec_filing_tickers)))
            sec_filing_frames, sec_filing_failures = _fetch_sec_filing_financials(
                sec_filing_client,
                sec_filing_mapping,
                year=year,
            )
            run_failures["sec_filing"].extend(sec_filing_failures)
            sec_filing_year = _filter_financial_year(_concat_or_empty(sec_filing_frames), year=year)
        yfinance_financial_tickers = _identify_yfinance_financial_fallback_tickers(
            tickers=ticker_list,
            sec_companyfacts=sec_year,
            sec_filing=sec_filing_year,
        )
        yahoo_financial_year = (
            yahoo_client.fetch_quarterly_financials(yfinance_financial_tickers).filter(pl.col("date").str.starts_with(str(year)))
            if yfinance_financial_tickers
            else _empty_raw_financial_base()
        )
        try:
            simfin_year = simfin_client.fetch_quarterly_financials(ticker_list, year) if simfin_client.enabled else _empty_raw_financial_base()
        except Exception as exc:
            simfin_year = _empty_raw_financial_base()
            run_failures["simfin"].append({"year": str(year), "error": str(exc)})
        run_failures["simfin"].extend(simfin_client.last_fetch_failures)

        sec_financial_deltas.append(
            _with_financial_ingestion_metadata(sec_year, dataset="financials_sec_companyfacts", run_id=run_id, ingested_at=ingested_at)
        )
        sec_filing_deltas.append(
            _with_financial_ingestion_metadata(sec_filing_year, dataset="financials_sec_filing", run_id=run_id, ingested_at=ingested_at)
        )
        simfin_deltas.append(
            _with_financial_ingestion_metadata(simfin_year, dataset="financials_simfin", run_id=run_id, ingested_at=ingested_at)
        )
        yahoo_financial_deltas.append(
            _with_financial_ingestion_metadata(yahoo_financial_year, dataset="financials_yfinance", run_id=run_id, ingested_at=ingested_at)
        )

    raw_earnings = upsert_parquet(
        paths.raw_dir / "earnings_yfinance.parquet",
        earnings_delta,
        key_cols=["ticker", "reportDate", "source"],
        order_cols=["ingested_at"],
    )
    raw_earnings_sec_calendar = upsert_parquet(
        paths.raw_dir / "earnings_sec_calendar.parquet",
        earnings_sec_calendar_delta,
        key_cols=["ticker", "period_end", "reportDate", "accession_number", "source"],
        order_cols=["ingested_at"],
    )
    raw_earnings_sec_actuals = upsert_parquet(
        paths.raw_dir / "earnings_sec_actuals.parquet",
        earnings_sec_actuals_delta,
        key_cols=["ticker", "period_end", "reportDate", "source"],
        order_cols=["ingested_at"],
    )
    raw_sec_financials = _upsert_financial_dataset(
        paths=paths,
        run_id=run_id,
        file_name="financials_sec_companyfacts.parquet",
        deltas=sec_financial_deltas,
    )
    raw_sec_filing_financials = _upsert_financial_dataset(
        paths=paths,
        run_id=run_id,
        file_name="financials_sec_filing.parquet",
        deltas=sec_filing_deltas,
    )
    raw_simfin_financials = _upsert_financial_dataset(
        paths=paths,
        run_id=run_id,
        file_name="financials_simfin.parquet",
        deltas=simfin_deltas,
    )
    raw_yahoo_financials = _upsert_financial_dataset(
        paths=paths,
        run_id=run_id,
        file_name="financials_yfinance.parquet",
        deltas=yahoo_financial_deltas,
    )

    clean_prices = raw_prices.select(["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]).sort(
        ["ticker", "date"]
    )
    clean_benchmark_prices = raw_benchmark_prices.select(
        ["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]
    ).sort(["ticker", "date"])
    clean_earnings, clean_earnings_lineage, clean_earnings_long = consolidate_earnings(
        sec_calendar=raw_earnings_sec_calendar.select(
            ["ticker", "period_end", "reportDate", "earningsDatetime", "accession_number", "form", "fiscal_period", "fiscal_year", "source", "source_label"]
        ),
        yahoo_earnings=raw_earnings.select(
            ["ticker", "period_end", "reportDate", "earningsDatetime", "epsEstimate", "epsActual", "surprisePercent", "source"]
        ),
        sec_actuals=raw_earnings_sec_actuals.select(
            ["ticker", "period_end", "reportDate", "epsActual", "source", "source_label", "form", "fiscal_period", "fiscal_year"]
        ),
    )

    consolidated_financials, consolidated_lineage, source_summary = consolidate_financial_sources(
        [
            FinancialSourceInput(
                source_name="sec_companyfacts",
                frame=raw_sec_financials.select(_clean_financial_columns()),
                priority=1,
            ),
            FinancialSourceInput(
                source_name="sec_filing",
                frame=raw_sec_filing_financials.select(_clean_financial_columns()),
                priority=2,
            ),
            FinancialSourceInput(
                source_name="simfin",
                frame=raw_simfin_financials.select(_clean_financial_columns()),
                priority=3,
            ),
            FinancialSourceInput(
                source_name="yfinance",
                frame=raw_yahoo_financials.select(_clean_financial_columns()),
                priority=4,
            ),
        ]
    )

    clean_prices.write_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_benchmark_prices.write_parquet(paths.clean_dir / "benchmark_prices_open_source.parquet")
    clean_earnings.write_parquet(paths.clean_dir / "earnings_open_source_consolidated.parquet")
    clean_earnings_lineage.write_parquet(paths.clean_dir / "earnings_open_source_lineage.parquet")
    clean_earnings_long.write_parquet(paths.clean_dir / "earnings_open_source_long.parquet")
    consolidated_financials.write_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage.write_parquet(paths.clean_dir / "financials_open_source_lineage.parquet")
    source_summary.write_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")
    general_reference.write_parquet(paths.clean_dir / "general_reference.parquet")
    general_reference_lineage.write_parquet(paths.clean_dir / "general_reference_lineage.parquet")

    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_consolidated=clean_earnings,
        earnings_lineage=clean_earnings_lineage,
        earnings_long_frame=clean_earnings_long,
        manifest={
            "run_id": run_id,
            "official_dir": str(paths.base_dir),
            "target_dir": str(paths.target_dir),
            "output_dir": str(paths.output_dir),
            "legacy_dir": str(paths.legacy_dir),
        },
        history_root=paths.root_dir / "history" / "output",
    )

    audit_dirs: list[Path] = []
    for year in audit_years:
        audit_dirs.append(
            _write_live_audit(
                paths=paths,
                reference_data_dir=reference_data_dir,
                year=year,
                tickers=ticker_list,
                threshold_pct=threshold_pct,
            )
        )

    manifest = {
        "run_id": run_id,
        "mode": mode,
        "ingested_at": ingested_at,
        "price_window": {"start_date": price_start, "end_date": end_date},
        "price_backfill_ticker_count": len(price_backfill_tickers),
        "price_backfill_ticker_examples": list(price_backfill_tickers[:20]),
        "financial_years_refreshed": list(refreshed_years),
        "ticker_count": len(ticker_list),
        "official_dir": str(paths.base_dir),
        "live_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "raw_outputs": {
            "general_reference": "raw/general_reference.parquet",
            "general_reference_lineage": "raw/general_reference_lineage.parquet",
            "prices_yfinance": "raw/prices_yfinance.parquet",
            "prices_spy_yfinance": "raw/prices_spy_yfinance.parquet",
            "earnings_yfinance": "raw/earnings_yfinance.parquet",
            "earnings_sec_calendar": "raw/earnings_sec_calendar.parquet",
            "earnings_sec_actuals": "raw/earnings_sec_actuals.parquet",
            "financials_sec_companyfacts": "raw/financials_sec_companyfacts.parquet",
            "financials_sec_filing": "raw/financials_sec_filing.parquet",
            "financials_simfin": "raw/financials_simfin.parquet",
            "financials_yfinance": "raw/financials_yfinance.parquet",
        },
        "clean_outputs": {
            "prices_open_source": "target/prices_open_source.parquet",
            "benchmark_prices_open_source": "target/benchmark_prices_open_source.parquet",
            "general_reference": "target/general_reference.parquet",
            "general_reference_lineage": "target/general_reference_lineage.parquet",
            "earnings_open_source_consolidated": "target/earnings_open_source_consolidated.parquet",
            "earnings_open_source_lineage": "target/earnings_open_source_lineage.parquet",
            "earnings_open_source_long": "target/earnings_open_source_long.parquet",
            "financials_open_source_consolidated": "target/financials_open_source_consolidated.parquet",
            "financials_open_source_lineage": "target/financials_open_source_lineage.parquet",
            "financials_open_source_source_summary": "target/financials_open_source_source_summary.parquet",
        },
        "legacy_outputs": {name: str(path.relative_to(paths.base_dir)) for name, path in legacy_paths.items()},
        "published_output": {name: str(path.relative_to(paths.root_dir)) for name, path in published_output_paths.published_paths.items()},
        "published_output_snapshot": (
            str(published_output_paths.snapshot_dir.relative_to(paths.root_dir))
            if published_output_paths.snapshot_dir is not None
            else None
        ),
        "failures": run_failures,
        "audit_dirs": [str(path.relative_to(paths.root_dir)) for path in audit_dirs],
    }
    write_run_manifest(paths, run_id, manifest)

    return OpenSourceIngestionResult(
        mode=mode,
        run_id=run_id,
        live_dir=paths.base_dir,
        raw_dir=paths.raw_dir,
        target_dir=paths.target_dir,
        clean_dir=paths.clean_dir,
        legacy_dir=paths.legacy_dir,
        output_dir=paths.output_dir,
        output_lineage_dir=paths.output_lineage_dir,
        output_snapshot_dir=published_output_paths.snapshot_dir,
        audit_dirs=tuple(audit_dirs),
        ticker_count=len(ticker_list),
        price_start_date=price_start,
        price_end_date=end_date,
        refreshed_years=tuple(refreshed_years),
        price_rows=clean_prices.height,
        consolidated_rows=consolidated_financials.height,
        lineage_rows=consolidated_lineage.height,
    )


def _write_live_audit(
    *,
    paths: OpenSourceLivePaths,
    reference_data_dir: Path,
    year: int,
    tickers: tuple[str, ...],
    threshold_pct: float,
) -> Path:
    output_dir = paths.audit_dir / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)
    sp500_tickers = load_sp500_tickers_for_year(reference_data_dir, year)
    benchmark_tickers = tuple(ticker for ticker in tickers if ticker in set(sp500_tickers))

    clean_prices = pl.read_parquet(paths.clean_dir / "prices_open_source.parquet").filter(pl.col("date").str.starts_with(str(year)))
    clean_earnings = pl.read_parquet(paths.clean_dir / "earnings_open_source_long.parquet").filter(pl.col("date").str.starts_with(str(year)))
    consolidated_financials = pl.read_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet").filter(
        pl.col("date").str.starts_with(str(year))
    )
    consolidated_lineage = pl.read_parquet(paths.clean_dir / "financials_open_source_lineage.parquet").filter(
        pl.col("date").str.starts_with(str(year))
    )
    source_summary = pl.read_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")

    raw_sec_companyfacts = pl.read_parquet(paths.raw_dir / "financials_sec_companyfacts.parquet").filter(pl.col("date").str.starts_with(str(year)))
    raw_sec_filing = pl.read_parquet(paths.raw_dir / "financials_sec_filing.parquet").filter(pl.col("date").str.starts_with(str(year)))
    raw_simfin = pl.read_parquet(paths.raw_dir / "financials_simfin.parquet").filter(pl.col("date").str.starts_with(str(year)))
    raw_yfinance = pl.read_parquet(paths.raw_dir / "financials_yfinance.parquet").filter(pl.col("date").str.starts_with(str(year)))
    general_reference = pl.read_parquet(paths.raw_dir / "general_reference.parquet")

    yahoo_availability = (
        clean_prices.select(
            [
                pl.col("ticker"),
                pl.col("ticker").str.replace(r"\.US$", "").alias("ticker_root"),
                pl.lit(True).alias("yahoo_price_available"),
            ]
        )
        .unique()
        .sort("ticker")
    )
    sec_mapping = general_reference.select(
        [
            pl.col("ticker").str.replace(r"\.US$", "").alias("ticker"),
            pl.col("name"),
            pl.col("exchange"),
            pl.col("cik"),
        ]
    )
    coverage = build_coverage_audit(
        sp500_tickers=sp500_tickers,
        benchmark_tickers=benchmark_tickers,
        sec_mapping=sec_mapping,
        yahoo_availability=yahoo_availability,
    )

    eodhd_prices = load_eodhd_prices(reference_data_dir, benchmark_tickers, year)
    eodhd_financials = normalize_eodhd_financials(reference_data_dir, benchmark_tickers, year)
    eodhd_earnings = normalize_eodhd_earnings(reference_data_dir, benchmark_tickers, year)
    price_alignment = build_price_alignment(eodhd_prices, clean_prices.filter(pl.col("ticker").is_in([f"{ticker}.US" for ticker in benchmark_tickers])))
    financial_alignment = pl.concat(
        [
            build_financial_alignment(eodhd_financials, raw_sec_companyfacts.select(_clean_financial_columns()), "sec_companyfacts"),
            *([build_financial_alignment(eodhd_financials, raw_sec_filing.select(_clean_financial_columns()), "sec_filing")] if not raw_sec_filing.is_empty() else []),
            *([build_financial_alignment(eodhd_financials, raw_simfin.select(_clean_financial_columns()), "simfin")] if not raw_simfin.is_empty() else []),
            *([build_financial_alignment(eodhd_financials, raw_yfinance.select(_clean_financial_columns()), "yfinance")] if not raw_yfinance.is_empty() else []),
            *(
                [build_financial_alignment(eodhd_financials, consolidated_financials, "open_source_consolidated")]
                if not consolidated_financials.is_empty()
                else []
            ),
            *([build_earnings_alignment(eodhd_earnings, clean_earnings, open_source="open_source_earnings")] if not clean_earnings.is_empty() else []),
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
        include_yfinance_financials=not raw_yfinance.is_empty(),
        include_yfinance_earnings=not clean_earnings.is_empty(),
        include_sec_filing_financials=not raw_sec_filing.is_empty(),
        include_simfin_financials=not raw_simfin.is_empty(),
        include_open_source_consolidated=not consolidated_financials.is_empty(),
    )
    coverage.write_parquet(output_dir / f"ticker_coverage_{year}.parquet")
    audited_metric_catalog.write_parquet(output_dir / "audited_metric_catalog.parquet")
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
        tickers=benchmark_tickers,
        price_alignment=price_alignment,
        financial_alignment=financial_alignment,
        output_path=output_dir / "summary.json",
    )
    write_html_report(
        output_path=output_dir / "report.html",
        year=year,
        threshold_pct=threshold_pct,
        benchmark_tickers=benchmark_tickers,
        coverage=coverage,
        audited_metric_catalog=audited_metric_catalog,
        consolidation_source_summary=source_summary,
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
    return output_dir


def _with_price_ingestion_metadata(frame: pl.DataFrame, *, dataset: str, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_price_frame()
    return frame.with_columns(
        [
            pl.lit("yfinance").alias("source"),
            pl.lit(dataset).alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(RAW_PRICE_SCHEMA))


def _with_financial_ingestion_metadata(frame: pl.DataFrame, *, dataset: str, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_FINANCIAL_SCHEMA)
    expressions: list[pl.Expr] = [
        pl.lit(dataset).alias("dataset"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ]
    if "form" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Utf8).alias("form"))
    if "fiscal_period" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Utf8).alias("fiscal_period"))
    if "fiscal_year" not in frame.columns:
        expressions.append(pl.lit(None).cast(pl.Int64).alias("fiscal_year"))
    return frame.with_columns(expressions).select(list(RAW_FINANCIAL_SCHEMA))


def _with_earnings_ingestion_metadata(frame: pl.DataFrame, *, dataset: str, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_EARNINGS_SCHEMA)
    expressions: list[pl.Expr] = [
        pl.lit(dataset).alias("dataset"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(ingested_at).alias("ingested_at"),
    ]
    for column, dtype in RAW_EARNINGS_SCHEMA.items():
        if column in frame.columns or column in {"dataset", "ingestion_run_id", "ingested_at"}:
            continue
        expressions.append(pl.lit(None).cast(dtype).alias(column))
    return frame.with_columns(expressions).select(list(RAW_EARNINGS_SCHEMA))


def _with_general_ingestion_metadata(frame: pl.DataFrame, *, run_id: str, ingested_at: str) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=RAW_GENERAL_SCHEMA)
    return frame.with_columns(
        [
            pl.col("cik").cast(pl.Utf8, strict=False),
            pl.lit("general_reference").alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(RAW_GENERAL_SCHEMA))


def _with_general_lineage_ingestion_metadata(frame: pl.DataFrame, *, run_id: str, ingested_at: str) -> pl.DataFrame:
    schema = {column: pl.String for column in empty_general_reference_lineage_frame().columns}
    schema.update({"dataset": pl.String, "ingestion_run_id": pl.String, "ingested_at": pl.String})
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    return frame.with_columns(
        [
            pl.lit("general_reference_lineage").alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(schema))


def _resolve_price_start(*, mode: str, explicit_start_date: str, raw_price_path: Path, lookback_days: int) -> str:
    if mode == "bootstrap" or not raw_price_path.exists():
        return explicit_start_date
    existing = pl.read_parquet(raw_price_path)
    if existing.is_empty():
        return explicit_start_date
    max_date = existing.select(pl.col("date").max()).item()
    if max_date is None:
        return explicit_start_date
    start = datetime.strptime(str(max_date), "%Y-%m-%d").date() - timedelta(days=lookback_days)
    return max(start.isoformat(), explicit_start_date)


def _identify_price_history_backfill_tickers(
    *,
    requested_tickers: Sequence[str],
    existing_prices: pl.DataFrame,
    explicit_start_date: str,
    mode: str,
    recent_first_date_window_days: int = 365,
) -> tuple[str, ...]:
    if mode == "bootstrap" or existing_prices.is_empty():
        return ()

    max_date = existing_prices.select(pl.col("date").max()).item()
    if max_date is None:
        return tuple(sorted(set(requested_tickers)))

    recent_cutoff = (
        datetime.strptime(str(max_date), "%Y-%m-%d").date() - timedelta(days=recent_first_date_window_days)
    ).isoformat()
    coverage = (
        existing_prices.select(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("date").cast(pl.Utf8),
            ]
        )
        .group_by("ticker")
        .agg(
            [
                pl.col("date").min().alias("first_date"),
                pl.col("date").max().alias("last_date"),
                pl.len().alias("row_count"),
            ]
        )
    )

    backfill: list[str] = []
    for ticker in requested_tickers:
        full_ticker = f"{ticker}.US"
        row = coverage.filter(pl.col("ticker") == full_ticker)
        if row.is_empty():
            backfill.append(ticker)
            continue
        first_date = row.select(pl.col("first_date")).item()
        if first_date is None:
            backfill.append(ticker)
            continue
        if str(first_date) > explicit_start_date and str(first_date) >= recent_cutoff:
            backfill.append(ticker)
    return tuple(sorted(set(backfill)))


def _identify_general_reference_refresh_tickers(
    *,
    requested_tickers: Sequence[str],
    existing_general_reference: pl.DataFrame,
    mode: str,
) -> tuple[str, ...]:
    if mode == "bootstrap" or existing_general_reference.is_empty():
        return tuple(requested_tickers)
    sort_cols = [column for column in ["ticker", "ingested_at"] if column in existing_general_reference.columns]
    existing = existing_general_reference.select(
        [
            pl.col("ticker").cast(pl.Utf8),
            pl.col("Sector").cast(pl.Utf8, strict=False).alias("Sector"),
            pl.col("industry").cast(pl.Utf8, strict=False).alias("industry"),
            *([pl.col("ingested_at").cast(pl.Utf8, strict=False)] if "ingested_at" in existing_general_reference.columns else []),
        ]
    )
    if sort_cols:
        existing = existing.sort(sort_cols)
    existing = existing.unique(subset=["ticker"], keep="last", maintain_order=True)
    missing: list[str] = []
    for ticker in requested_tickers:
        full_ticker = f"{ticker}.US"
        row = existing.filter(pl.col("ticker") == full_ticker)
        if row.is_empty():
            missing.append(ticker)
            continue
        sector = row.select(pl.col("Sector")).head(1).item()
        industry = row.select(pl.col("industry")).head(1).item()
        if sector in {None, "", "Unknown"} or industry in {None, ""}:
            missing.append(ticker)
    return tuple(sorted(set(missing)))


def _resolve_refreshed_years(*, mode: str, start_date: str, end_date: str, lookback_years: int) -> tuple[int, ...]:
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    if mode == "bootstrap":
        return tuple(range(start_year, end_year + 1))
    first_year = max(start_year, end_year - lookback_years + 1)
    return tuple(range(first_year, end_year + 1))


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


def _load_existing_open_source_tickers(paths: OpenSourceLivePaths, reference_data_dir: Path) -> tuple[str, ...]:
    candidate_paths = (
        paths.output_dir / "US_Finalprice.parquet",
        paths.clean_dir / "prices_open_source.parquet",
        paths.raw_dir / "prices_yfinance.parquet",
        reference_data_dir / "US_Finalprice.parquet",
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        if frame.is_empty() or "ticker" not in frame.columns:
            continue
        return tuple(
            frame.select(pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"))
            .unique()
            .sort("ticker")
            .to_series()
            .to_list()
        )
    return ()


def _load_existing_price_tickers(paths: OpenSourceLivePaths) -> tuple[str, ...]:
    candidate_paths = (
        paths.output_dir / "US_Finalprice.parquet",
        paths.clean_dir / "prices_open_source.parquet",
        paths.raw_dir / "prices_yfinance.parquet",
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        if frame.is_empty() or "ticker" not in frame.columns:
            continue
        return tuple(
            frame.select(pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"))
            .unique()
            .sort("ticker")
            .to_series()
            .to_list()
        )
    return ()


def _canonicalize_price_tickers(frame: pl.DataFrame, *, ticker_list: Sequence[str]) -> pl.DataFrame:
    alias_map = {
        f"{ticker.replace('.', '-')}.US": f"{ticker}.US"
        for ticker in ticker_list
        if "." in ticker
    }
    if frame.is_empty() or not alias_map:
        return frame
    return (
        frame.with_columns(pl.col("ticker").replace_strict(alias_map, default=pl.col("ticker")).alias("ticker"))
        .sort(["ticker", "date", "source", "dataset", "ingested_at"])
        .unique(subset=["ticker", "date", "source"], keep="last", maintain_order=True)
        .sort(["ticker", "date"])
    )


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
        key_cols=["ticker", "statement", "metric", "date", "source"],
        order_cols=["filing_date", "ingested_at"],
    )


def _concat_or_empty(frames: Sequence[pl.DataFrame], *, empty: pl.DataFrame | None = None) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return empty if empty is not None else _empty_raw_financial_base()
    return pl.concat(non_empty, how="vertical")


def _empty_raw_financial_base() -> pl.DataFrame:
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


def _empty_raw_earnings_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RAW_EARNINGS_SCHEMA)


def _empty_raw_price_frame() -> pl.DataFrame:
    return pl.DataFrame(schema=RAW_PRICE_SCHEMA)


def _empty_sec_profile_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "cik": pl.String,
            "sic": pl.String,
            "sic_description": pl.String,
        }
    )


def _clean_financial_columns() -> list[str]:
    return ["ticker", "statement", "metric", "date", "filing_date", "value", "source", "source_label", "form", "fiscal_period", "fiscal_year"]


def _filter_financial_year(frame: pl.DataFrame, *, year: int) -> pl.DataFrame:
    if frame.is_empty():
        return _empty_raw_financial_base()
    return frame.filter(pl.col("date").str.starts_with(str(year)))


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
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "earnings_sec_actuals"})
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
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                frames.append(future.result())
            except Exception as exc:
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
                failures.append({"ticker": ticker, "error": str(exc), "dataset": "general_reference"})
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
            except Exception as exc:
                failures.append({"ticker": ticker, "error": str(exc)})
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
