from __future__ import annotations

import shutil
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.observability import set_run_log_context

from alpharank.data.open_source.benchmark import (
    build_audited_metric_catalog,
    build_coverage_audit,
    build_earnings_alignment,
    build_error_detail_tables,
    build_error_summary_tables,
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
from alpharank.data.open_source.config import PRICE_COLUMNS
from alpharank.data.open_source.consolidation import (
    FinancialSourceInput,
    consolidate_financial_sources_with_share_quality,
)
from alpharank.data.open_source.earnings import (
    empty_earnings_actuals_frame,
    empty_earnings_calendar_frame,
)
from alpharank.data.open_source.freshness import (
    build_data_freshness_summary,
    validate_data_freshness,
)
from alpharank.data.open_source.fundamental_quality import (
    audit_fundamental_quality,
    quarantine_implausible_share_candidates,
    validate_fundamental_quality,
)
from alpharank.data.open_source.general_reference import (
    build_general_reference,
    empty_general_reference_frame,
    empty_general_reference_lineage_frame,
)
from alpharank.data.open_source.ingestion_frames import (
    RAW_EARNINGS_SCHEMA as RAW_EARNINGS_SCHEMA,
)
from alpharank.data.open_source.ingestion_frames import (
    RAW_FINANCIAL_SCHEMA as RAW_FINANCIAL_SCHEMA,
)
from alpharank.data.open_source.ingestion_frames import (
    RAW_GENERAL_SCHEMA as RAW_GENERAL_SCHEMA,
)
from alpharank.data.open_source.ingestion_frames import (
    RAW_PRICE_SCHEMA as RAW_PRICE_SCHEMA,
)
from alpharank.data.open_source.ingestion_frames import (
    _audit_and_validate_historical_revisions as _audit_and_validate_historical_revisions,
)
from alpharank.data.open_source.ingestion_frames import (
    _clean_financial_columns as _clean_financial_columns,
)
from alpharank.data.open_source.ingestion_frames import (
    _concat_or_empty as _concat_or_empty,
)
from alpharank.data.open_source.ingestion_frames import (
    _empty_raw_earnings_frame as _empty_raw_earnings_frame,
)
from alpharank.data.open_source.ingestion_frames import (
    _empty_raw_financial_base as _empty_raw_financial_base,
)
from alpharank.data.open_source.ingestion_frames import (
    _empty_raw_price_frame as _empty_raw_price_frame,
)
from alpharank.data.open_source.ingestion_frames import (
    _empty_sec_profile_frame as _empty_sec_profile_frame,
)
from alpharank.data.open_source.ingestion_frames import (
    _filter_financial_year as _filter_financial_year,
)
from alpharank.data.open_source.ingestion_frames import (
    _filter_financial_years as _filter_financial_years,
)
from alpharank.data.open_source.ingestion_frames import (
    _with_earnings_ingestion_metadata as _with_earnings_ingestion_metadata,
)
from alpharank.data.open_source.ingestion_frames import (
    _with_financial_ingestion_metadata as _with_financial_ingestion_metadata,
)
from alpharank.data.open_source.ingestion_frames import (
    _with_general_ingestion_metadata as _with_general_ingestion_metadata,
)
from alpharank.data.open_source.ingestion_frames import (
    _with_general_lineage_ingestion_metadata as _with_general_lineage_ingestion_metadata,
)
from alpharank.data.open_source.ingestion_frames import (
    _with_price_ingestion_metadata as _with_price_ingestion_metadata,
)
from alpharank.data.open_source.ingestion_prices import (
    _canonicalize_price_tickers as _canonicalize_price_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _complete_yahoo_history_against_validated as _complete_yahoo_history_against_validated,
)
from alpharank.data.open_source.ingestion_prices import (
    _confirmed_terminal_price_tickers as _confirmed_terminal_price_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _consolidate_price_sources as _consolidate_price_sources,
)
from alpharank.data.open_source.ingestion_prices import (
    _download_yahoo_price_history as _download_yahoo_price_history,
)
from alpharank.data.open_source.ingestion_prices import (
    _drop_refreshed_partitions as _drop_refreshed_partitions,
)
from alpharank.data.open_source.ingestion_prices import (
    _historical_yahoo_key_gaps as _historical_yahoo_key_gaps,
)
from alpharank.data.open_source.ingestion_prices import (
    _identify_general_reference_refresh_tickers as _identify_general_reference_refresh_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _identify_price_history_backfill_tickers as _identify_price_history_backfill_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _identify_simfin_price_fallback_tickers as _identify_simfin_price_fallback_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _identify_stockanalysis_price_fallback_tickers as _identify_stockanalysis_price_fallback_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_existing_open_source_tickers as _load_existing_open_source_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_existing_price_history_frame as _load_existing_price_history_frame,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_existing_price_tickers as _load_existing_price_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_latest_sp500_tickers as _load_latest_sp500_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_reference_tickers as _load_reference_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _load_retained_open_price_vintages as _load_retained_open_price_vintages,
)
from alpharank.data.open_source.ingestion_prices import (
    _merge_prospective_price_sources as _merge_prospective_price_sources,
)
from alpharank.data.open_source.ingestion_prices import (
    _network_price_refresh_coverage as _network_price_refresh_coverage,
)
from alpharank.data.open_source.ingestion_prices import (
    _prepare_canonical_hybrid_price_merge as _prepare_canonical_hybrid_price_merge,
)
from alpharank.data.open_source.ingestion_prices import (
    _prepare_validated_stock_price_merge as _prepare_validated_stock_price_merge,
)
from alpharank.data.open_source.ingestion_prices import (
    _price_source_priority_expr as _price_source_priority_expr,
)
from alpharank.data.open_source.ingestion_prices import (
    _required_failure_tickers as _required_failure_tickers,
)
from alpharank.data.open_source.ingestion_prices import (
    _resolve_price_start as _resolve_price_start,
)
from alpharank.data.open_source.ingestion_prices import (
    _resolve_refreshed_years as _resolve_refreshed_years,
)
from alpharank.data.open_source.ingestion_prices import (
    _resolve_sec_mapping_coverage as _resolve_sec_mapping_coverage,
)
from alpharank.data.open_source.ingestion_prices import (
    _write_yahoo_attempt_audit as _write_yahoo_attempt_audit,
)
from alpharank.data.open_source.ingestion_reference import (
    _build_clean_earnings as _build_clean_earnings,
)
from alpharank.data.open_source.ingestion_reference import (
    _canonicalize_general_outputs as _canonicalize_general_outputs,
)
from alpharank.data.open_source.ingestion_reference import (
    _extract_sec_companyfacts_earnings_actuals as _extract_sec_companyfacts_earnings_actuals,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_company_profiles as _fetch_sec_company_profiles,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_companyfacts_bundle as _fetch_sec_companyfacts_bundle,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_earnings_actuals as _fetch_sec_earnings_actuals,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_earnings_calendar as _fetch_sec_earnings_calendar,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_filing_earnings_actuals as _fetch_sec_filing_earnings_actuals,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_filing_financials as _fetch_sec_filing_financials,
)
from alpharank.data.open_source.ingestion_reference import (
    _fetch_sec_financials as _fetch_sec_financials,
)
from alpharank.data.open_source.ingestion_reference import (
    _filter_earnings_years as _filter_earnings_years,
)
from alpharank.data.open_source.ingestion_reference import (
    _identify_metric_gap_tickers as _identify_metric_gap_tickers,
)
from alpharank.data.open_source.ingestion_reference import (
    _identify_sec_filing_fallback_tickers as _identify_sec_filing_fallback_tickers,
)
from alpharank.data.open_source.ingestion_reference import (
    _identify_yahoo_earnings_repair_tickers as _identify_yahoo_earnings_repair_tickers,
)
from alpharank.data.open_source.ingestion_reference import (
    _identify_yfinance_financial_fallback_tickers as _identify_yfinance_financial_fallback_tickers,
)
from alpharank.data.open_source.ingestion_reference import (
    _repair_yahoo_earnings as _repair_yahoo_earnings,
)
from alpharank.data.open_source.ingestion_reference import (
    _upsert_financial_dataset as _upsert_financial_dataset,
)
from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs
from alpharank.data.open_source.price_quality import (
    build_split_detection_prices,
    find_extreme_adjusted_price_moves,
    repair_confirmed_split_discontinuities,
)
from alpharank.data.open_source.publishing import publish_open_source_output_package
from alpharank.data.open_source.refresh_policy import (
    PRODUCTION_SOURCE_REFRESH_POLICY,
    SourceRefreshPolicy,
)
from alpharank.data.open_source.sec import SecCompanyFactsClient
from alpharank.data.open_source.sec_filing import SecFilingFactsClient
from alpharank.data.open_source.sec_mapping import resolve_sec_company_mapping
from alpharank.data.open_source.simfin import SimFinClient
from alpharank.data.open_source.stockanalysis import StockAnalysisClient
from alpharank.data.open_source.storage import (
    OpenSourceLivePaths,
    append_run_delta,
    new_run_id,
    upsert_parquet,
    utc_now_iso,
    write_json,
    write_run_manifest,
)
from alpharank.data.open_source.transaction import OpenSourceStoreTransaction
from alpharank.data.open_source.yahoo import YahooFinanceClient
from alpharank.data.prices import (
    combine_stock_split_evidence,
    load_confirmed_stock_splits,
    resolve_previous_validated_price_lineage,
)
from alpharank.data.warehouse import WarehousePaths


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
    sec_companyfacts_years: tuple[int, ...] = ()


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


def _resolve_open_source_data_layout(
    *,
    project_root: Path,
    live_dir: Path | None,
    reference_data_dir: Path | None,
) -> tuple[Path, Path, Path, Path]:
    """Resolve one physical data root while code remains in its worktree."""

    official_dir = (
        live_dir
        if live_dir is not None
        else project_root / "data" / "open_source" / "official"
    ).resolve()
    open_source_root = official_dir.parent
    data_root = open_source_root.parent
    resolved_reference_dir = (
        reference_data_dir if reference_data_dir is not None else data_root
    ).resolve()
    return official_dir, open_source_root, data_root, resolved_reference_dir


def repair_open_source_price_history(
    *,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    tickers: Sequence[str] | None = None,
    live_dir: Path | None = None,
    reference_data_dir: Path | None = None,
    audit_years: Sequence[int] = (),
    threshold_pct: float = 0.5,
    source_refresh_policy: SourceRefreshPolicy = PRODUCTION_SOURCE_REFRESH_POLICY,
) -> OpenSourceIngestionResult:
    project_root = Path(__file__).resolve().parents[4]
    official_dir, open_source_root, _, reference_data_dir = (
        _resolve_open_source_data_layout(
            project_root=project_root,
            live_dir=live_dir,
            reference_data_dir=reference_data_dir,
        )
    )
    paths = OpenSourceLivePaths(
        official_dir,
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = new_run_id()
    ingested_at = utc_now_iso()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    source_refresh_contract = source_refresh_policy.to_manifest(
        mode="price_history_repair",
        price_start_date=start_date,
        price_end_date=end_date,
        financial_years=(),
        snapshot_scope="price_history_repair",
    )
    if tickers is None:
        current_sp500 = set(_load_latest_sp500_tickers(reference_data_dir))
        existing_price_tickers = set(_load_existing_price_tickers(paths))
        ticker_list = tuple(sorted(current_sp500 | existing_price_tickers))
    else:
        ticker_list = tuple(tickers)
        current_sp500 = set(_load_latest_sp500_tickers(reference_data_dir))
    price_quality_tickers = tuple(sorted(current_sp500.intersection(ticker_list))) or ticker_list
    price_refresh_tickers = (
        price_quality_tickers
        if source_refresh_policy.refresh_full_price_history
        else ticker_list
    )
    retained_inactive_price_tickers = tuple(sorted(set(ticker_list) - set(price_refresh_tickers)))

    yahoo_client = YahooFinanceClient(
        cache_dir=open_source_root / "_cache" / "yfinance"
    )
    simfin_client = SimFinClient(
        data_dir=open_source_root / "_cache" / "simfin",
        refresh_days=source_refresh_policy.simfin_refresh_days,
    )
    stockanalysis_client = StockAnalysisClient(
        cache_dir=open_source_root / "_cache" / "stockanalysis",
        refresh_cache=source_refresh_policy.refresh_stockanalysis,
        persist_cache=source_refresh_policy.persist_stockanalysis_payloads,
    )
    existing_price_history = _load_existing_price_history_frame(paths)
    backfill_tickers = _identify_price_history_backfill_tickers(
        requested_tickers=ticker_list,
        existing_prices=existing_price_history,
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
    simfin_price_tickers = _identify_simfin_price_fallback_tickers(
        requested_tickers=ticker_list,
        yahoo_prices_delta=_concat_or_empty(price_deltas, empty=_empty_raw_price_frame()),
        backfill_tickers=backfill_tickers,
    )
    simfin_prices_delta = _with_price_ingestion_metadata(
        simfin_client.fetch_daily_prices(simfin_price_tickers, start_date, end_date)
        if simfin_client.enabled and simfin_price_tickers
        else _empty_raw_price_frame(),
        dataset="prices_simfin_repair",
        source="simfin",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    stockanalysis_price_tickers = _identify_stockanalysis_price_fallback_tickers(
        requested_tickers=ticker_list,
        covered_prices_delta=_concat_or_empty(
            [_concat_or_empty(price_deltas, empty=_empty_raw_price_frame()), simfin_prices_delta],
            empty=_empty_raw_price_frame(),
        ),
        backfill_tickers=backfill_tickers,
    )
    stockanalysis_prices_delta = _with_price_ingestion_metadata(
        stockanalysis_client.fetch_daily_prices(stockanalysis_price_tickers, start_date, end_date)
        if stockanalysis_price_tickers
        else _empty_raw_price_frame(),
        dataset="prices_stockanalysis_repair",
        source="stockanalysis",
        run_id=run_id,
        ingested_at=ingested_at,
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

    (
        raw_yahoo_prices,
        raw_simfin_prices,
        raw_stockanalysis_prices,
        clean_prices,
        clean_price_lineage,
    ) = _prepare_validated_stock_price_merge(
        paths=paths,
        yahoo_delta=prices_delta,
        simfin_delta=simfin_prices_delta,
        stockanalysis_delta=stockanalysis_prices_delta,
        ticker_list=ticker_list,
        event_since=start_date,
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_yfinance.parquet", prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_simfin.parquet", simfin_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_stockanalysis.parquet", stockanalysis_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_spy_yfinance.parquet", benchmark_delta)
    raw_yahoo_prices.write_parquet(paths.raw_dir / "prices_yfinance.parquet")
    raw_simfin_prices.write_parquet(paths.raw_dir / "prices_simfin.parquet")
    raw_stockanalysis_prices.write_parquet(paths.raw_dir / "prices_stockanalysis.parquet")
    raw_benchmark_prices = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )

    clean_benchmark_prices = raw_benchmark_prices.select(
        ["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]
    ).sort(["ticker", "date"])
    clean_prices.write_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_price_lineage.write_parquet(paths.clean_dir / "prices_open_source_lineage.parquet")
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
        consolidated_lineage=consolidated_lineage,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    _audit_and_validate_historical_revisions(
        paths=paths,
        run_id=run_id,
        legacy_paths=legacy_paths,
        expected_through=end_date,
        source_refresh_policy=source_refresh_policy,
        source_refresh_contract=source_refresh_contract,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        prices_lineage=clean_price_lineage,
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
            "simfin_price_fallback_ticker_count": len(simfin_price_tickers),
            "simfin_price_fallback_ticker_examples": list(simfin_price_tickers[:20]),
            "stockanalysis_price_fallback_ticker_count": len(stockanalysis_price_tickers),
            "stockanalysis_price_fallback_ticker_examples": list(stockanalysis_price_tickers[:20]),
            "source_refresh_contract": source_refresh_contract,
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
        "simfin_price_fallback_ticker_count": len(simfin_price_tickers),
        "simfin_price_fallback_ticker_examples": list(simfin_price_tickers[:20]),
        "stockanalysis_price_fallback_ticker_count": len(stockanalysis_price_tickers),
        "stockanalysis_price_fallback_ticker_examples": list(stockanalysis_price_tickers[:20]),
        "price_window": {"start_date": start_date, "end_date": end_date},
        "source_refresh_contract": source_refresh_contract,
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
    source_refresh_policy: SourceRefreshPolicy = PRODUCTION_SOURCE_REFRESH_POLICY,
) -> OpenSourceReferenceRefreshResult:
    project_root = Path(__file__).resolve().parents[4]
    official_dir, open_source_root, _, reference_data_dir = (
        _resolve_open_source_data_layout(
            project_root=project_root,
            live_dir=live_dir,
            reference_data_dir=reference_data_dir,
        )
    )
    paths = OpenSourceLivePaths(
        official_dir,
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = new_run_id()
    ingested_at = utc_now_iso()
    final_end_year = end_year or date.today().year
    refreshed_years = tuple(range(start_year, final_end_year + 1))
    source_refresh_contract = source_refresh_policy.to_manifest(
        mode="reference_refresh",
        price_start_date="not_refreshed",
        price_end_date="not_refreshed",
        financial_years=refreshed_years,
        snapshot_scope="reference_refresh",
    )
    ticker_list = tuple(tickers) if tickers is not None else _load_existing_open_source_tickers(paths, reference_data_dir)

    yahoo_client = YahooFinanceClient(
        cache_dir=open_source_root / "_cache" / "yfinance"
    )
    sec_client = SecCompanyFactsClient(
        user_agent=user_agent,
        cache_dir=open_source_root / "_cache" / "sec_companyfacts",
        refresh_cache=source_refresh_policy.refresh_sec_companyfacts,
        persist_cache=source_refresh_policy.persist_sec_companyfacts_payloads,
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=user_agent,
        cache_dir=open_source_root / "_cache" / "sec_filing",
        refresh_mutable_cache=source_refresh_policy.refresh_sec_submissions,
        persist_metadata_cache=source_refresh_policy.persist_sec_filing_metadata,
        persist_filing_documents=source_refresh_policy.persist_sec_filing_documents,
    )

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
    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = resolve_sec_company_mapping(
        requested_tickers=ticker_list,
        sec_mapping_all=sec_mapping_all,
        reference_data_dir=reference_data_dir,
        existing_general_reference_lineage=existing_general_reference_lineage,
    )
    mapped_sec_tickers, required_sec_tickers, missing_active_sec_mappings = (
        _resolve_sec_mapping_coverage(
            sec_mapping=sec_mapping,
            required_tickers=price_quality_tickers,
        )
    )
    source_refresh_contract["source_semantics"]["sec_companyfacts"].update(
        {
            "active_mapping_count": len(required_sec_tickers) - len(missing_active_sec_mappings),
            "active_mapping_missing_tickers": list(missing_active_sec_mappings),
        }
    )
    if missing_active_sec_mappings:
        raise RuntimeError(
            "SEC mapping is incomplete for the active universe; "
            f"missing={list(missing_active_sec_mappings)}. No package was published."
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

    raw_yahoo_earnings, clean_earnings, clean_earnings_lineage, clean_earnings_long, earnings_repair_tickers = _repair_yahoo_earnings(
        paths=paths,
        run_id=run_id,
        ingested_at=ingested_at,
        yahoo_client=yahoo_client,
        raw_yahoo_earnings=raw_yahoo_earnings,
        raw_earnings_sec_calendar=raw_earnings_sec_calendar,
        raw_earnings_sec_actuals=raw_earnings_sec_actuals,
        candidate_tickers=ticker_list,
        years=refreshed_years,
    )

    clean_prices = pl.read_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_price_lineage = (
        pl.read_parquet(paths.clean_dir / "prices_open_source_lineage.parquet")
        if (paths.clean_dir / "prices_open_source_lineage.parquet").exists()
        else clean_prices
    )
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
        consolidated_lineage=consolidated_lineage,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    _audit_and_validate_historical_revisions(
        paths=paths,
        run_id=run_id,
        legacy_paths=legacy_paths,
        expected_through=end_date,
        source_refresh_policy=source_refresh_policy,
        source_refresh_contract=source_refresh_contract,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        prices_lineage=clean_price_lineage,
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
            "source_refresh_contract": source_refresh_contract,
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
        "source_refresh_contract": source_refresh_contract,
        "general_rows": general_reference.height,
        "general_sector_non_null_rows": general_reference.filter(pl.col("Sector").is_not_null() & (pl.col("Sector") != "")).height,
        "earnings_rows": clean_earnings.height,
        "earnings_tickers": clean_earnings.select(pl.col("ticker").n_unique()).item() if not clean_earnings.is_empty() else 0,
        "earnings_repair_ticker_count": len(earnings_repair_tickers),
        "earnings_repair_ticker_examples": list(earnings_repair_tickers[:20]),
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
    source_refresh_policy: SourceRefreshPolicy = PRODUCTION_SOURCE_REFRESH_POLICY,
    eodhd_price_seed_path: Path | None = None,
    run_id: str | None = None,
) -> OpenSourceIngestionResult:
    project_root = Path(__file__).resolve().parents[4]
    official_dir = (
        live_dir
        if live_dir is not None
        else project_root / "data" / "open_source" / "official"
    ).resolve()
    try:
        with OpenSourceStoreTransaction(official_dir=official_dir):
            return _run_open_source_ingestion_in_place(
                mode=mode,
                start_date=start_date,
                end_date=end_date,
                tickers=tickers,
                live_dir=official_dir,
                reference_data_dir=reference_data_dir,
                user_agent=user_agent,
                simfin_api_key=simfin_api_key,
                price_lookback_days=price_lookback_days,
                financial_lookback_years=financial_lookback_years,
                audit_years=audit_years,
                threshold_pct=threshold_pct,
                source_refresh_policy=source_refresh_policy,
                eodhd_price_seed_path=eodhd_price_seed_path,
                run_id=run_id,
            )
    finally:
        transport_cache = official_dir.parent / "_cache"
        shutil.rmtree(transport_cache, ignore_errors=True)
        transport_cache.mkdir(parents=True, exist_ok=True)


def _run_open_source_ingestion_in_place(
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
    source_refresh_policy: SourceRefreshPolicy = PRODUCTION_SOURCE_REFRESH_POLICY,
    eodhd_price_seed_path: Path | None = None,
    run_id: str | None = None,
) -> OpenSourceIngestionResult:
    project_root = Path(__file__).resolve().parents[4]
    official_dir, open_source_root, data_root, reference_data_dir = (
        _resolve_open_source_data_layout(
            project_root=project_root,
            live_dir=live_dir,
            reference_data_dir=reference_data_dir,
        )
    )
    eodhd_price_seed_path = (
        eodhd_price_seed_path.resolve()
        if eodhd_price_seed_path is not None
        else data_root / "eodhd" / "output" / "US_Finalprice.parquet"
    )
    paths = OpenSourceLivePaths(
        official_dir,
        audit_root_dir=open_source_root / "audit",
    )
    paths.ensure()

    run_id = run_id or new_run_id()
    set_run_log_context(
        run_id=run_id,
        snapshot_id="not_applicable",
        component=__name__,
        step="open_source_ingestion",
    )
    ingested_at = utc_now_iso()
    end_date = end_date or date.today().strftime("%Y-%m-%d")
    if tickers is None:
        current_sp500 = set(_load_latest_sp500_tickers(reference_data_dir))
        existing_price_tickers = set(_load_existing_price_tickers(paths))
        ticker_list = tuple(sorted(current_sp500 | existing_price_tickers))
    else:
        ticker_list = tuple(tickers)
        current_sp500 = set(_load_latest_sp500_tickers(reference_data_dir))
    price_quality_tickers = tuple(sorted(current_sp500.intersection(ticker_list))) or ticker_list
    constituent_registry_path = (
        project_root / "configs" / "data_quality" / "sp500_constituent_changes_2026.json"
    )
    terminal_price_tickers = _confirmed_terminal_price_tickers(
        registry_path=constituent_registry_path,
        active_tickers=price_quality_tickers,
        expected_through=end_date,
    )
    terminal_price_roots = {
        ticker.upper().removesuffix(".US") for ticker in terminal_price_tickers
    }
    price_refresh_tickers = (
        tuple(
            ticker
            for ticker in price_quality_tickers
            if ticker.upper().removesuffix(".US") not in terminal_price_roots
        )
        if source_refresh_policy.refresh_full_price_history
        else ticker_list
    )
    retained_inactive_price_tickers = tuple(
        sorted(set(ticker_list) - set(price_refresh_tickers))
    )
    existing_price_history = _load_existing_price_history_frame(paths)
    rolling_price_start = _resolve_price_start(
        mode=mode,
        explicit_start_date=start_date,
        raw_price_path=paths.raw_dir / "prices_yfinance.parquet",
        lookback_days=price_lookback_days,
        existing_prices=existing_price_history,
    )
    price_start = start_date if source_refresh_policy.refresh_full_price_history else rolling_price_start
    price_backfill_tickers = (
        ()
        if source_refresh_policy.refresh_full_price_history
        else _identify_price_history_backfill_tickers(
            requested_tickers=ticker_list,
            existing_prices=existing_price_history,
            explicit_start_date=start_date,
            mode=mode,
        )
    )
    refreshed_years = _resolve_refreshed_years(
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        lookback_years=financial_lookback_years,
    )
    companyfacts_years = (
        tuple(range(int(start_date[:4]), int(end_date[:4]) + 1))
        if source_refresh_policy.refresh_full_sec_companyfacts_history
        else refreshed_years
    )
    source_refresh_contract = source_refresh_policy.to_manifest(
        mode=mode,
        price_start_date=price_start,
        price_end_date=end_date,
        financial_years=companyfacts_years,
    )
    source_refresh_contract["source_semantics"]["yfinance_prices"].update(
        {
            "refreshed_ticker_count": len(price_refresh_tickers),
            "retained_inactive_ticker_count": len(retained_inactive_price_tickers),
            "retained_inactive_ticker_examples": list(retained_inactive_price_tickers[:20]),
            "inactive_history_semantics": "retained official raw; upstream symbol no longer assumed downloadable",
        }
    )
    source_refresh_contract["source_semantics"]["active_universe"] = {
        "ticker_count": len(price_quality_tickers),
        "mutable_yahoo_layers": ["prices", "earnings", "general_reference", "financial_fallback"],
        "inactive_history": "retained in official raw; SEC full-company payloads still refreshed when a CIK resolves",
    }
    source_refresh_contract["source_semantics"]["terminal_price_history"] = {
        "tickers": list(terminal_price_tickers),
        "constituent_registry": str(constituent_registry_path),
        "semantics": (
            "A ticker still present in the latest monthly constituent snapshot but "
            "covered by a sourced removal event effective by the price cutoff keeps "
            "its preceding validated price history byte-stable."
        ),
    }

    yahoo_client = YahooFinanceClient(
        cache_dir=open_source_root / "_cache" / "yfinance"
    )
    sec_client = SecCompanyFactsClient(
        user_agent=user_agent,
        cache_dir=open_source_root / "_cache" / "sec_companyfacts",
        refresh_cache=source_refresh_policy.refresh_sec_companyfacts,
        persist_cache=source_refresh_policy.persist_sec_companyfacts_payloads,
    )
    sec_filing_client = SecFilingFactsClient(
        user_agent=user_agent,
        cache_dir=open_source_root / "_cache" / "sec_filing",
        refresh_mutable_cache=source_refresh_policy.refresh_sec_submissions,
        persist_metadata_cache=source_refresh_policy.persist_sec_filing_metadata,
        persist_filing_documents=source_refresh_policy.persist_sec_filing_documents,
    )
    simfin_client = SimFinClient(
        api_key=simfin_api_key,
        data_dir=open_source_root / "_cache" / "simfin",
        refresh_days=source_refresh_policy.simfin_refresh_days,
    )
    stockanalysis_client = StockAnalysisClient(
        cache_dir=open_source_root / "_cache" / "stockanalysis",
        refresh_cache=source_refresh_policy.refresh_stockanalysis,
        persist_cache=source_refresh_policy.persist_stockanalysis_payloads,
    )
    run_failures: dict[str, list[dict[str, str]]] = {
        "sec_companyfacts": [],
        "sec_filing": [],
        "simfin": [],
        "stockanalysis": [],
        "yfinance_earnings": [],
    }

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
    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = resolve_sec_company_mapping(
        requested_tickers=ticker_list,
        sec_mapping_all=sec_mapping_all,
        reference_data_dir=reference_data_dir,
        existing_general_reference_lineage=existing_general_reference_lineage,
    )
    mapped_sec_tickers, required_sec_tickers, missing_active_sec_mappings = (
        _resolve_sec_mapping_coverage(
            sec_mapping=sec_mapping,
            required_tickers=price_quality_tickers,
        )
    )
    source_refresh_contract["source_semantics"]["sec_companyfacts"].update(
        {
            "active_mapping_count": len(required_sec_tickers) - len(missing_active_sec_mappings),
            "active_mapping_missing_tickers": list(missing_active_sec_mappings),
        }
    )
    if missing_active_sec_mappings:
        raise RuntimeError(
            "SEC mapping is incomplete for the active universe; "
            f"missing={list(missing_active_sec_mappings)}. No package was published."
        )
    general_refresh_tickers = _identify_general_reference_refresh_tickers(
        requested_tickers=ticker_list,
        existing_general_reference=existing_general_reference,
        mode=mode,
    )
    general_refresh_tickers = tuple(
        sorted(set(general_refresh_tickers).intersection(price_quality_tickers))
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

    previous_price_source = resolve_previous_validated_price_lineage(
        data_root / "model_inputs" / "manifests" / "latest.json"
    )
    previous_validated_price_lineage = pl.read_parquet(previous_price_source.lineage_path)
    yahoo_full_history, yahoo_key_coverage = _complete_yahoo_history_against_validated(
        yahoo_client,
        initial_prices=_download_yahoo_price_history(
            yahoo_client,
            tickers=price_refresh_tickers,
            start_date=price_start,
            end_date=end_date,
        ),
        previous_validated_lineage=previous_validated_price_lineage,
        active_tickers=price_refresh_tickers,
        start_date=price_start,
        end_date=end_date,
        run_dir=paths.run_dir(run_id),
        run_id=run_id,
        ingested_at=ingested_at,
        raw_archive_dir=(
            WarehousePaths(data_root / "warehouse").raw / "yahoo" / "prices"
        ),
    )
    source_refresh_contract["source_semantics"]["yfinance_prices"][
        "validated_key_coverage"
    ] = yahoo_key_coverage
    yahoo_price_deltas = [
        _with_price_ingestion_metadata(
            yahoo_full_history,
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
    refreshed_price_roots, missing_network_price_tickers = _network_price_refresh_coverage(
        yahoo_prices_delta,
        requested_tickers=price_refresh_tickers,
    )
    source_refresh_contract["source_semantics"]["yfinance_prices"].update(
        {
            "network_refreshed_ticker_count": len(refreshed_price_roots),
            "network_missing_tickers": list(missing_network_price_tickers),
        }
    )
    if source_refresh_policy.refresh_full_price_history and missing_network_price_tickers:
        raise RuntimeError(
            "Full active-universe Yahoo history refresh is incomplete; "
            f"missing={list(missing_network_price_tickers)}. No package was published."
        )
    preliminary_prices = build_split_detection_prices(
        existing_prices=existing_price_history.select(list(PRICE_COLUMNS)),
        fresh_prices=yahoo_prices_delta.select(list(PRICE_COLUMNS)),
        full_history_refresh=source_refresh_policy.refresh_full_price_history,
    )
    preliminary_split_findings = find_extreme_adjusted_price_moves(
        preliminary_prices,
        event_since=rolling_price_start,
        tickers=[f"{ticker}.US" for ticker in price_quality_tickers],
    )
    split_repairs: list[dict[str, object]] = []
    if not preliminary_split_findings.is_empty():
        split_tickers = (
            preliminary_split_findings.select(
                pl.col("ticker").str.replace(r"\.US$", "").unique().sort()
            )
            .to_series()
            .to_list()
        )
        registry_splits, registry_manifest = load_confirmed_stock_splits(
            project_root
            / "configs"
            / "data_quality"
            / "confirmed_corporate_actions.json",
            tickers=[f"{ticker}.US" for ticker in split_tickers],
        )
        split_evidence = combine_stock_split_evidence(
            yahoo_client.fetch_stock_splits(split_tickers),
            registry_splits,
        )
        source_refresh_contract["corporate_action_registry"] = registry_manifest
        yahoo_prices_delta, split_repairs = repair_confirmed_split_discontinuities(
            yahoo_prices_delta,
            findings=preliminary_split_findings,
            splits=split_evidence,
        )
    source_refresh_contract["corporate_action_repairs"] = split_repairs
    simfin_price_tickers = _identify_simfin_price_fallback_tickers(
        requested_tickers=price_refresh_tickers,
        yahoo_prices_delta=yahoo_prices_delta,
        backfill_tickers=price_backfill_tickers,
    )
    simfin_prices_delta = _with_price_ingestion_metadata(
        simfin_client.fetch_daily_prices(simfin_price_tickers, start_date, end_date)
        if simfin_client.enabled and simfin_price_tickers
        else _empty_raw_price_frame(),
        dataset="prices_simfin",
        source="simfin",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    if simfin_client.last_fetch_failures:
        run_failures["simfin"].extend(simfin_client.last_fetch_failures)
    stockanalysis_price_tickers = _identify_stockanalysis_price_fallback_tickers(
        requested_tickers=price_refresh_tickers,
        covered_prices_delta=_concat_or_empty([yahoo_prices_delta, simfin_prices_delta], empty=_empty_raw_price_frame()),
        backfill_tickers=price_backfill_tickers,
    )
    stockanalysis_prices_delta = _with_price_ingestion_metadata(
        stockanalysis_client.fetch_daily_prices(stockanalysis_price_tickers, start_date, end_date)
        if stockanalysis_price_tickers
        else _empty_raw_price_frame(),
        dataset="prices_stockanalysis",
        source="stockanalysis",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    if stockanalysis_client.last_fetch_failures:
        run_failures["stockanalysis"].extend(stockanalysis_client.last_fetch_failures)
    benchmark_prices_delta = _with_price_ingestion_metadata(
        _download_yahoo_price_history(
            yahoo_client,
            tickers=("SPY",),
            start_date=price_start,
            end_date=end_date,
        ),
        dataset="prices_spy_yfinance",
        run_id=run_id,
        ingested_at=ingested_at,
    )
    benchmark_refreshed, benchmark_missing = _network_price_refresh_coverage(
        benchmark_prices_delta,
        requested_tickers=("SPY",),
    )
    source_refresh_contract["source_semantics"]["yfinance_prices"].update(
        {
            "benchmark_network_refreshed": "SPY" in benchmark_refreshed,
            "benchmark_network_missing_tickers": list(benchmark_missing),
        }
    )
    if source_refresh_policy.refresh_full_price_history and benchmark_missing:
        raise RuntimeError(
            "Full Yahoo benchmark history refresh is incomplete; "
            f"missing={list(benchmark_missing)}. No package was published."
        )
    if source_refresh_policy.refresh_full_price_history:
        _drop_refreshed_partitions(
            paths.raw_dir / "prices_yfinance.parquet",
            tickers=price_refresh_tickers,
            date_column="date",
            start_date=price_start,
            end_date=end_date,
        )
        _drop_refreshed_partitions(
            paths.raw_dir / "prices_simfin.parquet",
            tickers=simfin_price_tickers,
            date_column="date",
            start_date=price_start,
            end_date=end_date,
        )
        _drop_refreshed_partitions(
            paths.raw_dir / "prices_stockanalysis.parquet",
            tickers=stockanalysis_price_tickers,
            date_column="date",
            start_date=price_start,
            end_date=end_date,
        )
        _drop_refreshed_partitions(
            paths.raw_dir / "prices_spy_yfinance.parquet",
            tickers=("SPY",),
            date_column="date",
            start_date=price_start,
            end_date=end_date,
        )
    (
        raw_yahoo_prices,
        raw_simfin_prices,
        raw_stockanalysis_prices,
        clean_prices,
        clean_price_lineage,
        persistent_price_history_registry,
    ) = _prepare_canonical_hybrid_price_merge(
        paths=paths,
        yahoo_delta=yahoo_prices_delta,
        simfin_delta=simfin_prices_delta,
        stockanalysis_delta=stockanalysis_prices_delta,
        ticker_list=ticker_list,
        active_tickers=price_quality_tickers,
        event_since=rolling_price_start,
        start_date=start_date,
        expected_through=end_date,
        eodhd_seed_path=eodhd_price_seed_path,
        run_id=run_id,
        source_refresh_policy=source_refresh_policy,
        source_refresh_contract=source_refresh_contract,
        latest_composed_manifest_path=(
            data_root / "model_inputs" / "manifests" / "latest.json"
        ),
        preserved_terminal_tickers=terminal_price_tickers,
        reviewed_extreme_price_move_registry_path=(
            project_root
            / "configs"
            / "data_quality"
            / "reviewed_extreme_price_moves.json"
        ),
    )
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_yfinance.parquet", yahoo_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_simfin.parquet", simfin_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_stockanalysis.parquet", stockanalysis_prices_delta)
    append_run_delta(paths.run_dir(run_id) / "raw" / "prices_spy_yfinance.parquet", benchmark_prices_delta)
    raw_yahoo_prices.write_parquet(paths.raw_dir / "prices_yfinance.parquet")
    raw_simfin_prices.write_parquet(paths.raw_dir / "prices_simfin.parquet")
    raw_stockanalysis_prices.write_parquet(paths.raw_dir / "prices_stockanalysis.parquet")
    raw_benchmark_prices = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_prices_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )

    earnings_delta = _empty_raw_earnings_frame()
    earnings_sec_calendar_delta = _empty_raw_earnings_frame()
    earnings_sec_actuals_delta = _empty_raw_earnings_frame()
    sec_financials_all = _empty_raw_financial_base()
    sec_filing_deltas: list[pl.DataFrame] = []
    simfin_deltas: list[pl.DataFrame] = []
    yahoo_financial_deltas: list[pl.DataFrame] = []
    if refreshed_years:
        try:
            earnings_fetched = yahoo_client.fetch_earnings_dates(price_quality_tickers, limit=100)
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
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
        active_sec_calendar_failures = _required_failure_tickers(
            sec_calendar_failures,
            required_tickers=price_quality_tickers,
        )
        source_refresh_contract["source_semantics"]["sec_submissions"].update(
            {
                "active_network_failure_tickers": list(active_sec_calendar_failures),
                "active_network_complete": not active_sec_calendar_failures,
            }
        )
        if active_sec_calendar_failures:
            raise RuntimeError(
                "SEC submissions refresh is incomplete for the active universe; "
                f"failed={list(active_sec_calendar_failures)}. No package was published."
            )
        earnings_sec_calendar_delta = _with_earnings_ingestion_metadata(
            _concat_or_empty(sec_calendar_frames, empty=empty_earnings_calendar_frame()),
            dataset="earnings_sec_calendar",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_calendar.parquet", earnings_sec_calendar_delta)
        sec_frames, sec_actual_frames, sec_actual_failures = _fetch_sec_companyfacts_bundle(
            sec_client,
            sec_mapping,
        )
        run_failures["sec_companyfacts"].extend(sec_actual_failures)
        active_sec_actual_failures = _required_failure_tickers(
            sec_actual_failures,
            required_tickers=price_quality_tickers,
        )
        source_refresh_contract["source_semantics"]["sec_companyfacts"].update(
            {
                "active_network_failure_tickers": list(active_sec_actual_failures),
                "active_network_complete": not active_sec_actual_failures,
            }
        )
        if active_sec_actual_failures:
            raise RuntimeError(
                "SEC companyfacts refresh is incomplete for the active universe; "
                f"failed={list(active_sec_actual_failures)}. No package was published."
            )
        earnings_sec_actuals_delta = _with_earnings_ingestion_metadata(
            _filter_earnings_years(
                _concat_or_empty(sec_actual_frames, empty=empty_earnings_actuals_frame()),
                companyfacts_years,
            ),
            dataset="earnings_sec_actuals",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_sec_actuals.parquet", earnings_sec_actuals_delta)
        sec_financials_all = _concat_or_empty(sec_frames)

    successful_companyfacts_tickers = tuple(
        sorted(
            mapped_sec_tickers
            - {
                str(item["ticker"]).upper().removesuffix(".US")
                for item in run_failures["sec_companyfacts"]
                if item.get("ticker")
            }
        )
    )
    successful_submission_tickers = tuple(
        sorted(
            mapped_sec_tickers
            - {
                str(item["ticker"]).upper().removesuffix(".US")
                for item in run_failures["sec_filing"]
                if item.get("ticker") and item.get("dataset") == "earnings_sec_calendar"
            }
        )
    )
    if source_refresh_policy.refresh_sec_companyfacts:
        companyfacts_start = f"{min(companyfacts_years):04d}-01-01"
        companyfacts_end = f"{max(companyfacts_years):04d}-12-31"
        _drop_refreshed_partitions(
            paths.raw_dir / "earnings_sec_actuals.parquet",
            tickers=successful_companyfacts_tickers,
            date_column="period_end",
            start_date=companyfacts_start,
            end_date=companyfacts_end,
        )
        _drop_refreshed_partitions(
            paths.raw_dir / "financials_sec_companyfacts.parquet",
            tickers=successful_companyfacts_tickers,
            date_column="date",
            start_date=companyfacts_start,
            end_date=companyfacts_end,
        )
    if source_refresh_policy.refresh_sec_submissions:
        refreshed_start = f"{min(refreshed_years):04d}-01-01"
        refreshed_end = f"{max(refreshed_years):04d}-12-31"
        _drop_refreshed_partitions(
            paths.raw_dir / "earnings_sec_calendar.parquet",
            tickers=successful_submission_tickers,
            date_column="period_end",
            start_date=refreshed_start,
            end_date=refreshed_end,
        )

    sec_financial_deltas: list[pl.DataFrame] = [
        _with_financial_ingestion_metadata(
            _filter_financial_years(sec_financials_all, years=companyfacts_years),
            dataset="financials_sec_companyfacts",
            run_id=run_id,
            ingested_at=ingested_at,
        )
    ]

    for year in refreshed_years:
        sec_year = _filter_financial_year(sec_financials_all, year=year)
        sec_filing_tickers = _identify_sec_filing_fallback_tickers(
            tickers=price_quality_tickers,
            sec_companyfacts=sec_year,
        )
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
        yfinance_financial_tickers = tuple(
            sorted(set(yfinance_financial_tickers).intersection(price_quality_tickers))
        )
        yahoo_financial_year = (
            yahoo_client.fetch_quarterly_financials(yfinance_financial_tickers).filter(pl.col("date").str.starts_with(str(year)))
            if yfinance_financial_tickers
            else _empty_raw_financial_base()
        )
        try:
            simfin_year = simfin_client.fetch_quarterly_financials(ticker_list, year) if simfin_client.enabled else _empty_raw_financial_base()
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            simfin_year = _empty_raw_financial_base()
            run_failures["simfin"].append({"year": str(year), "error": str(exc)})
        run_failures["simfin"].extend(simfin_client.last_fetch_failures)

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

    clean_benchmark_prices = raw_benchmark_prices.select(
        ["date", "open", "high", "low", "close", "volume", "adjusted_close", "ticker"]
    ).sort(["ticker", "date"])
    raw_earnings, clean_earnings, clean_earnings_lineage, clean_earnings_long, earnings_repair_tickers = _repair_yahoo_earnings(
        paths=paths,
        run_id=run_id,
        ingested_at=ingested_at,
        yahoo_client=yahoo_client,
        raw_yahoo_earnings=raw_earnings,
        raw_earnings_sec_calendar=raw_earnings_sec_calendar,
        raw_earnings_sec_actuals=raw_earnings_sec_actuals,
        candidate_tickers=price_quality_tickers,
        years=refreshed_years,
    )

    financial_source_frames: list[tuple[str, pl.DataFrame, int]] = [
        ("sec_companyfacts", raw_sec_financials, 1),
        ("sec_filing", raw_sec_filing_financials, 2),
        ("simfin", raw_simfin_financials, 3),
        ("yfinance", raw_yahoo_financials, 4),
    ]
    sanitized_financial_sources: list[FinancialSourceInput] = []
    share_candidate_quarantine: dict[str, object] = {}
    for source_name, source_frame, priority in financial_source_frames:
        sanitized, quarantine_report = quarantine_implausible_share_candidates(
            source_frame.select(_clean_financial_columns())
        )
        share_candidate_quarantine[source_name] = quarantine_report
        sanitized_financial_sources.append(
            FinancialSourceInput(
                source_name=source_name,
                frame=sanitized,
                priority=priority,
            )
        )
    source_refresh_contract["share_candidate_quarantine"] = share_candidate_quarantine
    write_json(
        paths.run_dir(run_id) / "share_candidate_quarantine.json",
        share_candidate_quarantine,
    )

    (
        consolidated_financials,
        consolidated_lineage,
        source_summary,
        share_selection_quality,
    ) = consolidate_financial_sources_with_share_quality(
        sanitized_financial_sources
    )
    source_refresh_contract["share_selection_quality"] = share_selection_quality
    write_json(
        paths.run_dir(run_id) / "share_selection_quality.json",
        share_selection_quality,
    )

    fundamental_quality = audit_fundamental_quality(consolidated_financials)
    source_refresh_contract["fundamental_quality_guard"] = fundamental_quality
    write_json(
        paths.run_dir(run_id) / "fundamental_quality_guard.json",
        fundamental_quality,
    )
    validate_fundamental_quality(fundamental_quality)

    clean_prices.write_parquet(paths.clean_dir / "prices_open_source.parquet")
    clean_price_lineage.write_parquet(paths.clean_dir / "prices_open_source_lineage.parquet")
    persistent_price_history_registry.write_parquet(
        paths.clean_dir / "persistent_price_history_registry.parquet"
    )
    clean_benchmark_prices.write_parquet(paths.clean_dir / "benchmark_prices_open_source.parquet")
    clean_earnings.write_parquet(paths.clean_dir / "earnings_open_source_consolidated.parquet")
    clean_earnings_lineage.write_parquet(paths.clean_dir / "earnings_open_source_lineage.parquet")
    clean_earnings_long.write_parquet(paths.clean_dir / "earnings_open_source_long.parquet")
    consolidated_financials.write_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage.write_parquet(paths.clean_dir / "financials_open_source_lineage.parquet")
    source_summary.write_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")
    general_reference.write_parquet(paths.clean_dir / "general_reference.parquet")
    general_reference_lineage.write_parquet(paths.clean_dir / "general_reference_lineage.parquet")

    constituents_frame = pl.read_csv(reference_data_dir / "SP500_Constituents.csv", try_parse_dates=True)
    data_freshness = build_data_freshness_summary(
        prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        financials=consolidated_financials,
        earnings_sec_calendar=raw_earnings_sec_calendar,
        constituents=constituents_frame,
        terminal_tickers=terminal_price_tickers,
    )
    validate_data_freshness(data_freshness, expected_through=end_date)

    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=clean_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=paths.legacy_dir,
    )
    _audit_and_validate_historical_revisions(
        paths=paths,
        run_id=run_id,
        legacy_paths=legacy_paths,
        expected_through=end_date,
        source_refresh_policy=source_refresh_policy,
        source_refresh_contract=source_refresh_contract,
    )
    published_output_paths = publish_open_source_output_package(
        output_dir=paths.output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        prices_lineage=clean_price_lineage,
        persistent_price_history_registry=persistent_price_history_registry,
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
            "source_refresh_contract": source_refresh_contract,
            "data_freshness": data_freshness,
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
        "simfin_price_fallback_ticker_count": len(simfin_price_tickers),
        "simfin_price_fallback_ticker_examples": list(simfin_price_tickers[:20]),
        "stockanalysis_price_fallback_ticker_count": len(stockanalysis_price_tickers),
        "stockanalysis_price_fallback_ticker_examples": list(stockanalysis_price_tickers[:20]),
        "financial_years_refreshed": list(refreshed_years),
        "sec_companyfacts_years_refreshed": list(companyfacts_years),
        "source_refresh_contract": source_refresh_contract,
        "data_freshness": data_freshness,
        "corporate_action_repairs": split_repairs,
        "ticker_count": len(ticker_list),
        "earnings_repair_ticker_count": len(earnings_repair_tickers),
        "earnings_repair_ticker_examples": list(earnings_repair_tickers[:20]),
        "official_dir": str(paths.base_dir),
        "live_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "raw_outputs": {
            "general_reference": "raw/general_reference.parquet",
            "general_reference_lineage": "raw/general_reference_lineage.parquet",
            "prices_yfinance": "raw/prices_yfinance.parquet",
            "prices_simfin": "raw/prices_simfin.parquet",
            "prices_stockanalysis": "raw/prices_stockanalysis.parquet",
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
            "prices_open_source_lineage": "target/prices_open_source_lineage.parquet",
            "persistent_price_history_registry": "target/persistent_price_history_registry.parquet",
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
    (paths.manifests_dir / "raw_store_quarantine.json").unlink(missing_ok=True)

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
        sec_companyfacts_years=companyfacts_years,
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
