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
from alpharank.data.open_source.config import METRIC_SPECS
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
    "reportDate": pl.String,
    "earningsDatetime": pl.String,
    "period_end": pl.String,
    "epsEstimate": pl.Float64,
    "epsActual": pl.Float64,
    "surprisePercent": pl.Float64,
    "source": pl.String,
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
    refreshed_years = _resolve_refreshed_years(
        mode=mode,
        start_date=start_date,
        end_date=end_date,
        lookback_years=financial_lookback_years,
    )

    yahoo_client = YahooFinanceClient()
    sec_client = SecCompanyFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_companyfacts")
    sec_filing_client = SecFilingFactsClient(user_agent=user_agent, cache_dir=project_root / "data" / "open_source" / "_cache" / "sec_filing")
    simfin_client = SimFinClient(api_key=simfin_api_key, data_dir=project_root / "data" / "open_source" / "_cache" / "simfin")

    sec_mapping_all = sec_client.fetch_company_mapping()
    sec_mapping = sec_mapping_all.filter(pl.col("ticker").is_in(list(ticker_list)))
    general_reference_delta = _with_general_ingestion_metadata(
        yahoo_client.fetch_general_reference(ticker_list, sec_mapping),
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

    yahoo_prices_delta = _with_price_ingestion_metadata(
        yahoo_client.download_prices(ticker_list, price_start, end_date),
        dataset="prices_yfinance",
        run_id=run_id,
        ingested_at=ingested_at,
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
    raw_benchmark_prices = upsert_parquet(
        paths.raw_dir / "prices_spy_yfinance.parquet",
        benchmark_prices_delta,
        key_cols=["ticker", "date", "source"],
        order_cols=["ingested_at"],
    )

    earnings_delta = _empty_raw_earnings_frame()
    sec_financial_deltas: list[pl.DataFrame] = []
    sec_filing_deltas: list[pl.DataFrame] = []
    simfin_deltas: list[pl.DataFrame] = []
    yahoo_financial_deltas: list[pl.DataFrame] = []
    run_failures: dict[str, list[dict[str, str]]] = {
        "sec_companyfacts": [],
        "sec_filing": [],
        "simfin": [],
    }

    if refreshed_years:
        earnings_delta = _with_earnings_ingestion_metadata(
            yahoo_client.fetch_earnings_dates(ticker_list, limit=max(8, len(refreshed_years) * 4)),
            dataset="earnings_yfinance",
            run_id=run_id,
            ingested_at=ingested_at,
        )
        append_run_delta(paths.run_dir(run_id) / "raw" / "earnings_yfinance.parquet", earnings_delta)

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
    clean_earnings = raw_earnings.select(
        ["ticker", "reportDate", "earningsDatetime", "period_end", "epsEstimate", "epsActual", "surprisePercent", "source"]
    ).sort(["ticker", "reportDate"])
    clean_earnings_long = yahoo_client.normalize_earnings_long(clean_earnings)

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
    clean_earnings.write_parquet(paths.clean_dir / "earnings_open_source.parquet")
    clean_earnings_long.write_parquet(paths.clean_dir / "earnings_open_source_long.parquet")
    consolidated_financials.write_parquet(paths.clean_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage.write_parquet(paths.clean_dir / "financials_open_source_lineage.parquet")
    source_summary.write_parquet(paths.clean_dir / "financials_open_source_source_summary.parquet")

    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=clean_benchmark_prices,
        general_reference=general_reference.select(["ticker", "name", "exchange", "cik", "source"]),
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
        general_reference=general_reference.select(["ticker", "name", "exchange", "cik", "source"]),
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_frame=clean_earnings,
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
        "financial_years_refreshed": list(refreshed_years),
        "ticker_count": len(ticker_list),
        "official_dir": str(paths.base_dir),
        "live_dir": str(paths.base_dir),
        "target_dir": str(paths.target_dir),
        "raw_outputs": {
            "general_reference": "raw/general_reference.parquet",
            "prices_yfinance": "raw/prices_yfinance.parquet",
            "prices_spy_yfinance": "raw/prices_spy_yfinance.parquet",
            "earnings_yfinance": "raw/earnings_yfinance.parquet",
            "financials_sec_companyfacts": "raw/financials_sec_companyfacts.parquet",
            "financials_sec_filing": "raw/financials_sec_filing.parquet",
            "financials_simfin": "raw/financials_simfin.parquet",
            "financials_yfinance": "raw/financials_yfinance.parquet",
        },
        "clean_outputs": {
            "prices_open_source": "target/prices_open_source.parquet",
            "benchmark_prices_open_source": "target/benchmark_prices_open_source.parquet",
            "earnings_open_source": "target/earnings_open_source.parquet",
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
            *([build_earnings_alignment(eodhd_earnings, clean_earnings)] if not clean_earnings.is_empty() else []),
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
        return pl.DataFrame(schema=RAW_PRICE_SCHEMA)
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
    return frame.with_columns(
        [
            pl.lit(dataset).alias("dataset"),
            pl.lit(run_id).alias("ingestion_run_id"),
            pl.lit(ingested_at).alias("ingested_at"),
        ]
    ).select(list(RAW_EARNINGS_SCHEMA))


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


def _concat_or_empty(frames: Sequence[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    if not non_empty:
        return _empty_raw_financial_base()
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
