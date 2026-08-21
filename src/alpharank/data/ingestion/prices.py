"""Price acquisition, retention, consolidation and validation stage."""

from __future__ import annotations

import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Sequence

import polars as pl

from alpharank.data.ingestion.config import PRICE_COLUMNS
from alpharank.data.sources.constituents import load_constituent_change_registry
from alpharank.data.publishing.definitive_prices import (
    bootstrap_definitive_prices,
    build_definitive_prices,
    stage_yahoo_prices,
)
from alpharank.data.ingestion.frames import (
    _concat_or_empty,
    _empty_raw_price_frame,
    _with_price_ingestion_metadata,
)
from alpharank.data.open_source.price_quality import (
    assert_no_extreme_adjusted_price_moves,
    load_reviewed_extreme_price_moves,
)
from alpharank.data.ingestion.raw_archive import RAW_DELTA_CONTRACT, archive_raw_frame_delta
from alpharank.data.ingestion.refresh_policy import SourceRefreshPolicy
from alpharank.data.ingestion.storage import (
    OpenSourceLivePaths,
    merge_upsert_frames,
    utc_now_iso,
    write_json,
)
from alpharank.data.sources.yahoo import YahooFinanceClient
from alpharank.data.prices import (
    audit_price_candidate,
    build_persistent_price_history_registry,
    compose_hybrid_price_history,
    load_eodhd_seed,
    persistent_history_summary,
    resolve_previous_validated_price_lineage,
    roll_forward_validated_price_history,
    validate_price_candidate,
)


def _resolve_price_start(
    *,
    mode: str,
    explicit_start_date: str,
    raw_price_path: Path,
    lookback_days: int,
    existing_prices: pl.DataFrame | None = None,
) -> str:
    if mode == "bootstrap":
        return explicit_start_date
    if existing_prices is not None:
        existing = existing_prices
    elif raw_price_path.exists():
        existing = pl.read_parquet(raw_price_path)
    else:
        return explicit_start_date
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


def _identify_simfin_price_fallback_tickers(
    *,
    requested_tickers: Sequence[str],
    yahoo_prices_delta: pl.DataFrame,
    backfill_tickers: Sequence[str],
) -> tuple[str, ...]:
    if yahoo_prices_delta.is_empty():
        return tuple(sorted(set(requested_tickers)))
    yahoo_covered = set(
        yahoo_prices_delta.select(pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"))
        .unique()
        .to_series()
        .to_list()
    )
    fallback = set(backfill_tickers)
    fallback.update(ticker for ticker in requested_tickers if ticker not in yahoo_covered)
    return tuple(sorted(fallback))


def _network_price_refresh_coverage(
    prices_delta: pl.DataFrame,
    *,
    requested_tickers: Sequence[str],
) -> tuple[set[str], tuple[str, ...]]:
    refreshed = (
        set(
            prices_delta.select(
                pl.col("ticker")
                .cast(pl.String)
                .str.replace(r"\.US$", "")
                .str.to_uppercase()
            )
            .unique()
            .to_series()
            .to_list()
        )
        if not prices_delta.is_empty()
        else set()
    )
    requested = {str(ticker).upper().removesuffix(".US") for ticker in requested_tickers}
    return refreshed, tuple(sorted(requested - refreshed))


def _drop_refreshed_partitions(
    path: Path,
    *,
    tickers: Sequence[str],
    date_column: str,
    start_date: str,
    end_date: str,
) -> None:
    """Remove only the partitions a successful full network fetch will replace."""
    if not path.exists() or not tickers:
        return
    frame = pl.read_parquet(path)
    if frame.is_empty():
        return
    ticker_roots = sorted(
        {str(ticker).upper().removesuffix(".US") for ticker in tickers}
    )
    ticker_root = (
        pl.col("ticker")
        .cast(pl.String)
        .str.to_uppercase()
        .str.replace(r"\.US$", "")
    )
    normalized_date = pl.col(date_column).cast(pl.String, strict=False).str.slice(0, 10)
    replaced_partition = (
        ticker_root.is_in(ticker_roots)
        & (normalized_date >= start_date)
        & (normalized_date <= end_date)
    )
    frame.filter(replaced_partition.not_()).write_parquet(path)


def _required_failure_tickers(
    failures: Sequence[dict[str, str]],
    *,
    required_tickers: Sequence[str],
) -> tuple[str, ...]:
    required = {
        str(ticker).upper().removesuffix(".US") for ticker in required_tickers
    }
    failed = {
        str(item["ticker"]).upper().removesuffix(".US")
        for item in failures
        if item.get("ticker")
    }
    return tuple(sorted(required.intersection(failed)))


def _resolve_sec_mapping_coverage(
    *,
    sec_mapping: pl.DataFrame,
    required_tickers: Sequence[str],
) -> tuple[set[str], set[str], tuple[str, ...]]:
    mapped = {
        str(ticker).upper().removesuffix(".US")
        for ticker in sec_mapping.get_column("ticker").to_list()
    }
    required = {
        str(ticker).upper().removesuffix(".US") for ticker in required_tickers
    }
    return mapped, required, tuple(sorted(required - mapped))


def _download_yahoo_price_history(
    yahoo_client: YahooFinanceClient,
    *,
    tickers: Sequence[str],
    start_date: str,
    end_date: str,
    max_attempts: int = 3,
) -> pl.DataFrame:
    remaining = tuple(sorted({str(ticker).upper().removesuffix(".US") for ticker in tickers}))
    frames: list[pl.DataFrame] = []
    for attempt in range(max_attempts):
        if not remaining:
            break
        fetched = yahoo_client.download_prices(remaining, start_date, end_date)
        frames.append(fetched)
        refreshed, _ = _network_price_refresh_coverage(
            _concat_or_empty(frames, empty=fetched),
            requested_tickers=tickers,
        )
        remaining = tuple(sorted(set(remaining) - refreshed))
        if remaining and attempt + 1 < max_attempts:
            time.sleep(min(4.0, 2.0**attempt))
    if not frames:
        return pl.DataFrame()
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .sort(["ticker", "date"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        .sort(["ticker", "date"])
    )


def _historical_yahoo_key_gaps(
    *,
    previous_validated_lineage: pl.DataFrame,
    fresh_prices: pl.DataFrame,
    active_tickers: Sequence[str],
    start_date: str,
    end_date: str,
    recent_mutable_calendar_days: int = 7,
) -> pl.DataFrame:
    """Return validated active keys absent from a purported full Yahoo vintage."""

    active = sorted(
        {f"{str(ticker).upper().removesuffix('.US')}.US" for ticker in active_tickers}
    )
    cutoff = date.fromisoformat(end_date) - timedelta(days=recent_mutable_calendar_days)
    previous_keys = (
        previous_validated_lineage.select(
            pl.col("ticker").cast(pl.String).str.to_uppercase(),
            pl.col("date").cast(pl.Date, strict=False),
        )
        .filter(
            pl.col("ticker").is_in(active)
            & (pl.col("date") >= pl.lit(date.fromisoformat(start_date)))
            & (pl.col("date") < pl.lit(cutoff))
        )
        .unique()
    )
    fresh_keys = (
        fresh_prices.filter(
            pl.col("adjusted_close").cast(pl.Float64, strict=False).is_not_null()
            & (pl.col("adjusted_close").cast(pl.Float64, strict=False) > 0.0)
        )
        .select(
            pl.col("ticker").cast(pl.String).str.to_uppercase(),
            pl.col("date").cast(pl.Date, strict=False),
        )
        .unique()
        if not fresh_prices.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "date": pl.Date})
    )
    return previous_keys.join(fresh_keys, on=["ticker", "date"], how="anti").sort(
        ["ticker", "date"]
    )


def _complete_yahoo_history_against_validated(
    yahoo_client: YahooFinanceClient,
    *,
    initial_prices: pl.DataFrame,
    previous_validated_lineage: pl.DataFrame,
    active_tickers: Sequence[str],
    start_date: str,
    end_date: str,
    max_repair_rounds: int = 2,
    first_round_chunk_size: int = 10,
    run_dir: Path | None = None,
    run_id: str | None = None,
    ingested_at: str | None = None,
    raw_archive_dir: Path | None = None,
) -> tuple[pl.DataFrame, dict[str, object]]:
    """Retry active Yahoo histories until every old validated key is present."""

    frames = [initial_prices] if not initial_prices.is_empty() else []
    candidate = initial_prices
    initial_gaps = _historical_yahoo_key_gaps(
        previous_validated_lineage=previous_validated_lineage,
        fresh_prices=candidate,
        active_tickers=active_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    gaps = initial_gaps
    retried_tickers: set[str] = set()
    for repair_round in range(max_repair_rounds):
        if gaps.is_empty():
            break
        missing_tickers = gaps.select(
            pl.col("ticker").str.replace(r"\.US$", "").unique().sort()
        ).to_series().to_list()
        retried_tickers.update(missing_tickers)
        chunk_size = first_round_chunk_size if repair_round == 0 else 1
        for start_index in range(0, len(missing_tickers), chunk_size):
            chunk = tuple(missing_tickers[start_index : start_index + chunk_size])
            fetched = yahoo_client.download_prices(chunk, start_date, end_date)
            if not fetched.is_empty():
                frames.append(fetched)
        candidate = (
            pl.concat(frames, how="diagonal_relaxed")
            .sort(["ticker", "date"])
            .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
            .sort(["ticker", "date"])
            if frames
            else pl.DataFrame()
        )
        gaps = _historical_yahoo_key_gaps(
            previous_validated_lineage=previous_validated_lineage,
            fresh_prices=candidate,
            active_tickers=active_tickers,
            start_date=start_date,
            end_date=end_date,
        )

    report = {
        "contract": "complete_active_yahoo_vintage_against_previous_validated_keys_v1",
        "run_id": run_id,
        "initial_missing_key_count": initial_gaps.height,
        "initial_missing_ticker_count": initial_gaps.select(
            pl.col("ticker").n_unique()
        ).item(),
        "retried_ticker_count": len(retried_tickers),
        "remaining_missing_key_count": gaps.height,
        "remaining_missing_ticker_count": gaps.select(
            pl.col("ticker").n_unique()
        ).item(),
        "provider_complete": gaps.is_empty(),
    }
    incomplete_provider_tickers = (
        gaps.select(pl.col("ticker").unique().sort()).to_series().to_list()
        if not gaps.is_empty()
        else []
    )
    provider_candidate = candidate
    if raw_archive_dir is not None:
        if run_id is None or ingested_at is None:
            raise ValueError("run_id and ingested_at are required for the immutable raw archive")
        raw_archive = archive_raw_frame_delta(
            archive_dir=raw_archive_dir,
            run_id=run_id,
            frame=candidate,
            key_columns=("ticker", "date"),
            source="yahoo",
            dataset="prices",
            observed_at=ingested_at,
            request={
                "start_date": start_date,
                "end_date": end_date,
                "active_ticker_count": len(active_tickers),
                "repair_round_count": max_repair_rounds,
                "retried_ticker_count": len(retried_tickers),
            },
        )
        report["raw_archive"] = {
            "contract": RAW_DELTA_CONTRACT,
            "run_id": raw_archive.run_id,
            "manifest_path": str(raw_archive.manifest_path),
            "parent_run_id": raw_archive.parent_run_id,
            "input_row_count": raw_archive.input_row_count,
            "stored_content_row_count": raw_archive.stored_content_row_count,
            "unchanged_row_count": raw_archive.unchanged_row_count,
            "inserted_row_count": raw_archive.inserted_row_count,
            "updated_row_count": raw_archive.updated_row_count,
            "restored_row_count": raw_archive.restored_row_count,
            "missing_row_count": raw_archive.missing_row_count,
            "snapshot_sha256": raw_archive.snapshot_sha256,
        }
    staged_current = stage_yahoo_prices(
        provider_candidate,
        run_id=run_id or "unknown",
        observed_at=ingested_at or utc_now_iso(),
    )
    definitive = build_definitive_prices(
        staged_current=staged_current,
        previous_definitive=bootstrap_definitive_prices(
            _with_price_ingestion_metadata(
                previous_validated_lineage,
                dataset="previous_validated_price_lineage",
                source="validated_price_lineage",
                run_id="previous_validated_price_lineage",
                ingested_at="unknown",
            )
        ),
        requested_tickers=active_tickers,
        freeze_previous_prefix_tickers=incomplete_provider_tickers,
    )
    definitive_gaps = _historical_yahoo_key_gaps(
        previous_validated_lineage=previous_validated_lineage,
        fresh_prices=definitive.frame,
        active_tickers=active_tickers,
        start_date=start_date,
        end_date=end_date,
    )
    report["definitive_resolution"] = {
        "contract": "exact_key_last_valid_raw_v1",
        "current_row_count": definitive.current_row_count,
        "carried_forward_row_count": definitive.carried_forward_row_count,
        "unresolved_row_count": definitive.unresolved_row_count,
        "frozen_previous_prefix_tickers": incomplete_provider_tickers,
        "frozen_previous_prefix_row_count": definitive.audit.filter(
            pl.col("selection_reason")
            == "carried_forward_incomplete_ticker_prefix"
        ).height,
        "remaining_previous_validated_key_count": definitive_gaps.height,
        "passed": definitive_gaps.is_empty(),
    }
    report["passed"] = definitive_gaps.is_empty()
    candidate = definitive.frame
    if run_dir is not None:
        definitive.audit.write_parquet(run_dir / "def_price_selection_audit.parquet")
        _write_yahoo_attempt_audit(
            run_dir=run_dir,
            candidate=provider_candidate,
            initial_gaps=initial_gaps,
            remaining_gaps=gaps,
            report=report,
            run_id=run_id,
            ingested_at=ingested_at,
        )
    if not definitive_gaps.is_empty():
        examples = definitive_gaps.head(20).with_columns(pl.col("date").cast(pl.String)).to_dicts()
        raise RuntimeError(
            "DEF price resolution is missing previously validated exact ticker/date "
            f"keys after RAW retries and prior-value resolution; report={report}; examples={examples}. "
            "No package was published."
        )
    return candidate, report


def _write_yahoo_attempt_audit(
    *,
    run_dir: Path,
    candidate: pl.DataFrame,
    initial_gaps: pl.DataFrame,
    remaining_gaps: pl.DataFrame,
    report: dict[str, object],
    run_id: str | None,
    ingested_at: str | None,
) -> None:
    """Persist an immutable run-scoped record before any publication decision."""

    raw_dir = run_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    attempted = _with_price_ingestion_metadata(
        candidate,
        dataset="prices_yfinance_attempted",
        run_id=run_id or "unknown",
        ingested_at=ingested_at or utc_now_iso(),
    )
    attempted.write_parquet(raw_dir / "prices_yfinance_attempted.parquet")
    initial_gaps.write_parquet(run_dir / "price_validated_key_gaps_initial.parquet")
    remaining_gaps.write_parquet(run_dir / "price_validated_key_gaps_remaining.parquet")
    write_json(run_dir / "price_validated_key_coverage.json", report)


def _confirmed_terminal_price_tickers(
    *,
    registry_path: Path,
    active_tickers: Sequence[str],
    expected_through: str,
) -> tuple[str, ...]:
    """Resolve sourced removals that post-date the latest monthly snapshot."""

    registry = load_constituent_change_registry(registry_path)
    active = {
        str(ticker).upper().removesuffix(".US") for ticker in active_tickers
    }
    cutoff = date.fromisoformat(expected_through)
    confirmed = {
        str(operation["ticker"]).upper().removesuffix(".US")
        for event in registry["events"]
        if date.fromisoformat(str(event["effective_date"])) <= cutoff
        for operation in event.get("operations", [])
        if operation.get("action") == "remove"
        and str(operation.get("ticker", "")).upper().removesuffix(".US") in active
    }
    return tuple(sorted(f"{ticker}.US" for ticker in confirmed))


def _identify_stockanalysis_price_fallback_tickers(
    *,
    requested_tickers: Sequence[str],
    covered_prices_delta: pl.DataFrame,
    backfill_tickers: Sequence[str],
) -> tuple[str, ...]:
    if covered_prices_delta.is_empty():
        return tuple(sorted(set(requested_tickers)))
    covered = set(
        covered_prices_delta.select(pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"))
        .unique()
        .to_series()
        .to_list()
    )
    fallback = set(backfill_tickers)
    fallback.update(ticker for ticker in requested_tickers if ticker not in covered)
    return tuple(sorted(fallback))


def _resolve_refreshed_years(*, mode: str, start_date: str, end_date: str, lookback_years: int) -> tuple[int, ...]:
    start_year = int(start_date[:4])
    end_year = int(end_date[:4])
    if mode == "bootstrap":
        return tuple(range(start_year, end_year + 1))
    first_year = max(start_year, end_year - lookback_years + 1)
    return tuple(range(first_year, end_year + 1))


def _load_reference_tickers(reference_data_dir: Path, *, start_date: str) -> tuple[str, ...]:
    start_date_value = date.fromisoformat(start_date)
    return tuple(
        pl.read_parquet(reference_data_dir / "US_Finalprice.parquet")
        .filter(pl.col("date").cast(pl.Date, strict=False) >= pl.lit(start_date_value))
        .select(pl.col("ticker").str.replace(r"\.US$", "").alias("ticker"))
        .unique()
        .sort("ticker")
        .to_series()
        .to_list()
    )


def _load_latest_sp500_tickers(reference_data_dir: Path) -> tuple[str, ...]:
    constituents = pl.read_csv(reference_data_dir / "SP500_Constituents.csv", try_parse_dates=True).select(
        pl.col("Date").cast(pl.Date, strict=False),
        pl.col("Ticker").cast(pl.String).str.to_uppercase(),
    )
    latest_month = constituents.select(pl.col("Date").max()).item()
    return tuple(
        constituents.filter(pl.col("Date") == latest_month)
        .select(pl.col("Ticker").unique().sort())
        .to_series()
        .to_list()
    )


def _load_existing_open_source_tickers(paths: OpenSourceLivePaths, reference_data_dir: Path) -> tuple[str, ...]:
    candidate_paths = (
        paths.output_dir / "US_Finalprice.parquet",
        paths.clean_dir / "prices_open_source.parquet",
        paths.raw_dir / "prices_yfinance.parquet",
        paths.raw_dir / "prices_simfin.parquet",
        paths.raw_dir / "prices_stockanalysis.parquet",
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


def _load_existing_price_history_frame(paths: OpenSourceLivePaths) -> pl.DataFrame:
    clean_candidates = (
        paths.clean_dir / "prices_open_source.parquet",
        paths.output_dir / "US_Finalprice.parquet",
    )
    for path in clean_candidates:
        if path.exists():
            frame = pl.read_parquet(path)
            if not frame.is_empty():
                return frame.select(list(PRICE_COLUMNS))

    raw_frames: list[pl.DataFrame] = []
    for path in (
        paths.raw_dir / "prices_yfinance.parquet",
        paths.raw_dir / "prices_simfin.parquet",
        paths.raw_dir / "prices_stockanalysis.parquet",
    ):
        if path.exists():
            frame = pl.read_parquet(path)
            if not frame.is_empty():
                raw_frames.append(frame)
    if not raw_frames:
        return _empty_raw_price_frame().select(list(PRICE_COLUMNS))
    clean_prices, _ = _consolidate_price_sources(raw_frames, ticker_list=())
    return clean_prices


def _load_existing_price_tickers(paths: OpenSourceLivePaths) -> tuple[str, ...]:
    candidate_paths = (
        paths.output_dir / "US_Finalprice.parquet",
        paths.clean_dir / "prices_open_source.parquet",
        paths.raw_dir / "prices_yfinance.parquet",
        paths.raw_dir / "prices_simfin.parquet",
        paths.raw_dir / "prices_stockanalysis.parquet",
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


def _prepare_canonical_hybrid_price_merge(
    *,
    paths: OpenSourceLivePaths,
    yahoo_delta: pl.DataFrame,
    simfin_delta: pl.DataFrame,
    stockanalysis_delta: pl.DataFrame,
    ticker_list: Sequence[str],
    active_tickers: Sequence[str],
    event_since: str,
    start_date: str,
    expected_through: str,
    eodhd_seed_path: Path,
    run_id: str,
    source_refresh_policy: SourceRefreshPolicy,
    source_refresh_contract: dict[str, object],
    latest_composed_manifest_path: Path | None = None,
    preserved_terminal_tickers: Sequence[str] = (),
    reviewed_extreme_price_move_registry_path: Path | None = None,
) -> tuple[
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
    pl.DataFrame,
]:
    prospective = _merge_prospective_price_sources(
        paths=paths,
        yahoo_delta=yahoo_delta,
        simfin_delta=simfin_delta,
        stockanalysis_delta=stockanalysis_delta,
        ticker_list=ticker_list,
    )
    seed = load_eodhd_seed(eodhd_seed_path, start_date=start_date)
    price_policy = source_refresh_policy.price_gate_policy()
    if latest_composed_manifest_path is not None:
        previous_source = resolve_previous_validated_price_lineage(
            latest_composed_manifest_path
        )
        previous_lineage = pl.read_parquet(previous_source.lineage_path)
        previous_prices = previous_lineage.select(list(PRICE_COLUMNS))
        hybrid = roll_forward_validated_price_history(
            previous_validated_lineage=previous_lineage,
            active_yahoo_vintage=yahoo_delta,
            active_tickers=active_tickers,
            preserved_terminal_tickers=preserved_terminal_tickers,
            active_resolution_vintage_id=run_id,
        )
        source_refresh_contract["previous_validated_price_lineage"] = {
            "path": str(previous_source.lineage_path),
            "price_manifest_path": str(previous_source.price_manifest_path),
            "snapshot_dir": str(previous_source.snapshot_dir),
            "composition_id": previous_source.composition_id,
            "resolution": "latest_composed_model_snapshot",
        }
    else:
        previous_path = paths.output_dir / "US_Finalprice.parquet"
        previous_prices = (
            pl.read_parquet(previous_path) if previous_path.exists() else None
        )
        retained_open_history = _load_retained_open_price_vintages(
            paths=paths,
            prospective=prospective,
            active_tickers=active_tickers,
            ticker_list=ticker_list,
        )
        hybrid = compose_hybrid_price_history(
            eodhd_seed=seed.frame,
            active_yahoo_vintage=yahoo_delta,
            retained_open_history=retained_open_history,
            active_tickers=active_tickers,
            policy=price_policy,
        )

    persistent_registry = build_persistent_price_history_registry(
        hybrid.lineage,
        active_tickers=active_tickers,
        preserved_terminal_tickers=preserved_terminal_tickers,
    )
    persistent_summary = persistent_history_summary(persistent_registry)
    source_refresh_contract["persistent_price_history"] = {
        **persistent_summary,
        "semantics": (
            "Every ticker/date in the preceding validated lineage is retained "
            "when the ticker leaves the active refresh universe, including "
            "histories first acquired from Yahoo and absent from EODHD."
        ),
        "routine_deletion_allowed": False,
    }

    gate = audit_price_candidate(
        previous_prices=previous_prices,
        candidate_prices=hybrid.prices,
        candidate_lineage=hybrid.lineage,
        active_tickers=tuple(
            ticker
            for ticker in active_tickers
            if f"{str(ticker).upper().removesuffix('.US')}.US"
            not in {
                f"{str(terminal).upper().removesuffix('.US')}.US"
                for terminal in preserved_terminal_tickers
            }
        ),
        expected_eodhd_keys=seed.frame.select("ticker", "date"),
        expected_through=expected_through,
        policy=price_policy,
        active_resolution_vintage_id=run_id,
    )

    source_refresh_contract["eodhd_price_seed"] = seed.manifest()
    source_refresh_contract["price_composition"] = hybrid.composition_report
    source_refresh_contract["price_revision_guard"] = gate.report
    run_dir = paths.run_dir(run_id)
    write_json(run_dir / "price_composition.json", hybrid.composition_report)
    write_json(run_dir / "price_revision_guard.json", gate.report)
    persistent_registry.write_parquet(
        run_dir / "persistent_price_history_registry.parquet"
    )
    gate.daily_return_revisions.write_parquet(
        run_dir / "price_daily_return_revisions.parquet"
    )
    gate.transition_factor_findings.write_parquet(
        run_dir / "price_transition_factor_findings.parquet"
    )
    gate.historical_key_removals.write_parquet(
        run_dir / "price_historical_key_removals.parquet"
    )
    validate_price_candidate(gate)

    terminal_set = {
        f"{str(ticker).upper().removesuffix('.US')}.US"
        for ticker in preserved_terminal_tickers
    }
    quality_tickers = [
        normalized
        for ticker in active_tickers
        if (normalized := f"{str(ticker).upper().removesuffix('.US')}.US")
        not in terminal_set
    ]
    reviewed_moves = None
    reviewed_move_manifest: dict[str, object] | None = None
    if reviewed_extreme_price_move_registry_path is not None:
        reviewed_moves, reviewed_move_manifest = load_reviewed_extreme_price_moves(
            reviewed_extreme_price_move_registry_path
        )
    reviewed_findings = assert_no_extreme_adjusted_price_moves(
        hybrid.prices,
        event_since=event_since,
        tickers=quality_tickers,
        reviewed_moves=reviewed_moves,
    )
    if reviewed_move_manifest is not None:
        reviewed_move_report = {
            **reviewed_move_manifest,
            "matched_count": reviewed_findings.height,
            "matched_events": reviewed_findings.with_columns(
                pl.col("date").cast(pl.String)
            ).to_dicts(),
        }
        source_refresh_contract["reviewed_extreme_price_moves"] = reviewed_move_report
        write_json(run_dir / "reviewed_extreme_price_moves.json", reviewed_move_report)
    return (
        prospective[0],
        prospective[1],
        prospective[2],
        hybrid.prices,
        hybrid.lineage,
        persistent_registry,
    )


def _prepare_validated_stock_price_merge(
    *,
    paths: OpenSourceLivePaths,
    yahoo_delta: pl.DataFrame,
    simfin_delta: pl.DataFrame,
    stockanalysis_delta: pl.DataFrame,
    ticker_list: Sequence[str],
    quality_tickers: Sequence[str] | None = None,
    event_since: str,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    prospective = _merge_prospective_price_sources(
        paths=paths,
        yahoo_delta=yahoo_delta,
        simfin_delta=simfin_delta,
        stockanalysis_delta=stockanalysis_delta,
        ticker_list=ticker_list,
    )

    clean, lineage = _consolidate_price_sources(prospective, ticker_list=ticker_list)
    quality_tickers = [
        f"{str(ticker).upper().removesuffix('.US')}.US"
        for ticker in (quality_tickers if quality_tickers is not None else ticker_list)
    ]
    assert_no_extreme_adjusted_price_moves(
        clean,
        event_since=event_since,
        tickers=quality_tickers,
    )
    return prospective[0], prospective[1], prospective[2], clean, lineage


def _merge_prospective_price_sources(
    *,
    paths: OpenSourceLivePaths,
    yahoo_delta: pl.DataFrame,
    simfin_delta: pl.DataFrame,
    stockanalysis_delta: pl.DataFrame,
    ticker_list: Sequence[str],
) -> list[pl.DataFrame]:
    sources = (
        ("prices_yfinance.parquet", yahoo_delta),
        ("prices_simfin.parquet", simfin_delta),
        ("prices_stockanalysis.parquet", stockanalysis_delta),
    )
    prospective: list[pl.DataFrame] = []
    for file_name, delta in sources:
        path = paths.raw_dir / file_name
        existing = pl.read_parquet(path) if path.exists() else _empty_raw_price_frame()
        merged = merge_upsert_frames(
            existing,
            delta,
            key_cols=["ticker", "date", "source"],
            order_cols=["ingested_at"],
        )
        prospective.append(_canonicalize_price_tickers(merged, ticker_list=ticker_list))
    return prospective


def _load_retained_open_price_vintages(
    *,
    paths: OpenSourceLivePaths,
    prospective: Sequence[pl.DataFrame],
    active_tickers: Sequence[str],
    ticker_list: Sequence[str],
) -> pl.DataFrame:
    active = [
        f"{str(ticker).upper().removesuffix('.US')}.US" for ticker in active_tickers
    ]
    archived_paths = sorted(
        paths.runs_dir.glob("*/raw/prices_yfinance.parquet")
    )
    archived = (
        pl.concat(
            [pl.scan_parquet(path) for path in archived_paths],
            how="diagonal_relaxed",
        )
        .filter(~pl.col("ticker").cast(pl.String).str.to_uppercase().is_in(active))
        .collect()
        if archived_paths
        else _empty_raw_price_frame()
    )
    current = _concat_or_empty(
        list(prospective),
        empty=_empty_raw_price_frame(),
    ).filter(~pl.col("ticker").cast(pl.String).str.to_uppercase().is_in(active))
    return _canonicalize_price_tickers(
        _concat_or_empty([archived, current], empty=_empty_raw_price_frame()),
        ticker_list=ticker_list,
    )


def _consolidate_price_sources(
    frames: Sequence[pl.DataFrame],
    *,
    ticker_list: Sequence[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    combined = _concat_or_empty(list(frames), empty=_empty_raw_price_frame())
    if combined.is_empty():
        empty = _empty_raw_price_frame()
        return empty.select(list(PRICE_COLUMNS)), empty
    combined = _canonicalize_price_tickers(combined, ticker_list=ticker_list)
    prioritized = combined.with_columns(_price_source_priority_expr().alias("source_priority"))
    lineage = (
        prioritized.sort(
            ["ticker", "date", "source_priority", "ingested_at"],
            descending=[False, False, False, True],
        )
        .unique(subset=["ticker", "date"], keep="first", maintain_order=True)
        .drop("source_priority")
        .sort(["ticker", "date"])
    )
    clean = lineage.select(list(PRICE_COLUMNS)).sort(["ticker", "date"])
    return clean, lineage


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


def _price_source_priority_expr() -> pl.Expr:
    return (
        pl.when(pl.col("source") == "yfinance")
        .then(pl.lit(1))
        .when(pl.col("source") == "simfin")
        .then(pl.lit(2))
        .when(pl.col("source") == "stockanalysis")
        .then(pl.lit(3))
        .otherwise(pl.lit(99))
    )
