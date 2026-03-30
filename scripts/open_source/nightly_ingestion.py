#!/usr/bin/env python3
from __future__ import annotations

from datetime import date
from datetime import datetime
import os
from pathlib import Path
import traceback

import polars as pl

from alpharank.data.open_source import run_open_source_ingestion
from alpharank.data.open_source.benchmark import load_sp500_tickers_for_year
from alpharank.data.open_source.storage import read_json, release_json_lock, try_acquire_json_lock, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these values directly in Python.
START_DATE = "2005-01-01"
LIVE_DIR = PROJECT_ROOT / "data" / "open_source" / "official"
REFERENCE_DATA_DIR = PROJECT_ROOT / "data"
AUDIT_YEARS = (2025,)
THRESHOLD_PCT = 0.5
PRICE_LOOKBACK_DAYS = 7
FINANCIAL_LOOKBACK_YEARS = 2
USER_AGENT = "Florian Bouttier florianbouttier@example.com"
TICKERS: tuple[str, ...] | None = None


RAW_TICKER_FILES = ("prices_yfinance.parquet",)

LOCK_PATH = LIVE_DIR / "manifests" / "nightly.lock.json"
STATUS_PATH = LIVE_DIR / "manifests" / "nightly_status.json"


def load_existing_live_tickers(live_dir: Path = LIVE_DIR) -> tuple[str, ...]:
    raw_dir = live_dir / "raw"
    tickers: set[str] = set()
    for file_name in RAW_TICKER_FILES:
        path = raw_dir / file_name
        if not path.exists():
            continue
        frame = pl.read_parquet(path, columns=["ticker"])
        tickers.update(
            ticker
            for ticker in (
                frame.select(pl.col("ticker").cast(pl.Utf8, strict=False).str.replace(r"\.US$", "")).to_series().to_list()
            )
            if ticker
        )
    if not tickers:
        output_path = live_dir.parent / "output" / "US_Finalprice.parquet"
        if output_path.exists():
            frame = pl.read_parquet(output_path, columns=["ticker"])
            tickers.update(
                ticker
                for ticker in (
                    frame.select(pl.col("ticker").cast(pl.Utf8, strict=False).str.replace(r"\.US$", "")).to_series().to_list()
                )
                if ticker
            )
    return tuple(sorted(tickers))


def default_nightly_tickers(
    *,
    reference_data_dir: Path = REFERENCE_DATA_DIR,
    live_dir: Path = LIVE_DIR,
) -> tuple[str, ...]:
    current_year = date.today().year
    current_sp500 = set(load_sp500_tickers_for_year(reference_data_dir, current_year))
    existing_live = set(load_existing_live_tickers(live_dir))
    return tuple(sorted(current_sp500 | existing_live))


def main() -> None:
    current_sp500 = load_sp500_tickers_for_year(REFERENCE_DATA_DIR, date.today().year)
    existing_live = load_existing_live_tickers(LIVE_DIR)
    active_tickers = TICKERS or default_nightly_tickers(reference_data_dir=REFERENCE_DATA_DIR, live_dir=LIVE_DIR)
    started_at = datetime.now().isoformat(timespec="seconds")
    lock_payload = {
        "pid": os.getpid(),
        "started_at": started_at,
        "script": str(Path(__file__).resolve()),
        "live_dir": str(LIVE_DIR),
    }
    acquired, existing_lock = try_acquire_json_lock(LOCK_PATH, lock_payload)
    if not acquired:
        current_status = read_json(STATUS_PATH)
        should_write_skip_status = not (
            isinstance(current_status, dict)
            and current_status.get("status") == "running"
            and isinstance(existing_lock, dict)
            and current_status.get("pid") == existing_lock.get("pid")
        )
        if should_write_skip_status:
            write_json(
                STATUS_PATH,
                {
                    "status": "skipped_locked",
                    "checked_at": started_at,
                    "lock_path": str(LOCK_PATH),
                    "existing_lock": existing_lock,
                },
            )
        print(f"[{started_at}] Nightly ingestion skipped: another run already holds {LOCK_PATH}")
        if isinstance(existing_lock, dict):
            print(f"Existing lock pid: {existing_lock.get('pid')}")
            print(f"Existing lock started_at: {existing_lock.get('started_at')}")
        return

    write_json(
        STATUS_PATH,
        {
            "status": "running",
            "started_at": started_at,
            "pid": os.getpid(),
            "lock_path": str(LOCK_PATH),
            "live_dir": str(LIVE_DIR),
        },
    )
    print(f"[{started_at}] Starting nightly ingestion")
    print(f"Universe tickers: {len(active_tickers)}")
    if TICKERS is None:
        print(f"Current S&P 500 tickers: {len(current_sp500)}")
        print(f"Already tracked live tickers: {len(existing_live)}")
        print("Nightly default universe = union(current S&P 500, existing live tickers)")
    try:
        result = run_open_source_ingestion(
            mode="daily",
            start_date=START_DATE,
            tickers=active_tickers,
            live_dir=LIVE_DIR,
            reference_data_dir=REFERENCE_DATA_DIR,
            audit_years=AUDIT_YEARS,
            threshold_pct=THRESHOLD_PCT,
            price_lookback_days=PRICE_LOOKBACK_DAYS,
            financial_lookback_years=FINANCIAL_LOOKBACK_YEARS,
            user_agent=USER_AGENT,
        )
    except Exception as exc:
        finished_at = datetime.now().isoformat(timespec="seconds")
        write_json(
            STATUS_PATH,
            {
                "status": "failed",
                "started_at": started_at,
                "finished_at": finished_at,
                "pid": os.getpid(),
                "lock_path": str(LOCK_PATH),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise
    finally:
        release_json_lock(LOCK_PATH)

    finished_at = datetime.now().isoformat(timespec="seconds")
    write_json(
        STATUS_PATH,
        {
            "status": "success",
            "started_at": started_at,
            "finished_at": finished_at,
            "pid": os.getpid(),
            "run_id": result.run_id,
            "official_dir": str(result.live_dir),
            "target_dir": str(result.target_dir),
            "output_dir": str(result.output_dir),
            "output_lineage_dir": str(result.output_lineage_dir),
            "output_snapshot_dir": (str(result.output_snapshot_dir) if result.output_snapshot_dir is not None else None),
            "ticker_count": result.ticker_count,
            "price_window": {
                "start_date": result.price_start_date,
                "end_date": result.price_end_date,
            },
            "financial_years_refreshed": list(result.refreshed_years),
            "audit_dirs": [str(path) for path in result.audit_dirs],
        },
    )
    print(f"Nightly ingestion completed: {result.run_id}")
    print(f"Official dir: {result.live_dir}")
    print(f"Target dir: {result.target_dir}")
    print(f"Output dir: {result.output_dir}")
    print(f"Output lineage dir: {result.output_lineage_dir}")
    if result.output_snapshot_dir is not None:
        print(f"Output snapshot dir: {result.output_snapshot_dir}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Price window: {result.price_start_date} -> {result.price_end_date}")
    print(f"Financial years refreshed: {', '.join(str(year) for year in result.refreshed_years)}")
    if result.audit_dirs:
        print("Audit dirs:")
        for audit_dir in result.audit_dirs:
            print(f"  - {audit_dir}")


if __name__ == "__main__":
    main()
