from __future__ import annotations

from pathlib import Path

import polars as pl

from scripts.open_source.install_nightly_launchd import build_plist
from scripts.open_source.nightly_ingestion import LIVE_DIR, START_DATE, default_nightly_tickers, load_existing_live_tickers


def test_nightly_ingestion_defaults_are_defined() -> None:
    assert START_DATE == "2005-01-01"
    assert isinstance(LIVE_DIR, Path)


def test_launchd_plist_points_to_repo_python_script() -> None:
    plist = build_plist()
    program_arguments = plist["ProgramArguments"]
    assert isinstance(program_arguments, list)
    assert str(program_arguments[0]).endswith("/.venv/bin/python")
    assert str(program_arguments[1]).endswith("/scripts/open_source/nightly_ingestion.py")
    env = plist["EnvironmentVariables"]
    assert env["HOME"]
    assert env["TMPDIR"] == "/tmp"


def test_load_existing_live_tickers_collects_union_from_raw_tables(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    pl.DataFrame({"ticker": ["AAPL.US", "MSFT.US"]}).write_parquet(raw_dir / "general_reference.parquet")
    pl.DataFrame({"ticker": ["OXY.US", "AAPL.US"]}).write_parquet(raw_dir / "prices_yfinance.parquet")

    assert load_existing_live_tickers(tmp_path) == ("AAPL", "MSFT", "OXY")


def test_default_nightly_tickers_preserves_existing_live_tickers_outside_current_sp500(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    pl.DataFrame({"ticker": ["ZZZ.US"]}).write_parquet(raw_dir / "general_reference.parquet")

    reference_data_dir = tmp_path / "reference"
    reference_data_dir.mkdir(parents=True)
    pl.DataFrame({"Date": ["2026-01-01", "2026-01-01"], "Ticker": ["AAPL", "MSFT"]}).write_csv(
        reference_data_dir / "SP500_Constituents.csv"
    )

    tickers = default_nightly_tickers(reference_data_dir=reference_data_dir, live_dir=tmp_path)
    assert tickers == ("AAPL", "MSFT", "ZZZ")
