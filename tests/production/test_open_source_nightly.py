from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from scripts.open_source import nightly_ingestion
from scripts.open_source.ingestion.install_nightly_launchd import build_plist
from scripts.open_source.nightly_ingestion import LIVE_DIR, START_DATE, default_nightly_tickers, load_existing_live_tickers


def test_nightly_ingestion_defaults_are_defined() -> None:
    assert START_DATE == "2005-01-01"
    assert isinstance(LIVE_DIR, Path)


def test_launchd_plist_points_to_repo_python_script() -> None:
    plist = build_plist()
    program_arguments = plist["ProgramArguments"]
    assert isinstance(program_arguments, list)
    assert str(program_arguments[0]).endswith("/.venv/bin/python")
    assert str(program_arguments[1]).endswith("/scripts/open_source/ingestion/nightly_ingestion.py")
    env = plist["EnvironmentVariables"]
    assert env["HOME"]
    assert env["TMPDIR"] == "/tmp"


def test_load_existing_live_tickers_uses_existing_price_universe(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)

    pl.DataFrame({"ticker": ["OXY.US", "AAPL.US"]}).write_parquet(raw_dir / "prices_yfinance.parquet")

    assert load_existing_live_tickers(tmp_path) == ("AAPL", "OXY")


def test_default_nightly_tickers_preserves_existing_live_tickers_outside_current_sp500(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True)
    pl.DataFrame({"ticker": ["ZZZ.US"]}).write_parquet(raw_dir / "prices_yfinance.parquet")

    reference_data_dir = tmp_path / "reference"
    reference_data_dir.mkdir(parents=True)
    pl.DataFrame({"Date": ["2026-01-01", "2026-01-01"], "Ticker": ["AAPL", "MSFT"]}).write_csv(
        reference_data_dir / "SP500_Constituents.csv"
    )

    tickers = default_nightly_tickers(reference_data_dir=reference_data_dir, live_dir=tmp_path)
    assert tickers == ("AAPL", "MSFT", "ZZZ")


def test_failed_nightly_status_keeps_the_run_id(tmp_path: Path, monkeypatch) -> None:
    live_dir = tmp_path / "official"
    manifests_dir = live_dir / "manifests"
    lock_path = manifests_dir / "nightly.lock.json"
    status_path = manifests_dir / "nightly_status.json"
    captured: dict[str, object] = {}

    def fail_ingestion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("provider failure")

    monkeypatch.setattr(nightly_ingestion, "LIVE_DIR", live_dir)
    monkeypatch.setattr(nightly_ingestion, "REFERENCE_DATA_DIR", tmp_path / "reference")
    monkeypatch.setattr(nightly_ingestion, "LOCK_PATH", lock_path)
    monkeypatch.setattr(nightly_ingestion, "STATUS_PATH", status_path)
    monkeypatch.setattr(nightly_ingestion, "TICKERS", ("AAPL",))
    monkeypatch.setattr(nightly_ingestion, "new_run_id", lambda: "20260819_120000")
    monkeypatch.setattr(nightly_ingestion, "_load_latest_sp500_tickers", lambda path: ("AAPL",))
    monkeypatch.setattr(nightly_ingestion, "load_existing_live_tickers", lambda path: ())
    monkeypatch.setattr(nightly_ingestion, "run_open_source_ingestion", fail_ingestion)

    with pytest.raises(RuntimeError, match="provider failure"):
        nightly_ingestion.main()

    status = json.loads(status_path.read_text())
    assert captured["run_id"] == "20260819_120000"
    assert status["run_id"] == "20260819_120000"
    assert status["status"] == "failed"
    assert not lock_path.exists()
