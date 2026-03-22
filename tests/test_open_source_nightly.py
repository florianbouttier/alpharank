from __future__ import annotations

from pathlib import Path

from scripts.open_source.install_nightly_launchd import build_plist
from scripts.open_source.nightly_ingestion import LIVE_DIR, START_DATE


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
