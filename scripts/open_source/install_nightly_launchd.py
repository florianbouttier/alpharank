#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import os
import plistlib
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABEL = "com.florianbouttier.alpharank.open_source_ingestion"
HOUR = 2
MINUTE = 15
PYTHON_BIN = PROJECT_ROOT / ".venv" / "bin" / "python"
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "open_source" / "nightly_ingestion.py"
LOG_DIR = PROJECT_ROOT / "logs" / "open_source_ingestion"
PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / f"{LABEL}.plist"


def build_plist() -> dict[str, object]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return {
        "Label": LABEL,
        "ProgramArguments": [str(PYTHON_BIN), str(SCRIPT_PATH)],
        "WorkingDirectory": str(PROJECT_ROOT),
        "EnvironmentVariables": {
            "HOME": str(Path.home()),
            "PATH": f"{PROJECT_ROOT / '.venv' / 'bin'}:/usr/bin:/bin:/usr/sbin:/sbin",
            "TMPDIR": "/tmp",
        },
        "RunAtLoad": False,
        "StartCalendarInterval": {
            "Hour": HOUR,
            "Minute": MINUTE,
        },
        "StandardOutPath": str(LOG_DIR / "stdout.log"),
        "StandardErrorPath": str(LOG_DIR / "stderr.log"),
    }


def install() -> Path:
    domain = f"gui/{os.getuid()}"
    PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PLIST_PATH.open("wb") as handle:
        plistlib.dump(build_plist(), handle)

    bootout = subprocess.run(
        ["launchctl", "bootout", domain, str(PLIST_PATH)],
        check=False,
        capture_output=True,
        text=True,
    )
    if bootout.returncode not in {0, 5}:
        print(bootout.stderr.strip() or bootout.stdout.strip())
    subprocess.run(["launchctl", "bootstrap", domain, str(PLIST_PATH)], check=True)
    subprocess.run(["launchctl", "enable", f"{domain}/{LABEL}"], check=True)
    return PLIST_PATH


def main() -> None:
    plist_path = install()
    print(f"Installed launchd agent: {plist_path}")
    print(f"Nightly schedule: {HOUR:02d}:{MINUTE:02d}")
    print(f"Python: {PYTHON_BIN}")
    print(f"Script: {SCRIPT_PATH}")
    print(f"Logs: {LOG_DIR}")


if __name__ == "__main__":
    main()
