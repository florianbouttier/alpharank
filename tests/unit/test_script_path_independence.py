from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPRESENTATIVE_COMMANDS = (
    "scripts/build_legacy_common_replay.py",
    "scripts/experiments/modeling/analyze_ema_anchor_recomposition_gap.py",
    "scripts/open_source/publication/build_sec_metric_hybrid_package.py",
    "scripts/run_legacy.py",
)


def test_no_tracked_python_file_mutates_sys_path() -> None:
    completed = subprocess.run(
        ["git", "grep", "-nE", r"sys\.path\.(insert|append)", "--", "*.py"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 1, completed.stdout


def test_representative_commands_start_outside_repository(tmp_path) -> None:
    for relative_path in REPRESENTATIVE_COMMANDS:
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative_path), "--help"],
            cwd=tmp_path,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, f"{relative_path}: {completed.stderr}"


def test_compatibility_facade_resolves_to_the_implementation_module() -> None:
    module = importlib.import_module("scripts.open_source.nightly_ingestion")

    assert Path(module.__file__).resolve() == (
        ROOT / "scripts/open_source/ingestion/nightly_ingestion.py"
    ).resolve()
