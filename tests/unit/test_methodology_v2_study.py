from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = PROJECT_ROOT / "scripts/render_methodology_v2_study.py"
SPEC = importlib.util.spec_from_file_location("render_methodology_v2_study", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_drawdown_details_reports_recovery() -> None:
    details = MODULE._drawdown_details(np.asarray([0.10, -0.20, 0.30, 0.05]))

    assert details == {
        "drawdown_peak_index": 0,
        "drawdown_trough_index": 1,
        "drawdown_recovery_index": 2,
        "drawdown_duration_months": 2,
    }


def test_approved_study_builds_complete_report(tmp_path: Path) -> None:
    report = MODULE.build_report(
        MODULE.DEFAULT_COMMON,
        MODULE.DEFAULT_RECONCILIATION,
        MODULE.DEFAULT_BOOSTING,
        tmp_path,
    )
    manifest = json.loads((tmp_path / "html/methodology_v2_study_manifest.json").read_text())
    content = report.read_text()

    assert manifest["comparison_eligible"] is True
    assert manifest["promotion_eligible"] is True
    assert manifest["historical_scope_only"] is True
    assert manifest["live_portfolio_signal"] is False
    assert manifest["counts"] == {
        "strategies": 4,
        "months_per_strategy": 178,
        "monthly_rows": 712,
        "holding_rows": 6305,
        "terminal_rows": 7,
    }
    assert "Positions à chaque mois" in content
    assert "720 / 720 lignes mensuelles divergentes expliquées" in content
    assert (tmp_path / "html/downloads/performance_kpis.csv").is_file()
    assert (tmp_path / "html/downloads/monthly_positions.csv").is_file()
