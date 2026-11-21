from __future__ import annotations

import json
import shutil
import subprocess

import pytest

from alpharank.reporting._performance_report_chart_script import PERFORMANCE_REPORT_CHART_SCRIPT
from alpharank.reporting._performance_report_composer_script import (
    PERFORMANCE_REPORT_COMPOSER_SCRIPT,
)
from alpharank.reporting._performance_report_script import PERFORMANCE_REPORT_SCRIPT
from alpharank.reporting._performance_report_studio_script import PERFORMANCE_REPORT_STUDIO_SCRIPT
from alpharank.reporting.analysis_ui import load_analysis_ui_assets


def _project(*, mode: str, view: str, missing_benchmark: bool = False) -> dict:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node is required for the shared frontend projection contract")
    scripts = "\n".join(
        [
            PERFORMANCE_REPORT_COMPOSER_SCRIPT,
            PERFORMANCE_REPORT_STUDIO_SCRIPT,
            PERFORMANCE_REPORT_CHART_SCRIPT,
            PERFORMANCE_REPORT_SCRIPT.rsplit("boot();", 1)[0],
        ]
    )
    payload = {
        "strategy_order": ["Alpha", "SPY · Total return"],
        "strategies": [
            {"label": "Alpha", "color": "#2563eb"},
            {"label": "SPY · Total return", "color": "#64748b"},
        ],
        "monthly": [
            {"strategy": strategy, "holding_month": month, "net_return": value}
            for strategy, values in [
                ("Alpha", [99.0, 0.5, -0.1, 99.0]),
                ("SPY · Total return", [0.9, 0.1, 0.05, 0.9]),
            ]
            for month, value in zip(
                ["2025-12-01", "2026-01-01", "2026-02-01", "2026-03-01"], values, strict=True
            )
            if not (
                missing_benchmark and strategy == "SPY · Total return" and month == "2026-02-01"
            )
        ],
        "portfolio_composer": {
            "strategy_order": ["Alpha"],
            "combination_masks": [1],
            "months": ["2026-01-01", "2026-02-01"],
            "monthly_returns": [[0.5], [-0.1]],
        },
    }
    program = """
const fs=require('fs'), vm=require('vm');
const input=JSON.parse(fs.readFileSync(0,'utf8'));
const context=vm.createContext({console});
vm.runInContext(input.scripts,context);
vm.runInContext('state.data='+JSON.stringify(input.payload),context);
vm.runInContext('state.start="2026-01-01";state.end="2026-02-01";state.curves=["Alpha","SPY · Total return"];state.composerStrategies=["Alpha"]',context);
vm.runInContext('state.analysisMode='+JSON.stringify(input.mode)+';state.chartView='+JSON.stringify(input.view),context);
try { process.stdout.write(JSON.stringify({series:vm.runInContext('studioChartSeries()',context)})); }
catch(error) { process.stdout.write(JSON.stringify({error:error.message})); }
"""
    result = subprocess.run(
        [node, "-e", program],
        input=json.dumps({"scripts": scripts, "payload": payload, "mode": mode, "view": view}),
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout)


@pytest.mark.parametrize("mode", ["strategies", "composer"])
@pytest.mark.parametrize("view", ["wealth", "relative", "drawdown"])
def test_monthly_chart_starts_before_first_return_without_erasing_it(mode: str, view: str) -> None:
    result = _project(mode=mode, view=view)
    points = result["series"][0]["values"]

    assert len(points) == 3
    assert points[0]["date"] == "2025-12-31"
    assert points[0]["value"] == (0.0 if view == "drawdown" else 1.0)
    assert points[1]["date"] == "2026-01-31"
    expected_first = 0.0 if view == "drawdown" else 1.5 / 1.1 if view == "relative" else 1.5
    expected_last = -0.1 if view == "drawdown" else 1.35 / 1.155 if view == "relative" else 1.35
    assert points[1]["value"] == pytest.approx(expected_first, abs=1e-12)
    assert points[-1]["value"] == pytest.approx(expected_last, abs=1e-12)
    assert "initial" in points[0]["label"]


def test_relative_chart_rejects_missing_benchmark_instead_of_drawing_nan() -> None:
    result = _project(mode="strategies", view="relative", missing_benchmark=True)
    assert "Mois SPY absent" in result["error"]


def test_reusable_chart_adapter_is_offline_and_contains_no_custom_canvas_engine() -> None:
    assets = load_analysis_ui_assets()
    assert "Plotly.react" in assets.script
    assert "renderTimeSeries" in assets.script and "renderFrontier" in assets.script
    assert "getContext" not in assets.script
    assert "fetch(" not in assets.script
    assert "XMLHttpRequest" not in assets.script
    assert assets.runtime_version
    assert "plotly.js" in assets.runtime.lower()
    assert "Permission is hereby granted" in assets.runtime
