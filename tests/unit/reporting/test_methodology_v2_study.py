from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts/research/render_methodology_v2_study.py"
SPEC = importlib.util.spec_from_file_location("render_methodology_v2_study", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _write_study_inputs(root: Path) -> tuple[Path, Path, Path]:
    common_dir = root / "common"
    reconciliation_dir = root / "reconciliation"
    boosting_dir = root / "boosting"
    for directory in (common_dir, reconciliation_dir, boosting_dir):
        directory.mkdir(parents=True)

    months = [date(2024, month, 1) for month in range(1, 13)]
    monthly_rows = []
    for strategy_index, strategy in enumerate(MODULE.STRATEGY_ORDER):
        for month_index, holding_month in enumerate(months):
            monthly_rows.append(
                {
                    "strategy": strategy,
                    "decision_month": date(2023, 12, 1)
                    if month_index == 0
                    else months[month_index - 1],
                    "holding_month": holding_month,
                    "gross_return": 0.012 + strategy_index * 0.001,
                    "net_return": 0.01 + strategy_index * 0.001,
                    "benchmark_return": 0.005,
                    "active_return": 0.005 + strategy_index * 0.001,
                    "turnover": 0.2,
                    "transaction_cost": 0.002,
                    "n_positions": 1,
                    "maximum_position_weight": 1.0,
                }
            )
    pl.DataFrame(monthly_rows).write_parquet(common_dir / "common_v2_monthly.parquet")

    holdings = pl.DataFrame(
        [
            {
                "strategy": strategy,
                "holding_month": holding_month,
                "ticker": f"TEST{strategy_index}.US",
                "target_weight": 1.0,
                "realized_return": 0.01,
                "selection_rank": 1,
            }
            for strategy_index, strategy in enumerate(MODULE.STRATEGY_ORDER[:-1])
            for holding_month in months
        ]
    )
    holdings.write_parquet(common_dir / "common_v2_holdings.parquet")
    pl.DataFrame(
        {
            "strategy": ["Legacy"],
            "holding_month": [months[-1]],
            "ticker": ["TEST2.US"],
            "terminal_event_type": ["cash_consideration"],
        }
    ).write_parquet(common_dir / "terminal_resolution_journal.parquet")
    pl.DataFrame(
        {
            "strategy": list(MODULE.STRATEGY_ORDER[:-1]),
            "terminal_holding_rows": [0, 0, 1],
            "terminal_annualized_log_contribution": [0.0, 0.0, 0.001],
            "terminal_marginal_cagr_impact": [0.0, 0.0, 0.001],
        }
    ).write_csv(common_dir / "terminal_cagr_attribution.csv")
    pl.DataFrame(
        {
            "strategy": list(MODULE.STRATEGY_ORDER[:-1]),
            "v1_cagr": [0.1, 0.1, 0.1],
            "v2_cagr": [0.11, 0.11, 0.11],
        }
    ).write_csv(reconciliation_dir / "metrics_reconciliation.csv")
    pl.DataFrame({"folds": [2], "test_rows": [24], "spearman_ic": [0.1]}).write_csv(
        boosting_dir / "model_horizon_summary.csv"
    )

    (common_dir / "manifest.json").write_text(
        json.dumps(
            {
                "comparison_eligible": True,
                "promotion_eligible": True,
                "composition_id": "synthetic-test-composition",
            }
        ),
        encoding="utf-8",
    )
    (reconciliation_dir / "manifest.json").write_text(
        json.dumps({"promotion_eligible": True}),
        encoding="utf-8",
    )
    (boosting_dir / "manifest.json").write_text("{}", encoding="utf-8")
    return common_dir, reconciliation_dir, boosting_dir


def test_drawdown_details_reports_recovery() -> None:
    details = MODULE._drawdown_details(np.asarray([0.10, -0.20, 0.30, 0.05]))

    assert details == {
        "drawdown_peak_index": 0,
        "drawdown_trough_index": 1,
        "drawdown_recovery_index": 2,
        "drawdown_duration_months": 2,
    }


def test_approved_study_builds_complete_report(tmp_path: Path) -> None:
    common_dir, reconciliation_dir, boosting_dir = _write_study_inputs(tmp_path)
    output_dir = tmp_path / "report"
    report = MODULE.build_report(
        common_dir,
        reconciliation_dir,
        boosting_dir,
        output_dir,
    )
    manifest = json.loads((output_dir / "html/methodology_v2_study_manifest.json").read_text())
    content = report.read_text()

    assert manifest["comparison_eligible"] is True
    assert manifest["promotion_eligible"] is True
    assert manifest["historical_scope_only"] is True
    assert manifest["live_portfolio_signal"] is False
    assert manifest["counts"] == {
        "strategies": 4,
        "months_per_strategy": 12,
        "monthly_rows": 48,
        "holding_rows": 36,
        "terminal_rows": 1,
    }
    assert "Positions à chaque mois" in content
    assert (output_dir / "html/downloads/performance_kpis.csv").is_file()
    assert (output_dir / "html/downloads/monthly_positions.csv").is_file()
