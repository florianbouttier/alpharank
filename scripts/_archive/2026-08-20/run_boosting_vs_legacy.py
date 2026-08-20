from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from run_backtest_application import (
    ApplicationBacktestResult,
    build_application_configs,
    build_spy_curve,
    compare_application_backtests,
    default_report_path,
    latest_run_dir,
    load_legacy_curves,
    run_application_backtests,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Edit this block, then run the script directly.
RUN_DIR: str | Path | None = None
OUTPUT_DIR: str | Path | None = None
REPORT_PATH: str | Path | None = None
REPORT_NAME = "boosting_vs_legacy_comparison.html"
REPORT_TITLE = "Boosting vs Legacy Comparison"
START_YEAR: int | None = None
END_YEAR: int | None = None
RISK_FREE_RATE = 0.02
INCLUDE_SPY = True

BOOSTING_SCENARIO_SPECS: list[dict[str, Any]] = [
    {"name": "boosting_top_n_5", "selection_mode": "top_n", "top_n": 5},
    {"name": "boosting_top_n_10", "selection_mode": "top_n", "top_n": 10},
    {"name": "boosting_top_n_20", "selection_mode": "top_n", "top_n": 20},
    {"name": "boosting_prediction_gt_0_40", "selection_mode": "prediction_threshold", "prediction_threshold": 0.40},
]

LEGACY_CURVE_SPECS: list[dict[str, str]] = [
    {"label": "legacy_combined_frequency", "checkpoint_name": "polars_combined_frequency.parquet"},
    {"label": "legacy_combined_equal", "checkpoint_name": "polars_combined_equal.parquet"},
    {"label": "legacy_optuna_11", "checkpoint_name": "polars_optuna_output_11_aggregated.parquet"},
    {"label": "legacy_optuna_12", "checkpoint_name": "polars_optuna_output_12_aggregated.parquet"},
    {"label": "legacy_optuna_21", "checkpoint_name": "polars_optuna_output_21_aggregated.parquet"},
    {"label": "legacy_optuna_22", "checkpoint_name": "polars_optuna_output_22_aggregated.parquet"},
]


def main() -> None:
    resolved_run_dir = Path(RUN_DIR).expanduser().resolve() if RUN_DIR else latest_run_dir(OUTPUT_DIR)
    report_path = (
        Path(REPORT_PATH).expanduser().resolve()
        if REPORT_PATH is not None
        else default_report_path(resolved_run_dir, REPORT_NAME)
    )

    boosting_configs = build_application_configs(BOOSTING_SCENARIO_SPECS)
    if not boosting_configs:
        raise ValueError("BOOSTING_SCENARIO_SPECS is empty. Add at least one boosting scenario before running.")

    boosting_results = run_application_backtests(
        resolved_run_dir,
        boosting_configs,
        risk_free_rate=RISK_FREE_RATE,
    )
    legacy_curves = load_legacy_curves(LEGACY_CURVE_SPECS)

    comparison_inputs: dict[str, ApplicationBacktestResult | pl.DataFrame | Any] = {
        **boosting_results,
        **legacy_curves,
    }
    if INCLUDE_SPY:
        comparison_inputs["SPY"] = build_spy_curve(boosting_results)

    comparison = compare_application_backtests(
        comparison_inputs,
        output_path=report_path,
        title=REPORT_TITLE,
        start_year=START_YEAR,
        end_year=END_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )

    print(f"Boosting run dir: {resolved_run_dir}")
    print(f"Report: {comparison.output_path}")
    print("Boosting scenarios:")
    for name, result in boosting_results.items():
        portfolio_kpis = result.kpis.filter(pl.col("strategy") == "Portfolio")
        total_return = portfolio_kpis.get_column("total_return").item() if not portfolio_kpis.is_empty() else 0.0
        sharpe = portfolio_kpis.get_column("sharpe_ratio").item() if not portfolio_kpis.is_empty() else 0.0
        print(
            f"- {name}: selected={result.selections.height} months={result.monthly_returns.height} "
            f"total_return={total_return:.4f} sharpe={sharpe:.4f}"
        )

    print("Legacy curves:")
    for spec in LEGACY_CURVE_SPECS:
        print(f"- {spec['label']}: {spec['checkpoint_name']}")

    if INCLUDE_SPY:
        print("- SPY: added from benchmark_return in boosting backtests")


if __name__ == "__main__":
    main()
