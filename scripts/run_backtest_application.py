from __future__ import annotations

from pathlib import Path
import subprocess
import sys
from typing import Any

import polars as pl

from alpharank.backtest.application import (
    ApplicationBacktestConfig,
    ApplicationBacktestResult,
    BacktestComparisonResult,
    compare_backtest_curves,
    run_application_backtest,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# Edit this block, then run the script directly.
RUN_DIR: str | Path | None = None
OUTPUT_DIR: str | Path | None = None
REPORT_PATH: str | Path | None = None
REPORT_NAME = "application_backtest_comparison.html"
REPORT_TITLE = "Application Backtest Comparison"
START_YEAR: int | None = None
END_YEAR: int | None = None
RISK_FREE_RATE = 0.02
INCLUDE_SPY = True
INCLUDE_LEGACY = True
LEGACY_LABEL = "legacy_combined_frequency"
LEGACY_CHECKPOINTS_DIR: str | Path = PROJECT_ROOT / "outputs" / "checkpoints"
LEGACY_CHECKPOINT_NAME = "polars_combined_frequency.parquet"
LEGACY_AUTO_RUN_IF_MISSING = True

SCENARIO_SPECS: list[dict[str, Any]] = [
    {"name": "top_n_5", "selection_mode": "top_n", "top_n": 5},
    {"name": "top_n_10", "selection_mode": "top_n", "top_n": 10},
    {"name": "top_n_20", "selection_mode": "top_n", "top_n": 20},
    {"name": "prediction_gt_0_40", "selection_mode": "prediction_threshold", "prediction_threshold": 0.40},
    # Example:
    # {"name": "prediction_gt_0_35_stale_1m", "selection_mode": "prediction_threshold", "prediction_threshold": 0.35, "max_price_staleness_months": 1},
]


def latest_run_dir(output_dir: str | Path | None = None) -> Path:
    active_output_dir = Path(output_dir) if output_dir is not None else PROJECT_ROOT / "outputs"
    run_dirs = sorted(
        [path for path in active_output_dir.glob("xgboost_timefold_backtest_*") if path.is_dir()],
        reverse=True,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No backtest run found under {active_output_dir}")
    return run_dirs[0]


def load_predictions(run_dir: str | Path) -> pl.DataFrame:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    path = resolved_run_dir / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {path}")
    return pl.read_parquet(path)


def build_application_configs(scenario_specs: list[dict[str, Any]] | None = None) -> list[ApplicationBacktestConfig]:
    specs = SCENARIO_SPECS if scenario_specs is None else scenario_specs
    return [ApplicationBacktestConfig(**spec) for spec in specs]


def run_application_backtests(
    run_dir: str | Path,
    configs: list[ApplicationBacktestConfig],
    *,
    risk_free_rate: float = 0.02,
) -> dict[str, ApplicationBacktestResult]:
    predictions = load_predictions(run_dir)
    return {
        config.name: run_application_backtest(predictions, config, risk_free_rate=risk_free_rate)
        for config in configs
    }


def build_spy_curve(backtests: dict[str, ApplicationBacktestResult]) -> pl.DataFrame:
    if not backtests:
        raise ValueError("Cannot build SPY curve from empty backtests.")

    first_result = next(iter(backtests.values()))
    monthly_returns = first_result.monthly_returns
    required_cols = {"year_month", "benchmark_return"}
    missing = required_cols - set(monthly_returns.columns)
    if missing:
        raise ValueError("Missing columns for SPY curve: " + ", ".join(sorted(missing)))

    return monthly_returns.select(
        [
            "year_month",
            pl.col("benchmark_return").alias("monthly_return"),
        ]
    )


def legacy_checkpoint_path(checkpoints_dir: str | Path | None = None) -> Path:
    base_dir = Path(checkpoints_dir).expanduser().resolve() if checkpoints_dir else Path(LEGACY_CHECKPOINTS_DIR).expanduser().resolve()
    return base_dir / LEGACY_CHECKPOINT_NAME


def ensure_legacy_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    resolved_checkpoint_path = (
        Path(checkpoint_path).expanduser().resolve()
        if checkpoint_path is not None
        else legacy_checkpoint_path()
    )
    if resolved_checkpoint_path.exists():
        return resolved_checkpoint_path

    if not LEGACY_AUTO_RUN_IF_MISSING:
        raise FileNotFoundError(f"Missing legacy checkpoint: {resolved_checkpoint_path}")

    resolved_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_script = PROJECT_ROOT / "scripts" / "run_legacy.py"
    print(f"Legacy checkpoint missing. Running {legacy_script} to generate comparison artifacts...")
    subprocess.run(
        [sys.executable, str(legacy_script)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    if not resolved_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Legacy run finished but checkpoint was not created: {resolved_checkpoint_path}"
        )
    return resolved_checkpoint_path


def load_legacy_curve(checkpoint_path: str | Path | None = None) -> pl.DataFrame:
    path = ensure_legacy_checkpoint(checkpoint_path)
    frame = pl.read_parquet(path)
    required_cols = {"year_month", "monthly_return"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Legacy checkpoint {path} is missing columns: {', '.join(sorted(missing))}")

    keep_cols = ["year_month", "monthly_return"]
    if "n" in frame.columns:
        keep_cols.append("n")
    return frame.select(keep_cols)


def compare_application_backtests(
    backtests: dict[str, ApplicationBacktestResult | pl.DataFrame | Any],
    *,
    output_path: str | Path | None = None,
    title: str = "Backtest Application Comparison",
    start_year: int | None = None,
    end_year: int | None = None,
    risk_free_rate: float = 0.02,
    return_column: str | None = None,
) -> BacktestComparisonResult:
    return compare_backtest_curves(
        backtests,
        output_path=output_path,
        title=title,
        start_year=start_year,
        end_year=end_year,
        risk_free_rate=risk_free_rate,
        return_column=return_column,
    )


def default_report_path(run_dir: str | Path, report_name: str = "application_backtest_comparison.html") -> Path:
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    return resolved_run_dir / report_name


def main() -> BacktestComparisonResult:
    resolved_run_dir = Path(RUN_DIR).expanduser().resolve() if RUN_DIR else latest_run_dir(OUTPUT_DIR)
    report_path = (
        Path(REPORT_PATH).expanduser().resolve()
        if REPORT_PATH is not None
        else default_report_path(resolved_run_dir, REPORT_NAME)
    )

    configs = build_application_configs()
    if not configs:
        raise ValueError("SCENARIO_SPECS is empty. Add at least one scenario before running the script.")
    results = run_application_backtests(
        resolved_run_dir,
        configs,
        risk_free_rate=RISK_FREE_RATE,
    )

    comparison_inputs: dict[str, ApplicationBacktestResult | pl.DataFrame | Any] = dict(results)
    if INCLUDE_SPY:
        comparison_inputs["SPY"] = build_spy_curve(results)
    if INCLUDE_LEGACY:
        comparison_inputs[LEGACY_LABEL] = load_legacy_curve()

    comparison = compare_application_backtests(
        comparison_inputs,
        output_path=report_path,
        title=REPORT_TITLE,
        start_year=START_YEAR,
        end_year=END_YEAR,
        risk_free_rate=RISK_FREE_RATE,
    )

    print(f"Run dir: {resolved_run_dir}")
    print(f"Report: {comparison.output_path}")
    print(f"Scenarios: {len(configs)}")
    for name, result in results.items():
        portfolio_kpis = result.kpis.filter(pl.col("strategy") == "Portfolio")
        total_return = portfolio_kpis.get_column("total_return").item() if not portfolio_kpis.is_empty() else 0.0
        sharpe = portfolio_kpis.get_column("sharpe_ratio").item() if not portfolio_kpis.is_empty() else 0.0
        print(
            f"- {name}: selected={result.selections.height} months={result.monthly_returns.height} "
            f"total_return={total_return:.4f} sharpe={sharpe:.4f}"
        )
    if INCLUDE_SPY:
        print("- SPY: added from benchmark_return in application backtests")
    if INCLUDE_LEGACY:
        print(f"- {LEGACY_LABEL}: loaded from {legacy_checkpoint_path()}")
    return comparison


if __name__ == "__main__":
    main()
