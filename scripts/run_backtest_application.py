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
LEGACY_CHECKPOINTS_DIR: str | Path = PROJECT_ROOT / "outputs" / "checkpoints"
LEGACY_AUTO_RUN_IF_MISSING = True
LEGACY_CURVE_SPECS: list[dict[str, str]] = [
    {"label": "legacy_combined_frequency", "checkpoint_name": "polars_combined_frequency.parquet"},
    {"label": "legacy_combined_equal", "checkpoint_name": "polars_combined_equal.parquet"},
    # Examples:
    # {"label": "legacy_optuna_11", "checkpoint_name": "polars_optuna_output_11_aggregated.parquet"},
    # {"label": "legacy_optuna_12", "checkpoint_name": "polars_optuna_output_12_aggregated.parquet"},
]

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


def legacy_checkpoint_path(
    checkpoint_name: str,
    checkpoints_dir: str | Path | None = None,
) -> Path:
    base_dir = (
        Path(checkpoints_dir).expanduser().resolve()
        if checkpoints_dir is not None
        else Path(LEGACY_CHECKPOINTS_DIR).expanduser().resolve()
    )
    return base_dir / checkpoint_name


def ensure_legacy_checkpoints(
    curve_specs: list[dict[str, str]] | None = None,
    *,
    checkpoints_dir: str | Path | None = None,
) -> dict[str, Path]:
    specs = LEGACY_CURVE_SPECS if curve_specs is None else curve_specs
    checkpoint_paths = {
        spec["label"]: legacy_checkpoint_path(spec["checkpoint_name"], checkpoints_dir=checkpoints_dir)
        for spec in specs
    }
    missing = {label: path for label, path in checkpoint_paths.items() if not path.exists()}
    if not missing:
        return checkpoint_paths

    if not LEGACY_AUTO_RUN_IF_MISSING:
        missing_display = ", ".join(f"{label} -> {path}" for label, path in missing.items())
        raise FileNotFoundError(f"Missing legacy checkpoints: {missing_display}")

    first_missing = next(iter(missing.values()))
    first_missing.parent.mkdir(parents=True, exist_ok=True)
    legacy_script = PROJECT_ROOT / "scripts" / "run_legacy.py"
    print(f"Legacy checkpoint missing. Running {legacy_script} to generate comparison artifacts...")
    subprocess.run(
        [sys.executable, str(legacy_script)],
        check=True,
        cwd=PROJECT_ROOT,
    )
    still_missing = {label: path for label, path in checkpoint_paths.items() if not path.exists()}
    if still_missing:
        missing_display = ", ".join(f"{label} -> {path}" for label, path in still_missing.items())
        raise FileNotFoundError(f"Legacy run finished but checkpoints are still missing: {missing_display}")
    return checkpoint_paths


def load_legacy_curve(
    checkpoint_name: str,
    *,
    checkpoints_dir: str | Path | None = None,
) -> pl.DataFrame:
    path = legacy_checkpoint_path(checkpoint_name, checkpoints_dir=checkpoints_dir)
    frame = pl.read_parquet(path)
    required_cols = {"year_month", "monthly_return"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Legacy checkpoint {path} is missing columns: {', '.join(sorted(missing))}")

    keep_cols = ["year_month", "monthly_return"]
    if "n" in frame.columns:
        keep_cols.append("n")
    return frame.select(keep_cols)


def load_legacy_curves(
    curve_specs: list[dict[str, str]] | None = None,
    *,
    checkpoints_dir: str | Path | None = None,
) -> dict[str, pl.DataFrame]:
    specs = LEGACY_CURVE_SPECS if curve_specs is None else curve_specs
    checkpoint_paths = ensure_legacy_checkpoints(specs, checkpoints_dir=checkpoints_dir)
    return {
        label: load_legacy_curve(path.name, checkpoints_dir=path.parent)
        for label, path in checkpoint_paths.items()
    }


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
        comparison_inputs.update(load_legacy_curves())

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
        for spec in LEGACY_CURVE_SPECS:
            checkpoint_path = legacy_checkpoint_path(spec["checkpoint_name"])
            print(f"- {spec['label']}: loaded from {checkpoint_path}")
    return comparison


if __name__ == "__main__":
    main()
