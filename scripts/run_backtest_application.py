from __future__ import annotations

import argparse
from pathlib import Path
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


def default_application_configs() -> list[ApplicationBacktestConfig]:
    return [
        ApplicationBacktestConfig(
            name="top_n_10",
            selection_mode="top_n",
            top_n=10,
        ),
        ApplicationBacktestConfig(
            name="prediction_gt_0_60",
            selection_mode="prediction_threshold",
            prediction_threshold=0.60,
        ),
    ]


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run dedicated application-level backtest comparisons.")
    parser.add_argument("--run-dir", type=str, default=None, help="Backtest run directory containing predictions.parquet.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory containing xgboost_timefold_backtest_* runs.")
    parser.add_argument("--report-path", type=str, default=None, help="Optional explicit HTML output path.")
    parser.add_argument("--start-year", type=int, default=None, help="Optional comparison start year.")
    parser.add_argument("--end-year", type=int, default=None, help="Optional comparison end year.")
    parser.add_argument("--risk-free-rate", type=float, default=0.02, help="Risk-free rate used for KPI comparison.")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N policy size.")
    parser.add_argument(
        "--prediction-threshold",
        type=float,
        default=0.60,
        help="Prediction threshold policy cutoff.",
    )
    parser.add_argument(
        "--max-price-staleness-months",
        type=int,
        default=None,
        help="Optional max staleness in months between decision_month and decision_asof_date.",
    )
    return parser.parse_args()


def _cli_configs(args: argparse.Namespace) -> list[ApplicationBacktestConfig]:
    suffix = (
        f"_stale_le_{args.max_price_staleness_months}m"
        if args.max_price_staleness_months is not None
        else ""
    )
    return [
        ApplicationBacktestConfig(
            name=f"top_n_{args.top_n}{suffix}",
            selection_mode="top_n",
            top_n=args.top_n,
            max_price_staleness_months=args.max_price_staleness_months,
        ),
        ApplicationBacktestConfig(
            name=f"prediction_gt_{str(args.prediction_threshold).replace('.', '_')}{suffix}",
            selection_mode="prediction_threshold",
            prediction_threshold=args.prediction_threshold,
            max_price_staleness_months=args.max_price_staleness_months,
        ),
    ]


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve() if args.run_dir else latest_run_dir(args.output_dir)
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path is not None
        else default_report_path(run_dir)
    )

    configs = _cli_configs(args)
    results = run_application_backtests(
        run_dir,
        configs,
        risk_free_rate=args.risk_free_rate,
    )
    comparison = compare_application_backtests(
        results,
        output_path=report_path,
        title="Application Backtest Comparison",
        start_year=args.start_year,
        end_year=args.end_year,
        risk_free_rate=args.risk_free_rate,
    )

    print(f"Run dir: {run_dir}")
    print(f"Report: {comparison.output_path}")
    print("Backtests:")
    for name, result in results.items():
        portfolio_kpis = result.kpis.filter(pl.col("strategy") == "Portfolio")
        total_return = portfolio_kpis.get_column("total_return").item() if not portfolio_kpis.is_empty() else 0.0
        sharpe = portfolio_kpis.get_column("sharpe_ratio").item() if not portfolio_kpis.is_empty() else 0.0
        print(
            f"  {name}: eligible={result.eligible_predictions.height} "
            f"selected={result.selections.height} months={result.monthly_returns.height} "
            f"total_return={total_return:.4f} sharpe={sharpe:.4f}"
        )


if __name__ == "__main__":
    main()
