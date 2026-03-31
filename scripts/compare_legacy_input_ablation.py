#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "outputs" / ".mplconfig"))

from alpharank.strategy.legacy import StrategyLearner
from run_legacy import run_pipeline


DEFAULT_FIRST_DATE = "2025-01"


@dataclass(frozen=True)
class ScenarioResult:
    scenario: str
    duration_seconds: float
    run_day_dir: str
    combined_frequency_total_return: str | None
    combined_equal_total_return: str | None
    combined_frequency_holdings: int
    combined_equal_holdings: int
    latest_selection_tickers: int
    latest_selection_month: str
    frequency_latest_tickers: list[str]
    equal_latest_tickers: list[str]


def _scenario_mapping() -> dict[str, dict[str, str]]:
    all_names = {
        "SP500Price.parquet": "eodhd",
        "SP500_Constituents.csv": "eodhd",
        "US_Balance_sheet.parquet": "eodhd",
        "US_Cash_flow.parquet": "eodhd",
        "US_Earnings.parquet": "eodhd",
        "US_Finalprice.parquet": "eodhd",
        "US_General.parquet": "eodhd",
        "US_Income_statement.parquet": "eodhd",
    }
    return {
        "eodhd_baseline": dict(all_names),
        "swap_earnings_only": {**all_names, "US_Earnings.parquet": "open"},
        "swap_sector_only": {**all_names, "US_General.parquet": "open"},
        "swap_price_only": {**all_names, "US_Finalprice.parquet": "open", "SP500Price.parquet": "open"},
        "swap_fundamentals_only": {
            **all_names,
            "US_Income_statement.parquet": "open",
            "US_Balance_sheet.parquet": "open",
            "US_Cash_flow.parquet": "open",
            "US_Earnings.parquet": "open",
        },
        "open_source_full": {name: "open" for name in all_names},
    }


def _latest_compare_dir() -> Path:
    candidates = sorted(
        PROJECT_ROOT.glob("outputs/legacy_db_compare_*"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError("No `outputs/legacy_db_compare_*` directory found.")
    return candidates[0]


def _source_file(compare_dir: Path, source_key: str, file_name: str) -> Path:
    base = compare_dir / "aligned_data" / ("open_source" if source_key == "open" else "eodhd")
    return base / file_name


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def _prepare_dataset_dir(compare_dir: Path, root: Path, mapping: dict[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for file_name, source_key in mapping.items():
        _link_or_copy(_source_file(compare_dir, source_key, file_name), root / file_name)
    return root


def _read_total_return(metrics_path: Path, model: str) -> str | None:
    metrics = pl.read_parquet(metrics_path)
    row = metrics.filter(pl.col("model") == model).head(1)
    if row.is_empty() or "Total Return" not in row.columns:
        return None
    value = row.select("Total Return").item()
    return None if value is None else str(value)


def _latest_tickers(frame: pl.DataFrame) -> tuple[str, list[str]]:
    if frame.is_empty():
        return "n/a", []
    latest_month = frame.select(pl.col("year_month").max()).item()
    latest = frame.filter(pl.col("year_month") == latest_month).sort("ticker")
    return str(latest_month), latest.select("ticker").to_series().to_list()


def _run_scenario(name: str, data_dir: Path, output_root: Path, *, n_trials: int, first_date: str) -> ScenarioResult:
    start = perf_counter()
    result = run_pipeline(
        n_trials=n_trials,
        n_jobs=1,
        first_date=first_date,
        data_dir=data_dir,
        output_dir=output_root / "runs" / name,
        checkpoints_dir=output_root / "checkpoints" / name,
    )
    duration = perf_counter() - start
    run_day_dir = Path(result.artifacts["metrics"]).parent

    portfolio_frequency = StrategyLearner.get_portfolio_at_month(result.combined_frequency)
    portfolio_equal = StrategyLearner.get_portfolio_at_month(result.combined_equal)
    latest_selection_month, latest_selection_tickers = _latest_tickers(result.stocks_selections)

    return ScenarioResult(
        scenario=name,
        duration_seconds=duration,
        run_day_dir=str(run_day_dir),
        combined_frequency_total_return=_read_total_return(Path(result.artifacts["metrics"]), "Combined_Frequency"),
        combined_equal_total_return=_read_total_return(Path(result.artifacts["metrics"]), "Combined_Equal"),
        combined_frequency_holdings=len(portfolio_frequency),
        combined_equal_holdings=len(portfolio_equal),
        latest_selection_tickers=len(latest_selection_tickers),
        latest_selection_month=latest_selection_month,
        frequency_latest_tickers=sorted(portfolio_frequency["ticker"].tolist()),
        equal_latest_tickers=sorted(portfolio_equal["ticker"].tolist()),
    )


def _jaccard(left: list[str], right: list[str]) -> float | None:
    lset = set(left)
    rset = set(right)
    union = lset | rset
    if not union:
        return None
    return len(lset & rset) / len(union)


def _report_table(results: list[ScenarioResult]) -> pl.DataFrame:
    baseline = next(result for result in results if result.scenario == "eodhd_baseline")
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "scenario": result.scenario,
                "combined_frequency_total_return": result.combined_frequency_total_return,
                "combined_equal_total_return": result.combined_equal_total_return,
                "combined_frequency_holdings": result.combined_frequency_holdings,
                "combined_equal_holdings": result.combined_equal_holdings,
                "latest_selection_tickers": result.latest_selection_tickers,
                "latest_selection_month": result.latest_selection_month,
                "frequency_jaccard_vs_eodhd": _jaccard(result.frequency_latest_tickers, baseline.frequency_latest_tickers),
                "equal_jaccard_vs_eodhd": _jaccard(result.equal_latest_tickers, baseline.equal_latest_tickers),
                "duration_min": round(result.duration_seconds / 60.0, 2),
            }
        )
    return pl.DataFrame(rows)


def _portfolio_diff_table(results: list[ScenarioResult]) -> pl.DataFrame:
    baseline = next(result for result in results if result.scenario == "eodhd_baseline")
    rows: list[dict[str, Any]] = []
    for result in results:
        baseline_only = sorted(set(baseline.frequency_latest_tickers) - set(result.frequency_latest_tickers))
        scenario_only = sorted(set(result.frequency_latest_tickers) - set(baseline.frequency_latest_tickers))
        rows.append(
            {
                "scenario": result.scenario,
                "baseline_only_frequency": ", ".join(baseline_only[:20]),
                "scenario_only_frequency": ", ".join(scenario_only[:20]),
            }
        )
    return pl.DataFrame(rows)


def _write_report(output_root: Path, summary_frame: pl.DataFrame, portfolio_diff: pl.DataFrame) -> None:
    report_path = output_root / "report.md"
    report = "\n".join(
        [
            "# Legacy Input Ablation",
            "",
            f"Generated at: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "This isolates source families on top of the same aligned cutoff and ticker universe used by the main legacy compare.",
            "",
            "## Scenario Summary",
            "",
            summary_frame.write_csv(file=None),
            "",
            "## Final Frequency Portfolio Diffs",
            "",
            portfolio_diff.write_csv(file=None),
            "",
            "## Interpretation",
            "",
            "- `swap_earnings_only`: swaps only `US_Earnings.parquet`.",
            "- `swap_sector_only`: swaps only `US_General.parquet`.",
            "- `swap_price_only`: swaps only `US_Finalprice.parquet` and `SP500Price.parquet`.",
            "- `swap_fundamentals_only`: swaps `US_Income_statement`, `US_Balance_sheet`, `US_Cash_flow`, and `US_Earnings`.",
            "- `open_source_full`: all open-source files.",
            "",
            "Read `frequency_jaccard_vs_eodhd` first. A scenario close to `1.0` behaves like EODHD; close to `0.0` it diverges strongly.",
            "",
        ]
    )
    report_path.write_text(report, encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run legacy source-family ablation on an aligned compare directory.")
    parser.add_argument(
        "--compare-dir",
        type=Path,
        default=None,
        help="Path to a `legacy_db_compare_*` directory. Defaults to the latest one under `outputs/`.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Directory where ablation outputs should be written. Defaults to `outputs/legacy_input_ablation_<timestamp>`.",
    )
    parser.add_argument(
        "--first-date",
        default=DEFAULT_FIRST_DATE,
        help="First YYYY-MM month passed to `run_legacy.py`. Default: 2025-01.",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=30,
        help="Optuna trials passed to each scenario run. Default: 30.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    compare_dir = args.compare_dir.resolve() if args.compare_dir else _latest_compare_dir()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() if args.output_root else (PROJECT_ROOT / "outputs" / f"legacy_input_ablation_{timestamp}")
    datasets_root = output_root / "datasets"
    results: list[ScenarioResult] = []
    for scenario, mapping in _scenario_mapping().items():
        data_dir = _prepare_dataset_dir(compare_dir, datasets_root / scenario, mapping)
        results.append(_run_scenario(scenario, data_dir, output_root, n_trials=args.n_trials, first_date=args.first_date))

    summary_frame = _report_table(results)
    portfolio_diff = _portfolio_diff_table(results)
    summary_frame.write_parquet(output_root / "scenario_summary.parquet")
    portfolio_diff.write_parquet(output_root / "portfolio_diff_summary.parquet")
    _write_report(output_root, summary_frame, portfolio_diff)
    (output_root / "scenario_summary.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2),
        encoding="utf-8",
    )
    print(f"Compare dir: {compare_dir}")
    print(f"Ablation written to: {output_root}")


if __name__ == "__main__":
    main()
