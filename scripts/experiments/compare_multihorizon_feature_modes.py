#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from alpharank.portfolio.performance import performance_statistics


def _run(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError("--run values must use LABEL=PATH.")
    return label, Path(raw_path)


def _monthly_path(run_dir: Path, method: str, horizon: int) -> Path:
    return run_dir / f"{method}_h{horizon:02d}" / "trading_monthly.csv"


def _statistics(frame: pl.DataFrame, column: str, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_{key}": value
        for key, value in performance_statistics(frame[column].to_numpy()).items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare multi-horizon feature modes on native and shared test periods."
    )
    parser.add_argument("--run", action="append", type=_run, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    runs = dict(args.run)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_parts: list[pl.DataFrame] = []
    native_parts: list[pl.DataFrame] = []
    combinations: dict[tuple[str, int, int], dict[str, pl.DataFrame]] = {}
    for label, run_dir in runs.items():
        model_parts.append(
            pl.read_csv(run_dir / "model_horizon_summary.csv").with_columns(
                pl.lit(label).alias("feature_mode")
            )
        )
        native_parts.append(
            pl.read_csv(run_dir / "trading_backtest_all.csv").with_columns(
                pl.lit(label).alias("feature_mode")
            )
        )
        summary = pl.read_csv(run_dir / "model_horizon_summary.csv")
        for method, horizon in summary.select("method", "horizon").iter_rows():
            monthly_path = _monthly_path(run_dir, method, int(horizon))
            if not monthly_path.exists():
                continue
            monthly = pl.read_csv(monthly_path, try_parse_dates=True)
            for top_n in monthly["top_n"].unique().sort().to_list():
                combinations.setdefault((method, int(horizon), int(top_n)), {})[
                    label
                ] = monthly.filter(pl.col("top_n") == top_n)

    pl.concat(model_parts, how="diagonal_relaxed").write_csv(
        args.output_dir / "model_metrics_all_modes.csv"
    )
    pl.concat(native_parts, how="diagonal_relaxed").write_csv(
        args.output_dir / "trading_native_all_modes.csv"
    )

    common_rows: list[dict] = []
    expected_labels = set(runs)
    for (method, horizon, top_n), by_label in sorted(combinations.items()):
        if set(by_label) != expected_labels:
            continue
        common_months = set.intersection(
            *[
                set(frame["decision_month"].to_list())
                for frame in by_label.values()
            ]
        )
        if not common_months:
            continue
        for label, frame in by_label.items():
            common = frame.filter(pl.col("decision_month").is_in(common_months)).sort(
                "decision_month"
            )
            common_rows.append(
                {
                    "feature_mode": label,
                    "method": method,
                    "horizon": horizon,
                    "top_n": top_n,
                    "start_decision_month": common["decision_month"].min(),
                    "end_decision_month": common["decision_month"].max(),
                    "months": common.height,
                    **_statistics(common, "net_return", "model"),
                    **_statistics(common, "benchmark_return", "benchmark"),
                    **_statistics(common, "legacy_return", "legacy"),
                    "average_monthly_turnover": float(common["turnover"].mean()),
                }
            )
    pl.DataFrame(common_rows).write_csv(
        args.output_dir / "trading_common_across_modes.csv"
    )


if __name__ == "__main__":
    main()
