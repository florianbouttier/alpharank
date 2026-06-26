from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import polars as pl

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.kpis import compute_backtest_kpis
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n


DEFAULT_PREDICTION_RUN = Path("outputs/tradable_ema_regression_optuna_20260621_003954")
DEFAULT_LEGACY_MONTHLY_RETURNS = Path("outputs/2026-06-07/legacy_monthly_returns_polars.parquet")
DEFAULT_RISK_FREE_RATE = 0.02
DEFAULT_TOP_N_VALUES = (5, 7, 10)
LEGACY_MODELS = ("Combined_Equal", "Combined_Frequency")


@dataclass(frozen=True)
class TradingBacktestConfig:
    prediction_run: Path = DEFAULT_PREDICTION_RUN
    source_run: Path | None = None
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    output_dir: Path = Path("outputs")
    score_col: str = "tradable_ema_regression"
    top_n_values: tuple[int, ...] = DEFAULT_TOP_N_VALUES
    include_legacy_k: bool = True
    risk_free_rate: float = DEFAULT_RISK_FREE_RATE


def _prediction_path(prediction_run: Path) -> Path:
    path = prediction_run / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {path}")
    return path


def _source_run_from_metadata(prediction_run: Path) -> Path:
    metadata_path = prediction_run / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing prediction metadata: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_run = metadata.get("source_run")
    if not source_run:
        raise ValueError(f"Prediction metadata does not contain `source_run`: {metadata_path}")
    return Path(source_run)


def load_backtest_predictions(config: TradingBacktestConfig) -> pl.DataFrame:
    prediction_run = config.prediction_run
    source_run = config.source_run or _source_run_from_metadata(prediction_run)
    source_frame_path = source_run / "model_frame.parquet"
    if not source_frame_path.exists():
        raise FileNotFoundError(f"Missing source model_frame: {source_frame_path}")

    predictions = pl.read_parquet(_prediction_path(prediction_run)).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    )
    if config.score_col not in predictions.columns:
        raise ValueError(f"Missing score column `{config.score_col}` in {prediction_run}")

    source = (
        pl.read_parquet(source_frame_path)
        .with_columns(
            pl.col("ticker").cast(pl.Utf8),
            pl.col("year_month").cast(pl.Date),
            pl.col("holding_month").cast(pl.Date),
        )
        .select(
            [
                "ticker",
                "year_month",
                "decision_month",
                "decision_asof_date",
                "holding_month",
                "future_return",
                "benchmark_future_return",
                "future_excess_return",
            ]
        )
    )

    joined = predictions.select(["ticker", "year_month", "holding_month", "legacy_selected", config.score_col, "fold"]).join(
        source,
        on=["ticker", "year_month", "holding_month"],
        how="inner",
    )
    return (
        joined.with_columns(
            pl.col(config.score_col).alias("prediction"),
            (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
        )
        .select(
            [
                "ticker",
                "year_month",
                "decision_month",
                "decision_asof_date",
                "holding_month",
                "future_return",
                "benchmark_future_return",
                "future_excess_return",
                "target_label",
                "prediction",
                "legacy_selected",
                "fold",
            ]
        )
        .sort(["year_month", "ticker"])
    )


def select_dynamic_legacy_k(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions.with_columns(pl.lit(None).alias("rank")).head(0)

    legacy_counts = (
        predictions.group_by("year_month")
        .agg(pl.col("legacy_selected").sum().cast(pl.Int64).alias("legacy_k"))
        .filter(pl.col("legacy_k") > 0)
    )
    ranked = predictions.with_columns(
        pl.col("prediction").rank(method="ordinal", descending=True).over("year_month").alias("rank")
    ).join(legacy_counts, on="year_month", how="inner")
    return ranked.filter(pl.col("rank") <= pl.col("legacy_k")).sort(["year_month", "rank"])


def run_model_scenario(
    predictions: pl.DataFrame,
    *,
    name: str,
    top_n: int | None,
    risk_free_rate: float,
) -> dict[str, Any]:
    selections = select_dynamic_legacy_k(predictions) if top_n is None else select_top_n(predictions, top_n=top_n)
    monthly_returns = compute_monthly_portfolio_returns(selections)
    kpis = compute_backtest_kpis(monthly_returns, risk_free_rate=risk_free_rate).with_columns(
        pl.lit(name).alias("model")
    )
    return {
        "name": name,
        "selections": selections.with_columns(pl.lit(name).alias("model")),
        "monthly_returns": monthly_returns.with_columns(pl.lit(name).alias("model")),
        "kpis": kpis,
    }


def build_spy_curve(predictions: pl.DataFrame) -> pl.DataFrame:
    return (
        predictions.group_by("holding_month")
        .agg(pl.mean("benchmark_future_return").alias("monthly_return"))
        .rename({"holding_month": "year_month"})
        .sort("year_month")
    )


def load_legacy_curves(path: Path, months: Sequence[Any]) -> dict[str, pl.DataFrame]:
    month_frame = pl.DataFrame({"year_month": list(months)}).with_columns(pl.col("year_month").cast(pl.Date))
    legacy = (
        pl.read_parquet(path)
        .with_columns(pl.col("year_month").cast(pl.Date))
        .filter(pl.col("model").is_in(LEGACY_MODELS))
        .join(month_frame, on="year_month", how="inner")
        .select(["model", "year_month", "monthly_return"])
        .sort(["model", "year_month"])
    )
    curves: dict[str, pl.DataFrame] = {}
    for model_key, frame in legacy.partition_by("model", as_dict=True).items():
        model = model_key[0] if isinstance(model_key, tuple) else model_key
        curves[str(model)] = frame.select(["year_month", "monthly_return"])
    return curves


def _curve_from_monthly(monthly_returns: pl.DataFrame) -> pl.DataFrame:
    return monthly_returns.select(
        [
            "year_month",
            pl.col("portfolio_return").alias("monthly_return"),
            pl.col("n_positions").alias("n"),
        ]
    )


def run(config: TradingBacktestConfig) -> Path:
    run_dir = config.output_dir / f"tradable_ema_regression_trading_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = load_backtest_predictions(config)
    scenarios = [
        run_model_scenario(
            predictions,
            name=f"tradable_ema_top_{top_n}",
            top_n=top_n,
            risk_free_rate=config.risk_free_rate,
        )
        for top_n in config.top_n_values
    ]
    if config.include_legacy_k:
        scenarios.append(
            run_model_scenario(
                predictions,
                name="tradable_ema_legacy_k",
                top_n=None,
                risk_free_rate=config.risk_free_rate,
            )
        )

    monthly_returns = pl.concat([scenario["monthly_returns"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    model_kpis = pl.concat([scenario["kpis"] for scenario in scenarios], how="vertical")

    months = predictions.select(pl.col("holding_month").alias("year_month")).unique().sort("year_month").get_column("year_month").to_list()
    comparison_inputs: dict[str, pl.DataFrame] = {
        scenario["name"]: _curve_from_monthly(scenario["monthly_returns"])
        for scenario in scenarios
    }
    comparison_inputs["SPY"] = build_spy_curve(predictions)
    comparison_inputs.update(load_legacy_curves(config.legacy_monthly_returns, months))

    comparison = compare_backtest_curves(
        comparison_inputs,
        output_path=run_dir / "trading_backtest_comparison.html",
        title="Tradable EMA Regression Trading Backtest vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )

    predictions.write_parquet(run_dir / "application_predictions.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    model_kpis.write_csv(run_dir / "model_kpis.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "prediction_run": str(config.prediction_run),
                "source_run": str(config.source_run or _source_run_from_metadata(config.prediction_run)),
                "legacy_monthly_returns": str(config.legacy_monthly_returns),
                "score_col": config.score_col,
                "top_n_values": list(config.top_n_values),
                "include_legacy_k": config.include_legacy_k,
                "risk_free_rate": config.risk_free_rate,
                "months": len(months),
                "start_month": str(min(months)) if months else None,
                "end_month": str(max(months)) if months else None,
                "outputs": {
                    "html_report": str(comparison.output_path),
                    "comparison_metrics": "comparison_metrics.csv",
                    "model_kpis": "model_kpis.csv",
                    "monthly_returns": "monthly_returns.parquet",
                    "selections": "selections.parquet",
                },
            },
            indent=2,
            default=str,
        )
    )

    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest tradable EMA regression predictions vs Legacy monthly returns.")
    parser.add_argument("--prediction-run", type=Path, default=DEFAULT_PREDICTION_RUN)
    parser.add_argument("--source-run", type=Path, default=None)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--score-col", default="tradable_ema_regression")
    parser.add_argument("--top-n", type=int, nargs="*", default=list(DEFAULT_TOP_N_VALUES))
    parser.add_argument("--no-legacy-k", action="store_true")
    parser.add_argument("--risk-free-rate", type=float, default=DEFAULT_RISK_FREE_RATE)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        TradingBacktestConfig(
            prediction_run=args.prediction_run,
            source_run=args.source_run,
            legacy_monthly_returns=args.legacy_monthly_returns,
            output_dir=args.output_dir,
            score_col=args.score_col,
            top_n_values=tuple(args.top_n),
            include_legacy_k=not args.no_legacy_k,
            risk_free_rate=args.risk_free_rate,
        )
    )


if __name__ == "__main__":
    main()
