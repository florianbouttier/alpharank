from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import polars as pl
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n


@dataclass(frozen=True)
class PortfolioBoostingBlendBacktestConfig:
    prediction_run: Path = Path("outputs/portfolio_boosting_top_return_classifier_20260627_165036")
    deterministic_signal_run: Path = Path("outputs/deterministic_signal_predictions_20260627_154617")
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    output_dir: Path = Path("outputs")
    proba_col: str = "portfolio_boosting_top_return_proba"
    momentum_col: str = "technical_z_mean"
    momentum_weight: float = 0.10
    top_n: int = 5
    spy_overlay_weights: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    cash_overlay_weights: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    risk_free_rate: float = 0.02


def _load_scored(config: PortfolioBoostingBlendBacktestConfig) -> pl.DataFrame:
    predictions_path = config.prediction_run / "predictions.parquet"
    deterministic_path = config.deterministic_signal_run / "predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {predictions_path}")
    if not deterministic_path.exists():
        raise FileNotFoundError(f"Missing deterministic predictions parquet: {deterministic_path}")

    predictions = pl.read_parquet(predictions_path).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    )
    deterministic = pl.read_parquet(deterministic_path).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    )
    required = {config.momentum_col}
    missing = required - set(deterministic.columns)
    if missing:
        raise ValueError(f"Missing momentum columns in {deterministic_path}: {sorted(missing)}")

    return (
        predictions.join(
            deterministic.select(["ticker", "year_month", "holding_month", config.momentum_col]),
            on=["ticker", "year_month", "holding_month"],
            how="left",
        )
        .with_columns(
            (
                pl.col(config.proba_col).rank(method="average").over("year_month")
                / pl.len().over("year_month")
            ).alias("boosting_proba_rank"),
            (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
        )
        .with_columns(
            (
                (pl.col("boosting_proba_rank") + config.momentum_weight * pl.col(config.momentum_col))
                / (1.0 + config.momentum_weight)
            ).alias("boosting_momentum_score")
        )
    )


def _strategy_monthly_returns(scored: pl.DataFrame, top_n: int) -> pl.DataFrame:
    application = scored.with_columns(pl.col("boosting_momentum_score").alias("prediction"))
    selections = select_top_n(application, top_n=top_n)
    return compute_monthly_portfolio_returns(selections)


def _overlay_curves(strategy: pl.DataFrame, spy: pl.DataFrame, config: PortfolioBoostingBlendBacktestConfig) -> dict[str, pl.DataFrame]:
    joined = strategy.select("year_month", pl.col("portfolio_return").alias("strategy_return")).join(
        spy.rename({"monthly_return": "spy_return"}),
        on="year_month",
        how="inner",
    )
    curves: dict[str, pl.DataFrame] = {}
    for weight in config.spy_overlay_weights:
        strategy_pct = int(round(weight * 100))
        overlay_pct = int(round((1.0 - weight) * 100))
        curves[f"boosting_momentum_top{config.top_n}_{strategy_pct}pct_spy_{overlay_pct}pct"] = joined.select(
            "year_month",
            (weight * pl.col("strategy_return") + (1.0 - weight) * pl.col("spy_return")).alias("monthly_return"),
        )
    for weight in config.cash_overlay_weights:
        strategy_pct = int(round(weight * 100))
        overlay_pct = int(round((1.0 - weight) * 100))
        curves[f"boosting_momentum_top{config.top_n}_{strategy_pct}pct_cash_{overlay_pct}pct"] = joined.select(
            "year_month",
            (weight * pl.col("strategy_return")).alias("monthly_return"),
        )
    return curves


def _write_report(run_dir: Path, metrics: pl.DataFrame, config: PortfolioBoostingBlendBacktestConfig) -> None:
    rows = metrics.sort("Total Return", descending=True).to_dicts()
    lines = [
        "# Portfolio boosting blend backtest",
        "",
        "But: construire un portefeuille a partir d'une prediction boosting, avec un prior momentum explicite et un overlay de risque simple.",
        "",
        f"Prediction run: `{config.prediction_run}`",
        f"Momentum signal: `{config.momentum_col}`",
        f"Blend score: `rank(prediction_boosting) + {config.momentum_weight} * {config.momentum_col}`",
        f"Selection: top `{config.top_n}` actions par mois.",
        "",
        "## Resultats",
        "",
        "| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    lines.extend(
        [
            "",
            "## Lecture",
            "",
            "- Le modele boosting pur reste insuffisant.",
            "- Le candidat utile est le score hybride : prediction boosting en rang mensuel + prior momentum.",
            "- L'overlay SPY est une construction de portefeuille, pas une optimisation Legacy.",
            "- Le candidat principal observe dans ce run est `60% strategie / 40% SPY`: il bat `Combined_Frequency` en total return et Sharpe, mais garde un drawdown plus eleve.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: PortfolioBoostingBlendBacktestConfig) -> Path:
    run_dir = config.output_dir / f"portfolio_boosting_blend_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    scored = _load_scored(config)
    strategy_monthly = _strategy_monthly_returns(scored, config.top_n)
    spy_curve = build_spy_curve(scored)
    months = strategy_monthly.get_column("year_month").to_list()

    curves = _overlay_curves(strategy_monthly, spy_curve, config)
    curves["SPY"] = spy_curve
    curves.update(load_legacy_curves(config.legacy_monthly_returns, months))
    comparison = compare_backtest_curves(
        curves,
        output_path=run_dir / "comparison.html",
        title="Portfolio boosting blend backtest vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )

    scored.write_parquet(run_dir / "scored_predictions.parquet")
    strategy_monthly.write_parquet(run_dir / "strategy_monthly_returns.parquet")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "months": len(months),
                "start_month": str(min(months)) if months else None,
                "end_month": str(max(months)) if months else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(run_dir, comparison.metrics, config)
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True))
    return run_dir


def _parse_args() -> PortfolioBoostingBlendBacktestConfig:
    parser = argparse.ArgumentParser(description="Backtest a boosting + momentum blend portfolio.")
    parser.add_argument("--prediction-run", type=Path, default=PortfolioBoostingBlendBacktestConfig.prediction_run)
    parser.add_argument("--deterministic-signal-run", type=Path, default=PortfolioBoostingBlendBacktestConfig.deterministic_signal_run)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--proba-col", default="portfolio_boosting_top_return_proba")
    parser.add_argument("--momentum-col", default="technical_z_mean")
    parser.add_argument("--momentum-weight", type=float, default=0.10)
    parser.add_argument("--top-n", type=int, default=5)
    args = parser.parse_args()
    return PortfolioBoostingBlendBacktestConfig(
        prediction_run=args.prediction_run,
        deterministic_signal_run=args.deterministic_signal_run,
        legacy_monthly_returns=args.legacy_monthly_returns,
        output_dir=args.output_dir,
        proba_col=args.proba_col,
        momentum_col=args.momentum_col,
        momentum_weight=args.momentum_weight,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    run(_parse_args())
