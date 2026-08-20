from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import polars as pl
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n


@dataclass(frozen=True)
class PortfolioBoostingRiskOverlayConfig:
    prediction_run: Path
    score_col: str
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    output_dir: Path = Path("outputs")
    top_n_values: tuple[int, ...] = (10, 20, 30, 50)
    fixed_strategy_weights: tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 1.0)
    confidence_min_exposure: float = 0.30
    confidence_max_exposure: float = 1.00
    confidence_default_exposure: float = 0.60
    confidence_min_history: int = 24
    confidence_low_quantile: float = 0.25
    confidence_high_quantile: float = 0.75
    risk_free_rate: float = 0.02


def _load_predictions(config: PortfolioBoostingRiskOverlayConfig) -> pl.DataFrame:
    path = config.prediction_run / "predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {path}")
    predictions = pl.read_parquet(path).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col(config.score_col).cast(pl.Float64),
    )
    required = {
        "ticker",
        "year_month",
        "holding_month",
        "future_return",
        "benchmark_future_return",
        config.score_col,
    }
    missing = required - set(predictions.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return predictions.with_columns((pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"))


def _load_fold_validation_quality(prediction_run: Path) -> pl.DataFrame | None:
    trials_path = prediction_run / "optuna_trials.csv"
    if not trials_path.exists():
        return None
    trials = pl.read_csv(trials_path)
    if "fold" not in trials.columns or "val_metric" not in trials.columns:
        return None
    return (
        trials.filter(pl.col("val_metric").is_not_null())
        .group_by("fold")
        .agg(pl.max("val_metric").alias("validation_quality"))
    )


def _strategy_frame(predictions: pl.DataFrame, score_col: str, top_n: int) -> pl.DataFrame:
    application = predictions.with_columns(pl.col(score_col).alias("prediction"))
    selections = select_top_n(application, top_n=top_n)
    monthly = compute_monthly_portfolio_returns(selections)
    confidence = (
        predictions.group_by("year_month")
        .agg(
            (pl.col(score_col).sort(descending=True).head(top_n).mean() - pl.col(score_col).median()).alias("score_spread"),
            pl.std(score_col).alias("score_std"),
        )
        .with_columns((pl.col("score_spread") / (pl.col("score_std") + 1e-12)).alias("score_confidence"))
    )
    return monthly.join(confidence, on="year_month", how="left")


def _fixed_overlay(
    strategy: pl.DataFrame,
    spy: pl.DataFrame,
    *,
    strategy_weight: float,
    overlay: str,
) -> pl.DataFrame:
    joined = strategy.select("year_month", pl.col("portfolio_return").alias("strategy_return")).join(
        spy.rename({"monthly_return": "spy_return"}),
        on="year_month",
        how="inner",
    )
    if overlay == "spy":
        monthly_return = strategy_weight * pl.col("strategy_return") + (1.0 - strategy_weight) * pl.col("spy_return")
    elif overlay == "cash":
        monthly_return = strategy_weight * pl.col("strategy_return")
    else:
        raise ValueError(f"Unsupported overlay={overlay!r}")
    return joined.select("year_month", monthly_return.alias("monthly_return"))


def _expanding_exposure(values: Iterable[float], config: PortfolioBoostingRiskOverlayConfig) -> list[float]:
    history: list[float] = []
    exposures: list[float] = []
    for raw in values:
        value = float(raw) if raw is not None and np.isfinite(float(raw)) else np.nan
        clean_history = [item for item in history if np.isfinite(item)]
        if len(clean_history) < config.confidence_min_history or not np.isfinite(value):
            exposure = config.confidence_default_exposure
        else:
            low = float(np.quantile(clean_history, config.confidence_low_quantile))
            high = float(np.quantile(clean_history, config.confidence_high_quantile))
            if high <= low + 1e-12:
                exposure = config.confidence_default_exposure
            else:
                raw_exposure = (value - low) / (high - low)
                exposure = config.confidence_min_exposure + raw_exposure * (
                    config.confidence_max_exposure - config.confidence_min_exposure
                )
                exposure = float(np.clip(exposure, config.confidence_min_exposure, config.confidence_max_exposure))
        exposures.append(exposure)
        history.append(value)
    return exposures


def _confidence_overlay(
    strategy: pl.DataFrame,
    spy: pl.DataFrame,
    *,
    confidence_col: str,
    overlay: str,
    config: PortfolioBoostingRiskOverlayConfig,
) -> pl.DataFrame:
    joined = (
        strategy.select("year_month", pl.col("portfolio_return").alias("strategy_return"), confidence_col)
        .join(spy.rename({"monthly_return": "spy_return"}), on="year_month", how="inner")
        .sort("year_month")
    )
    exposures = _expanding_exposure(joined.get_column(confidence_col).to_list(), config)
    joined = joined.with_columns(pl.Series("strategy_exposure", exposures, dtype=pl.Float64))
    if overlay == "spy":
        monthly_return = pl.col("strategy_exposure") * pl.col("strategy_return") + (
            1.0 - pl.col("strategy_exposure")
        ) * pl.col("spy_return")
    elif overlay == "cash":
        monthly_return = pl.col("strategy_exposure") * pl.col("strategy_return")
    else:
        raise ValueError(f"Unsupported overlay={overlay!r}")
    return joined.select("year_month", monthly_return.alias("monthly_return"), "strategy_exposure")


def _build_curves(
    predictions: pl.DataFrame,
    config: PortfolioBoostingRiskOverlayConfig,
) -> tuple[dict[str, pl.DataFrame], pl.DataFrame]:
    spy = build_spy_curve(predictions)
    fold_quality = _load_fold_validation_quality(config.prediction_run)
    curves: dict[str, pl.DataFrame] = {"SPY": spy}
    exposure_frames: list[pl.DataFrame] = []

    for top_n in config.top_n_values:
        strategy = _strategy_frame(predictions, config.score_col, top_n)
        if fold_quality is not None and "fold" in predictions.columns:
            fold_months = predictions.select("year_month", "fold").unique()
            strategy = strategy.join(fold_months.join(fold_quality, on="fold", how="left"), on="year_month", how="left")

        curves[f"boosting_top{top_n}_100pct"] = strategy.select(
            "year_month", pl.col("portfolio_return").alias("monthly_return")
        )
        for weight in config.fixed_strategy_weights:
            weight_label = int(round(weight * 100))
            for overlay in ("spy", "cash"):
                curves[f"boosting_top{top_n}_{weight_label}pct_{overlay}"] = _fixed_overlay(
                    strategy,
                    spy,
                    strategy_weight=weight,
                    overlay=overlay,
                )

        for confidence_col in ("score_confidence", "validation_quality"):
            if confidence_col not in strategy.columns:
                continue
            for overlay in ("spy", "cash"):
                name = f"boosting_top{top_n}_dynamic_{confidence_col}_{overlay}"
                curve = _confidence_overlay(strategy, spy, confidence_col=confidence_col, overlay=overlay, config=config)
                curves[name] = curve.select("year_month", "monthly_return")
                exposure_frames.append(curve.with_columns(pl.lit(name).alias("model")))

    exposures = pl.concat(exposure_frames, how="vertical") if exposure_frames else pl.DataFrame()
    return curves, exposures


def _write_report(run_dir: Path, metrics: pl.DataFrame, config: PortfolioBoostingRiskOverlayConfig) -> None:
    rows = metrics.sort("Total Return", descending=True).to_dicts()
    lines = [
        "# Portfolio boosting risk overlay",
        "",
        "But: tester si le probleme du boosting pur vient surtout du sizing/concentration.",
        "",
        f"Prediction run: `{config.prediction_run}`",
        f"Score: `{config.score_col}`",
        "Aucune variable Legacy n'est utilisee pour construire les courbes.",
        "",
        "## Resultats",
        "",
        "| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:30]:
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
            "- `*_100pct` est le top N boosting pur sans overlay.",
            "- `*_pct_spy` garde le capital non alloue dans SPY.",
            "- `*_pct_cash` garde le capital non alloue en cash a 0% mensuel.",
            "- `dynamic_score_confidence` module l'exposition avec la dispersion des scores du mois, calibree seulement sur les mois passes.",
            "- `dynamic_validation_quality` module l'exposition avec la meilleure performance validation du fold, connue avant le mois de test.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: PortfolioBoostingRiskOverlayConfig) -> Path:
    run_dir = config.output_dir / f"portfolio_boosting_risk_overlay_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = _load_predictions(config)
    curves, exposures = _build_curves(predictions, config)
    months = predictions.select(pl.col("holding_month").alias("year_month")).unique().sort("year_month").get_column("year_month").to_list()
    curves.update(load_legacy_curves(config.legacy_monthly_returns, months))
    comparison = compare_backtest_curves(
        curves,
        output_path=run_dir / "comparison.html",
        title="Portfolio boosting risk overlay vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )

    predictions.write_parquet(run_dir / "input_predictions.parquet")
    exposures.write_parquet(run_dir / "dynamic_exposures.parquet")
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
    print(comparison.metrics.sort("Total Return", descending=True).head(20))
    return run_dir


def _parse_args() -> PortfolioBoostingRiskOverlayConfig:
    parser = argparse.ArgumentParser(description="Backtest pure boosting risk overlays.")
    parser.add_argument("--prediction-run", type=Path, required=True)
    parser.add_argument("--score-col", required=True)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top-n", type=int, nargs="*", default=[10, 20, 30, 50])
    parser.add_argument("--fixed-strategy-weights", type=float, nargs="*", default=[0.4, 0.5, 0.6, 0.7, 0.8, 1.0])
    parser.add_argument("--confidence-min-exposure", type=float, default=0.30)
    parser.add_argument("--confidence-max-exposure", type=float, default=1.00)
    parser.add_argument("--confidence-default-exposure", type=float, default=0.60)
    parser.add_argument("--confidence-min-history", type=int, default=24)
    args = parser.parse_args()
    return PortfolioBoostingRiskOverlayConfig(
        prediction_run=args.prediction_run,
        score_col=args.score_col,
        legacy_monthly_returns=args.legacy_monthly_returns,
        output_dir=args.output_dir,
        top_n_values=tuple(args.top_n),
        fixed_strategy_weights=tuple(args.fixed_strategy_weights),
        confidence_min_exposure=args.confidence_min_exposure,
        confidence_max_exposure=args.confidence_max_exposure,
        confidence_default_exposure=args.confidence_default_exposure,
        confidence_min_history=args.confidence_min_history,
    )


if __name__ == "__main__":
    run(_parse_args())
