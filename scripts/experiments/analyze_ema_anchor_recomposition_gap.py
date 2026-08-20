from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import polars as pl
from run_ema_anchor_residual_strategy import EmaAnchorResidualConfig, _load_frame  # noqa: E402
from run_ema_rich_future_target_models import (  # noqa: E402
    _recomposition_by_month,
    _recomposition_summary,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n

DEFAULT_PREDICTION_RUN = Path("outputs/ema_anchor_residual_strategy_20260628_194954")
DEFAULT_LEGACY_DETAILED_RETURNS = Path("outputs/2026-06-07/legacy_detailed_returns_polars.parquet")


@dataclass(frozen=True)
class EmaAnchorGapConfig:
    prediction_run: Path = DEFAULT_PREDICTION_RUN
    legacy_detailed_returns: Path = DEFAULT_LEGACY_DETAILED_RETURNS
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    output_dir: Path = Path("outputs")
    top_n_values: tuple[int, ...] = (10, 20, 30, 50)
    risk_free_rate: float = 0.02


def _legacy_labels(path: Path) -> pl.DataFrame:
    legacy = pl.read_parquet(path).with_columns(
        pl.col("year_month").dt.date().alias("holding_month"),
        pl.col("ticker").cast(pl.Utf8),
    )
    return (
        legacy.filter(pl.col("portfolio_model").is_in(["Combined_Equal", "Combined_Frequency"]))
        .select(["holding_month", "ticker"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )


def _scored_frame(config: EmaAnchorGapConfig) -> pl.DataFrame:
    predictions = pl.read_parquet(config.prediction_run / "predictions.parquet").with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    )
    months = predictions.select("holding_month").unique().get_column("holding_month").to_list()
    frame_config = EmaAnchorResidualConfig(anchor_mode="legacy_exact_dominant", residual_mode="init_score")
    frame, *_ = _load_frame(frame_config)
    legacy = _legacy_labels(config.legacy_detailed_returns)
    return (
        frame.filter(pl.col("holding_month").is_in(months))
        .join(legacy, on=["holding_month", "ticker"], how="left")
        .with_columns(pl.col("legacy_selected").fill_null(0).cast(pl.Int8))
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
                "legacy_selected",
                "legacy_exact_primary_mtr",
            ]
        )
        .join(
            predictions.select(
                [
                    "ticker",
                    "year_month",
                    "holding_month",
                    "ema_anchor_prediction",
                    "ema_anchor_residual_prediction",
                ]
            ),
            on=["ticker", "year_month", "holding_month"],
            how="inner",
        )
        .with_columns((pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"))
        .sort(["year_month", "ticker"])
    )


def _atomic_overlap(config: EmaAnchorGapConfig, months: Sequence[Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    month_frame = pl.DataFrame({"holding_month": list(months)}).with_columns(pl.col("holding_month").cast(pl.Date))
    legacy = (
        pl.read_parquet(config.legacy_detailed_returns)
        .with_columns(pl.col("year_month").dt.date().alias("holding_month"), pl.col("ticker").cast(pl.Utf8))
        .join(month_frame, on="holding_month", how="inner")
    )
    labels = _legacy_labels(config.legacy_detailed_returns).join(month_frame, on="holding_month", how="inner")
    atomic = legacy.filter(pl.col("portfolio_model").str.starts_with("Legacy_Optuna"))
    dominant_key = (
        atomic.group_by(["holding_month", "selected_model", "n_short", "n_long"])
        .agg(pl.len().alias("rows"))
        .sort(["holding_month", "rows", "selected_model"], descending=[False, True, False])
        .group_by("holding_month", maintain_order=True)
        .head(1)
        .select(["holding_month", "selected_model"])
    )
    dominant = (
        atomic.join(dominant_key, on=["holding_month", "selected_model"], how="inner")
        .select(["holding_month", "ticker"])
        .unique()
        .with_columns(pl.lit(1).alias("dominant_selected"))
    )
    union = atomic.select(["holding_month", "ticker"]).unique().with_columns(pl.lit(1).alias("union_selected"))
    universe = (
        labels.select(["holding_month", "ticker"])
        .join(dominant, on=["holding_month", "ticker"], how="full", coalesce=True)
        .join(union, on=["holding_month", "ticker"], how="full", coalesce=True)
        .join(labels, on=["holding_month", "ticker"], how="left")
        .with_columns(
            pl.col("legacy_selected").fill_null(0).cast(pl.Int8),
            pl.col("dominant_selected").fill_null(0).cast(pl.Int8),
            pl.col("union_selected").fill_null(0).cast(pl.Int8),
        )
    )
    rows: list[dict[str, Any]] = []
    for month_df in universe.partition_by("holding_month", maintain_order=True):
        month = month_df.get_column("holding_month")[0]
        legacy_tickers = set(month_df.filter(pl.col("legacy_selected") == 1).get_column("ticker").to_list())
        if not legacy_tickers:
            continue
        for col in ["dominant_selected", "union_selected"]:
            selected = set(month_df.filter(pl.col(col) == 1).get_column("ticker").to_list())
            common = legacy_tickers & selected
            rows.append(
                {
                    "model": col,
                    "holding_month": month,
                    "common_count": len(common),
                    "legacy_count": len(legacy_tickers),
                    "recomposition_pct": len(common) / len(legacy_tickers),
                }
            )
    by_month = pl.DataFrame(rows).sort(["model", "holding_month"])
    summary = (
        by_month.group_by("model")
        .agg(
            pl.sum("common_count").alias("common_count"),
            pl.sum("legacy_count").alias("legacy_count"),
            (pl.sum("common_count") / pl.sum("legacy_count")).alias("recomposition_pct"),
            pl.mean("recomposition_pct").alias("mean_monthly_recomposition_pct"),
            pl.median("recomposition_pct").alias("median_monthly_recomposition_pct"),
            pl.len().alias("months"),
        )
        .sort("recomposition_pct", descending=True)
    )
    return by_month, summary


def _prediction_dispersion(scored: pl.DataFrame) -> pl.DataFrame:
    return (
        scored.group_by("year_month")
        .agg(
            pl.std("legacy_exact_primary_mtr").alias("raw_ema_std"),
            pl.n_unique("legacy_exact_primary_mtr").alias("raw_ema_unique"),
            pl.std("ema_anchor_prediction").alias("ema_boosting_std"),
            pl.n_unique("ema_anchor_prediction").alias("ema_boosting_unique"),
            pl.std("ema_anchor_residual_prediction").alias("residual_boosting_std"),
            pl.n_unique("ema_anchor_residual_prediction").alias("residual_boosting_unique"),
            pl.len().alias("universe_size"),
        )
        .sort("year_month")
    )


def _curves(scored: pl.DataFrame, config: EmaAnchorGapConfig) -> dict[str, pl.DataFrame]:
    curves: dict[str, pl.DataFrame] = {}
    score_cols = ["legacy_exact_primary_mtr", "ema_anchor_prediction", "ema_anchor_residual_prediction"]
    for score_col in score_cols:
        for top_n in config.top_n_values:
            selected = select_top_n(scored.with_columns(pl.col(score_col).alias("prediction")), top_n=top_n)
            monthly = compute_monthly_portfolio_returns(selected)
            curves[f"{score_col}_top{top_n}"] = monthly.select(
                "year_month",
                pl.col("portfolio_return").alias("monthly_return"),
                pl.col("n_positions").alias("n"),
            )
    curves["SPY"] = build_spy_curve(scored)
    months = scored.select("holding_month").unique().sort("holding_month").get_column("holding_month").to_list()
    curves.update(load_legacy_curves(config.legacy_monthly_returns, months))
    return curves


def _write_report(
    run_dir: Path,
    *,
    config: EmaAnchorGapConfig,
    recomposition_summary: pl.DataFrame,
    atomic_summary: pl.DataFrame,
    dispersion: pl.DataFrame,
    comparison_metrics: pl.DataFrame,
) -> None:
    lines = [
        "# EMA anchor recomposition gap diagnostic",
        "",
        "But: expliquer pourquoi le run EMA mensuelle + boosting affiche une performance EMA seule faible alors que les diagnostics EMA reconstituent bien Legacy.",
        "",
        f"Prediction run: `{config.prediction_run}`",
        "",
        "## Conclusion",
        "",
        "- Le signal EMA brut est bien fort et recompose Legacy correctement.",
        "- Le modele boosting base entraine uniquement sur cette EMA ecrase le ranking : ses predictions ont tres peu de valeurs distinctes par mois.",
        "- La mauvaise performance `ema_anchor_top20` vient donc du booster base, pas de l'EMA Legacy.",
        "- Pour garder l'esprit de la methode, le score primaire doit rester le score EMA brut ou une calibration monotone du score brut, puis seulement apres on apprend un residu.",
        "",
        "## Recomposition avec K Legacy dynamique",
        "",
        "| modele | actions communes | actions Legacy | recomposition | mediane mensuelle |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in recomposition_summary.to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['common_count']} | {row['legacy_count']} | "
            f"{row['recomposition_pct'] * 100:.1f}% | {row['median_monthly_recomposition_pct'] * 100:.1f}% |"
        )
    lines.extend(["", "## Recomposition des briques Legacy", "", "| modele | actions communes | actions Legacy | recomposition | mediane mensuelle |", "|---|---:|---:|---:|---:|"])
    for row in atomic_summary.to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['common_count']} | {row['legacy_count']} | "
            f"{row['recomposition_pct'] * 100:.1f}% | {row['median_monthly_recomposition_pct'] * 100:.1f}% |"
        )
    lines.extend(["", "## Backtest", "", "| modele | total return | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |", "|---|---:|---:|---:|---:|---:|---:|"])
    for row in comparison_metrics.sort("Total Return", descending=True).to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    disp_summary = dispersion.select(
        pl.min("ema_boosting_unique").alias("ema_boosting_min_unique"),
        pl.median("ema_boosting_unique").alias("ema_boosting_median_unique"),
        pl.median("raw_ema_unique").alias("raw_ema_median_unique"),
        pl.median("residual_boosting_unique").alias("residual_boosting_median_unique"),
    ).to_dicts()[0]
    lines.extend(
        [
            "",
            "## Dispersion des scores",
            "",
            f"- mediane valeurs distinctes EMA brute par mois : `{disp_summary['raw_ema_median_unique']:.0f}`",
            f"- mediane valeurs distinctes prediction boosting EMA par mois : `{disp_summary['ema_boosting_median_unique']:.0f}`",
            f"- minimum valeurs distinctes prediction boosting EMA par mois : `{disp_summary['ema_boosting_min_unique']:.0f}`",
            f"- mediane valeurs distinctes prediction residuelle par mois : `{disp_summary['residual_boosting_median_unique']:.0f}`",
            "",
            "Lecture : quand le booster EMA sort 1 a 3 scores differents pour plusieurs centaines d'actions, le top K devient quasi arbitraire. C'est incompatible avec la recomposition Legacy.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: EmaAnchorGapConfig) -> Path:
    run_dir = config.output_dir / f"ema_anchor_recomposition_gap_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scored = _scored_frame(config)
    score_cols = ["legacy_exact_primary_mtr", "ema_anchor_prediction", "ema_anchor_residual_prediction"]
    recomposition = _recomposition_by_month(scored.select(["ticker", "holding_month", "legacy_selected"] + score_cols), score_cols)
    recomposition_summary = _recomposition_summary(recomposition)
    months = scored.select("holding_month").unique().sort("holding_month").get_column("holding_month").to_list()
    atomic_by_month, atomic_summary = _atomic_overlap(config, months)
    dispersion = _prediction_dispersion(scored)
    comparison = compare_backtest_curves(
        _curves(scored, config),
        output_path=run_dir / "comparison.html",
        title="EMA anchor recomposition gap diagnostic",
        risk_free_rate=config.risk_free_rate,
    )

    scored.write_parquet(run_dir / "scored_frame.parquet")
    recomposition.write_csv(run_dir / "recomposition_by_month.csv")
    recomposition_summary.write_csv(run_dir / "recomposition_summary.csv")
    atomic_by_month.write_csv(run_dir / "legacy_atomic_recomposition_by_month.csv")
    atomic_summary.write_csv(run_dir / "legacy_atomic_recomposition_summary.csv")
    dispersion.write_csv(run_dir / "score_dispersion_by_month.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    _write_report(
        run_dir,
        config=config,
        recomposition_summary=recomposition_summary,
        atomic_summary=atomic_summary,
        dispersion=dispersion,
        comparison_metrics=comparison.metrics,
    )
    (run_dir / "metadata.json").write_text(
        __import__("json").dumps({**asdict(config), "prediction_run": str(config.prediction_run)}, default=str, indent=2),
        encoding="utf-8",
    )
    print(f"RUN_DIR={run_dir}")
    print(recomposition_summary)
    print(comparison.metrics.sort("Total Return", descending=True).head(12))
    return run_dir


def _parse_args() -> EmaAnchorGapConfig:
    parser = argparse.ArgumentParser(description="Diagnose raw EMA vs boosting recomposition gap.")
    parser.add_argument("--prediction-run", type=Path, default=DEFAULT_PREDICTION_RUN)
    parser.add_argument("--legacy-detailed-returns", type=Path, default=DEFAULT_LEGACY_DETAILED_RETURNS)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--top-n", type=int, nargs="*", default=[10, 20, 30, 50])
    args = parser.parse_args()
    return EmaAnchorGapConfig(
        prediction_run=args.prediction_run,
        legacy_detailed_returns=args.legacy_detailed_returns,
        legacy_monthly_returns=args.legacy_monthly_returns,
        output_dir=args.output_dir,
        top_n_values=tuple(args.top_n),
    )


if __name__ == "__main__":
    run(_parse_args())
