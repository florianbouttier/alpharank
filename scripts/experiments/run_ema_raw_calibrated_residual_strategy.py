from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl
from run_ema_anchor_residual_strategy import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    RESIDUAL_MODEL_PARAMS,
    EmaAnchorResidualConfig,
    _fit_xgb_regressor_with_base_margin,
    _load_frame,
    _predict_xgb_with_base_margin,
    _prediction_metrics,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows


@dataclass(frozen=True)
class EmaRawCalibratedResidualConfig:
    output_dir: Path = Path("outputs")
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    anchor_mode: str = "legacy_exact_dominant"
    calibration_mode: str = "rank_linear_positive"
    target_clip: float = 0.30
    min_train_months: int = 168
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    max_windows: int = 999
    top_n_values: tuple[int, ...] = (5, 7, 10, 20, 30, 50)
    residual_shrinkages: tuple[float, ...] = (0.10, 0.25, 0.50, 1.00)
    min_calibration_slope: float = 1e-6
    risk_free_rate: float = 0.02
    seed: int = 42


def _target(frame: pl.DataFrame, clip: float) -> np.ndarray:
    return np.clip(frame.get_column("future_excess_return").to_numpy(), -clip, clip).astype(np.float32)


def _calibration_input(frame: pl.DataFrame) -> np.ndarray:
    values = frame.get_column("legacy_exact_primary_mtr_rank_month").to_numpy().astype(float)
    if np.isfinite(values).any():
        median = float(np.nanmedian(values[np.isfinite(values)]))
    else:
        median = 0.5
    return np.where(np.isfinite(values), values, median).astype(np.float32)


def _fit_positive_linear_calibration(x: np.ndarray, y: np.ndarray, min_slope: float) -> dict[str, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3 or float(np.var(x[mask])) < 1e-12:
        slope = max(0.0, min_slope)
        intercept = float(np.nanmean(y[mask]) - slope * np.nanmean(x[mask])) if mask.any() else 0.0
        return {"intercept": intercept, "slope": slope, "raw_slope": 0.0}
    x_m = x[mask].astype(float)
    y_m = y[mask].astype(float)
    raw_slope = float(np.cov(x_m, y_m, bias=True)[0, 1] / np.var(x_m))
    slope = max(min_slope, raw_slope)
    intercept = float(np.mean(y_m) - slope * np.mean(x_m))
    return {"intercept": intercept, "slope": slope, "raw_slope": raw_slope}


def _predict_positive_linear(calibration: dict[str, float], x: np.ndarray) -> np.ndarray:
    return (float(calibration["intercept"]) + float(calibration["slope"]) * x).astype(np.float32)


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).fill_null(0.0).to_numpy().astype(np.float32)


def _prediction_frame(
    test_df: pl.DataFrame,
    *,
    fold: str,
    base_prediction: np.ndarray,
    residual_prediction: np.ndarray,
    final_prediction: np.ndarray,
    shrink_predictions: dict[float, np.ndarray],
) -> pl.DataFrame:
    frame = test_df.select(
        [
            "ticker",
            "year_month",
            "decision_month",
            "decision_asof_date",
            "holding_month",
            "future_return",
            "benchmark_future_return",
            "future_excess_return",
            "legacy_exact_primary_mtr",
            "legacy_exact_primary_mtr_rank_month",
        ]
    ).with_columns(
        pl.lit(fold).alias("fold"),
        pl.Series("ema_raw_calibrated_prediction", base_prediction, dtype=pl.Float64),
        pl.Series("residual_prediction", residual_prediction, dtype=pl.Float64),
        pl.Series("ema_raw_calibrated_residual_prediction", final_prediction, dtype=pl.Float64),
        (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
    )
    for shrinkage, prediction in shrink_predictions.items():
        suffix = str(shrinkage).replace(".", "_")
        frame = frame.with_columns(
            pl.Series(f"ema_raw_calibrated_residual_s{suffix}_prediction", prediction, dtype=pl.Float64)
        )
    return frame


def _run_scenario(predictions: pl.DataFrame, score_col: str, name: str, top_n: int) -> dict[str, pl.DataFrame]:
    selections = select_top_n(predictions.with_columns(pl.col(score_col).alias("prediction")), top_n=top_n)
    monthly = compute_monthly_portfolio_returns(selections)
    return {
        "monthly": monthly.with_columns(pl.lit(name).alias("model")),
        "selections": selections.with_columns(pl.lit(name).alias("model")),
    }


def _write_report(
    run_dir: Path,
    *,
    config: EmaRawCalibratedResidualConfig,
    comparison_metrics: pl.DataFrame,
    prediction_metrics: pl.DataFrame,
    fold_metrics: pl.DataFrame,
) -> None:
    lines = [
        "# EMA raw calibrated residual strategy",
        "",
        "But: garder le ranking de l'EMA brute, la convertir en prediction de rendement relatif attendu, puis apprendre seulement le residu.",
        "",
        f"Mode EMA: `{config.anchor_mode}`",
        f"Calibration: `{config.calibration_mode}`",
        f"Target: `future_excess_return` clippe a +/-{config.target_clip:.2f}",
        "",
        "## Backtest",
        "",
        "| modele | total return | CAGR | Sharpe | max DD | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_metrics.sort("Total Return", descending=True).to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    lines.extend(["", "## Metriques prediction", "", "| modele | metrique | valeur |", "|---|---|---:|"])
    for row in prediction_metrics.to_dicts():
        value = row["value"]
        formatted = f"{value * 100:.2f}%" if row["metric"].endswith(("hit_rate", "excess")) else f"{value:.4f}"
        lines.append(f"| `{row['model']}` | `{row['metric']}` | {formatted} |")
    slope_summary = fold_metrics.select(
        pl.mean("calibration_raw_slope").alias("avg_raw_slope"),
        pl.mean("calibration_slope").alias("avg_slope"),
        pl.median("calibration_slope").alias("median_slope"),
        pl.min("calibration_slope").alias("min_slope"),
        pl.max("calibration_slope").alias("max_slope"),
        (pl.col("calibration_raw_slope") <= 0).sum().alias("non_positive_raw_slopes"),
        pl.len().alias("fold_count"),
    ).to_dicts()[0]
    lines.extend(
        [
            "",
            "## Calibration",
            "",
            f"- pente moyenne : `{slope_summary['avg_slope']:.6f}`",
            f"- pente mediane : `{slope_summary['median_slope']:.6f}`",
            f"- pente min/max : `{slope_summary['min_slope']:.6f}` / `{slope_summary['max_slope']:.6f}`",
            f"- pentes brutes non positives : `{int(slope_summary['non_positive_raw_slopes'])}` / `{int(slope_summary['fold_count'])}`",
            "",
            "Lecture : la calibration lineaire positive transforme le rang EMA en prediction numerique utilisable comme `base_margin`. Une pente plancher conserve l'ordre EMA meme quand la pente train brute est negative ou nulle.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: EmaRawCalibratedResidualConfig) -> Path:
    run_dir = config.output_dir / f"ema_raw_calibrated_residual_strategy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    frame_config = EmaAnchorResidualConfig(
        anchor_mode=config.anchor_mode,
        residual_mode="init_score",
        target_clip=config.target_clip,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
        top_n_values=config.top_n_values,
        seed=config.seed,
    )
    frame, source_features, _, _ = _load_frame(frame_config)
    if "legacy_exact_primary_mtr" not in frame.columns:
        raise ValueError("This experiment requires `legacy_exact_primary_mtr`; use anchor_mode=legacy_exact_dominant.")
    months = frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )
    residual_source_features = [feature for feature in source_features if feature != "legacy_exact_primary_mtr"]
    residual_features: list[str] = []
    for feature in residual_source_features:
        for suffix in ["", "_rank_month", "_z_month", "_top25_flag", "_bottom25_flag"]:
            col = f"{feature}{suffix}"
            if col in frame.columns:
                residual_features.append(col)
    residual_features = list(dict.fromkeys(residual_features))

    predictions: list[pl.DataFrame] = []
    fold_rows: list[dict[str, Any]] = []
    for position, window in enumerate(windows, start=1):
        fold = f"fold_{window.fold_index:03d}"
        train_df = filter_by_months(frame, window.train_months)
        test_df = filter_by_months(frame, window.test_months)
        if train_df.is_empty() or test_df.is_empty():
            continue
        y_train = _target(train_df, config.target_clip)
        x_train = _calibration_input(train_df)
        calibration = _fit_positive_linear_calibration(x_train, y_train, config.min_calibration_slope)
        base_train = _predict_positive_linear(calibration, x_train)
        x_test = _calibration_input(test_df)
        base_test = _predict_positive_linear(calibration, x_test)
        seed = config.seed + position
        residual_model = _fit_xgb_regressor_with_base_margin(
            params=RESIDUAL_MODEL_PARAMS,
            X=_matrix(train_df, residual_features),
            y=y_train,
            base_margin=base_train,
            seed=seed,
        )
        final_test = _predict_xgb_with_base_margin(
            residual_model,
            _matrix(test_df, residual_features),
            base_margin=base_test,
        )
        residual_test = final_test - base_test
        shrink_predictions = {
            shrinkage: base_test + float(shrinkage) * residual_test for shrinkage in config.residual_shrinkages
        }
        predictions.append(
            _prediction_frame(
                test_df,
                fold=fold,
                base_prediction=base_test,
                residual_prediction=residual_test,
                final_prediction=final_test,
                shrink_predictions=shrink_predictions,
            )
        )
        fold_rows.append(
            {
                "fold": fold,
                "test_start": str(min(window.test_months)),
                "test_end": str(max(window.test_months)),
                "calibration_intercept": calibration["intercept"],
                "calibration_slope": calibration["slope"],
                "calibration_raw_slope": calibration["raw_slope"],
                "residual_feature_count": len(residual_features),
            }
        )
        print(
            f"{fold}: test={window.test_months[0]} slope={calibration['slope']:.6f} intercept={calibration['intercept']:.6f}",
            flush=True,
        )

    prediction_frame = pl.concat(predictions, how="vertical")
    fold_metrics = pl.DataFrame(fold_rows)
    scenarios: list[dict[str, pl.DataFrame]] = []
    score_map = {
        "legacy_exact_primary_mtr": "ema_raw",
        "ema_raw_calibrated_prediction": "ema_calibrated",
        "ema_raw_calibrated_residual_prediction": "ema_calibrated_residual",
    }
    for shrinkage in config.residual_shrinkages:
        suffix = str(shrinkage).replace(".", "_")
        score_map[f"ema_raw_calibrated_residual_s{suffix}_prediction"] = f"ema_calibrated_residual_s{suffix}"
    for top_n in config.top_n_values:
        for score_col, label in score_map.items():
            scenarios.append(_run_scenario(prediction_frame, score_col, f"{label}_top{top_n}", top_n))
    monthly_returns = pl.concat([scenario["monthly"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    months_out = prediction_frame.select("holding_month").unique().sort("holding_month").get_column("holding_month").to_list()
    curves = {
        scenario["monthly"].get_column("model")[0]: scenario["monthly"].select(
            "year_month",
            pl.col("portfolio_return").alias("monthly_return"),
            pl.col("n_positions").alias("n"),
        )
        for scenario in scenarios
    }
    curves["SPY"] = build_spy_curve(prediction_frame)
    curves.update(load_legacy_curves(config.legacy_monthly_returns, months_out))
    comparison = compare_backtest_curves(
        curves,
        output_path=run_dir / "comparison.html",
        title="EMA raw calibrated residual strategy",
        risk_free_rate=config.risk_free_rate,
    )
    prediction_metrics = pl.concat(
        [
            _prediction_metrics(prediction_frame, "ema_raw_calibrated_prediction", "ema_calibrated", config.top_n_values),
            *[
                _prediction_metrics(prediction_frame, score_col, label, config.top_n_values)
                for score_col, label in score_map.items()
                if label.startswith("ema_calibrated_residual")
            ],
        ],
        how="vertical",
    )
    prediction_frame.write_parquet(run_dir / "predictions.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    fold_metrics.write_csv(run_dir / "fold_metrics.csv")
    prediction_metrics.write_csv(run_dir / "prediction_metrics.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "residual_model_params": RESIDUAL_MODEL_PARAMS,
                "residual_feature_count": len(residual_features),
                "residual_shrinkages": list(config.residual_shrinkages),
                "calibration_note": "positive linear calibration from monthly EMA rank percentile to clipped future_excess_return",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(
        run_dir,
        config=config,
        comparison_metrics=comparison.metrics,
        prediction_metrics=prediction_metrics,
        fold_metrics=fold_metrics,
    )
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True).head(20))
    return run_dir


def _parse_args() -> EmaRawCalibratedResidualConfig:
    parser = argparse.ArgumentParser(description="Run raw EMA calibrated base-margin residual strategy.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--anchor-mode", choices=["legacy_exact_dominant"], default="legacy_exact_dominant")
    parser.add_argument("--target-clip", type=float, default=0.30)
    parser.add_argument("--min-train-months", type=int, default=168)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--top-n", type=int, nargs="*", default=[5, 7, 10, 20, 30, 50])
    parser.add_argument("--residual-shrinkages", type=float, nargs="*", default=[0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--min-calibration-slope", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return EmaRawCalibratedResidualConfig(
        output_dir=args.output_dir,
        legacy_monthly_returns=args.legacy_monthly_returns,
        anchor_mode=args.anchor_mode,
        target_clip=args.target_clip,
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        max_windows=args.max_windows,
        top_n_values=tuple(args.top_n),
        residual_shrinkages=tuple(args.residual_shrinkages),
        min_calibration_slope=args.min_calibration_slope,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(_parse_args())
