from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import optuna
import polars as pl
from run_ema_anchor_residual_strategy import EmaAnchorResidualConfig, _load_frame  # noqa: E402
from run_portfolio_boosting_rank_regression import (  # noqa: E402
    _fit_mlcraft_regressor,
    _load_warm_starts,
    _matrix,
    _portfolio_metric,
    _predict,
    _run_model_scenario,
    _scored_frame,
    _target,
    _tune_fold,
    _write_warm_start_candidates,
)
from run_signal_copy_models import (  # noqa: E402
    DEFAULT_LEGACY_PATH,
    _append_legacy,
    _load_legacy_labels,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.portfolio import select_top_n
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows


@dataclass(frozen=True)
class PortfolioBoostingExactEmaRankConfig:
    output_dir: Path = Path("outputs")
    legacy_path: Path = DEFAULT_LEGACY_PATH
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    anchor_mode: str = "validation_exact"
    feature_mode: str = "exact_ema_family"
    score_col: str = "portfolio_boosting_exact_ema_rank"
    n_trials: int = 4
    startup_trials: int = 2
    max_windows: int = 999
    min_train_months: int = 168
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    objective_top_k: int = 10
    objective_mode: str = "mean_return"
    lambda_gap: float = 0.0
    top_n_values: tuple[int, ...] = (5, 7, 10, 20, 30, 50)
    warm_start_path: Path | None = None
    warm_start_output_count: int = 30
    risk_free_rate: float = 0.02
    seed: int = 42


def _feature_family(frame: pl.DataFrame, feature: str) -> list[str]:
    candidates = [
        feature,
        f"{feature}_rank_month",
        f"{feature}_z_month",
        f"{feature}_top25_flag",
        f"{feature}_bottom25_flag",
    ]
    return [col for col in candidates if col in frame.columns]


def _select_features(frame: pl.DataFrame, ema_features: Sequence[str], mode: str) -> list[str]:
    if mode != "exact_ema_family":
        raise ValueError(f"Unsupported feature_mode={mode!r}.")
    features: list[str] = []
    for feature in ema_features:
        if feature.startswith("legacy_ema_ratio_short"):
            features.extend(_feature_family(frame, feature))
    features = list(dict.fromkeys(features))
    if not features:
        raise ValueError("No exact EMA feature family found for boosting.")
    return features


def _load_exact_ema_rank_frame(config: PortfolioBoostingExactEmaRankConfig) -> tuple[pl.DataFrame, list[str]]:
    anchor_config = EmaAnchorResidualConfig(
        anchor_mode=config.anchor_mode,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
        seed=config.seed,
    )
    frame, _, ema_features, _ = _load_frame(anchor_config)
    frame = _append_legacy(frame, _load_legacy_labels(config.legacy_path))
    frame = frame.with_columns(
        (
            pl.col("future_excess_return").rank(method="average").over("year_month")
            / pl.len().over("year_month")
        ).alias("future_excess_rank_target")
    )
    features = _select_features(frame, ema_features, config.feature_mode)
    return frame, features


def _prediction_metrics(predictions: pl.DataFrame, score_col: str, top_n_values: Sequence[int]) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for top_n in top_n_values:
        selected = select_top_n(
            predictions.with_columns(
                pl.col(score_col).alias("prediction"),
                (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
            ),
            top_n=top_n,
        )
        if selected.is_empty():
            continue
        monthly = (
            selected.group_by("year_month")
            .agg(
                pl.mean("future_excess_return").alias("avg_excess"),
                (pl.col("future_excess_return") > 0.0).mean().alias("hit_rate"),
            )
            .sort("year_month")
        )
        rows.extend(
            [
                {
                    "top_n": top_n,
                    "metric": "avg_monthly_future_excess_return",
                    "value": float(monthly.get_column("avg_excess").mean()),
                },
                {"top_n": top_n, "metric": "hit_rate_gt0", "value": float(monthly.get_column("hit_rate").mean())},
            ]
        )
    return pl.DataFrame(rows)


def _write_exact_report(
    run_dir: Path,
    comparison_metrics: pl.DataFrame,
    prediction_metrics: pl.DataFrame,
    config: PortfolioBoostingExactEmaRankConfig,
) -> None:
    lines = [
        "# Portfolio boosting exact EMA rank",
        "",
        "But: tester du boosting seul sur les familles EMA exactes.",
        "",
        "Contraintes:",
        "",
        "- score final = prediction du modele `mlcraft` XGBoost uniquement ;",
        "- pas de score EMA brut comme sortie finale ;",
        "- pas de blend momentum ;",
        "- pas d'objectif Legacy dans l'entrainement.",
        "",
        f"Target: rang percentile mensuel futur de `future_excess_return`.",
        f"Feature mode: `{config.feature_mode}`.",
        f"Objectif Optuna validation: `{config.objective_mode}` sur top `{config.objective_top_k}`.",
        "",
        "## Backtest",
        "",
        "| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle | mois positifs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in comparison_metrics.sort("Total Return", descending=True).to_dicts():
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% | {row['Positive Periods %'] * 100:.1f}% |"
        )
    lines.extend(["", "## Top-K prediction", "", "| top N | metrique | valeur |", "|---:|---|---:|"])
    for row in prediction_metrics.to_dicts():
        value = row["value"]
        formatted = f"{value * 100:.2f}%" if row["metric"] in {"avg_monthly_future_excess_return", "hit_rate_gt0"} else f"{value:.4f}"
        lines.append(f"| {row['top_n']} | `{row['metric']}` | {formatted} |")
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: PortfolioBoostingExactEmaRankConfig) -> Path:
    run_dir = config.output_dir / f"portfolio_boosting_exact_ema_rank_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, features = _load_exact_ema_rank_frame(config)
    warm_starts = _load_warm_starts(config.warm_start_path)
    months = frame.select("year_month").unique().sort("year_month").get_column("year_month").to_list()
    windows = walk_forward_windows(
        months,
        min_train_months=config.min_train_months,
        val_months=config.val_months,
        test_months=config.test_months,
        step_months=config.step_months,
        max_windows=config.max_windows,
    )

    prediction_frames: list[pl.DataFrame] = []
    trial_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []

    for position, window in enumerate(windows, start=1):
        fold_label = f"fold_{window.fold_index:03d}"
        train_df = filter_by_months(frame, window.train_months)
        val_df = filter_by_months(frame, window.val_months)
        test_df = filter_by_months(frame, window.test_months)
        if train_df.is_empty() or val_df.is_empty() or test_df.is_empty():
            continue
        seed = config.seed + position
        best_params, rows = _tune_fold(
            train_df=train_df,
            val_df=val_df,
            features=features,
            config=config,
            warm_starts=warm_starts,
            seed=seed,
            fold_label=fold_label,
        )
        trial_rows.extend(rows)
        fit_df = pl.concat([train_df, val_df], how="vertical")
        model = _fit_mlcraft_regressor(params=best_params, X=_matrix(fit_df, features), y=_target(fit_df), seed=seed)
        test_scores = _predict(model, _matrix(test_df, features))
        test_predictions = _scored_frame(test_df, test_scores, config.score_col).with_columns(pl.lit(fold_label).alias("fold"))
        prediction_frames.append(test_predictions)
        fold_metric = _portfolio_metric(test_predictions, config.score_col, config.objective_top_k, config.objective_mode)
        fold_rows.append(
            {
                "fold": fold_label,
                "train_start": str(min(window.train_months)),
                "train_end": str(max(window.train_months)),
                "val_start": str(min(window.val_months)),
                "val_end": str(max(window.val_months)),
                "test_start": str(min(window.test_months)),
                "test_end": str(max(window.test_months)),
                "test_metric": fold_metric,
                **{f"param_{key}": value for key, value in best_params.items()},
            }
        )
        print(f"{fold_label}: test={window.test_months[0]} metric={fold_metric:.4f}", flush=True)

    predictions = pl.concat(prediction_frames, how="vertical")
    predictions.write_parquet(run_dir / "predictions.parquet")
    pl.DataFrame(trial_rows).write_csv(run_dir / "optuna_trials.csv")
    pl.DataFrame(fold_rows).write_csv(run_dir / "fold_metrics.csv")
    _write_warm_start_candidates(run_dir, trial_rows, limit=config.warm_start_output_count)

    scenarios = [
        _run_model_scenario(
            predictions,
            config.score_col,
            f"portfolio_boosting_exact_ema_rank_top_{top_n}",
            top_n,
            config.risk_free_rate,
        )
        for top_n in config.top_n_values
    ]
    monthly_returns = pl.concat([scenario["monthly_returns"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    model_kpis = pl.concat([scenario["kpis"] for scenario in scenarios], how="vertical")

    months_out = (
        predictions.select(pl.col("holding_month").alias("year_month"))
        .unique()
        .sort("year_month")
        .get_column("year_month")
        .to_list()
    )
    comparison_inputs = {
        scenario["name"]: scenario["monthly_returns"].select(
            "year_month",
            pl.col("portfolio_return").alias("monthly_return"),
            pl.col("n_positions").alias("n"),
        )
        for scenario in scenarios
    }
    comparison_inputs["SPY"] = build_spy_curve(predictions)
    comparison_inputs.update(load_legacy_curves(config.legacy_monthly_returns, months_out))
    comparison = compare_backtest_curves(
        comparison_inputs,
        output_path=run_dir / "comparison.html",
        title="Portfolio boosting exact EMA rank vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )
    prediction_metrics = _prediction_metrics(predictions, config.score_col, config.top_n_values)

    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    model_kpis.write_csv(run_dir / "model_kpis.csv")
    prediction_metrics.write_csv(run_dir / "prediction_metrics.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "feature_count": len(features),
                "features": features,
                "target": "future_excess_rank_target",
                "score_semantics": "Final portfolio score is mlcraft XGBoost prediction only.",
                "warm_start_path": str(config.warm_start_path) if config.warm_start_path else None,
                "warm_start_count": len(warm_starts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_exact_report(run_dir, comparison.metrics, prediction_metrics, config)
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True))
    return run_dir


def _parse_args() -> PortfolioBoostingExactEmaRankConfig:
    parser = argparse.ArgumentParser(description="Train a mlcraft boosting rank model on exact EMA feature families only.")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--anchor-mode", choices=["validation_exact"], default="validation_exact")
    parser.add_argument("--feature-mode", choices=["exact_ema_family"], default="exact_ema_family")
    parser.add_argument("--n-trials", type=int, default=4)
    parser.add_argument("--startup-trials", type=int, default=2)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--min-train-months", type=int, default=168)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--objective-top-k", type=int, default=10)
    parser.add_argument(
        "--objective-mode",
        choices=["mean_return", "mean_active", "sharpe_return", "sharpe_active"],
        default="mean_return",
    )
    parser.add_argument("--lambda-gap", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, nargs="*", default=[5, 7, 10, 20, 30, 50])
    parser.add_argument("--warm-start-path", type=Path, default=None)
    parser.add_argument("--warm-start-output-count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PortfolioBoostingExactEmaRankConfig(
        output_dir=args.output_dir,
        legacy_path=args.legacy_path,
        legacy_monthly_returns=args.legacy_monthly_returns,
        anchor_mode=args.anchor_mode,
        feature_mode=args.feature_mode,
        n_trials=args.n_trials,
        startup_trials=args.startup_trials,
        max_windows=args.max_windows,
        min_train_months=args.min_train_months,
        val_months=args.val_months,
        test_months=args.test_months,
        step_months=args.step_months,
        objective_top_k=args.objective_top_k,
        objective_mode=args.objective_mode,
        lambda_gap=args.lambda_gap,
        top_n_values=tuple(args.top_n),
        warm_start_path=args.warm_start_path,
        warm_start_output_count=args.warm_start_output_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    run(_parse_args())
