from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler
from run_signal_copy_models import (  # noqa: E402
    DEFAULT_LEGACY_PATH,
    DEFAULT_SOURCE_RUN,
    _append_legacy,
    _load_legacy_labels,
)
from run_tradable_ema_regression_optuna import (  # noqa: E402
    _add_cross_sectional_features,
    _base_features_for_set,
)
from run_tradable_ema_regression_trading_backtest import (  # noqa: E402
    DEFAULT_LEGACY_MONTHLY_RETURNS,
    build_spy_curve,
    load_legacy_curves,
)

from alpharank.backtest.application import compare_backtest_curves
from alpharank.backtest.kpis import compute_backtest_kpis
from alpharank.backtest.mlcraft_adapter import (
    ensure_mlcraft_importable,
    to_mlcraft_model_and_fit_params,
    to_mlcraft_search_space,
)
from alpharank.backtest.portfolio import compute_monthly_portfolio_returns, select_top_n
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows


@dataclass(frozen=True)
class PortfolioBoostingRankRegressionConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    legacy_monthly_returns: Path = DEFAULT_LEGACY_MONTHLY_RETURNS
    output_dir: Path = Path("outputs")
    feature_set: str = "technical"
    score_col: str = "portfolio_boosting_rank_regression"
    n_trials: int = 2
    startup_trials: int = 1
    max_windows: int = 999
    min_train_months: int = 168
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    objective_top_k: int = 5
    objective_mode: str = "mean_return"
    lambda_gap: float = 0.0
    top_n_values: tuple[int, ...] = (5, 7, 10, 20, 30, 50)
    warm_start_path: Path | None = None
    warm_start_output_count: int = 20
    risk_free_rate: float = 0.02
    seed: int = 42


BASE_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 180,
    "learning_rate": 0.03,
    "max_depth": 2,
    "subsample": 0.75,
    "colsample_bytree": 0.70,
    "min_child_weight": 10.0,
    "gamma": 1.0,
    "reg_alpha": 2.0,
    "reg_lambda": 8.0,
    "n_jobs": -1,
}

SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "n_estimators": ("int", 60, 320),
    "learning_rate": ("loguniform", 0.004, 0.08),
    "max_depth": ("int", 1, 4),
    "subsample": ("float", 0.55, 0.95),
    "colsample_bytree": ("float", 0.45, 0.95),
    "min_child_weight": ("float", 3.0, 30.0),
    "gamma": ("float", 0.0, 10.0),
    "reg_alpha": ("float", 0.0, 12.0),
    "reg_lambda": ("float", 1.0, 20.0),
}


def _load_frame(config: PortfolioBoostingRankRegressionConfig) -> tuple[pl.DataFrame, list[str]]:
    metadata = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(metadata["features_used"])
    base_features = _base_features_for_set(source_features, config.feature_set)
    if not base_features:
        raise ValueError(f"No features found for feature_set={config.feature_set!r}.")

    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    frame, features = _add_cross_sectional_features(frame, base_features, prefix=config.feature_set)
    frame = _append_legacy(frame, _load_legacy_labels(config.legacy_path))
    frame = frame.filter(pl.col("future_return").is_not_null(), pl.col("future_excess_return").is_not_null())
    frame = frame.with_columns(
        (
            pl.col("future_excess_return").rank(method="average").over("year_month")
            / pl.len().over("year_month")
        ).alias("future_excess_rank_target")
    )
    return frame, features


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).fill_null(0.0).to_numpy().astype(np.float32)


def _target(frame: pl.DataFrame) -> np.ndarray:
    return frame.get_column("future_excess_rank_target").to_numpy().astype(np.float32)


def _split_mlcraft_params(
    params: dict[str, Any],
    search_space: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    model_params: dict[str, Any] = {}
    fit_params: dict[str, Any] = {}
    for name, value in params.items():
        target = str(search_space.get(name, {}).get("target", "model"))
        if target == "fit":
            fit_params[name] = value
        else:
            model_params[name] = value
    return model_params, fit_params


def _fit_mlcraft_regressor(*, params: dict[str, Any], X: np.ndarray, y: np.ndarray, seed: int):
    ensure_mlcraft_importable()
    from mlcraft.core.task import TaskSpec
    from mlcraft.models.factory import ModelFactory

    base_model_params, base_fit_params = to_mlcraft_model_and_fit_params(BASE_PARAMS)
    trial_model_params, trial_fit_params = _split_mlcraft_params(params, to_mlcraft_search_space(SEARCH_SPACE))
    fit_params = {**base_fit_params, **trial_fit_params}
    if "num_boost_round" not in fit_params:
        fit_params["num_boost_round"] = int(BASE_PARAMS["n_estimators"])
    model = ModelFactory.create(
        "xgboost",
        task_spec=TaskSpec(task_type="regression"),
        model_params={**base_model_params, **trial_model_params},
        fit_params=fit_params,
        random_state=seed,
    )
    model.fit(X, y)
    return model


def _predict(model: Any, X: np.ndarray) -> np.ndarray:
    pred = model.predict(X)
    if isinstance(pred, tuple):
        pred = pred[0]
    return np.asarray(pred, dtype=float).reshape(-1)


def _sample_params(trial: optuna.Trial, search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
    return params


def _base_trial_params(search_space: dict[str, dict[str, Any]]) -> dict[str, Any]:
    base_model_params, base_fit_params = to_mlcraft_model_and_fit_params(BASE_PARAMS)
    raw = {**base_model_params, **base_fit_params}
    trial_params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if name not in raw:
            continue
        value = raw[name]
        trial_params[name] = int(value) if spec["type"] == "int" else float(value)
    return trial_params


def _load_warm_starts(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        candidates = payload.get("candidates", payload.get("warm_starts", []))
    else:
        candidates = payload
    warm_starts: list[dict[str, Any]] = []
    for candidate in candidates:
        params = candidate.get("params", candidate) if isinstance(candidate, dict) else {}
        if isinstance(params, dict):
            warm_starts.append(params)
    return warm_starts


def _enqueue_warm_starts(
    study: optuna.Study,
    warm_starts: Sequence[dict[str, Any]],
    search_space: dict[str, dict[str, Any]],
) -> None:
    study.enqueue_trial(_base_trial_params(search_space))
    for params in warm_starts:
        filtered: dict[str, Any] = {}
        for name, spec in search_space.items():
            if name not in params:
                continue
            value = params[name]
            filtered[name] = int(round(float(value))) if spec["type"] == "int" else float(value)
        if filtered:
            study.enqueue_trial(filtered)


def _write_warm_start_candidates(
    run_dir: Path,
    trial_rows: Sequence[dict[str, Any]],
    *,
    limit: int,
) -> None:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, Any], ...]] = set()
    sorted_rows = sorted(
        (row for row in trial_rows if row.get("objective") is not None),
        key=lambda row: float(row["objective"]),
        reverse=True,
    )
    for row in sorted_rows:
        params = {key.removeprefix("param_"): value for key, value in row.items() if key.startswith("param_")}
        signature = tuple(sorted(params.items()))
        if not params or signature in seen:
            continue
        seen.add(signature)
        candidates.append(
            {
                "fold": row.get("fold"),
                "objective": row.get("objective"),
                "train_metric": row.get("train_metric"),
                "val_metric": row.get("val_metric"),
                "params": params,
            }
        )
        if len(candidates) >= limit:
            break
    (run_dir / "warm_start_candidates.json").write_text(
        json.dumps({"candidates": candidates}, indent=2),
        encoding="utf-8",
    )


def _scored_frame(frame: pl.DataFrame, scores: np.ndarray, score_col: str) -> pl.DataFrame:
    return frame.select(
        [
            "ticker",
            "year_month",
            "decision_month",
            "decision_asof_date",
            "holding_month",
            "future_return",
            "benchmark_future_return",
            "future_excess_return",
            "future_excess_rank_target",
            "legacy_selected",
        ]
    ).with_columns(pl.Series(score_col, scores, dtype=pl.Float64))


def _topk_monthly(scored: pl.DataFrame, score_col: str, top_k: int) -> pl.DataFrame:
    selected = (
        scored.with_columns(pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("rank"))
        .filter(pl.col("rank") <= int(top_k))
        .sort(["year_month", "rank"])
        .with_columns(
            pl.col(score_col).alias("prediction"),
            (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
        )
    )
    return compute_monthly_portfolio_returns(selected)


def _portfolio_metric(scored: pl.DataFrame, score_col: str, top_k: int, mode: str) -> float:
    monthly = _topk_monthly(scored, score_col, top_k)
    if monthly.is_empty():
        return -999.0
    returns = monthly.get_column("portfolio_return").to_numpy().astype(float)
    active = monthly.get_column("active_return").to_numpy().astype(float)
    if mode == "mean_return":
        return float(np.nanmean(returns))
    if mode == "mean_active":
        return float(np.nanmean(active))
    if mode == "sharpe_return":
        return float(np.nanmean(returns) / (np.nanstd(returns) + 1e-8))
    if mode == "sharpe_active":
        return float(np.nanmean(active) / (np.nanstd(active) + 1e-8))
    raise ValueError(f"Unsupported objective_mode={mode!r}")


def _tune_fold(
    *,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: Sequence[str],
    config: PortfolioBoostingRankRegressionConfig,
    warm_starts: Sequence[dict[str, Any]],
    seed: int,
    fold_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X_train = _matrix(train_df, features)
    y_train = _target(train_df)
    X_val = _matrix(val_df, features)
    search_space = to_mlcraft_search_space(SEARCH_SPACE)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, n_startup_trials=config.startup_trials))
    _enqueue_warm_starts(study, warm_starts, search_space)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, search_space)
        model = _fit_mlcraft_regressor(params=params, X=X_train, y=y_train, seed=seed)
        train_scores = _predict(model, X_train)
        val_scores = _predict(model, X_val)
        train_scored = _scored_frame(train_df, train_scores, "_score")
        val_scored = _scored_frame(val_df, val_scores, "_score")
        train_metric = _portfolio_metric(train_scored, "_score", config.objective_top_k, config.objective_mode)
        val_metric = _portfolio_metric(val_scored, "_score", config.objective_top_k, config.objective_mode)
        score = val_metric - config.lambda_gap * abs(train_metric - val_metric)
        trial.set_user_attr("train_metric", train_metric)
        trial.set_user_attr("val_metric", val_metric)
        return float(score)

    study.optimize(objective, n_trials=config.n_trials)
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "fold": fold_label,
            "trial_number": trial.number,
            "objective": trial.value,
            "train_metric": trial.user_attrs.get("train_metric"),
            "val_metric": trial.user_attrs.get("val_metric"),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    complete = [trial for trial in study.trials if trial.value is not None]
    if not complete:
        raise ValueError(f"No complete trial for {fold_label}")
    return dict(max(complete, key=lambda trial: float(trial.value)).params), rows


def _run_model_scenario(predictions: pl.DataFrame, score_col: str, name: str, top_n: int, risk_free_rate: float) -> dict[str, Any]:
    application = predictions.with_columns(
        pl.col(score_col).alias("prediction"),
        (pl.col("future_excess_return") > 0.0).cast(pl.Int8).alias("target_label"),
    )
    selections = select_top_n(application, top_n=top_n)
    monthly_returns = compute_monthly_portfolio_returns(selections)
    kpis = compute_backtest_kpis(monthly_returns, risk_free_rate=risk_free_rate).with_columns(pl.lit(name).alias("model"))
    return {
        "name": name,
        "selections": selections.with_columns(pl.lit(name).alias("model")),
        "monthly_returns": monthly_returns.with_columns(pl.lit(name).alias("model")),
        "kpis": kpis,
    }


def _write_report(run_dir: Path, comparison_metrics: pl.DataFrame, config: PortfolioBoostingRankRegressionConfig) -> None:
    rows = comparison_metrics.sort("Total Return", descending=True).to_dicts()
    lines = [
        "# Portfolio boosting rank regression",
        "",
        "But: faire marcher le boosting seul en apprenant le rang mensuel futur du rendement relatif.",
        "",
        "Target: percentile mensuel de `future_excess_return`.",
        f"Objectif Optuna validation: `{config.objective_mode}` sur top `{config.objective_top_k}`.",
        f"Feature set: `{config.feature_set}`.",
        "",
        "## Backtest",
        "",
        "| modele | total return | CAGR | Sharpe | max drawdown | vol mensuelle |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row['Total Return'] * 100:.1f}% | {row['CAGR'] * 100:.1f}% | "
            f"{row['Sharpe Ratio']:.2f} | {row['Max Drawdown'] * 100:.1f}% | "
            f"{row['Monthly Volatility'] * 100:.1f}% |"
        )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: PortfolioBoostingRankRegressionConfig) -> Path:
    run_dir = config.output_dir / f"portfolio_boosting_rank_regression_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, features = _load_frame(config)
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
        test_predictions = _scored_frame(test_df, _predict(model, _matrix(test_df, features)), config.score_col).with_columns(
            pl.lit(fold_label).alias("fold")
        )
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

    predictions = pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame()
    predictions.write_parquet(run_dir / "predictions.parquet")
    pl.DataFrame(trial_rows).write_csv(run_dir / "optuna_trials.csv")
    pl.DataFrame(fold_rows).write_csv(run_dir / "fold_metrics.csv")
    _write_warm_start_candidates(run_dir, trial_rows, limit=config.warm_start_output_count)

    scenarios = [
        _run_model_scenario(
            predictions,
            config.score_col,
            f"portfolio_boosting_rank_regression_top_{top_n}",
            top_n,
            config.risk_free_rate,
        )
        for top_n in config.top_n_values
    ]
    monthly_returns = pl.concat([scenario["monthly_returns"] for scenario in scenarios], how="vertical")
    selections = pl.concat([scenario["selections"] for scenario in scenarios], how="diagonal_relaxed")
    model_kpis = pl.concat([scenario["kpis"] for scenario in scenarios], how="vertical")

    months_out = predictions.select(pl.col("holding_month").alias("year_month")).unique().sort("year_month").get_column("year_month").to_list()
    comparison_inputs: dict[str, pl.DataFrame] = {
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
        title="Portfolio boosting rank regression vs Legacy",
        risk_free_rate=config.risk_free_rate,
    )

    monthly_returns.write_parquet(run_dir / "monthly_returns.parquet")
    selections.write_parquet(run_dir / "selections.parquet")
    model_kpis.write_csv(run_dir / "model_kpis.csv")
    comparison.metrics.write_csv(run_dir / "comparison_metrics.csv")
    comparison.annual_returns.write_csv(run_dir / "annual_returns.csv")
    comparison.correlation_matrix.write_csv(run_dir / "correlation_matrix.csv")
    comparison.worst_periods.write_csv(run_dir / "worst_periods.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "features": features,
                "feature_count": len(features),
                "warm_start_path": str(config.warm_start_path) if config.warm_start_path else None,
                "warm_start_count": len(warm_starts),
                "months": len(months_out),
                "start_month": str(min(months_out)) if months_out else None,
                "end_month": str(max(months_out)) if months_out else None,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(run_dir, comparison.metrics, config)
    print(f"RUN_DIR={run_dir}")
    print(comparison.metrics.sort("Total Return", descending=True))
    return run_dir


def _parse_args() -> PortfolioBoostingRankRegressionConfig:
    parser = argparse.ArgumentParser(description="Train a mlcraft boosting rank-regression portfolio model.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--legacy-monthly-returns", type=Path, default=DEFAULT_LEGACY_MONTHLY_RETURNS)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=["ema", "technical"], default="technical")
    parser.add_argument("--n-trials", type=int, default=2)
    parser.add_argument("--startup-trials", type=int, default=1)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--min-train-months", type=int, default=168)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--objective-top-k", type=int, default=5)
    parser.add_argument(
        "--objective-mode",
        choices=["mean_return", "mean_active", "sharpe_return", "sharpe_active"],
        default="mean_return",
    )
    parser.add_argument("--lambda-gap", type=float, default=0.0)
    parser.add_argument("--top-n", type=int, nargs="*", default=[5, 7, 10, 20, 30, 50])
    parser.add_argument("--warm-start-path", type=Path, default=None)
    parser.add_argument("--warm-start-output-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return PortfolioBoostingRankRegressionConfig(
        source_run=args.source_run,
        legacy_path=args.legacy_path,
        legacy_monthly_returns=args.legacy_monthly_returns,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
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
