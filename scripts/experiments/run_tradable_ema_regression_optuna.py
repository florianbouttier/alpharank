from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import optuna
import polars as pl
from optuna.samplers import TPESampler

from alpharank.backtest.mlcraft_adapter import (
    ensure_mlcraft_importable,
    to_mlcraft_model_and_fit_params,
    to_mlcraft_search_space,
)
from alpharank.backtest.time_folds import filter_by_months, walk_forward_windows

sys.path.insert(0, str(Path(__file__).parent))

from run_ema_rich_future_target_models import _recomposition_by_month, _recomposition_summary  # noqa: E402
from run_signal_copy_models import DEFAULT_LEGACY_PATH, DEFAULT_SOURCE_RUN, _append_legacy, _load_legacy_labels  # noqa: E402


@dataclass(frozen=True)
class TradableEmaRegressionConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    warm_start_json: Path | None = None
    warm_start_top_k: int = 20
    n_trials: int = 40
    startup_trials: int = 12
    max_windows: int = 999
    min_train_months: int = 60
    val_months: int = 12
    test_months: int = 1
    step_months: int = 1
    target_clip: float = 0.30
    lambda_gap: float = 0.25
    trial_selection_policy: str = "best_objective"
    feature_set: str = "ema"
    seed: int = 42


BASE_PARAMS: dict[str, Any] = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "n_estimators": 500,
    "learning_rate": 0.02,
    "max_depth": 2,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 8.0,
    "gamma": 1.0,
    "reg_alpha": 1.0,
    "reg_lambda": 4.0,
    "n_jobs": -1,
}

SEARCH_SPACE: dict[str, tuple[str, float, float]] = {
    "n_estimators": ("int", 150, 1500),
    "learning_rate": ("loguniform", 0.002, 0.08),
    "max_depth": ("int", 1, 5),
    "subsample": ("float", 0.55, 1.0),
    "colsample_bytree": ("float", 0.55, 1.0),
    "min_child_weight": ("float", 2.0, 25.0),
    "gamma": ("float", 0.0, 8.0),
    "reg_alpha": ("float", 0.0, 8.0),
    "reg_lambda": ("float", 0.5, 12.0),
}


def _ema_base_features(columns: Iterable[str]) -> list[str]:
    return [
        column
        for column in columns
        if column.startswith("ema_ratio_") or column.startswith("price_to_ema_")
    ]


def _technical_base_features(columns: Iterable[str]) -> list[str]:
    technical_prefixes = (
        "price_roc_",
        "ema_ratio_",
        "price_to_ema_",
        "rsi_",
        "rsi_ratio_",
        "bollinger_",
        "stoch_",
        "dist_to_",
        "range_position_",
        "volatility_",
        "volatility_ratio_",
    )
    return [column for column in columns if column.startswith(technical_prefixes)]


def _base_features_for_set(columns: Iterable[str], feature_set: str) -> list[str]:
    if feature_set == "ema":
        return _ema_base_features(columns)
    if feature_set == "technical":
        return _technical_base_features(columns)
    raise ValueError(f"Unsupported feature set: {feature_set!r}")


def _score_col(config: TradableEmaRegressionConfig) -> str:
    if config.feature_set == "ema":
        return "tradable_ema_regression"
    if config.feature_set == "technical":
        return "tradable_technical_regression"
    raise ValueError(f"Unsupported feature set: {config.feature_set!r}")


def _add_cross_sectional_features(
    frame: pl.DataFrame,
    base_features: Sequence[str],
    *,
    prefix: str,
) -> tuple[pl.DataFrame, list[str]]:
    rank_cols = [f"{feature}_rank_month" for feature in base_features]
    z_cols = [f"{feature}_z_month" for feature in base_features]
    top_cols = [f"{feature}_top25_flag" for feature in base_features]
    bottom_cols = [f"{feature}_bottom25_flag" for feature in base_features]

    ranked = frame.with_columns(
        [
            (pl.col(feature).rank(method="average").over("year_month") / pl.len().over("year_month")).alias(rank_col)
            for feature, rank_col in zip(base_features, rank_cols, strict=True)
        ]
    )
    zscored = ranked.with_columns(
        [
            pl.when(pl.col(feature).std().over("year_month") > 1e-12)
            .then((pl.col(feature) - pl.col(feature).mean().over("year_month")) / pl.col(feature).std().over("year_month"))
            .otherwise(0.0)
            .alias(z_col)
            for feature, z_col in zip(base_features, z_cols, strict=True)
        ]
    )
    flagged = zscored.with_columns(
        [
            (pl.col(rank_col) >= 0.75).cast(pl.Int8).alias(top_col)
            for rank_col, top_col in zip(rank_cols, top_cols, strict=True)
        ]
        + [
            (pl.col(rank_col) <= 0.25).cast(pl.Int8).alias(bottom_col)
            for rank_col, bottom_col in zip(rank_cols, bottom_cols, strict=True)
        ]
    )
    enriched = flagged.with_columns(
        pl.mean_horizontal(rank_cols).alias(f"{prefix}_rank_mean"),
        pl.max_horizontal(rank_cols).alias(f"{prefix}_rank_max"),
        pl.min_horizontal(rank_cols).alias(f"{prefix}_rank_min"),
        pl.mean_horizontal(z_cols).alias(f"{prefix}_z_mean"),
        pl.max_horizontal(z_cols).alias(f"{prefix}_z_max"),
        pl.min_horizontal(z_cols).alias(f"{prefix}_z_min"),
        pl.sum_horizontal(top_cols).alias(f"{prefix}_top25_vote_count"),
        pl.sum_horizontal(bottom_cols).alias(f"{prefix}_bottom25_vote_count"),
    )
    features = (
        list(base_features)
        + rank_cols
        + z_cols
        + top_cols
        + bottom_cols
        + [
            f"{prefix}_rank_mean",
            f"{prefix}_rank_max",
            f"{prefix}_rank_min",
            f"{prefix}_z_mean",
            f"{prefix}_z_max",
            f"{prefix}_z_min",
            f"{prefix}_top25_vote_count",
            f"{prefix}_bottom25_vote_count",
        ]
    )
    return enriched, features


def _load_frame(config: TradableEmaRegressionConfig) -> tuple[pl.DataFrame, list[str]]:
    meta = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(meta["features_used"])
    base_features = _base_features_for_set(source_features, config.feature_set)
    if not base_features:
        raise ValueError(f"No tradable {config.feature_set} features found in source run metadata.")

    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    frame, features = _add_cross_sectional_features(frame, base_features, prefix=config.feature_set)
    legacy = _load_legacy_labels(config.legacy_path)
    return _append_legacy(frame, legacy), features


def _matrix(frame: pl.DataFrame, features: Sequence[str]) -> np.ndarray:
    return frame.select(list(features)).fill_null(0.0).to_numpy().astype(np.float32)


def _target(frame: pl.DataFrame, clip: float) -> np.ndarray:
    return np.clip(frame.get_column("future_excess_return").to_numpy(), -clip, clip).astype(np.float32)


def _legacy_overlap(frame: pl.DataFrame, score_col: str) -> float:
    recomposition = _recomposition_by_month(frame, [score_col])
    if recomposition.is_empty():
        return 0.0
    total_common = recomposition.get_column("common_count").sum()
    total_legacy = recomposition.get_column("legacy_count").sum()
    return float(total_common / total_legacy) if total_legacy else 0.0


def _load_warm_starts(path: Path | None, top_k: int) -> list[dict[str, Any]]:
    if path is None or top_k <= 0 or not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_trials = payload.get("warm_start_params", payload if isinstance(payload, list) else [])
    if not isinstance(raw_trials, list):
        return []
    return [dict(row.get("params", row)) for row in raw_trials[:top_k] if isinstance(row, dict)]


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


def _fit_mlcraft_regressor(
    *,
    params: dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    seed: int,
):
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
        ptype = spec["type"]
        if ptype == "int":
            params[name] = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
        else:
            params[name] = trial.suggest_float(
                name,
                float(spec["low"]),
                float(spec["high"]),
                log=bool(spec.get("log", False)),
            )
    return params


def _enqueue_warm_starts(study: optuna.Study, warm_starts: Sequence[dict[str, Any]], search_space: dict[str, dict[str, Any]]) -> None:
    for params in warm_starts:
        filtered: dict[str, Any] = {}
        for name, spec in search_space.items():
            if name not in params:
                continue
            value = params[name]
            filtered[name] = int(round(float(value))) if spec["type"] == "int" else float(value)
        if filtered:
            study.enqueue_trial(filtered)


def _select_trial_params(
    trials: Sequence[optuna.trial.FrozenTrial],
    *,
    policy: str,
    warm_start_count: int,
) -> dict[str, Any]:
    complete_trials = [trial for trial in trials if trial.value is not None]
    if not complete_trials:
        raise ValueError("No completed Optuna trials available for selection.")

    if policy == "best_objective":
        selected = max(complete_trials, key=lambda trial: float(trial.value))
        return dict(selected.params)

    if policy == "warm_only":
        warm_trials = [trial for trial in complete_trials if trial.number < warm_start_count]
        candidates = warm_trials or complete_trials
        selected = max(candidates, key=lambda trial: float(trial.value))
        return dict(selected.params)

    if policy == "top10_min_gap":
        top_trials = sorted(complete_trials, key=lambda trial: float(trial.value), reverse=True)[:10]

        def gap(trial: optuna.trial.FrozenTrial) -> tuple[float, float]:
            train_overlap = float(trial.user_attrs.get("train_overlap", 0.0))
            val_overlap = float(trial.user_attrs.get("val_overlap", 0.0))
            return (abs(train_overlap - val_overlap), -float(trial.value))

        selected = min(top_trials, key=gap)
        return dict(selected.params)

    raise ValueError(f"Unknown trial selection policy: {policy}")


def _score_predictions(frame: pl.DataFrame, scores: np.ndarray, score_col: str) -> pl.DataFrame:
    return frame.select(["ticker", "year_month", "holding_month", "future_excess_return", "legacy_selected"]).with_columns(
        pl.Series(score_col, scores, dtype=pl.Float64)
    )


def _tune_fold(
    *,
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    features: Sequence[str],
    config: TradableEmaRegressionConfig,
    warm_starts: Sequence[dict[str, Any]],
    seed: int,
    fold_label: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    X_train = _matrix(train_df, features)
    y_train = _target(train_df, config.target_clip)
    X_val = _matrix(val_df, features)
    y_val = _target(val_df, config.target_clip)
    mlcraft_space = to_mlcraft_search_space(SEARCH_SPACE)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=seed, n_startup_trials=config.startup_trials))
    _enqueue_warm_starts(study, warm_starts, mlcraft_space)

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(trial, mlcraft_space)
        model = _fit_mlcraft_regressor(params=params, X=X_train, y=y_train, seed=seed)
        train_scores = _predict(model, X_train)
        val_scores = _predict(model, X_val)
        train_overlap = _legacy_overlap(_score_predictions(train_df, train_scores, "_score"), "_score")
        val_overlap = _legacy_overlap(_score_predictions(val_df, val_scores, "_score"), "_score")
        objective_score = val_overlap - config.lambda_gap * abs(train_overlap - val_overlap)
        trial.set_user_attr("train_overlap", train_overlap)
        trial.set_user_attr("val_overlap", val_overlap)
        trial.set_user_attr("objective_score", objective_score)
        return float(objective_score)

    study.optimize(objective, n_trials=config.n_trials)

    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        row = {
            "fold": fold_label,
            "trial_number": trial.number,
            "objective": trial.value,
            "train_overlap": trial.user_attrs.get("train_overlap"),
            "val_overlap": trial.user_attrs.get("val_overlap"),
        }
        row.update({f"param_{key}": value for key, value in trial.params.items()})
        rows.append(row)
    best_params = _select_trial_params(
        study.trials,
        policy=config.trial_selection_policy,
        warm_start_count=len(warm_starts),
    )
    return best_params, rows


def _warm_start_payload(trials: pl.DataFrame, top_k: int = 50) -> dict[str, Any]:
    if trials.is_empty():
        return {"warm_start_params": []}
    param_cols = [column for column in trials.columns if column.startswith("param_")]
    rows = []
    for row in trials.sort("objective", descending=True, nulls_last=True).head(top_k).to_dicts():
        params = {column.removeprefix("param_"): row[column] for column in param_cols if row.get(column) is not None}
        rows.append(
            {
                "fold": row.get("fold"),
                "trial_number": row.get("trial_number"),
                "objective": row.get("objective"),
                "train_overlap": row.get("train_overlap"),
                "val_overlap": row.get("val_overlap"),
                "params": params,
            }
        )
    return {"warm_start_params": rows}


def run(config: TradableEmaRegressionConfig) -> Path:
    score_col = _score_col(config)
    run_dir = config.output_dir / f"{score_col}_optuna_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, features = _load_frame(config)
    warm_starts = _load_warm_starts(config.warm_start_json, config.warm_start_top_k)
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
        train_df = filter_by_months(frame, window.train_months).filter(pl.col("future_excess_return").is_not_null())
        val_df = filter_by_months(frame, window.val_months).filter(pl.col("future_excess_return").is_not_null())
        test_df = filter_by_months(frame, window.test_months).filter(pl.col("future_excess_return").is_not_null())
        if train_df.height < 250 or val_df.height < 80 or test_df.height < 80:
            continue
        print(f"[{position}/{len(windows)}] {fold_label} train={train_df.height} val={val_df.height} test={test_df.height}", flush=True)
        seed = config.seed + int(window.fold_index)
        best_params, fold_trials = _tune_fold(
            train_df=train_df,
            val_df=val_df,
            features=features,
            config=config,
            warm_starts=warm_starts,
            seed=seed,
            fold_label=fold_label,
        )
        trial_rows.extend(fold_trials)

        fit_df = pl.concat([train_df, val_df], how="vertical")
        model = _fit_mlcraft_regressor(
            params=best_params,
            X=_matrix(fit_df, features),
            y=_target(fit_df, config.target_clip),
            seed=seed,
        )
        test_predictions = _score_predictions(test_df, _predict(model, _matrix(test_df, features)), score_col)
        prediction_frames.append(test_predictions.with_columns(pl.lit(fold_label).alias("fold")))

        val_model = _fit_mlcraft_regressor(
            params=best_params,
            X=_matrix(train_df, features),
            y=_target(train_df, config.target_clip),
            seed=seed,
        )
        val_overlap = _legacy_overlap(_score_predictions(val_df, _predict(val_model, _matrix(val_df, features)), "_score"), "_score")
        test_overlap = _legacy_overlap(test_predictions, score_col)
        fold_rows.append(
            {
                "fold": fold_label,
                "val_overlap": val_overlap,
                "test_overlap": test_overlap,
                "train_month_start": str(window.train_months[0]),
                "train_month_end": str(window.train_months[-1]),
                "val_month_start": str(window.val_months[0]),
                "val_month_end": str(window.val_months[-1]),
                "test_month_start": str(window.test_months[0]),
                "test_month_end": str(window.test_months[-1]),
                **{f"best_{key}": value for key, value in best_params.items()},
            }
        )

    predictions = pl.concat(prediction_frames, how="vertical") if prediction_frames else pl.DataFrame()
    recomposition = _recomposition_by_month(predictions, [score_col]) if not predictions.is_empty() else pl.DataFrame()
    summary = _recomposition_summary(recomposition) if not recomposition.is_empty() else pl.DataFrame()
    trials = pl.DataFrame(trial_rows) if trial_rows else pl.DataFrame()
    folds = pl.DataFrame(fold_rows) if fold_rows else pl.DataFrame()

    predictions.write_parquet(run_dir / "predictions.parquet")
    recomposition.write_csv(run_dir / "recomposition_by_month.csv")
    summary.write_csv(run_dir / "recomposition_summary.csv")
    trials.write_csv(run_dir / "optuna_trials.csv")
    folds.write_csv(run_dir / "fold_metrics.csv")
    (run_dir / "warm_start_candidates.json").write_text(json.dumps(_warm_start_payload(trials), indent=2, default=str))
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "score_col": score_col,
                "feature_set": config.feature_set,
                "warm_start_json": str(config.warm_start_json) if config.warm_start_json else None,
                "warm_start_loaded": len(warm_starts),
                "warm_start_top_k": config.warm_start_top_k,
                "n_trials": config.n_trials,
                "startup_trials": config.startup_trials,
                "max_windows": config.max_windows,
                "min_train_months": config.min_train_months,
                "val_months": config.val_months,
                "test_months": config.test_months,
                "step_months": config.step_months,
                "lambda_gap": config.lambda_gap,
                "trial_selection_policy": config.trial_selection_policy,
                "sampler": "TPESampler with random startup trials, warm starts enqueued first",
                "target": f"future_excess_return clipped to +/-{config.target_clip}",
                "features": features,
                "feature_policy": (
                    "tradable EMA-only features; no legacy_atomic, no legacy_optuna, no legacy_selected in features"
                    if config.feature_set == "ema"
                    else "tradable technical features; no fundamentals, no legacy_atomic, no legacy_optuna, no legacy_selected in features"
                ),
                "primary_metric": "nombre d'actions communes entre regression et Legacy / nombre d'actions choisies par Legacy",
                "base_params": BASE_PARAMS,
                "search_space": SEARCH_SPACE,
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(summary)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tradable EMA-only future-excess-return regression tuned on Legacy overlap.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--warm-start-json", type=Path, default=None)
    parser.add_argument("--warm-start-top-k", type=int, default=20)
    parser.add_argument("--n-trials", type=int, default=40)
    parser.add_argument("--startup-trials", type=int, default=12)
    parser.add_argument("--max-windows", type=int, default=999)
    parser.add_argument("--min-train-months", type=int, default=60)
    parser.add_argument("--val-months", type=int, default=12)
    parser.add_argument("--test-months", type=int, default=1)
    parser.add_argument("--step-months", type=int, default=1)
    parser.add_argument("--target-clip", type=float, default=0.30)
    parser.add_argument("--lambda-gap", type=float, default=0.25)
    parser.add_argument(
        "--trial-selection-policy",
        choices=["best_objective", "warm_only", "top10_min_gap"],
        default="best_objective",
    )
    parser.add_argument("--feature-set", choices=["ema", "technical"], default="ema")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        TradableEmaRegressionConfig(
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            output_dir=args.output_dir,
            warm_start_json=args.warm_start_json,
            warm_start_top_k=args.warm_start_top_k,
            n_trials=args.n_trials,
            startup_trials=args.startup_trials,
            max_windows=args.max_windows,
            min_train_months=args.min_train_months,
            val_months=args.val_months,
            test_months=args.test_months,
            step_months=args.step_months,
            target_clip=args.target_clip,
            lambda_gap=args.lambda_gap,
            trial_selection_policy=args.trial_selection_policy,
            feature_set=args.feature_set,
        )
    )


if __name__ == "__main__":
    main()
