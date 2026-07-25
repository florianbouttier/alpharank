from __future__ import annotations

import json
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from alpharank.multihorizon.config import MultiHorizonConfig
from alpharank.multihorizon.data import build_research_frame
from alpharank.multihorizon.explain import compute_shap_sample, write_shap_outputs
from alpharank.multihorizon.metrics import score_predictions
from alpharank.multihorizon.modeling import fit_booster
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.splits import horizon_walk_forward_windows
from alpharank.multihorizon.tuning import tune_with_purged_cpcv


def _jsonable_config(config: MultiHorizonConfig) -> dict:
    payload = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in payload.items()
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _eligible(frame: pl.DataFrame, method: str, horizon: int) -> pl.DataFrame:
    target = "legacy_selected" if method == "teacher" else f"future_excess_return_{horizon}m"
    eligible = frame.filter(pl.col(target).is_not_null())
    if method == "teacher":
        eligible = eligible.filter(
            (pl.col("legacy_label_available") == 1)
            & pl.col("future_excess_return_1m").is_not_null()
        )
    return eligible.sort(["decision_month", "ticker"])


def run_multihorizon_research(config: MultiHorizonConfig) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_dir or config.output_dir / "multihorizon_boosting" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    research = build_research_frame(
        data_dir=config.data_dir,
        legacy_detailed_returns_path=config.legacy_detailed_returns_path,
        horizons=tuple(sorted(set(config.horizons) | {1})),
        start_month=config.start_month,
        excluded_tickers=config.excluded_tickers,
    )
    frame = research.frame
    if config.save_research_frame:
        frame.write_parquet(run_dir / "research_frame.parquet")
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": _jsonable_config(config),
        "input_paths": {
            key: {"path": str(value), "sha256": _sha256(value)}
            for key, value in research.input_paths.items()
        },
        "legacy_detailed_returns_path": str(config.legacy_detailed_returns_path),
        "legacy_detailed_returns_sha256": _sha256(config.legacy_detailed_returns_path),
        "legacy_monthly_returns_path": str(config.legacy_monthly_returns_path),
        "legacy_monthly_returns_sha256": _sha256(config.legacy_monthly_returns_path),
        "feature_count": len(research.feature_columns),
        "research_frame_rows": frame.height,
        "research_frame_columns": frame.width,
        "relative_ema_pairs": research.relative_ema_pairs,
        "protocol": {
            "outer": "strict expanding walk-forward; fixed model over each test block",
            "inner": "horizon-purged CPCV on pre-test data only",
            "preprocessing": "sparse filtering and fallback medians fitted inside each fold",
            "teacher": "Legacy Combined_Frequency basket at decision t for holding t+1",
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    all_summary: list[dict] = []
    for method in config.methods:
        method_horizons = (1,) if method == "teacher" else config.horizons
        for horizon in method_horizons:
            combination_dir = run_dir / f"{method}_h{horizon:02d}"
            combination_dir.mkdir(parents=True, exist_ok=True)
            panel = _eligible(frame, method, horizon)
            all_months = panel["decision_month"].unique().sort().to_list()
            windows = horizon_walk_forward_windows(
                all_months,
                horizon=1 if method == "teacher" else horizon,
                min_train_months=config.min_train_months,
                validation_months=config.validation_months,
                test_months=config.test_months,
                step_months=config.step_months,
                max_windows=config.max_windows,
            )
            prediction_parts: list[pl.DataFrame] = []
            portfolio_parts: list[pl.DataFrame] = []
            fold_rows: list[dict] = []
            shap_parts: list[pl.DataFrame] = []
            for window in windows:
                train = panel.filter(pl.col("decision_month").is_in(window.train_months))
                validation = panel.filter(pl.col("decision_month").is_in(window.validation_months))
                test = panel.filter(pl.col("decision_month").is_in(window.test_months))
                if train.is_empty() or validation.is_empty() or test.is_empty():
                    continue
                best_params, trials = tune_with_purged_cpcv(
                    frame=pl.concat([train, validation]),
                    candidate_features=research.feature_columns,
                    method=method,
                    horizon=horizon,
                    n_trials=config.n_trials,
                    n_groups=config.inner_cpcv_groups,
                    test_group_count=config.inner_test_groups,
                    missing_threshold=config.missing_feature_threshold,
                    positive_quantile=config.positive_quantile,
                    num_boost_round=config.num_boost_round,
                    seed=config.random_seed + window.fold,
                )
                trials.write_csv(combination_dir / f"fold_{window.fold:02d}_tuning.csv")
                preprocessor = fit_fold_preprocessor(
                    train,
                    research.feature_columns,
                    max_missing_ratio=config.missing_feature_threshold,
                )
                _, X_train = preprocessor.transform(train)
                _, X_validation = preprocessor.transform(validation)
                _, X_test = preprocessor.transform(test)
                fitted = fit_booster(
                    method=method,
                    horizon=horizon,
                    train_frame=train,
                    validation_frame=validation,
                    X_train=X_train,
                    X_validation=X_validation,
                    features=preprocessor.features,
                    positive_quantile=config.positive_quantile,
                    seed=config.random_seed + window.fold,
                    num_boost_round=config.num_boost_round,
                    params=best_params,
                )
                predictions = test.select(
                    "decision_month",
                    "ticker",
                    "legacy_selected",
                    *[
                        column
                        for column in test.columns
                        if column.startswith("future_") or column.startswith("benchmark_future_")
                    ],
                ).with_columns(
                    pl.Series("score", fitted.predict(X_test)),
                    pl.lit(window.fold).alias("fold"),
                    pl.lit(method).alias("method"),
                    pl.lit(horizon).alias("horizon"),
                )
                fold_metrics, portfolios = score_predictions(
                    predictions,
                    method=method,
                    horizon=horizon,
                    top_n_values=config.top_n_values,
                )
                fold_rows.append({"fold": window.fold, **fold_metrics})
                prediction_parts.append(predictions)
                portfolio_parts.append(
                    portfolios.with_columns(
                        pl.lit(window.fold).alias("fold"),
                        pl.lit(method).alias("method"),
                        pl.lit(horizon).alias("horizon"),
                    )
                )
                try:
                    shap_parts.append(
                        compute_shap_sample(
                            fitted=fitted,
                            X=X_test,
                            source=test,
                            fold=window.fold,
                            method=method,
                            horizon=horizon,
                            sample_size=config.shap_sample_per_fold,
                            seed=config.random_seed + window.fold,
                        )
                    )
                except Exception as exc:
                    (combination_dir / f"fold_{window.fold:02d}_shap_error.txt").write_text(
                        f"{type(exc).__name__}: {exc}\n"
                    )
            if not prediction_parts:
                continue
            predictions = pl.concat(prediction_parts)
            portfolios = pl.concat(portfolio_parts)
            fold_metrics = pl.DataFrame(fold_rows)
            overall, _ = score_predictions(
                predictions,
                method=method,
                horizon=horizon,
                top_n_values=config.top_n_values,
            )
            predictions.write_parquet(combination_dir / "predictions.parquet")
            portfolios.write_csv(combination_dir / "portfolio_monthly.csv")
            fold_metrics.write_csv(combination_dir / "fold_metrics.csv")
            if shap_parts:
                write_shap_outputs(
                    pl.concat(shap_parts, how="diagonal_relaxed"),
                    combination_dir,
                    top_features=config.shap_top_features,
                )
            all_summary.append(
                {
                    "method": method,
                    "horizon": horizon,
                    "folds": len(fold_rows),
                    "test_rows": predictions.height,
                    **overall,
                }
            )
    summary = pl.DataFrame(all_summary)
    summary.write_csv(run_dir / "model_horizon_summary.csv")
    return run_dir
