from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alpharank.governance import capture_runtime_provenance, reserve_run_directory
from alpharank.multihorizon.config import (
    APPROVED_TARGET_CENSORING_POLICY_ID,
    MultiHorizonConfig,
    load_approved_target_censoring_policy,
)
from alpharank.multihorizon.data import (
    RELATIVE_EMA_PAIRS,
    TRAINABLE_TARGET_STATUSES,
    build_research_frame,
    classify_training_target_status,
    mask_targets_after_completed_month,
    provisional_target_journal,
    require_resolved_training_targets,
    target_censoring_counts,
)
from alpharank.multihorizon.explain import compute_shap_sample, write_shap_outputs
from alpharank.multihorizon.legacy_ema import (
    add_active_legacy_oracle_features,
    legacy_winning_pairs,
    point_in_time_fold_features,
)
from alpharank.multihorizon.metrics import (
    build_prediction_portfolios,
    score_predictions,
)
from alpharank.multihorizon.modeling import fit_booster
from alpharank.multihorizon.preprocessing import fit_fold_preprocessor
from alpharank.multihorizon.splits import horizon_walk_forward_windows
from alpharank.multihorizon.tuning import tune_with_purged_cpcv
from alpharank.replay import validate_causal_v2_snapshot


def _prediction_frame(
    *,
    source: pl.DataFrame,
    matrix,
    fitted: Any,
    fold: int,
    method: str,
    horizon: int,
) -> pl.DataFrame:
    target_columns = [
        column
        for column in source.columns
        if column.startswith("future_") or column.startswith("benchmark_future_")
    ]
    status_columns = [
        column for column in source.columns if column.startswith("target_status_")
    ]
    output = source.select(
        "decision_month",
        "ticker",
        "legacy_selected",
        *target_columns,
        *status_columns,
    ).with_columns(
        pl.Series("score", fitted.predict_raw_score(matrix)),
        pl.lit(fold).alias("fold"),
        pl.lit(method).alias("method"),
        pl.lit(horizon).alias("horizon"),
    )
    target_column = (
        "legacy_selected"
        if method == "teacher"
        else f"future_excess_return_{horizon}m"
    )
    benchmark_target_column = (
        "legacy_selected"
        if method == "teacher"
        else f"benchmark_future_return_{horizon}m"
    )
    if method == "teacher":
        output = output.with_columns(
            pl.when(pl.col(target_column).is_not_null())
            .then(pl.lit("evaluable"))
            .otherwise(pl.lit("ticker_target_unavailable"))
            .alias("target_status")
        )
    else:
        output = output.with_columns(
            pl.col(f"target_status_{horizon}m").alias("target_status")
        )
    if method in {"classification", "teacher"}:
        output = output.with_columns(
            pl.Series("calibrated_probability", fitted.predict(matrix))
        )
    return output


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


def _eligible(
    frame: pl.DataFrame,
    method: str,
    horizon: int,
    feature_mode: str,
) -> pl.DataFrame:
    if method == "teacher":
        eligible = frame.filter(pl.col("legacy_selected").is_not_null())
    else:
        eligible = frame.filter(
            pl.col(f"target_status_{horizon}m").is_in(
                TRAINABLE_TARGET_STATUSES
            )
        )
    if method == "teacher" or feature_mode == "legacy_active_oracle":
        eligible = eligible.filter(pl.col("legacy_label_available") == 1)
    return eligible.sort(["decision_month", "ticker"])


def _fold_censoring_rows(
    *,
    frame: pl.DataFrame,
    windows: list[Any],
    method: str,
    horizon: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for window in windows:
        for split, months in (
            ("train", window.train_months),
            ("validation", window.validation_months),
            ("test", window.test_months),
        ):
            population = frame.filter(pl.col("decision_month").is_in(months))
            counts = target_censoring_counts(
                population,
                method=method,
                horizon=horizon,
            )
            rows.append(
                {
                    "fold": window.fold,
                    "split": split,
                    "population_rows": population.height,
                    "trainable_rows": sum(
                        counts[status] for status in TRAINABLE_TARGET_STATUSES
                    ),
                    **{f"{status}_rows": count for status, count in counts.items()},
                }
            )
    return rows


def _score_only_panel(
    frame: pl.DataFrame,
    *,
    method: str,
    feature_mode: str,
    end_month: str | None,
) -> pl.DataFrame | None:
    if end_month is None:
        return None
    cutoff = date.fromisoformat(f"{end_month}-01")
    eligible = frame.filter(pl.col("decision_month") <= cutoff)
    if method == "teacher" or feature_mode == "legacy_active_oracle":
        eligible = eligible.filter(pl.col("legacy_label_available") == 1)
    return eligible.sort(["decision_month", "ticker"])


def run_multihorizon_research(config: MultiHorizonConfig) -> Path:
    # Imported lazily to avoid the backtest/multihorizon package init cycle.
    from alpharank.backtest.model_artifacts import (
        load_serialized_fold_predictor,
        serialize_fold_model,
    )

    if config.feature_mode.startswith("legacy_winners_pit") and config.n_trials > 1:
        raise ValueError(
            "Point-in-time winner modes currently require n_trials <= 1: "
            "inner-CPCV tuning would otherwise need its own nested winner selection."
        )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = config.run_dir or config.output_dir / "multihorizon_boosting" / timestamp
    run_dir = reserve_run_directory(run_dir)
    project_root = Path(__file__).resolve().parents[3]
    methodology_identity: dict[str, Any] | None = None
    if config.methodology_manifest is not None:
        methodology_manifest = config.methodology_manifest.resolve()
        causal_package = methodology_manifest.parent
        methodology_identity = {
            **validate_causal_v2_snapshot(causal_package),
            "methodology_version": "v2-causal",
            "methodology_manifest": str(methodology_manifest),
            "methodology_manifest_sha256": _sha256(methodology_manifest),
        }
        if config.data_dir.resolve() != (causal_package / "input_snapshot").resolve():
            raise ValueError(
                "A causal methodology manifest requires data_dir to reference "
                "its sealed input_snapshot."
            )
    runtime_provenance = capture_runtime_provenance(
        project_root=project_root,
        entrypoint="scripts/experiments/run_multihorizon_boosting.py",
        command_argv=[sys.executable, *sys.argv],
        resolved_config=_jsonable_config(config),
        seeds={"random_seed": config.random_seed},
        critical_files=(
            "scripts/experiments/run_multihorizon_boosting.py",
            "src/alpharank/multihorizon/pipeline.py",
            "src/alpharank/multihorizon/data.py",
            "src/alpharank/multihorizon/modeling.py",
            "src/alpharank/multihorizon/preprocessing.py",
            "src/alpharank/multihorizon/splits.py",
            "src/alpharank/backtest/model_artifacts.py",
            "src/alpharank/portfolio/simulation.py",
            "src/alpharank/governance.py",
            "src/alpharank/governance_contracts/common.py",
            "src/alpharank/governance_contracts/contracts.py",
            "src/alpharank/governance_contracts/runtime_provenance.py",
        ),
        data_identifiers={
            "methodology_identity": methodology_identity or {"version": "research"},
            "data_dir": str(config.data_dir.resolve()),
            "legacy_detailed_sha256": _sha256(config.legacy_detailed_returns_path),
            "legacy_monthly_sha256": _sha256(config.legacy_monthly_returns_path),
        },
        patch_path=run_dir / "runtime_git_patch.json",
    )
    exact_winner_pairs = legacy_winning_pairs(config.legacy_detailed_returns_path)
    relative_ema_pairs = (
        RELATIVE_EMA_PAIRS
        if config.feature_mode == "broad"
        else exact_winner_pairs
    )
    research = build_research_frame(
        data_dir=config.data_dir,
        legacy_detailed_returns_path=config.legacy_detailed_returns_path,
        horizons=tuple(sorted(set(config.horizons) | {1})),
        start_month=config.start_month,
        excluded_tickers=config.excluded_tickers,
        relative_ema_pairs=relative_ema_pairs,
        minimum_monthly_price_observations=(
            config.minimum_monthly_price_observations
        ),
        minimum_monthly_median_dollar_volume=(
            config.minimum_monthly_median_dollar_volume
        ),
        maximum_monthly_ohlc_violation_rate=(
            config.maximum_monthly_ohlc_violation_rate
        ),
        mature_target_gap_policy=config.mature_target_gap_policy,
    )
    frame = research.frame
    completed_through_month = None
    if config.score_only_end_month is not None:
        completed_through_month = date.fromisoformat(
            f"{config.score_only_end_month}-01"
        )
        frame = mask_targets_after_completed_month(
            frame,
            horizons=tuple(sorted(set(config.horizons) | {1})),
            completed_through_month=completed_through_month,
        )
    if completed_through_month is None:
        completed_through_month = frame["decision_month"].max()
    frame = classify_training_target_status(
        frame,
        horizons=tuple(sorted(set(config.horizons) | {1})),
        completed_through_month=completed_through_month,
    )
    target_journal = provisional_target_journal(
        frame,
        horizons=tuple(sorted(set(config.horizons) | {1})),
    )
    target_policy_approved = (
        config.mature_target_gap_policy == APPROVED_TARGET_CENSORING_POLICY_ID
    )
    target_journal_stem = (
        "approved_censored_target_journal"
        if target_policy_approved
        else "provisional_target_journal"
    )
    target_journal_path = run_dir / f"{target_journal_stem}.parquet"
    target_journal_csv_path = run_dir / f"{target_journal_stem}.csv"
    target_journal.write_parquet(target_journal_path)
    target_journal.write_csv(target_journal_csv_path)
    oracle_features: tuple[str, ...] = ()
    if config.feature_mode == "legacy_active_oracle":
        frame, oracle_features = add_active_legacy_oracle_features(
            frame,
            legacy_path=config.legacy_detailed_returns_path,
            available_pairs=research.relative_ema_pairs,
        )
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
        "feature_mode": config.feature_mode,
        "candidate_feature_count": (
            len(oracle_features)
            if config.feature_mode == "legacy_active_oracle"
            else len(research.feature_columns)
        ),
        "research_frame_rows": frame.height,
        "research_frame_columns": frame.width,
        "relative_ema_pairs": research.relative_ema_pairs,
        "exact_legacy_winner_pairs": exact_winner_pairs,
        "oracle_features": oracle_features,
        "protocol": {
            "outer": "strict expanding walk-forward; fixed model over each test block",
            "inner": "horizon-purged CPCV on pre-test data only",
            "preprocessing": "sparse filtering and fallback medians fitted inside each fold",
            "score_only_tail": {
                "enabled": config.score_only_end_month is not None,
                "decision_end_month": config.score_only_end_month,
                "realized_targets_completed_through_month": (
                    completed_through_month
                ),
                "target_maturity_rule": (
                    "a future label is null when decision_month + horizon "
                    "extends past decision_end_month"
                ),
                "target_censoring_rule": (
                    "every row is classified; mature benchmark/ticker/terminal "
                    "missingness fails training closed and is never dropped silently"
                ),
                "model_metrics": "mature learning targets only",
                "portfolio_metrics": "all test months with a complete one-month return",
            },
            "teacher": "Legacy Combined_Frequency basket at decision t for holding t+1",
            "legacy_winners_pit": (
                "for each outer fold, only exact EMA pairs observed in Legacy output "
                "through the last training decision month are eligible"
            ),
            "legacy_active_oracle": (
                "diagnostic only: exposes the four EMA pairs selected by Legacy for "
                "the current decision month"
            ),
        },
        "methodology_identity": methodology_identity,
        "runtime_provenance": runtime_provenance,
        "terminal_target_policy": {
            "policy_id": config.mature_target_gap_policy,
            "status": (
                "approved_methodology_censoring"
                if target_policy_approved
                else (
                    "pending_manual_review"
                    if target_journal.height
                    else "no_censored_target"
                )
            ),
            "methodology_approval": (
                load_approved_target_censoring_policy()
                if target_policy_approved
                else None
            ),
            "journal_rows": target_journal.height,
            "journal_tickers": target_journal["ticker"].n_unique(),
            "journal_parquet": {
                "path": str(target_journal_path.resolve()),
                "sha256": _sha256(target_journal_path),
            },
            "journal_csv": {
                "path": str(target_journal_csv_path.resolve()),
                "sha256": _sha256(target_journal_csv_path),
            },
        },
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    all_summary: list[dict] = []
    for method in config.methods:
        method_horizons = (1,) if method == "teacher" else config.horizons
        for horizon in method_horizons:
            combination_dir = run_dir / f"{method}_h{horizon:02d}"
            combination_dir.mkdir(parents=True, exist_ok=True)
            panel = _eligible(frame, method, horizon, config.feature_mode)
            score_only_panel = _score_only_panel(
                frame,
                method=method,
                feature_mode=config.feature_mode,
                end_month=config.score_only_end_month,
            )
            test_panel = score_only_panel if score_only_panel is not None else panel
            censoring_panel = frame
            if method == "teacher" or config.feature_mode == "legacy_active_oracle":
                censoring_panel = censoring_panel.filter(
                    pl.col("legacy_label_available") == 1
                )
            all_months = test_panel["decision_month"].unique().sort().to_list()
            try:
                windows = horizon_walk_forward_windows(
                    all_months,
                    horizon=1 if method == "teacher" else horizon,
                    min_train_months=config.min_train_months,
                    validation_months=config.validation_months,
                    test_months=config.test_months,
                    step_months=config.step_months,
                    include_partial_test_window=(
                        config.include_partial_test_window
                        or score_only_panel is not None
                    ),
                    max_windows=config.max_windows,
                )
            except ValueError as exc:
                if "No mature horizon-aware outer window" not in str(exc):
                    raise
                (combination_dir / "unavailable.json").write_text(
                    json.dumps(
                        {
                            "reason": str(exc),
                            "eligible_start": min(all_months) if all_months else None,
                            "eligible_end": max(all_months) if all_months else None,
                            "eligible_months": len(all_months),
                        },
                        indent=2,
                        default=str,
                    )
                    + "\n"
                )
                continue
            censoring_rows = _fold_censoring_rows(
                frame=censoring_panel,
                windows=windows,
                method=method,
                horizon=horizon,
            )
            pl.DataFrame(censoring_rows).write_csv(
                combination_dir / "fold_target_censoring.csv"
            )
            prediction_parts: list[pl.DataFrame] = []
            portfolio_parts: list[pl.DataFrame] = []
            fold_rows: list[dict] = []
            feature_manifest_rows: list[dict] = []
            model_manifest_rows: list[dict] = []
            shap_parts: list[pl.DataFrame] = []
            for window in windows:
                train_population = censoring_panel.filter(
                    pl.col("decision_month").is_in(window.train_months)
                )
                validation_population = censoring_panel.filter(
                    pl.col("decision_month").is_in(window.validation_months)
                )
                require_resolved_training_targets(
                    train_population,
                    method=method,
                    horizon=horizon,
                    context=f"fold {window.fold} train",
                )
                require_resolved_training_targets(
                    validation_population,
                    method=method,
                    horizon=horizon,
                    context=f"fold {window.fold} validation",
                )
                train = panel.filter(pl.col("decision_month").is_in(window.train_months))
                validation = panel.filter(pl.col("decision_month").is_in(window.validation_months))
                test = test_panel.filter(
                    pl.col("decision_month").is_in(window.test_months)
                )
                if train.is_empty() or validation.is_empty() or test.is_empty():
                    continue
                train_cutoff = max(train["decision_month"])
                fold_pairs: tuple[tuple[int, int], ...] = research.relative_ema_pairs
                if config.feature_mode == "legacy_winners_pit_ema_only":
                    fold_features, fold_pairs = point_in_time_fold_features(
                        all_features=research.feature_columns,
                        legacy_path=config.legacy_detailed_returns_path,
                        train_decision_cutoff=train_cutoff,
                        include_non_relative_features=False,
                    )
                elif config.feature_mode == "legacy_winners_pit_ema_plus":
                    fold_features, fold_pairs = point_in_time_fold_features(
                        all_features=research.feature_columns,
                        legacy_path=config.legacy_detailed_returns_path,
                        train_decision_cutoff=train_cutoff,
                        include_non_relative_features=True,
                    )
                elif config.feature_mode == "legacy_active_oracle":
                    fold_features = oracle_features
                else:
                    fold_features = research.feature_columns
                best_params, trials = tune_with_purged_cpcv(
                    frame=pl.concat([train, validation]),
                    candidate_features=fold_features,
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
                    fold_features,
                    max_missing_ratio=config.missing_feature_threshold,
                )
                feature_manifest_rows.append(
                    {
                        "fold": window.fold,
                        "train_start": min(train["decision_month"]),
                        "train_cutoff": train_cutoff,
                        "validation_start": min(validation["decision_month"]),
                        "validation_end": max(validation["decision_month"]),
                        "test_start": min(test["decision_month"]),
                        "test_end": max(test["decision_month"]),
                        "train_rows": train.height,
                        "validation_rows": validation.height,
                        "test_rows": test.height,
                        "mature_test_rows": test.filter(
                            pl.col(
                                "legacy_selected"
                                if method == "teacher"
                                else f"future_excess_return_{horizon}m"
                            ).is_not_null()
                        ).height,
                        "winner_pair_count": len(fold_pairs),
                        "winner_pairs": json.dumps(fold_pairs),
                        "candidate_feature_count": len(fold_features),
                        "kept_feature_count": len(preprocessor.features),
                        "kept_features": json.dumps(preprocessor.features),
                    }
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
                split_predictions = {
                    "train": _prediction_frame(
                        source=train, matrix=X_train, fitted=fitted,
                        fold=window.fold, method=method, horizon=horizon,
                    ),
                    "validation": _prediction_frame(
                        source=validation, matrix=X_validation, fitted=fitted,
                        fold=window.fold, method=method, horizon=horizon,
                    ),
                    "test": _prediction_frame(
                        source=test, matrix=X_test, fitted=fitted,
                        fold=window.fold, method=method, horizon=horizon,
                    ),
                }
                fold_dir = combination_dir / f"fold_{window.fold:02d}"
                serialized = serialize_fold_model(
                    fold_dir=fold_dir,
                    model=fitted.model,
                    preprocessor=preprocessor,
                    seed=config.random_seed + window.fold,
                    fold_metadata={
                        "fold": window.fold,
                        "train_start": str(min(train["decision_month"])),
                        "train_cutoff": str(train_cutoff),
                        "validation_start": str(min(validation["decision_month"])),
                        "validation_end": str(max(validation["decision_month"])),
                        "test_start": str(min(test["decision_month"])),
                        "test_end": str(max(test["decision_month"])),
                    },
                )
                replay_frame = test.select(
                    "decision_month",
                    "ticker",
                    *preprocessor.features,
                ).with_columns(
                    pl.Series(
                        "expected_raw_score",
                        split_predictions["test"]["score"],
                    )
                )
                replay_path = fold_dir / "oos_replay.parquet"
                replay_frame.write_parquet(replay_path)
                replayed_scores = load_serialized_fold_predictor(fold_dir).predict(
                    replay_frame
                )
                expected_scores = replay_frame["expected_raw_score"].to_numpy()
                if not np.array_equal(replayed_scores, expected_scores):
                    maximum_error = float(
                        np.max(np.abs(replayed_scores - expected_scores))
                    )
                    raise RuntimeError(
                        "Serialized fold model changed OOS scores: "
                        f"fold={window.fold}, max_error={maximum_error}"
                    )
                replay_manifest = {
                    "fold": window.fold,
                    "rows": replay_frame.height,
                    "oos_replay_file": replay_path.name,
                    "oos_replay_sha256": _sha256(replay_path),
                    "model_sha256": serialized["model_sha256"],
                    "score_replay_maximum_absolute_error": 0.0,
                }
                (fold_dir / "oos_replay_manifest.json").write_text(
                    json.dumps(replay_manifest, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                model_manifest_rows.append(replay_manifest)
                predictions = split_predictions["test"]
                target_column = (
                    "legacy_selected"
                    if method == "teacher"
                    else f"future_excess_return_{horizon}m"
                )
                benchmark_target_column = (
                    "legacy_selected"
                    if method == "teacher"
                    else f"benchmark_future_return_{horizon}m"
                )
                mature_predictions = predictions.filter(
                    pl.col(target_column).is_not_null()
                )
                horizon_pending_predictions = predictions.filter(
                    pl.col(benchmark_target_column).is_null()
                )
                ticker_target_unavailable_predictions = predictions.filter(
                    pl.col(benchmark_target_column).is_not_null()
                    & pl.col(target_column).is_null()
                )
                test_metrics: dict[str, float] = {}
                if not mature_predictions.is_empty():
                    test_metrics, _ = score_predictions(
                        mature_predictions,
                        method=method,
                        horizon=horizon,
                        top_n_values=config.top_n_values,
                    )
                portfolios = build_prediction_portfolios(
                    predictions,
                    horizon=horizon,
                    top_n_values=config.top_n_values,
                )
                split_metrics: dict[str, float] = {}
                metric_splits = {
                    "train": split_predictions["train"],
                    "validation": split_predictions["validation"],
                    "test": mature_predictions,
                }
                for split_name, split_frame in metric_splits.items():
                    if split_frame.is_empty():
                        continue
                    metrics, _ = score_predictions(
                        split_frame,
                        method=method,
                        horizon=horizon,
                        top_n_values=config.top_n_values,
                    )
                    split_metrics.update(
                        {f"{split_name}_{key}": value for key, value in metrics.items()}
                    )
                fold_rows.append(
                    {
                        "fold": window.fold,
                        "mature_test_rows": mature_predictions.height,
                        "score_only_test_rows": horizon_pending_predictions.height,
                        "ticker_target_unavailable_rows": (
                            ticker_target_unavailable_predictions.height
                        ),
                        **test_metrics,
                        **split_metrics,
                    }
                )
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
                except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
                    (combination_dir / f"fold_{window.fold:02d}_shap_error.txt").write_text(
                        f"{type(exc).__name__}: {exc}\n"
                    )
            if not prediction_parts:
                continue
            predictions = pl.concat(prediction_parts)
            portfolios = pl.concat(portfolio_parts)
            fold_metrics = pl.DataFrame(fold_rows)
            target_column = (
                "legacy_selected"
                if method == "teacher"
                else f"future_excess_return_{horizon}m"
            )
            benchmark_target_column = (
                "legacy_selected"
                if method == "teacher"
                else f"benchmark_future_return_{horizon}m"
            )
            mature_predictions = predictions.filter(
                pl.col(target_column).is_not_null()
            )
            horizon_pending_predictions = predictions.filter(
                pl.col(benchmark_target_column).is_null()
            )
            ticker_target_unavailable_predictions = predictions.filter(
                pl.col(benchmark_target_column).is_not_null()
                & pl.col(target_column).is_null()
            )
            overall: dict[str, float] = {}
            if not mature_predictions.is_empty():
                overall, _ = score_predictions(
                    mature_predictions,
                    method=method,
                    horizon=horizon,
                    top_n_values=config.top_n_values,
                )
            predictions.write_parquet(combination_dir / "predictions.parquet")
            portfolios.write_csv(combination_dir / "portfolio_monthly.csv")
            fold_metrics.write_csv(combination_dir / "fold_metrics.csv")
            pl.DataFrame(feature_manifest_rows).write_csv(
                combination_dir / "fold_feature_manifest.csv"
            )
            pl.DataFrame(model_manifest_rows).write_csv(
                combination_dir / "fold_model_manifest.csv"
            )
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
                    "first_test_decision_month": predictions[
                        "decision_month"
                    ].min(),
                    "last_test_decision_month": predictions[
                        "decision_month"
                    ].max(),
                    "last_mature_target_decision_month": (
                        mature_predictions["decision_month"].max()
                        if not mature_predictions.is_empty()
                        else None
                    ),
                    "first_score_only_decision_month": (
                        horizon_pending_predictions["decision_month"].min()
                        if not horizon_pending_predictions.is_empty()
                        else None
                    ),
                    "mature_test_rows": mature_predictions.height,
                    "score_only_test_rows": horizon_pending_predictions.height,
                    "ticker_target_unavailable_rows": (
                        ticker_target_unavailable_predictions.height
                    ),
                    **overall,
                }
            )
    summary = pl.DataFrame(all_summary)
    summary.write_csv(run_dir / "model_horizon_summary.csv")
    manifest["results"] = {
        "combinations": summary.to_dicts(),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str) + "\n"
    )
    return run_dir
