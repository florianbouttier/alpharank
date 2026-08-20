"""Strict replay validation for causal multi-horizon Boosting runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from alpharank.backtest.model_artifacts import load_serialized_fold_predictor
from alpharank.multihorizon.config import validate_latest_common_comparison_profile
from alpharank.multihorizon.metrics import build_prediction_portfolios


def validate_boosting_v2_replay(
    run_dir: Path,
    *,
    expected_composition_id: str,
    require_clean_runtime: bool = True,
    tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Replay serialized folds and reconstruct Top-N output without return filters."""

    root = run_dir.resolve()
    manifest = _read_json(root / "manifest.json")
    identity = manifest.get("methodology_identity", {})
    if identity.get("methodology_version") != "v2-causal":
        raise RuntimeError("Boosting run is not bound to methodology v2-causal")
    if identity.get("composition_id") != expected_composition_id:
        raise RuntimeError("Boosting run composition differs from the causal snapshot")
    profile = validate_latest_common_comparison_profile(manifest.get("config", {}))
    if not profile["passed"]:
        raise RuntimeError(f"Boosting latest-common profile drifted: {profile['mismatches']}")
    runtime_git = manifest.get("runtime_provenance", {}).get("git", {})
    if require_clean_runtime and runtime_git.get("dirty") is not False:
        raise RuntimeError("Boosting v2 promotion run must come from a clean worktree")

    fold_count = 0
    replay_rows = 0
    maximum_score_error = 0.0
    portfolio_rows = 0
    unresolved_mature_rows = 0
    provisional_mature_rows = 0
    approved_censored_mature_rows = 0
    target_policy = manifest.get(
        "terminal_target_policy",
        manifest.get("provisional_target_policy", {}),
    )
    target_policy_status = target_policy.get("status")
    target_policy_approved = (
        target_policy_status == "approved_methodology_censoring"
    )
    if target_policy_approved:
        approval = target_policy.get("methodology_approval", {})
        approval_path = Path(approval.get("path", ""))
        if (
            approval.get("policy_id")
            != "approved_last_observation_censoring_v1"
            or not approval_path.is_file()
            or _sha256(approval_path) != approval.get("sha256")
        ):
            raise RuntimeError("Boosting approved target-censoring policy drifted")
    for journal_key in ("journal_parquet", "journal_csv"):
        record = target_policy.get(journal_key, {})
        journal_path = Path(record.get("path", ""))
        if not journal_path.is_file() or _sha256(journal_path) != record.get("sha256"):
            raise RuntimeError(f"Boosting terminal target {journal_key} hash mismatch")
    combinations = manifest.get("results", {}).get("combinations", [])
    if not combinations:
        raise RuntimeError("Boosting v2 manifest has no completed combination")
    for combination in combinations:
        method = str(combination["method"])
        horizon = int(combination["horizon"])
        combination_dir = root / f"{method}_h{horizon:02d}"
        predictions = pl.read_parquet(combination_dir / "predictions.parquet")
        expected_portfolios = pl.read_csv(
            combination_dir / "portfolio_monthly.csv", try_parse_dates=True
        )
        rebuilt_parts: list[pl.DataFrame] = []
        for fold_frame in predictions.partition_by("fold", maintain_order=True):
            fold = int(fold_frame["fold"][0])
            rebuilt_parts.append(
                build_prediction_portfolios(
                    fold_frame,
                    horizon=horizon,
                    top_n_values=manifest["config"]["top_n_values"],
                ).with_columns(
                    pl.lit(fold).alias("fold"),
                    pl.lit(method).alias("method"),
                    pl.lit(horizon).alias("horizon"),
                )
            )
        rebuilt_portfolios = pl.concat(rebuilt_parts)
        portfolio_keys = ["decision_month", "top_n", "fold", "method", "horizon"]
        portfolio_numeric = [
            "future_excess_return",
            "realized_one_month_excess",
            "legacy_overlap",
            "legacy_jaccard",
        ]
        joined = rebuilt_portfolios.join(
            expected_portfolios.select(
                *portfolio_keys,
                *[
                    pl.col(column).alias(f"saved_{column}")
                    for column in portfolio_numeric
                ],
            ),
            on=portfolio_keys,
            how="inner",
            validate="1:1",
        )
        if (
            joined.height != rebuilt_portfolios.height
            or joined.height != expected_portfolios.height
        ):
            raise RuntimeError("Boosting Top-N portfolio calendar is not reproducible")
        for column in portfolio_numeric:
            error = joined.select(
                (pl.col(column) - pl.col(f"saved_{column}")).abs().max()
            ).item()
            if error is not None and float(error) > tolerance:
                raise RuntimeError(f"Boosting Top-N replay mismatch for {column}")
        portfolio_rows += rebuilt_portfolios.height

        censoring = pl.read_csv(combination_dir / "fold_target_censoring.csv")
        status_columns = [column for column in censoring.columns if column.endswith("_rows")]
        status_columns = [
            column
            for column in status_columns
            if column not in {"population_rows", "trainable_rows"}
        ]
        census_total = pl.sum_horizontal([pl.col(column) for column in status_columns])
        if censoring.filter(census_total != pl.col("population_rows")).height:
            raise RuntimeError("Boosting target-censoring census is incomplete")
        unresolved_columns = [
            "benchmark_target_unavailable_rows",
            "ticker_target_unavailable_rows",
            "terminal_event_unresolved_rows",
        ]
        unresolved_mature_rows += int(
            censoring.filter(pl.col("split").is_in(["train", "validation"]))
            .select(pl.sum_horizontal(unresolved_columns).sum())
            .item()
        )
        if unresolved_mature_rows:
            raise RuntimeError("Boosting v2 contains unresolved mature train/validation targets")
        provisional_mature_rows += int(
            censoring.filter(pl.col("split").is_in(["train", "validation"]))
            .select(pl.col("provisional_last_observation_rows").sum())
            .item()
        )
        if "approved_censored_last_observation_rows" in censoring.columns:
            approved_censored_mature_rows += int(
                censoring.filter(
                    pl.col("split").is_in(["train", "validation"])
                )
                .select(
                    pl.col("approved_censored_last_observation_rows").sum()
                )
                .item()
            )

        for fold_dir in sorted(combination_dir.glob("fold_[0-9][0-9]")):
            replay_manifest = _read_json(fold_dir / "oos_replay_manifest.json")
            replay_path = fold_dir / replay_manifest["oos_replay_file"]
            if _sha256(replay_path) != replay_manifest["oos_replay_sha256"]:
                raise RuntimeError("Boosting OOS replay artifact hash mismatch")
            replay = pl.read_parquet(replay_path)
            expected = replay["expected_raw_score"].to_numpy()
            observed = load_serialized_fold_predictor(fold_dir).predict(replay)
            if expected.shape != observed.shape:
                raise RuntimeError("Boosting serialized prediction shape changed")
            error = float(np.max(np.abs(expected - observed))) if expected.size else 0.0
            maximum_score_error = max(maximum_score_error, error)
            if error > tolerance:
                raise RuntimeError(f"Boosting serialized OOS score mismatch: {error}")
            for month in replay.partition_by("decision_month", maintain_order=True):
                expected_order = np.lexsort(
                    (
                        np.asarray(month["ticker"].to_list()),
                        -month["expected_raw_score"].to_numpy(),
                    )
                )
                observed_order = np.lexsort(
                    (np.asarray(month["ticker"].to_list()), -observed[: month.height])
                )
                if not np.array_equal(expected_order, observed_order):
                    raise RuntimeError("Boosting serialized OOS rank mismatch")
                observed = observed[month.height :]
            fold_count += 1
            replay_rows += replay.height

    return {
        "passed": True,
        "composition_id": expected_composition_id,
        "combination_count": len(combinations),
        "fold_count": fold_count,
        "oos_replay_rows": replay_rows,
        "portfolio_rows": portfolio_rows,
        "unresolved_mature_rows": unresolved_mature_rows,
        "provisional_mature_rows": provisional_mature_rows,
        "approved_censored_mature_rows": approved_censored_mature_rows,
        "provisional_target_journal_rows": (
            0
            if target_policy_approved
            else int(target_policy.get("journal_rows", 0))
        ),
        "approved_censored_target_journal_rows": (
            int(target_policy.get("journal_rows", 0))
            if target_policy_approved
            else 0
        ),
        "terminal_target_policy_approved": target_policy_approved,
        "maximum_absolute_score_replay_error": maximum_score_error,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
