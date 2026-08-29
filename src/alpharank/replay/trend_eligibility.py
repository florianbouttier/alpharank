"""Causal trend eligibility for optional Boosting allocation diagnostics."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import polars as pl

CAUSAL_TREND_POLICY_ID = "causal_majority_relative_ema_v1"
_KEY_COLUMNS = ("decision_month", "ticker", "fold")


@dataclass(frozen=True, slots=True)
class CausalTrendEligibilityRegistry:
    """Eligibility rows plus the immutable fold inputs that produced them."""

    frame: pl.DataFrame
    source_manifest: dict[str, Any]


def build_causal_trend_eligibility_registry(
    boosting_run_dir: Path,
    candidates: pl.DataFrame,
) -> CausalTrendEligibilityRegistry:
    """Build one strict point-in-time trend decision for every candidate row."""

    classification_dir = boosting_run_dir / "classification_h06"
    manifest_path = classification_dir / "fold_feature_manifest.csv"
    fold_manifest = _load_fold_manifest(manifest_path)
    frames: list[pl.DataFrame] = []
    fold_sources: list[dict[str, Any]] = []
    for row in fold_manifest.iter_rows(named=True):
        fold_frame, fold_source = _build_fold_registry(classification_dir, row)
        frames.append(fold_frame)
        fold_sources.append(fold_source)
    registry = pl.concat(frames, how="diagonal_relaxed").sort(list(_KEY_COLUMNS))
    _require_exact_candidate_keys(candidates, registry)
    return CausalTrendEligibilityRegistry(
        frame=registry,
        source_manifest={
            "policy_id": CAUSAL_TREND_POLICY_ID,
            "decision_rule": (
                "all fold-specific raw relative EMA pairs observed and strictly more "
                "than half pointing to positive relative trend"
            ),
            "pair_catalogue_timing": (
                "winner pairs available through each outer-fold train cutoff"
            ),
            "feature_manifest": _source_record(manifest_path),
            "fold_oos_replays": fold_sources,
            "candidate_rows": candidates.height,
            "eligible_rows": registry.filter(pl.col("trend_eligible")).height,
        },
    )


def filter_predictions_to_causal_trend_universe(
    predictions: pl.DataFrame,
    registry: CausalTrendEligibilityRegistry,
) -> pl.DataFrame:
    """Filter before ranking, while preserving every original prediction column."""

    joined = predictions.join(
        registry.frame.select(*_KEY_COLUMNS, "trend_eligible"),
        on=list(_KEY_COLUMNS),
        how="left",
        validate="1:1",
    )
    if joined["trend_eligible"].null_count():
        raise ValueError("Trend eligibility is missing for at least one prediction row.")
    return joined.filter(pl.col("trend_eligible")).drop("trend_eligible")


def write_causal_trend_eligibility_artifacts(
    output_dir: Path,
    registry: CausalTrendEligibilityRegistry,
) -> dict[str, Path]:
    """Write the row-level audit and compact monthly coverage evidence."""

    registry_path = output_dir / "causal_trend_eligibility.parquet"
    summary_path = output_dir / "causal_trend_eligibility_by_month.csv"
    registry.frame.write_parquet(registry_path)
    (
        registry.frame.group_by("decision_month")
        .agg(
            pl.len().alias("candidate_rows"),
            pl.col("trend_eligible").sum().alias("eligible_rows"),
            pl.col("trend_required_pair_count").min().alias("required_pairs_min"),
            pl.col("trend_required_pair_count").max().alias("required_pairs_max"),
        )
        .with_columns(
            (pl.col("eligible_rows") / pl.col("candidate_rows")).alias("eligible_fraction")
        )
        .sort("decision_month")
        .write_csv(summary_path)
    )
    return {
        "causal_trend_eligibility": registry_path,
        "causal_trend_eligibility_by_month": summary_path,
    }


def _load_fold_manifest(path: Path) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    required = {"fold", "train_cutoff", "winner_pair_count", "winner_pairs"}
    frame = pl.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing or frame.is_empty():
        raise ValueError(f"Invalid fold feature manifest; missing={missing}.")
    duplicate_folds = frame.group_by("fold").len().filter(pl.col("len") != 1)
    if duplicate_folds.height:
        raise ValueError("Fold feature manifest must contain one row per fold.")
    return frame.sort("fold")


def _build_fold_registry(
    classification_dir: Path,
    manifest_row: Mapping[str, Any],
) -> tuple[pl.DataFrame, dict[str, Any]]:
    fold = int(manifest_row["fold"])
    pairs = _parse_winner_pairs(manifest_row["winner_pairs"])
    if int(manifest_row["winner_pair_count"]) != len(pairs):
        raise ValueError(f"Fold {fold} winner_pair_count does not match winner_pairs.")
    replay_path = classification_dir / f"fold_{fold:02d}" / "oos_replay.parquet"
    _validate_replay_sidecar(replay_path, fold)
    replay = pl.read_parquet(replay_path).with_columns(pl.lit(fold).cast(pl.Int32).alias("fold"))
    observed, positive = _trend_count_expressions(replay.columns, pairs)
    required_count = len(pairs)
    registry = replay.select(
        "decision_month",
        "ticker",
        "fold",
        pl.lit(str(manifest_row["train_cutoff"]))
        .str.to_date(strict=False)
        .alias("trend_pair_catalogue_known_through"),
        pl.lit(required_count).cast(pl.Int32).alias("trend_required_pair_count"),
        pl.sum_horizontal(observed).cast(pl.Int32).alias("trend_observed_pair_count"),
        pl.sum_horizontal(positive).cast(pl.Int32).alias("trend_positive_pair_count"),
    )
    return _finish_fold_registry(registry), {
        "fold": fold,
        "train_cutoff": str(manifest_row["train_cutoff"]),
        "winner_pairs": [list(pair) for pair in pairs],
        **_source_record(replay_path),
    }


def _finish_fold_registry(frame: pl.DataFrame) -> pl.DataFrame:
    complete = pl.col("trend_observed_pair_count") == pl.col("trend_required_pair_count")
    majority = pl.col("trend_positive_pair_count") * 2 > pl.col("trend_required_pair_count")
    return frame.with_columns(
        pl.lit(CAUSAL_TREND_POLICY_ID).alias("trend_policy_id"),
        (pl.col("trend_positive_pair_count") / pl.col("trend_required_pair_count")).alias(
            "trend_positive_pair_fraction"
        ),
        (complete & majority).alias("trend_eligible"),
        pl.when(~complete)
        .then(pl.lit("incomplete_pair_coverage"))
        .when(majority)
        .then(pl.lit("eligible_strict_majority"))
        .otherwise(pl.lit("non_positive_majority"))
        .alias("trend_eligibility_reason"),
    )


def _parse_winner_pairs(payload: object) -> tuple[tuple[int, int], ...]:
    decoded = json.loads(str(payload))
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in decoded)
    if not pairs or len(set(pairs)) != len(pairs) or any(left == right for left, right in pairs):
        raise ValueError("Winner pairs must be non-empty, unique and use distinct spans.")
    return pairs


def _trend_count_expressions(
    columns: Sequence[str],
    pairs: Sequence[tuple[int, int]],
) -> tuple[list[pl.Expr], list[pl.Expr]]:
    observed: list[pl.Expr] = []
    positive: list[pl.Expr] = []
    for left, right in pairs:
        column = f"relative_ema_ratio_{left}_{right}"
        if column not in columns:
            observed.append(pl.lit(0))
            positive.append(pl.lit(0))
            continue
        available = pl.col(column).is_not_null() & pl.col(column).is_finite()
        direction = 1.0 if left < right else -1.0
        observed.append(available.cast(pl.Int8))
        positive.append((available & ((pl.col(column) - 1.0) * direction > 0.0)).cast(pl.Int8))
    return observed, positive


def _validate_replay_sidecar(replay_path: Path, fold: int) -> None:
    sidecar_path = replay_path.with_name("oos_replay_manifest.json")
    if not replay_path.exists() or not sidecar_path.exists():
        raise FileNotFoundError(replay_path if not replay_path.exists() else sidecar_path)
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    if int(sidecar.get("fold", -1)) != fold or sidecar.get("oos_replay_file") != replay_path.name:
        raise ValueError(f"Fold {fold} OOS replay sidecar does not identify its replay.")
    if sidecar.get("oos_replay_sha256") != _hash(replay_path):
        raise ValueError(f"Fold {fold} OOS replay hash does not match its sidecar.")


def _require_exact_candidate_keys(candidates: pl.DataFrame, registry: pl.DataFrame) -> None:
    missing_columns = sorted(set(_KEY_COLUMNS) - set(candidates.columns))
    if missing_columns:
        raise ValueError(f"Predictions lack trend eligibility keys: {missing_columns}.")
    candidate_keys = candidates.select(*_KEY_COLUMNS).with_columns(pl.col("fold").cast(pl.Int32))
    if candidate_keys.n_unique(list(_KEY_COLUMNS)) != candidate_keys.height:
        raise ValueError("Predictions contain duplicate trend eligibility keys.")
    missing = candidate_keys.join(registry.select(*_KEY_COLUMNS), on=list(_KEY_COLUMNS), how="anti")
    extra = registry.select(*_KEY_COLUMNS).join(candidate_keys, on=list(_KEY_COLUMNS), how="anti")
    if missing.height or extra.height:
        raise ValueError(
            "Trend eligibility keys do not exactly match predictions: "
            f"missing={missing.height}, extra={extra.height}."
        )


def _source_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": _hash(path)}


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
