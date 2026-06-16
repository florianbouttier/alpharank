from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import polars as pl

from run_ema_rich_future_target_models import _recomposition_by_month, _recomposition_summary
from run_signal_copy_models import DEFAULT_LEGACY_PATH, DEFAULT_SOURCE_RUN


OPTUNA_MODELS = ("Legacy_Optuna_11", "Legacy_Optuna_12", "Legacy_Optuna_21", "Legacy_Optuna_22")


@dataclass(frozen=True)
class AtomicFeatureFrameConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")


def _load_model_frame(source_run: Path) -> pl.DataFrame:
    frame_path = source_run / "model_frame.parquet"
    frame = pl.read_parquet(frame_path)
    return frame.with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )


def _load_legacy(legacy_path: Path) -> pl.DataFrame:
    return pl.read_parquet(legacy_path).with_columns(
        pl.col("year_month").dt.date().alias("holding_month"),
        pl.col("ticker").cast(pl.Utf8),
    )


def _legacy_labels(legacy: pl.DataFrame) -> pl.DataFrame:
    return (
        legacy.filter(pl.col("portfolio_model").is_in(["Combined_Equal", "Combined_Frequency"]))
        .select(["holding_month", "ticker"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )


def _atomic_base_features(legacy: pl.DataFrame) -> pl.DataFrame:
    optuna = (
        legacy.filter(pl.col("portfolio_model").is_in(OPTUNA_MODELS))
        .select(
            [
                "holding_month",
                "ticker",
                "portfolio_model",
                "mtr",
                "quantile_mtr",
                "n_long",
                "n_short",
                "selected_n_asset",
                "selected_n_max_per_sector",
            ]
        )
        .unique()
    )
    return (
        optuna.group_by(["holding_month", "ticker"])
        .agg(
            pl.col("portfolio_model").n_unique().alias("legacy_atomic_vote_count"),
            pl.max("quantile_mtr").alias("legacy_atomic_max_quantile_mtr"),
            pl.mean("quantile_mtr").alias("legacy_atomic_mean_quantile_mtr"),
            pl.max("mtr").alias("legacy_atomic_max_mtr"),
            pl.mean("mtr").alias("legacy_atomic_mean_mtr"),
            pl.min("n_short").alias("legacy_atomic_min_n_short"),
            pl.max("n_short").alias("legacy_atomic_max_n_short"),
            pl.min("n_long").alias("legacy_atomic_min_n_long"),
            pl.max("n_long").alias("legacy_atomic_max_n_long"),
            pl.mean("selected_n_asset").alias("legacy_atomic_mean_selected_n_asset"),
            pl.mean("selected_n_max_per_sector").alias("legacy_atomic_mean_selected_n_max_per_sector"),
        )
    )


def _atomic_block_flags(legacy: pl.DataFrame) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for model in OPTUNA_MODELS:
        suffix = model.removeprefix("Legacy_Optuna_").lower()
        frames.append(
            legacy.filter(pl.col("portfolio_model") == model)
            .select(["holding_month", "ticker", "quantile_mtr", "mtr"])
            .unique()
            .group_by(["holding_month", "ticker"])
            .agg(
                pl.lit(1).cast(pl.Int8).alias(f"legacy_optuna_{suffix}_selected"),
                pl.max("quantile_mtr").alias(f"legacy_optuna_{suffix}_quantile_mtr"),
                pl.max("mtr").alias(f"legacy_optuna_{suffix}_mtr"),
            )
        )
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on=["holding_month", "ticker"], how="full", coalesce=True)
    return out


def _add_monthly_ranks(frame: pl.DataFrame, feature_cols: Sequence[str]) -> tuple[pl.DataFrame, list[str]]:
    rank_cols = [f"{column}_rank_month" for column in feature_cols]
    ranked = frame.with_columns(
        [
            (pl.col(column).rank(method="average").over("holding_month") / pl.len().over("holding_month")).alias(rank_col)
            for column, rank_col in zip(feature_cols, rank_cols, strict=True)
        ]
    )
    return ranked, rank_cols


def _fill_atomic_nulls(frame: pl.DataFrame, feature_cols: Sequence[str]) -> pl.DataFrame:
    return frame.with_columns([pl.col(column).fill_null(0.0).alias(column) for column in feature_cols])


def _build_frame(config: AtomicFeatureFrameConfig) -> tuple[pl.DataFrame, list[str]]:
    model_frame = _load_model_frame(config.source_run)
    legacy = _load_legacy(config.legacy_path)
    labels = _legacy_labels(legacy)
    atomic = _atomic_base_features(legacy).join(
        _atomic_block_flags(legacy),
        on=["holding_month", "ticker"],
        how="full",
        coalesce=True,
    )
    atomic_feature_cols = [column for column in atomic.columns if column not in {"holding_month", "ticker"}]
    frame = (
        model_frame.join(atomic, on=["holding_month", "ticker"], how="left")
        .pipe(_fill_atomic_nulls, atomic_feature_cols)
        .join(labels, on=["holding_month", "ticker"], how="left")
        .with_columns(pl.col("legacy_selected").fill_null(0).cast(pl.Int8))
    )
    rank_base_cols = [
        "legacy_atomic_vote_count",
        "legacy_atomic_max_quantile_mtr",
        "legacy_atomic_mean_quantile_mtr",
        "legacy_atomic_max_mtr",
        "legacy_atomic_mean_mtr",
    ]
    frame, rank_cols = _add_monthly_ranks(frame, rank_base_cols)
    return frame, atomic_feature_cols + rank_cols


def run(config: AtomicFeatureFrameConfig) -> Path:
    run_dir = config.output_dir / f"legacy_atomic_feature_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, feature_cols = _build_frame(config)
    frame.write_parquet(run_dir / "legacy_atomic_feature_frame.parquet")

    diagnostic = frame.select(
        [
            "ticker",
            "holding_month",
            "future_excess_return",
            "legacy_selected",
            "legacy_atomic_vote_count",
            "legacy_atomic_max_quantile_mtr",
        ]
    )
    recomposition = _recomposition_by_month(diagnostic, ["legacy_atomic_vote_count", "legacy_atomic_max_quantile_mtr"])
    summary = _recomposition_summary(recomposition)
    recomposition.write_csv(run_dir / "deterministic_recomposition_by_month.csv")
    summary.write_csv(run_dir / "deterministic_recomposition_summary.csv")

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "primary_metric": "recomposition_pct",
                "primary_metric_formula": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "notes": [
                    "Ce frame joint les signaux atomiques Legacy au model_frame mensuel.",
                    "Les colonnes atomiques sont des signaux de decision connus au mois de holding via le pipeline Legacy.",
                    "Le diagnostic deterministe sert de controle avant entrainement ML.",
                ],
            },
            indent=2,
            default=str,
        )
    )

    print(f"RUN_DIR={run_dir}")
    print(summary)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a monthly model frame enriched with atomic Legacy signals.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(AtomicFeatureFrameConfig(source_run=args.source_run, legacy_path=args.legacy_path, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
