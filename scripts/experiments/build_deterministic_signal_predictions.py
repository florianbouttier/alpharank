from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

import polars as pl
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


@dataclass(frozen=True)
class DeterministicSignalConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    feature_set: str = "technical"
    start_month: str = "2015-01-01"
    end_month: str | None = None
    score_cols: tuple[str, ...] = (
        "ema_ratio_2_12_rank_month",
        "ema_ratio_3_12_rank_month",
        "price_to_ema_12_rank_month",
        "technical_z_mean",
        "technical_rank_mean",
    )


def _parse_month(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _load_frame(config: DeterministicSignalConfig) -> tuple[pl.DataFrame, list[str]]:
    metadata = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(metadata["features_used"])
    base_features = _base_features_for_set(source_features, config.feature_set)
    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    frame, features = _add_cross_sectional_features(frame, base_features, prefix=config.feature_set)
    legacy = _load_legacy_labels(config.legacy_path)
    frame = _append_legacy(frame, legacy)

    start = _parse_month(config.start_month)
    end = _parse_month(config.end_month)
    if start is not None:
        frame = frame.filter(pl.col("holding_month") >= start)
    if end is not None:
        frame = frame.filter(pl.col("holding_month") <= end)
    return frame, features


def _validate_scores(score_cols: Sequence[str], available_features: Sequence[str]) -> None:
    available = set(available_features)
    missing = [score for score in score_cols if score not in available]
    if missing:
        raise ValueError(f"Score columns are not available: {missing}")


def run(config: DeterministicSignalConfig) -> Path:
    frame, features = _load_frame(config)
    _validate_scores(config.score_cols, features)
    run_dir = config.output_dir / f"deterministic_signal_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    predictions = (
        frame.select(
            [
                "ticker",
                "year_month",
                "holding_month",
                "legacy_selected",
                *config.score_cols,
            ]
        )
        .with_columns(pl.lit("deterministic_signal").alias("fold"))
        .sort(["year_month", "ticker"])
    )
    predictions.write_parquet(run_dir / "predictions.parquet")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "rows": predictions.height,
                "score_cols": list(config.score_cols),
                "feature_count": len(features),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(run_dir)
    return run_dir


def _parse_args() -> DeterministicSignalConfig:
    parser = argparse.ArgumentParser(description="Build deterministic tradable signal prediction runs.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--feature-set", choices=["ema", "technical"], default="technical")
    parser.add_argument("--start-month", default="2015-01-01")
    parser.add_argument("--end-month")
    parser.add_argument("--score-col", nargs="*", default=list(DeterministicSignalConfig.score_cols))
    args = parser.parse_args()
    return DeterministicSignalConfig(
        source_run=args.source_run,
        legacy_path=args.legacy_path,
        output_dir=args.output_dir,
        feature_set=args.feature_set,
        start_month=args.start_month,
        end_month=args.end_month,
        score_cols=tuple(args.score_col),
    )


if __name__ == "__main__":
    run(_parse_args())
