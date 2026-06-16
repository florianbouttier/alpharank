from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from build_generalized_ema_expert_frame import (  # noqa: E402
    DEFAULT_LEGACY_PATH,
    DEFAULT_SOURCE_RUN,
    _expert_scores,
    _fill_feature_nulls,
    _legacy_labels,
    _load_legacy,
    _load_model_frame,
    _ticker_features,
)
from run_ema_rich_future_target_models import _recomposition_by_month, _recomposition_summary  # noqa: E402


@dataclass(frozen=True)
class SweepConfig:
    expert_run_dir: Path
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    trailing_months: tuple[int, ...] = (12, 24, 36, 60)
    top_experts: tuple[int, ...] = (5, 10, 20, 30, 50)
    min_trailing_months: int = 6


def _load_expert_selections(run_dir: Path) -> pl.DataFrame:
    path = run_dir / "ema_expert_selections.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pl.read_parquet(path).with_columns(
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
        pl.col("expert_id").cast(pl.Utf8),
    )


def _score_config(
    *,
    selections: pl.DataFrame,
    model_frame: pl.DataFrame,
    labels: pl.DataFrame,
    trailing_months: int,
    min_trailing_months: int,
    top_experts: int,
) -> dict:
    expert_scores = _expert_scores(selections, model_frame, trailing_months, min_trailing_months)
    features = _ticker_features(selections, expert_scores, top_experts)
    feature_cols = [column for column in features.columns if column not in {"holding_month", "ticker"}]
    frame = (
        model_frame.select(["ticker", "holding_month"])
        .join(features, on=["holding_month", "ticker"], how="left")
        .pipe(_fill_feature_nulls, feature_cols)
        .join(labels, on=["holding_month", "ticker"], how="left")
        .with_columns(pl.col("legacy_selected").fill_null(0).cast(pl.Int8))
    )
    diagnostic = frame.select(["ticker", "holding_month", "legacy_selected"] + feature_cols)
    recomposition = _recomposition_by_month(diagnostic, ["learned_ema_expert_vote_count", "learned_ema_expert_score_sum"])
    summary = _recomposition_summary(recomposition)
    best = summary.sort("recomposition_pct", descending=True).row(0, named=True)
    return {
        "trailing_months": trailing_months,
        "top_experts": top_experts,
        "best_score_name": best["model"],
        "common_count": int(best["common_count"]),
        "legacy_count": int(best["legacy_count"]),
        "recomposition_pct": float(best["recomposition_pct"]),
        "mean_monthly_recomposition_pct": float(best["mean_monthly_recomposition_pct"]),
        "median_monthly_recomposition_pct": float(best["median_monthly_recomposition_pct"]),
        "months": int(best["months"]),
    }


def run(config: SweepConfig) -> Path:
    run_dir = config.output_dir / f"generalized_ema_expert_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    selections = _load_expert_selections(config.expert_run_dir)
    model_frame = _load_model_frame(config.source_run)
    labels = _legacy_labels(_load_legacy(config.legacy_path))

    rows: list[dict] = []
    for trailing in config.trailing_months:
        for top in config.top_experts:
            row = _score_config(
                selections=selections,
                model_frame=model_frame,
                labels=labels,
                trailing_months=trailing,
                min_trailing_months=config.min_trailing_months,
                top_experts=top,
            )
            rows.append(row)
            print(
                f"trailing={trailing} top={top} "
                f"{row['best_score_name']}={row['recomposition_pct']:.3f}",
                flush=True,
            )

    results = pl.DataFrame(rows).sort("recomposition_pct", descending=True)
    results.write_csv(run_dir / "sweep_results.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "expert_run_dir": str(config.expert_run_dir),
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "trailing_months": list(config.trailing_months),
                "top_experts": list(config.top_experts),
                "min_trailing_months": config.min_trailing_months,
                "primary_metric": "recomposition_pct",
            },
            indent=2,
            default=str,
        )
    )
    print(f"RUN_DIR={run_dir}")
    print(results.head(10))
    return run_dir


def _parse_int_tuple(raw: str) -> tuple[int, ...]:
    return tuple(int(value) for value in raw.split(",") if value.strip())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep top expert and trailing-window parameters for generalized EMA expert runs.")
    parser.add_argument("expert_run_dir", type=Path)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--trailing-months", default="12,24,36,60")
    parser.add_argument("--top-experts", default="5,10,20,30,50")
    parser.add_argument("--min-trailing-months", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        SweepConfig(
            expert_run_dir=args.expert_run_dir,
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            output_dir=args.output_dir,
            trailing_months=_parse_int_tuple(args.trailing_months),
            top_experts=_parse_int_tuple(args.top_experts),
            min_trailing_months=args.min_trailing_months,
        )
    )


if __name__ == "__main__":
    main()
