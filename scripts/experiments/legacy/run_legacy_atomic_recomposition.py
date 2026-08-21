from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Sequence

import polars as pl


DEFAULT_LEGACY_PATH = Path("outputs/2026-06-07/legacy_detailed_returns_polars.parquet")

COMBINED_MODELS = ("Combined_Equal", "Combined_Frequency")
OPTUNA_MODELS = ("Legacy_Optuna_11", "Legacy_Optuna_12", "Legacy_Optuna_21", "Legacy_Optuna_22")


@dataclass(frozen=True)
class AtomicRecompositionConfig:
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")


def _ticker_list(values: Iterable[str]) -> str:
    return ",".join(sorted(str(value) for value in values))


def _load_legacy(path: Path) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if "year_month" not in frame.columns or "ticker" not in frame.columns or "portfolio_model" not in frame.columns:
        raise ValueError(f"{path} does not look like a legacy detailed portfolio file.")
    return frame.with_columns(
        pl.col("year_month").dt.date().alias("holding_month"),
        pl.col("ticker").cast(pl.Utf8),
    )


def _combined_target(frame: pl.DataFrame) -> pl.DataFrame:
    return (
        frame.filter(pl.col("portfolio_model").is_in(COMBINED_MODELS))
        .select(["holding_month", "ticker"])
        .unique()
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )


def _optuna_rows(frame: pl.DataFrame) -> pl.DataFrame:
    required = [
        "holding_month",
        "ticker",
        "portfolio_model",
        "selected_model",
        "n_long",
        "n_short",
        "n_asset",
        "selected_n_asset",
        "selected_n_max_per_sector",
        "quantile_mtr",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing expected columns in legacy detailed file: {missing}")

    return (
        frame.filter(pl.col("portfolio_model").is_in(OPTUNA_MODELS))
        .select(required)
        .unique()
        .with_columns(
            pl.concat_str(
                [
                    pl.col("portfolio_model"),
                    pl.lit("|"),
                    pl.col("selected_model").fill_null("unknown"),
                ]
            ).alias("atomic_model")
        )
    )


def _recomposition_rows(
    *,
    candidates: pl.DataFrame,
    target: pl.DataFrame,
    group_cols: Sequence[str],
    model_col: str,
) -> pl.DataFrame:
    rows: list[dict] = []
    target_by_month = {
        month_df.get_column("holding_month")[0]: set(month_df.get_column("ticker").to_list())
        for month_df in target.partition_by("holding_month", maintain_order=True)
    }

    for candidate_df in candidates.partition_by(list(group_cols) + ["holding_month"], maintain_order=True):
        month = candidate_df.get_column("holding_month")[0]
        legacy_tickers = target_by_month.get(month, set())
        if not legacy_tickers:
            continue

        model_tickers = set(candidate_df.get_column("ticker").to_list())
        common_tickers = model_tickers & legacy_tickers
        first = candidate_df.row(0, named=True)
        rows.append(
            {
                "model": str(first[model_col]),
                "holding_month": month,
                "common_count": len(common_tickers),
                "legacy_count": len(legacy_tickers),
                "model_count": len(model_tickers),
                "recomposition_pct": len(common_tickers) / len(legacy_tickers),
                "legacy_tickers": _ticker_list(legacy_tickers),
                "model_tickers": _ticker_list(model_tickers),
                "common_tickers": _ticker_list(common_tickers),
                "missed_legacy_tickers": _ticker_list(legacy_tickers - model_tickers),
                "added_model_tickers": _ticker_list(model_tickers - legacy_tickers),
            }
        )

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows).sort(["model", "holding_month"])


def _summary(by_month: pl.DataFrame) -> pl.DataFrame:
    if by_month.is_empty():
        return by_month
    return (
        by_month.group_by("model")
        .agg(
            pl.sum("common_count").alias("common_count"),
            pl.sum("legacy_count").alias("legacy_count"),
            pl.sum("model_count").alias("model_count"),
            (pl.sum("common_count") / pl.sum("legacy_count")).alias("recomposition_pct"),
            pl.mean("recomposition_pct").alias("mean_monthly_recomposition_pct"),
            pl.median("recomposition_pct").alias("median_monthly_recomposition_pct"),
            pl.min("holding_month").alias("first_holding_month"),
            pl.max("holding_month").alias("last_holding_month"),
            pl.len().alias("months"),
        )
        .sort(["recomposition_pct", "months"], descending=[True, True])
    )


def _union_rows(optuna: pl.DataFrame, target: pl.DataFrame) -> pl.DataFrame:
    union = (
        optuna.select(["holding_month", "ticker"])
        .unique()
        .with_columns(pl.lit("all_optuna_union").alias("model"))
    )
    return _recomposition_rows(
        candidates=union,
        target=target,
        group_cols=["model"],
        model_col="model",
    )


def run(config: AtomicRecompositionConfig) -> Path:
    legacy = _load_legacy(config.legacy_path)
    target = _combined_target(legacy)
    optuna = _optuna_rows(legacy)

    run_dir = config.output_dir / f"legacy_atomic_recomposition_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    block_by_month = _recomposition_rows(
        candidates=optuna.with_columns(pl.col("portfolio_model").alias("model")),
        target=target,
        group_cols=["model"],
        model_col="model",
    )
    atomic_by_month = _recomposition_rows(
        candidates=optuna,
        target=target,
        group_cols=["atomic_model"],
        model_col="atomic_model",
    )
    union_by_month = _union_rows(optuna, target)

    block_summary = _summary(block_by_month)
    atomic_summary = _summary(atomic_by_month)
    union_summary = _summary(union_by_month)

    block_by_month.write_csv(run_dir / "optuna_block_recomposition_by_month.csv")
    block_summary.write_csv(run_dir / "optuna_block_recomposition_summary.csv")
    atomic_by_month.write_csv(run_dir / "atomic_model_recomposition_by_month.csv")
    atomic_summary.write_csv(run_dir / "atomic_model_recomposition_summary.csv")
    union_by_month.write_csv(run_dir / "all_optuna_union_recomposition_by_month.csv")
    union_summary.write_csv(run_dir / "all_optuna_union_recomposition_summary.csv")
    atomic_summary.head(30).write_csv(run_dir / "best_atomic_models.csv")

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "legacy_path": str(config.legacy_path),
                "target": list(COMBINED_MODELS),
                "optuna_models": list(OPTUNA_MODELS),
                "primary_metric": "recomposition_pct",
                "primary_metric_formula": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "notes": [
                    "Les blocs Optuna et modeles atomiques sont mesures avec leur nombre reel de tickers.",
                    "Le denominateur reste toujours le nombre de tickers du panier Legacy combine.",
                    "Cette experience ne fait aucun entrainement ML ; elle isole la mecanique Legacy.",
                ],
            },
            indent=2,
            default=str,
        )
    )

    print(f"RUN_DIR={run_dir}")
    print("all_optuna_union")
    print(union_summary)
    print("optuna_blocks")
    print(block_summary)
    print("best_atomic_models")
    print(atomic_summary.head(10))
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure recomposition of Legacy combined portfolios by atomic Optuna model.")
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(AtomicRecompositionConfig(legacy_path=args.legacy_path, output_dir=args.output_dir))


if __name__ == "__main__":
    main()
