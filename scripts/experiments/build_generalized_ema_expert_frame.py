from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from run_ema_rich_future_target_models import _recomposition_by_month, _recomposition_summary  # noqa: E402
from run_signal_copy_models import DEFAULT_LEGACY_PATH, DEFAULT_SOURCE_RUN  # noqa: E402


DEFAULT_PRICE_PATH = Path("outputs/checkpoints_open_source_20260607/polars_final_price_vs_index.parquet")
DEFAULT_STOCK_FILTER_PATH = Path("outputs/checkpoints_open_source_20260607/polars_stocks_selections.parquet")
DEFAULT_GENERAL_PATH = Path("data/eodhd/output/US_General.parquet")


@dataclass(frozen=True)
class GeneralizedEmaExpertConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    price_path: Path = DEFAULT_PRICE_PATH
    stock_filter_path: Path = DEFAULT_STOCK_FILTER_PATH
    general_path: Path = DEFAULT_GENERAL_PATH
    output_dir: Path = Path("outputs")
    trailing_months: int = 36
    min_trailing_months: int = 6
    top_experts: int = 10
    max_candidates: int | None = None
    candidate_mode: str = "observed"
    short_deltas: tuple[int, ...] = (-10, 0, 10)
    long_deltas: tuple[int, ...] = (-40, 0, 40)


def _ticker_list(values: Iterable[str]) -> str:
    return ",".join(sorted(str(value) for value in values))


def _load_model_frame(source_run: Path) -> pl.DataFrame:
    return pl.read_parquet(source_run / "model_frame.parquet").with_columns(
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


def _with_candidate_ids(candidates: pl.DataFrame) -> pl.DataFrame:
    return (
        candidates.unique()
        .sort(["n_long", "n_short", "selected_n_asset", "selected_n_max_per_sector"])
        .with_row_index("candidate_id")
        .with_columns(
            pl.concat_str(
                [
                    pl.lit("ema_s"),
                    pl.col("n_short").cast(pl.Utf8),
                    pl.lit("_l"),
                    pl.col("n_long").cast(pl.Utf8),
                    pl.lit("_a"),
                    pl.col("selected_n_asset").cast(pl.Utf8),
                    pl.lit("_sec"),
                    pl.col("selected_n_max_per_sector").cast(pl.Utf8),
                ]
            ).alias("expert_id")
        )
    )


def _observed_candidate_base(legacy: pl.DataFrame) -> pl.DataFrame:
    return (
        legacy.filter(pl.col("portfolio_model").str.starts_with("Legacy_Optuna"))
        .select(["n_short", "n_long", "selected_n_asset", "selected_n_max_per_sector"])
        .unique()
    )


def _observed_candidates(legacy: pl.DataFrame, max_candidates: int | None) -> pl.DataFrame:
    candidates = _with_candidate_ids(_observed_candidate_base(legacy))
    if max_candidates is not None:
        candidates = candidates.head(max_candidates)
    return candidates


def _neighbor_candidates(
    legacy: pl.DataFrame,
    *,
    short_deltas: tuple[int, ...],
    long_deltas: tuple[int, ...],
    max_candidates: int | None,
) -> pl.DataFrame:
    rows: list[dict[str, int]] = []
    for candidate in _observed_candidate_base(legacy).iter_rows(named=True):
        for short_delta in short_deltas:
            for long_delta in long_deltas:
                n_short = int(candidate["n_short"]) + int(short_delta)
                n_long = int(candidate["n_long"]) + int(long_delta)
                if n_short < 1 or n_short > 100 or n_long < 50 or n_long > 400:
                    continue
                rows.append(
                    {
                        "n_short": n_short,
                        "n_long": n_long,
                        "selected_n_asset": int(candidate["selected_n_asset"]),
                        "selected_n_max_per_sector": int(candidate["selected_n_max_per_sector"]),
                    }
                )
    candidates = _with_candidate_ids(pl.DataFrame(rows))
    if max_candidates is not None:
        candidates = candidates.head(max_candidates)
    return candidates


def _candidate_frame(config: GeneralizedEmaExpertConfig, legacy: pl.DataFrame) -> pl.DataFrame:
    if config.candidate_mode == "observed":
        return _observed_candidates(legacy, config.max_candidates)
    if config.candidate_mode == "neighbors":
        return _neighbor_candidates(
            legacy,
            short_deltas=config.short_deltas,
            long_deltas=config.long_deltas,
            max_candidates=config.max_candidates,
        )
    raise ValueError(f"Unsupported candidate mode: {config.candidate_mode}")


def _load_sector(path: Path) -> pl.DataFrame:
    general = pl.read_parquet(path)
    if "Code" not in general.columns or "Sector" not in general.columns:
        raise ValueError(f"{path} must contain Code and Sector columns.")
    return (
        general.select(["Code", "Sector"])
        .drop_nulls("Code")
        .with_columns((pl.col("Code").cast(pl.Utf8) + pl.lit(".US")).alias("ticker"))
        .select(["ticker", "Sector"])
        .unique("ticker")
    )


def _load_stock_filter(path: Path) -> pl.DataFrame:
    return (
        pl.read_parquet(path)
        .select(["year_month", "ticker"])
        .with_columns(
            pl.col("year_month").cast(pl.Date).dt.offset_by("1mo").alias("holding_month"),
            pl.col("ticker").cast(pl.Utf8),
        )
        .select(["holding_month", "ticker"])
        .unique()
    )


def _expert_selections_for_candidate(
    prices: pl.DataFrame,
    stock_filter: pl.DataFrame,
    sectors: pl.DataFrame,
    candidate: dict,
) -> pl.DataFrame:
    n_short = int(candidate["n_short"])
    n_long = int(candidate["n_long"])
    n_asset = int(candidate["selected_n_asset"])
    n_sector = int(candidate["selected_n_max_per_sector"])
    expert_id = str(candidate["expert_id"])

    monthly = (
        prices.sort(["ticker", "date"])
        .with_columns(
            pl.col("close_vs_index").ewm_mean(span=n_short, adjust=False).over("ticker").alias("_ema_short"),
            pl.col("close_vs_index").ewm_mean(span=n_long, adjust=False).over("ticker").alias("_ema_long"),
            pl.col("close_vs_index").cum_count().over("ticker").alias("_obs_count"),
            pl.col("date").dt.truncate("1mo").dt.date().alias("signal_month"),
        )
        .filter(pl.col("_obs_count") >= n_long)
        .with_columns((pl.col("_ema_short") / pl.col("_ema_long")).alias("mtr"))
        .drop_nulls("mtr")
        .group_by(["signal_month", "ticker"])
        .agg(pl.col("date").last(), pl.col("mtr").last())
        .with_columns(pl.col("signal_month").dt.offset_by("1mo").alias("holding_month"))
        .join(stock_filter, on=["holding_month", "ticker"], how="inner")
        .join(sectors, on="ticker", how="left")
        .with_columns(pl.col("Sector").fill_null("Unknown"))
        .sort(["holding_month", "Sector", "mtr", "ticker"], descending=[False, False, True, False])
        .with_columns(pl.col("mtr").rank(method="ordinal", descending=True).over(["holding_month", "Sector"]).alias("_sector_rank"))
        .filter(pl.col("_sector_rank") <= n_sector)
        .sort(["holding_month", "mtr", "ticker"], descending=[False, True, False])
        .with_columns(pl.col("mtr").rank(method="ordinal", descending=True).over("holding_month").alias("_asset_rank"))
        .filter(pl.col("_asset_rank") <= n_asset)
        .select(["holding_month", "ticker", "mtr", "_asset_rank"])
        .with_columns(
            pl.lit(expert_id).alias("expert_id"),
            pl.lit(n_short).alias("n_short"),
            pl.lit(n_long).alias("n_long"),
            pl.lit(n_asset).alias("n_asset"),
            pl.lit(n_sector).alias("n_max_per_sector"),
        )
    )
    return monthly


def _build_expert_selections(config: GeneralizedEmaExpertConfig, legacy: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    prices = pl.read_parquet(config.price_path).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("date").cast(pl.Datetime),
    )
    stock_filter = _load_stock_filter(config.stock_filter_path)
    sectors = _load_sector(config.general_path)
    candidates = _candidate_frame(config, legacy)

    frames: list[pl.DataFrame] = []
    for position, row in enumerate(candidates.iter_rows(named=True), start=1):
        print(f"[{position}/{candidates.height}] {row['expert_id']}", flush=True)
        frames.append(_expert_selections_for_candidate(prices, stock_filter, sectors, row))
    selections = pl.concat(frames, how="vertical") if frames else pl.DataFrame()
    return candidates, selections


def _expert_scores(selections: pl.DataFrame, model_frame: pl.DataFrame, trailing_months: int, min_trailing_months: int) -> pl.DataFrame:
    monthly_returns = (
        selections.join(
            model_frame.select(["holding_month", "ticker", "future_excess_return"]),
            on=["holding_month", "ticker"],
            how="left",
        )
        .group_by(["expert_id", "holding_month"])
        .agg(
            pl.mean("future_excess_return").alias("expert_future_excess_return"),
            pl.len().alias("selected_count"),
        )
        .sort(["expert_id", "holding_month"])
        .with_columns(
            pl.col("expert_future_excess_return")
            .shift(1)
            .rolling_mean(window_size=trailing_months, min_samples=min_trailing_months)
            .over("expert_id")
            .alias("expert_trailing_score")
        )
    )
    return monthly_returns


def _ticker_features(selections: pl.DataFrame, expert_scores: pl.DataFrame, top_experts: int) -> pl.DataFrame:
    active_experts = (
        expert_scores.drop_nulls("expert_trailing_score")
        .sort(["holding_month", "expert_trailing_score", "expert_id"], descending=[False, True, False])
        .with_columns(pl.col("expert_trailing_score").rank(method="ordinal", descending=True).over("holding_month").alias("expert_rank_month"))
        .filter(pl.col("expert_rank_month") <= top_experts)
        .select(["holding_month", "expert_id", "expert_trailing_score", "expert_rank_month"])
    )
    return (
        selections.join(active_experts, on=["holding_month", "expert_id"], how="inner")
        .group_by(["holding_month", "ticker"])
        .agg(
            pl.len().alias("learned_ema_expert_vote_count"),
            pl.sum("expert_trailing_score").alias("learned_ema_expert_score_sum"),
            pl.mean("expert_trailing_score").alias("learned_ema_expert_score_mean"),
            pl.max("expert_trailing_score").alias("learned_ema_expert_score_max"),
            pl.mean("mtr").alias("learned_ema_expert_mtr_mean"),
            pl.max("mtr").alias("learned_ema_expert_mtr_max"),
            pl.mean("expert_rank_month").alias("learned_ema_expert_rank_mean"),
        )
    )


def _fill_feature_nulls(frame: pl.DataFrame, feature_cols: list[str]) -> pl.DataFrame:
    return frame.with_columns([pl.col(column).fill_null(0.0).alias(column) for column in feature_cols])


def run(config: GeneralizedEmaExpertConfig) -> Path:
    run_dir = config.output_dir / f"generalized_ema_expert_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    model_frame = _load_model_frame(config.source_run)
    legacy = _load_legacy(config.legacy_path)
    labels = _legacy_labels(legacy)

    candidates, selections = _build_expert_selections(config, legacy)
    expert_scores = _expert_scores(selections, model_frame, config.trailing_months, config.min_trailing_months)
    features = _ticker_features(selections, expert_scores, config.top_experts)
    feature_cols = [column for column in features.columns if column not in {"holding_month", "ticker"}]

    frame = (
        model_frame.join(features, on=["holding_month", "ticker"], how="left")
        .pipe(_fill_feature_nulls, feature_cols)
        .join(labels, on=["holding_month", "ticker"], how="left")
        .with_columns(pl.col("legacy_selected").fill_null(0).cast(pl.Int8))
    )

    diagnostic = frame.select(["ticker", "holding_month", "legacy_selected"] + feature_cols)
    recomposition = _recomposition_by_month(diagnostic, ["learned_ema_expert_vote_count", "learned_ema_expert_score_sum"])
    summary = _recomposition_summary(recomposition)

    candidates.write_csv(run_dir / "ema_expert_candidates.csv")
    selections.write_parquet(run_dir / "ema_expert_selections.parquet")
    expert_scores.write_parquet(run_dir / "ema_expert_monthly_scores.parquet")
    frame.write_parquet(run_dir / "generalized_ema_expert_frame.parquet")
    recomposition.write_csv(run_dir / "deterministic_recomposition_by_month.csv")
    summary.write_csv(run_dir / "deterministic_recomposition_summary.csv")

    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "source_run": str(config.source_run),
                "legacy_path": str(config.legacy_path),
                "price_path": str(config.price_path),
                "stock_filter_path": str(config.stock_filter_path),
                "general_path": str(config.general_path),
                "candidate_count": candidates.height,
                "candidate_mode": config.candidate_mode,
                "short_deltas": list(config.short_deltas),
                "long_deltas": list(config.long_deltas),
                "top_experts": config.top_experts,
                "trailing_months": config.trailing_months,
                "min_trailing_months": config.min_trailing_months,
                "feature_count": len(feature_cols),
                "features": feature_cols,
                "primary_metric": "recomposition_pct",
                "primary_metric_formula": "nombre d'actions communes entre modele et Legacy / nombre d'actions choisies par Legacy",
                "notes": [
                    "Les experts EMA sont selectionnes par performance passee, pas par appartenance au panier Legacy.",
                    "Le mode observed utilise les parametres vus dans Legacy ; le mode neighbors ajoute des voisins EMA autour de ces parametres.",
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
    parser = argparse.ArgumentParser(description="Build generalized EMA expert features scored by trailing future excess returns.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--price-path", type=Path, default=DEFAULT_PRICE_PATH)
    parser.add_argument("--stock-filter-path", type=Path, default=DEFAULT_STOCK_FILTER_PATH)
    parser.add_argument("--general-path", type=Path, default=DEFAULT_GENERAL_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--trailing-months", type=int, default=36)
    parser.add_argument("--min-trailing-months", type=int, default=6)
    parser.add_argument("--top-experts", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--candidate-mode", choices=["observed", "neighbors"], default="observed")
    parser.add_argument("--short-deltas", default="-10,0,10")
    parser.add_argument("--long-deltas", default="-40,0,40")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(
        GeneralizedEmaExpertConfig(
            source_run=args.source_run,
            legacy_path=args.legacy_path,
            price_path=args.price_path,
            stock_filter_path=args.stock_filter_path,
            general_path=args.general_path,
            output_dir=args.output_dir,
            trailing_months=args.trailing_months,
            min_trailing_months=args.min_trailing_months,
            top_experts=args.top_experts,
            max_candidates=args.max_candidates,
            candidate_mode=args.candidate_mode,
            short_deltas=tuple(int(value) for value in args.short_deltas.split(",") if value.strip()),
            long_deltas=tuple(int(value) for value in args.long_deltas.split(",") if value.strip()),
        )
    )


if __name__ == "__main__":
    main()
