from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent))

from run_signal_copy_models import DEFAULT_LEGACY_PATH, DEFAULT_SOURCE_RUN  # noqa: E402
from run_tradable_ema_regression_optuna import (  # noqa: E402
    _add_cross_sectional_features,
    _base_features_for_set,
)


@dataclass(frozen=True)
class LegacyFactorExposureConfig:
    source_run: Path = DEFAULT_SOURCE_RUN
    legacy_path: Path = DEFAULT_LEGACY_PATH
    output_dir: Path = Path("outputs")
    portfolio_model: str = "Combined_Frequency"
    feature_set: str = "technical"
    start_month: str = "2015-01-01"
    end_month: str | None = None
    top_features: int = 35


def _parse_month(value: str | None) -> date | None:
    if value is None:
        return None
    return datetime.strptime(value, "%Y-%m-%d").date()


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.1f}%"


def _format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _load_feature_frame(config: LegacyFactorExposureConfig) -> tuple[pl.DataFrame, list[str], list[str]]:
    metadata = json.loads((config.source_run / "metadata.json").read_text(encoding="utf-8"))
    source_features = list(metadata["features_used"])
    base_features = _base_features_for_set(source_features, config.feature_set)
    if not base_features:
        raise ValueError(f"No features found for feature_set={config.feature_set!r}.")

    frame = pl.read_parquet(config.source_run / "model_frame.parquet").with_columns(
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col("ticker").cast(pl.Utf8),
    )
    frame, features = _add_cross_sectional_features(frame, base_features, prefix=config.feature_set)

    start = _parse_month(config.start_month)
    end = _parse_month(config.end_month)
    if start is not None:
        frame = frame.filter(pl.col("holding_month") >= start)
    if end is not None:
        frame = frame.filter(pl.col("holding_month") <= end)
    return frame, base_features, features


def _load_legacy_selection(config: LegacyFactorExposureConfig) -> pl.DataFrame:
    legacy = (
        pl.read_parquet(config.legacy_path)
        .filter(pl.col("portfolio_model") == config.portfolio_model)
        .with_columns(
            pl.col("year_month").dt.date().alias("holding_month"),
            pl.col("ticker").cast(pl.Utf8),
        )
    )
    start = _parse_month(config.start_month)
    end = _parse_month(config.end_month)
    if start is not None:
        legacy = legacy.filter(pl.col("holding_month") >= start)
    if end is not None:
        legacy = legacy.filter(pl.col("holding_month") <= end)
    return (
        legacy.group_by(["holding_month", "ticker"])
        .agg(
            pl.first("Sector").alias("sector"),
            pl.max("n_models").cast(pl.Float64).alias("legacy_n_models"),
            pl.max("weight_normalized").cast(pl.Float64).alias("legacy_weight"),
            pl.max("selected_n_asset").cast(pl.Float64).alias("selected_n_asset"),
            pl.max("selected_n_max_per_sector").cast(pl.Float64).alias("selected_n_max_per_sector"),
        )
        .with_columns(pl.lit(1).cast(pl.Int8).alias("legacy_selected"))
    )


def _join_legacy(frame: pl.DataFrame, legacy_selection: pl.DataFrame) -> pl.DataFrame:
    return frame.join(legacy_selection, on=["holding_month", "ticker"], how="left").with_columns(
        pl.col("legacy_selected").fill_null(0).cast(pl.Int8),
        pl.col("legacy_n_models").fill_null(0.0),
        pl.col("legacy_weight").fill_null(0.0),
    )


def _monthly_performance(scored: pl.DataFrame) -> pl.DataFrame:
    selected = scored.filter(pl.col("legacy_selected") == 1)
    universe = (
        scored.group_by("holding_month")
        .agg(
            pl.len().alias("universe_size"),
            pl.mean("future_return").alias("universe_equal_return"),
            pl.mean("future_excess_return").alias("universe_equal_excess_return"),
            (pl.col("future_excess_return") > 0.0).mean().alias("universe_positive_excess_rate"),
        )
    )
    legacy = (
        selected.group_by("holding_month")
        .agg(
            pl.len().alias("legacy_count"),
            pl.mean("future_return").alias("legacy_equal_return"),
            pl.mean("future_excess_return").alias("legacy_equal_excess_return"),
            (pl.col("future_excess_return") > 0.0).mean().alias("legacy_positive_excess_rate"),
            (pl.col("future_return") * pl.col("legacy_weight")).sum().alias("legacy_weighted_return"),
            (pl.col("future_excess_return") * pl.col("legacy_weight")).sum().alias("legacy_weighted_excess_return"),
            pl.sum("legacy_weight").alias("legacy_weight_sum"),
        )
        .with_columns(
            pl.when(pl.col("legacy_weight_sum").abs() > 1e-12)
            .then(pl.col("legacy_weighted_return") / pl.col("legacy_weight_sum"))
            .otherwise(None)
            .alias("legacy_weighted_return"),
            pl.when(pl.col("legacy_weight_sum").abs() > 1e-12)
            .then(pl.col("legacy_weighted_excess_return") / pl.col("legacy_weight_sum"))
            .otherwise(None)
            .alias("legacy_weighted_excess_return"),
        )
    )
    return (
        legacy.join(universe, on="holding_month", how="left")
        .with_columns(
            (pl.col("legacy_equal_return") - pl.col("universe_equal_return")).alias("legacy_vs_universe_return"),
            (pl.col("legacy_equal_excess_return") - pl.col("universe_equal_excess_return")).alias(
                "legacy_vs_universe_excess_return"
            ),
        )
        .sort("holding_month")
    )


def _perf_summary(monthly: pl.DataFrame) -> pl.DataFrame:
    return monthly.select(
        pl.len().alias("months"),
        pl.sum("legacy_count").alias("legacy_positions"),
        pl.mean("legacy_count").alias("avg_legacy_count"),
        pl.min("legacy_count").alias("min_legacy_count"),
        pl.max("legacy_count").alias("max_legacy_count"),
        pl.mean("legacy_equal_return").alias("avg_monthly_legacy_equal_return"),
        pl.mean("legacy_equal_excess_return").alias("avg_monthly_legacy_equal_excess_return"),
        pl.mean("legacy_weighted_return").alias("avg_monthly_legacy_weighted_return"),
        pl.mean("legacy_weighted_excess_return").alias("avg_monthly_legacy_weighted_excess_return"),
        pl.mean("universe_equal_return").alias("avg_monthly_universe_equal_return"),
        pl.mean("universe_equal_excess_return").alias("avg_monthly_universe_equal_excess_return"),
        pl.mean("legacy_vs_universe_return").alias("avg_monthly_legacy_vs_universe_return"),
        pl.mean("legacy_vs_universe_excess_return").alias("avg_monthly_legacy_vs_universe_excess_return"),
        pl.mean("legacy_positive_excess_rate").alias("avg_legacy_positive_excess_rate"),
        pl.mean("universe_positive_excess_rate").alias("avg_universe_positive_excess_rate"),
    )


def _ticker_concentration(scored: pl.DataFrame) -> pl.DataFrame:
    return (
        scored.filter(pl.col("legacy_selected") == 1)
        .group_by("ticker")
        .agg(
            pl.len().alias("selected_months"),
            pl.mean("future_return").alias("avg_future_return_when_selected"),
            pl.mean("future_excess_return").alias("avg_future_excess_when_selected"),
            pl.sum("legacy_weight").alias("total_weight"),
            pl.mean("legacy_weight").alias("avg_weight"),
        )
        .sort(["selected_months", "avg_future_excess_when_selected"], descending=[True, True])
    )


def _sector_exposure(scored: pl.DataFrame) -> pl.DataFrame:
    selected = scored.filter(pl.col("legacy_selected") == 1)
    if "sector" not in selected.columns:
        return pl.DataFrame()
    return (
        selected.group_by("sector")
        .agg(
            pl.len().alias("selected_positions"),
            pl.n_unique("holding_month").alias("active_months"),
            pl.sum("legacy_weight").alias("total_weight"),
            pl.mean("future_return").alias("avg_future_return"),
            pl.mean("future_excess_return").alias("avg_future_excess_return"),
        )
        .sort("selected_positions", descending=True)
    )


def _parse_selected_model(raw: str | None) -> dict[str, float | str | None]:
    if raw is None:
        return {"short_window": None, "long_window": None, "horizon": None, "n_asset": None, "sector_limit": None}
    match = re.match(
        r"(?P<short>[0-9.]+)-(?P<long>[0-9.]+)-(?P<horizon>[0-9.]+)\|asset=(?P<asset>[0-9.]+)\|sector=(?P<sector>[0-9.]+)",
        raw,
    )
    if match is None:
        return {"short_window": None, "long_window": None, "horizon": None, "n_asset": None, "sector_limit": None}
    return {
        "short_window": float(match.group("short")),
        "long_window": float(match.group("long")),
        "horizon": float(match.group("horizon")),
        "n_asset": float(match.group("asset")),
        "sector_limit": float(match.group("sector")),
    }


def _atomic_model_summary(config: LegacyFactorExposureConfig) -> pl.DataFrame:
    legacy = (
        pl.read_parquet(config.legacy_path)
        .filter(pl.col("portfolio_model").str.starts_with("Legacy_Optuna_"))
        .with_columns(pl.col("year_month").dt.date().alias("holding_month"))
    )
    start = _parse_month(config.start_month)
    end = _parse_month(config.end_month)
    if start is not None:
        legacy = legacy.filter(pl.col("holding_month") >= start)
    if end is not None:
        legacy = legacy.filter(pl.col("holding_month") <= end)
    if legacy.is_empty():
        return pl.DataFrame()

    parsed = pl.DataFrame(
        [_parse_selected_model(value) for value in legacy.get_column("selected_model").cast(pl.Utf8).to_list()]
    )
    legacy = legacy.with_columns(parsed)
    return (
        legacy.group_by(["portfolio_model", "selected_model", "short_window", "long_window", "horizon", "n_asset", "sector_limit"])
        .agg(
            pl.len().alias("selected_positions"),
            pl.n_unique("holding_month").alias("active_months"),
            pl.n_unique("ticker").alias("unique_tickers"),
            pl.mean("mtr").alias("avg_mtr"),
            pl.mean("dr").alias("avg_dr"),
        )
        .sort(["selected_positions", "active_months"], descending=[True, True])
    )


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    x = x[mask]
    y = y[mask]
    if np.nanstd(x) <= 1e-12 or np.nanstd(y) <= 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _feature_diagnostics(scored: pl.DataFrame, features: Sequence[str], top_n: int) -> pl.DataFrame:
    rows: list[dict[str, float | str | int]] = []
    minimal = scored.select(
        ["holding_month", "legacy_selected", "future_excess_return", *features]
    ).fill_null(0.0)

    for feature in features:
        monthly_rows: list[dict[str, float]] = []
        for month_df in minimal.select(["holding_month", "legacy_selected", "future_excess_return", feature]).partition_by(
            "holding_month", maintain_order=True
        ):
            if month_df.is_empty():
                continue
            values = month_df.get_column(feature).to_numpy().astype(float)
            selected = month_df.get_column("legacy_selected").to_numpy().astype(bool)
            returns = month_df.get_column("future_excess_return").to_numpy().astype(float)
            if selected.sum() == 0:
                continue
            ranks = month_df.select(
                (pl.col(feature).rank(method="average") / pl.len()).alias("_rank")
            ).get_column("_rank").to_numpy()
            non_selected = ~selected
            monthly_rows.append(
                {
                    "legacy_rank_mean": float(np.nanmean(ranks[selected])),
                    "rank_lift_vs_universe": float(np.nanmean(ranks[selected]) - np.nanmean(ranks)),
                    "value_lift_vs_non_selected": float(
                        np.nanmean(values[selected]) - np.nanmean(values[non_selected])
                        if non_selected.sum() > 0
                        else np.nan
                    ),
                    "feature_legacy_corr": _safe_corr(ranks, selected.astype(float)),
                    "feature_return_ic": _safe_corr(ranks, returns),
                }
            )
        if not monthly_rows:
            continue
        rows.append(
            {
                "feature": feature,
                "months": len(monthly_rows),
                "legacy_rank_mean": float(np.nanmean([row["legacy_rank_mean"] for row in monthly_rows])),
                "rank_lift_vs_universe": float(np.nanmean([row["rank_lift_vs_universe"] for row in monthly_rows])),
                "value_lift_vs_non_selected": float(np.nanmean([row["value_lift_vs_non_selected"] for row in monthly_rows])),
                "feature_legacy_corr": float(np.nanmean([row["feature_legacy_corr"] for row in monthly_rows])),
                "feature_return_ic": float(np.nanmean([row["feature_return_ic"] for row in monthly_rows])),
            }
        )

    if not rows:
        return pl.DataFrame()
    return (
        pl.DataFrame(rows)
        .with_columns(
            (pl.col("rank_lift_vs_universe").abs() * pl.col("feature_return_ic").abs()).alias("signed_signal_score")
        )
        .sort(["rank_lift_vs_universe", "signed_signal_score"], descending=[True, True])
        .head(top_n)
    )


def _write_markdown_report(
    *,
    run_dir: Path,
    config: LegacyFactorExposureConfig,
    perf_summary: pl.DataFrame,
    monthly: pl.DataFrame,
    features: pl.DataFrame,
    sectors: pl.DataFrame,
    tickers: pl.DataFrame,
    atomics: pl.DataFrame,
) -> None:
    summary = perf_summary.to_dicts()[0] if not perf_summary.is_empty() else {}
    best_features = features.head(12).to_dicts() if not features.is_empty() else []
    best_sectors = sectors.head(8).to_dicts() if not sectors.is_empty() else []
    best_tickers = tickers.head(10).to_dicts() if not tickers.is_empty() else []
    best_atomics = atomics.head(10).to_dicts() if not atomics.is_empty() else []

    lines = [
        "# Diagnostic Legacy factor exposures",
        "",
        f"Run directory: `{run_dir}`",
        f"Portfolio Legacy analyse: `{config.portfolio_model}`",
        f"Fenetre: `{config.start_month}` a `{config.end_month or 'fin disponible'}`",
        f"Feature set: `{config.feature_set}`",
        "",
        "## Lecture rapide",
        "",
        "- Ce diagnostic n'entraine pas de modele et n'utilise pas Legacy comme objectif.",
        "- Il mesure ce que Legacy achete, comment ca performe, et quelles features tradables ressemblent aux choix Legacy.",
        "- Les rendements utilises sont les rendements futurs deja presents dans la frame de backtest, donc uniquement pour l'analyse ex post.",
        "",
        "## Performance Legacy vs univers",
        "",
        f"- Mois couverts: `{int(summary.get('months', 0) or 0)}`.",
        f"- Positions Legacy: `{int(summary.get('legacy_positions', 0) or 0)}`.",
        f"- Nombre moyen de positions: `{_format_float(summary.get('avg_legacy_count'), 1)}` "
        f"(min `{_format_float(summary.get('min_legacy_count'), 0)}`, max `{_format_float(summary.get('max_legacy_count'), 0)}`).",
        f"- Rendement mensuel Legacy equal-weight: `{_format_pct(summary.get('avg_monthly_legacy_equal_return'))}`.",
        f"- Excess return mensuel Legacy equal-weight: `{_format_pct(summary.get('avg_monthly_legacy_equal_excess_return'))}`.",
        f"- Excess return mensuel univers equal-weight: `{_format_pct(summary.get('avg_monthly_universe_equal_excess_return'))}`.",
        f"- Lift Legacy vs univers en excess return: `{_format_pct(summary.get('avg_monthly_legacy_vs_universe_excess_return'))}`.",
        f"- Taux de positions Legacy avec excess return positif: `{_format_pct(summary.get('avg_legacy_positive_excess_rate'))}` "
        f"vs univers `{_format_pct(summary.get('avg_universe_positive_excess_rate'))}`.",
        "",
        "## Features tradables les plus alignees avec Legacy",
        "",
        "| feature | rang moyen Legacy | lift rang vs univers | corr feature/Legacy | IC futur excess return |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in best_features:
        lines.append(
            f"| `{row['feature']}` | {_format_float(row.get('legacy_rank_mean'))} | "
            f"{_format_float(row.get('rank_lift_vs_universe'))} | "
            f"{_format_float(row.get('feature_legacy_corr'))} | "
            f"{_format_float(row.get('feature_return_ic'))} |"
        )

    lines.extend(["", "## Secteurs Legacy", "", "| secteur | positions | poids total | excess return moyen |", "|---|---:|---:|---:|"])
    for row in best_sectors:
        lines.append(
            f"| `{row.get('sector')}` | {int(row.get('selected_positions') or 0)} | "
            f"{_format_float(row.get('total_weight'), 2)} | {_format_pct(row.get('avg_future_excess_return'))} |"
        )

    lines.extend(["", "## Tickers les plus frequents", "", "| ticker | mois selectionnes | excess return moyen selection | poids total |", "|---|---:|---:|---:|"])
    for row in best_tickers:
        lines.append(
            f"| `{row.get('ticker')}` | {int(row.get('selected_months') or 0)} | "
            f"{_format_pct(row.get('avg_future_excess_when_selected'))} | {_format_float(row.get('total_weight'), 2)} |"
        )

    lines.extend(["", "## Modeles atomiques Legacy les plus presents", "", "| bloc | modele EMA | positions | mois actifs | tickers |", "|---|---|---:|---:|---:|"])
    for row in best_atomics:
        lines.append(
            f"| `{row.get('portfolio_model')}` | `{row.get('selected_model')}` | "
            f"{int(row.get('selected_positions') or 0)} | {int(row.get('active_months') or 0)} | "
            f"{int(row.get('unique_tickers') or 0)} |"
        )

    lines.extend(
        [
            "",
            "## Fichiers produits",
            "",
            "- `monthly_legacy_performance.csv` : performance mensuelle Legacy vs univers.",
            "- `legacy_performance_summary.csv` : resume global.",
            "- `feature_diagnostics.csv` : exposition Legacy aux features tradables et IC futur.",
            "- `sector_exposure.csv` : exposition sectorielle des paniers Legacy.",
            "- `ticker_concentration.csv` : concentration par ticker.",
            "- `atomic_model_summary.csv` : decomposition des blocs EMA Legacy.",
            "",
            "## Decision R&D",
            "",
            "Ce run sert a separer trois sujets :",
            "",
            "1. ce que Legacy achete vraiment ;",
            "2. quelles features tradables semblent expliquer ce choix ;",
            "3. si ces features ont aussi un lien direct avec le rendement futur relatif.",
            "",
            "La prochaine experience utile doit partir des features qui cochent 2 et 3, puis optimiser un objectif de validation futur, pas une recomposition Legacy.",
        ]
    )

    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: LegacyFactorExposureConfig) -> Path:
    run_dir = config.output_dir / f"legacy_factor_exposure_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    frame, base_features, features = _load_feature_frame(config)
    legacy_selection = _load_legacy_selection(config)
    scored = _join_legacy(frame, legacy_selection)
    scored = scored.filter(pl.col("future_return").is_not_null(), pl.col("future_excess_return").is_not_null())

    monthly = _monthly_performance(scored)
    perf_summary = _perf_summary(monthly)
    tickers = _ticker_concentration(scored)
    sectors = _sector_exposure(scored)
    atomics = _atomic_model_summary(config)
    feature_diag = _feature_diagnostics(scored, features, top_n=config.top_features)

    monthly.write_csv(run_dir / "monthly_legacy_performance.csv")
    perf_summary.write_csv(run_dir / "legacy_performance_summary.csv")
    tickers.write_csv(run_dir / "ticker_concentration.csv")
    if not sectors.is_empty():
        sectors.write_csv(run_dir / "sector_exposure.csv")
    if not atomics.is_empty():
        atomics.write_csv(run_dir / "atomic_model_summary.csv")
    if not feature_diag.is_empty():
        feature_diag.write_csv(run_dir / "feature_diagnostics.csv")

    metadata = {
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "run_dir": str(run_dir),
        "base_feature_count": len(base_features),
        "diagnostic_feature_count": len(features),
        "rows": scored.height,
        "legacy_selected_rows": int(scored.get_column("legacy_selected").sum()),
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    _write_markdown_report(
        run_dir=run_dir,
        config=config,
        perf_summary=perf_summary,
        monthly=monthly,
        features=feature_diag,
        sectors=sectors,
        tickers=tickers,
        atomics=atomics,
    )
    print(run_dir)
    return run_dir


def _parse_args() -> LegacyFactorExposureConfig:
    parser = argparse.ArgumentParser(description="Diagnose Legacy portfolio factor exposures.")
    parser.add_argument("--source-run", type=Path, default=DEFAULT_SOURCE_RUN)
    parser.add_argument("--legacy-path", type=Path, default=DEFAULT_LEGACY_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--portfolio-model", default="Combined_Frequency")
    parser.add_argument("--feature-set", choices=["ema", "technical"], default="technical")
    parser.add_argument("--start-month", default="2015-01-01")
    parser.add_argument("--end-month")
    parser.add_argument("--top-features", type=int, default=35)
    args = parser.parse_args()
    return LegacyFactorExposureConfig(
        source_run=args.source_run,
        legacy_path=args.legacy_path,
        output_dir=args.output_dir,
        portfolio_model=args.portfolio_model,
        feature_set=args.feature_set,
        start_month=args.start_month,
        end_month=args.end_month,
        top_features=args.top_features,
    )


if __name__ == "__main__":
    run(_parse_args())
