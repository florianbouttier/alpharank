from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl


DEFAULT_RUNS = (
    Path("outputs/tradable_ema_regression_optuna_20260627_121144"),
    Path("outputs/tradable_ema_regression_optuna_20260627_125020"),
    Path("outputs/tradable_ema_regression_optuna_20260627_133211"),
    Path("outputs/tradable_ema_regression_optuna_20260627_140859"),
    Path("outputs/tradable_ema_regression_optuna_20260627_144652"),
)


@dataclass(frozen=True)
class ForecastCalibrationConfig:
    prediction_runs: tuple[Path, ...] = DEFAULT_RUNS
    output_dir: Path = Path("outputs")
    score_col: str | None = None
    top_k_values: tuple[int, ...] = (5, 10, 20, 30, 50)
    n_bins: int = 10


def _format_pct(value: float | None) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


def _format_float(value: float | None, digits: int = 4) -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def _metadata(path: Path) -> dict[str, Any]:
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def _score_col(path: Path, override: str | None) -> str:
    if override:
        return override
    metadata = _metadata(path)
    score_col = metadata.get("score_col")
    if not score_col:
        raise ValueError(f"Missing score_col in metadata and no --score-col override provided: {path}")
    return str(score_col)


def _load_predictions(path: Path, score_col: str) -> pl.DataFrame:
    predictions_path = path / "predictions.parquet"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing predictions parquet: {predictions_path}")
    frame = pl.read_parquet(predictions_path).with_columns(
        pl.col("ticker").cast(pl.Utf8),
        pl.col("year_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
        pl.col(score_col).cast(pl.Float64),
        pl.col("future_excess_return").cast(pl.Float64),
    )
    return frame.filter(pl.col(score_col).is_not_null(), pl.col("future_excess_return").is_not_null())


def _with_bins(frame: pl.DataFrame, score_col: str, n_bins: int) -> pl.DataFrame:
    rank_pct = pl.col(score_col).rank(method="average").over("year_month") / pl.len().over("year_month")
    return frame.with_columns(
        pl.min_horizontal(
            n_bins,
            pl.max_horizontal(1, (rank_pct * n_bins).ceil().cast(pl.Int64)),
        ).alias("score_bin")
    )


def _decile_calibration(frame: pl.DataFrame, score_col: str) -> pl.DataFrame:
    monthly = (
        frame.group_by(["holding_month", "score_bin"])
        .agg(
            pl.len().alias("rows"),
            pl.mean(score_col).alias("avg_prediction"),
            pl.mean("future_excess_return").alias("avg_future_excess_return"),
            pl.median("future_excess_return").alias("median_future_excess_return"),
            (pl.col("future_excess_return") > 0.0).mean().alias("positive_excess_rate"),
        )
        .sort(["holding_month", "score_bin"])
    )
    return (
        monthly.group_by("score_bin")
        .agg(
            pl.sum("rows").alias("rows"),
            pl.mean("avg_prediction").alias("avg_prediction_monthly"),
            pl.mean("avg_future_excess_return").alias("avg_future_excess_return_monthly"),
            pl.mean("median_future_excess_return").alias("median_future_excess_return_monthly"),
            pl.mean("positive_excess_rate").alias("positive_excess_rate_monthly"),
        )
        .sort("score_bin")
    )


def _top_k_summary(frame: pl.DataFrame, score_col: str, top_k_values: Sequence[int]) -> pl.DataFrame:
    universe = frame.group_by("holding_month").agg(
        pl.mean("future_excess_return").alias("universe_excess_return"),
        (pl.col("future_excess_return") > 0.0).mean().alias("universe_hit_rate"),
    )
    rows: list[dict[str, Any]] = []
    ranked = frame.with_columns(
        pl.col(score_col).rank(method="ordinal", descending=True).over("year_month").alias("rank")
    )
    for top_k in top_k_values:
        selected = ranked.filter(pl.col("rank") <= top_k)
        monthly = (
            selected.group_by("holding_month")
            .agg(
                pl.len().alias("n"),
                pl.mean(score_col).alias("avg_prediction"),
                pl.mean("future_excess_return").alias("avg_future_excess_return"),
                (pl.col("future_excess_return") > 0.0).mean().alias("positive_excess_rate"),
            )
            .join(universe, on="holding_month", how="left")
            .with_columns(
                (pl.col("avg_future_excess_return") - pl.col("universe_excess_return")).alias("excess_lift_vs_universe"),
                (pl.col("positive_excess_rate") - pl.col("universe_hit_rate")).alias("hit_rate_lift_vs_universe"),
            )
        )
        rows.append(
            {
                "top_k": top_k,
                "months": monthly.height,
                "avg_prediction": float(monthly.get_column("avg_prediction").mean()),
                "avg_future_excess_return": float(monthly.get_column("avg_future_excess_return").mean()),
                "positive_excess_rate": float(monthly.get_column("positive_excess_rate").mean()),
                "excess_lift_vs_universe": float(monthly.get_column("excess_lift_vs_universe").mean()),
                "hit_rate_lift_vs_universe": float(monthly.get_column("hit_rate_lift_vs_universe").mean()),
            }
        )
    return pl.DataFrame(rows)


def _calibration_summary(deciles: pl.DataFrame, top_k: pl.DataFrame) -> dict[str, Any]:
    if deciles.is_empty():
        return {}
    x = deciles.get_column("score_bin").to_numpy().astype(float)
    y = deciles.get_column("avg_future_excess_return_monthly").to_numpy().astype(float)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() >= 2 and np.nanstd(y[mask]) > 1e-12:
        bin_return_corr = float(np.corrcoef(x[mask], y[mask])[0, 1])
    else:
        bin_return_corr = float("nan")
    bottom = deciles.filter(pl.col("score_bin") == pl.col("score_bin").min()).select(
        pl.first("avg_future_excess_return_monthly")
    )[0, 0]
    top = deciles.filter(pl.col("score_bin") == pl.col("score_bin").max()).select(
        pl.first("avg_future_excess_return_monthly")
    )[0, 0]
    best_top_k = top_k.sort("avg_future_excess_return", descending=True).row(0, named=True) if not top_k.is_empty() else {}
    return {
        "bin_return_corr": bin_return_corr,
        "top_bin_future_excess_return": float(top),
        "bottom_bin_future_excess_return": float(bottom),
        "top_minus_bottom_future_excess_return": float(top - bottom),
        "best_top_k": best_top_k.get("top_k"),
        "best_top_k_future_excess_return": best_top_k.get("avg_future_excess_return"),
        "best_top_k_lift_vs_universe": best_top_k.get("excess_lift_vs_universe"),
    }


def _run_label(path: Path) -> str:
    metadata = _metadata(path)
    objective = metadata.get("objective_mode", "unknown")
    top_k = metadata.get("objective_top_k", "na")
    return f"{path.name}|{objective}|k={top_k}"


def _write_report(
    run_dir: Path,
    rows: list[dict[str, Any]],
    config: ForecastCalibrationConfig,
) -> None:
    lines = [
        "# Return forecast calibration",
        "",
        "Objectif de cette analyse : verifier si les scores de regression boosting quantifient vraiment le rendement futur relatif moyen.",
        "",
        "Lecture :",
        "",
        "- `top bin` = actions dans le meilleur decile mensuel selon la prediction.",
        "- `top minus bottom` = rendement relatif moyen du meilleur decile moins celui du pire decile.",
        "- `best top K` = panier mensuel top K qui maximise le rendement relatif moyen ex post dans ce diagnostic.",
        "- Aucun KPI Legacy n'est utilise ici comme objectif.",
        "",
        f"Bins: `{config.n_bins}`.",
        f"Top K testes: `{', '.join(str(value) for value in config.top_k_values)}`.",
        "",
        "## Synthese",
        "",
        "| run | objectif Optuna | corr decile/rendement | top decile excess | bottom decile excess | top-bottom | meilleur top K | excess top K | lift vs univers |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['run']}` | `{row['objective']}` | "
            f"{_format_float(row.get('bin_return_corr'))} | "
            f"{_format_pct(row.get('top_bin_future_excess_return'))} | "
            f"{_format_pct(row.get('bottom_bin_future_excess_return'))} | "
            f"{_format_pct(row.get('top_minus_bottom_future_excess_return'))} | "
            f"`{row.get('best_top_k')}` | "
            f"{_format_pct(row.get('best_top_k_future_excess_return'))} | "
            f"{_format_pct(row.get('best_top_k_lift_vs_universe'))} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Une prediction exploitable pour allocation doit montrer au minimum :",
            "",
            "1. une relation monotone entre score et rendement futur moyen ;",
            "2. un top decile clairement meilleur que le bottom decile ;",
            "3. un panier top K avec lift positif vs univers sur validation/test walk-forward ;",
            "4. ensuite seulement, un backtest portefeuille avec contraintes de risque.",
            "",
            "Si ces conditions ne tiennent pas, le modele ne quantifie pas assez bien le rendement moyen par action pour etre une base d'allocation.",
        ]
    )
    (run_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: ForecastCalibrationConfig) -> Path:
    run_dir = config.output_dir / f"return_forecast_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for prediction_run in config.prediction_runs:
        score_col = _score_col(prediction_run, config.score_col)
        metadata = _metadata(prediction_run)
        predictions = _with_bins(_load_predictions(prediction_run, score_col), score_col, config.n_bins)
        deciles = _decile_calibration(predictions, score_col)
        top_k = _top_k_summary(predictions, score_col, config.top_k_values)
        summary = _calibration_summary(deciles, top_k)
        label = _run_label(prediction_run)
        safe_label = prediction_run.name
        deciles.write_csv(run_dir / f"{safe_label}_decile_calibration.csv")
        top_k.write_csv(run_dir / f"{safe_label}_topk_summary.csv")
        summary_rows.append(
            {
                "run": prediction_run.name,
                "objective": f"{metadata.get('objective_mode', 'unknown')} k={metadata.get('objective_top_k', 'na')}",
                "score_col": score_col,
                "label": label,
                **summary,
            }
        )

    summary_frame = pl.DataFrame(summary_rows)
    summary_frame.write_csv(run_dir / "summary.csv")
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "prediction_runs": [str(path) for path in config.prediction_runs],
                "score_col_override": config.score_col,
                "top_k_values": list(config.top_k_values),
                "n_bins": config.n_bins,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_report(run_dir, summary_rows, config)
    print(run_dir)
    return run_dir


def _parse_args() -> ForecastCalibrationConfig:
    parser = argparse.ArgumentParser(description="Analyze whether prediction scores quantify future excess return.")
    parser.add_argument("--prediction-run", type=Path, nargs="*", default=list(DEFAULT_RUNS))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--score-col")
    parser.add_argument("--top-k", type=int, nargs="*", default=[5, 10, 20, 30, 50])
    parser.add_argument("--n-bins", type=int, default=10)
    args = parser.parse_args()
    return ForecastCalibrationConfig(
        prediction_runs=tuple(args.prediction_run),
        output_dir=args.output_dir,
        score_col=args.score_col,
        top_k_values=tuple(args.top_k),
        n_bins=args.n_bins,
    )


if __name__ == "__main__":
    run(_parse_args())
