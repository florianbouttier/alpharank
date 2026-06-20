#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import polars as pl


DEFAULT_SCENARIOS: tuple[tuple[str, str], ...] = (
    ("baseline", "outputs/sec_core_kpi_yearly_report_latest"),
    ("fix2", "outputs/sec_overlay_fix2_yearly"),
    ("q4_probe", "outputs/sec_q4_probe_yearly"),
    ("q4_fix2_combo", "outputs/sec_q4_fix2_combo_yearly"),
    ("q4_fix2_candidate", "outputs/sec_q4_fix2_candidate_combo_yearly_latest"),
    ("metric_hybrid", "outputs/sec_kpi_hybrid_yearly_latest"),
)


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Compare SEC KPI scenarios by worst-year missing percentages.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "sec_kpi_scenario_comparison_latest",
    )
    return parser.parse_args()


def _load_scenarios(*, project_root: Path) -> pl.DataFrame:
    frames: list[pl.DataFrame] = []
    for scenario, relative_dir in DEFAULT_SCENARIOS:
        csv_path = project_root / relative_dir / "worst_year_brief.csv"
        if not csv_path.exists():
            continue
        frames.append(
            pl.read_csv(csv_path)
            .with_columns(pl.lit(scenario).alias("scenario"))
            .select(
                [
                    "scenario",
                    "metric",
                    "metric_label",
                    "fiscal_year",
                    "missing_quarters",
                    "missing_pct",
                    "fill_pct",
                    "top_tickers",
                ]
            )
        )
    if not frames:
        return pl.DataFrame(
            schema={
                "scenario": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
                "fiscal_year": pl.Int64,
                "missing_quarters": pl.Int64,
                "missing_pct": pl.Float64,
                "fill_pct": pl.Float64,
                "top_tickers": pl.String,
            }
        )
    return pl.concat(frames, how="diagonal_relaxed")


def _build_metric_ranking(scenarios: pl.DataFrame) -> pl.DataFrame:
    if scenarios.is_empty():
        return scenarios
    return (
        scenarios.sort(["metric", "missing_pct", "missing_quarters", "scenario"], descending=[False, False, False, False])
        .with_columns(pl.int_range(1, pl.len() + 1).over("metric").alias("metric_rank"))
    )


def _build_global_summary(scenarios: pl.DataFrame) -> pl.DataFrame:
    if scenarios.is_empty():
        return pl.DataFrame(
            schema={
                "scenario": pl.String,
                "worst_metric_pct": pl.Float64,
                "worst_metric_missing_quarters": pl.Int64,
                "global_rank": pl.Int64,
            }
        )
    summary = (
        scenarios.group_by("scenario")
        .agg(
            [
                pl.col("missing_pct").max().alias("worst_metric_pct"),
                pl.col("missing_quarters").max().alias("worst_metric_missing_quarters"),
            ]
        )
        .sort(["worst_metric_pct", "worst_metric_missing_quarters", "scenario"], descending=[False, False, False])
        .with_columns(pl.int_range(1, pl.len() + 1).alias("global_rank"))
    )
    return summary


def _render_markdown(*, ranking: pl.DataFrame, global_summary: pl.DataFrame) -> str:
    lines = [
        "# Comparaison des scenarios KPI SEC",
        "",
        "Lecture:",
        "- pour chaque KPI, on regarde la pire année",
        "- `missing_pct` est le pourcentage de trimestres manquants sur cette pire année",
        "- plus c'est bas, mieux c'est",
        "",
        "## Classement global",
    ]
    for row in global_summary.to_dicts():
        lines.append(
            f"- {row['scenario']}: rang global {row['global_rank']}, pire KPI a {row['worst_metric_pct']:.2f}% "
            f"({row['worst_metric_missing_quarters']} trous)."
        )
    lines.append("")
    lines.append("## Detail par KPI")
    for metric in ["revenue", "net_income", "epsActual"]:
        metric_rows = ranking.filter(pl.col("metric") == metric)
        if metric_rows.is_empty():
            continue
        label = metric_rows.row(0, named=True)["metric_label"]
        lines.append(f"### {label}")
        for row in metric_rows.to_dicts():
            lines.append(
                f"- rang {row['metric_rank']}: {row['scenario']} -> année {row['fiscal_year']}, "
                f"{row['missing_quarters']} trous, {row['missing_pct']:.2f}% manquants, "
                f"{row['fill_pct']:.2f}% de couverture."
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = _load_scenarios(project_root=project_root)
    ranking = _build_metric_ranking(scenarios)
    global_summary = _build_global_summary(scenarios)

    scenarios.write_csv(output_dir / "scenario_briefs.csv")
    ranking.write_csv(output_dir / "metric_ranking.csv")
    global_summary.write_csv(output_dir / "global_ranking.csv")
    (output_dir / "summary.md").write_text(
        _render_markdown(ranking=ranking, global_summary=global_summary),
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(
        json.dumps(
            {
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "scenarios": [{"name": name, "source_dir": source_dir} for name, source_dir in DEFAULT_SCENARIOS],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(output_dir)
    print(output_dir / "summary.md")


if __name__ == "__main__":
    main()
