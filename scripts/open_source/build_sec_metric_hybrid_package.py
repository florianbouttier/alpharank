#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import polars as pl

from alpharank.data.open_source.legacy_export import export_legacy_compatible_fundamental_outputs
from alpharank.data.ingestion.storage import utc_now_iso
from alpharank.utils.module_loading import load_module_from_path

SCRIPT_DIR = Path(__file__).resolve().parent
_OUTPUT_PACKAGE_MODULE = load_module_from_path(
    "alpharank_local_build_sec_output_package",
    SCRIPT_DIR / "build_sec_output_package.py",
)
_OVERLAY_PACKAGE_MODULE = load_module_from_path(
    "alpharank_local_build_sec_overlay_package",
    SCRIPT_DIR / "build_sec_overlay_package.py",
)
publish_sec_output_package = _OUTPUT_PACKAGE_MODULE.publish_sec_output_package
_load_lineage_frames = _OVERLAY_PACKAGE_MODULE._load_lineage_frames
merge_earnings_long = _OVERLAY_PACKAGE_MODULE.merge_earnings_long
merge_financials = _OVERLAY_PACKAGE_MODULE.merge_financials


DEFAULT_FINANCIAL_METRIC_SOURCES: tuple[tuple[str, str], ...] = (
    ("revenue", "outputs/sec_overlay_fix2_output"),
    ("net_income", "outputs/sec_overlay_fix2_output"),
)

QUARTER_PERIODS = ("Q1", "Q2", "Q3", "Q4")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Build a hybrid SEC package by combining the best metric-level coverage from multiple SEC-only candidates."
    )
    parser.add_argument(
        "--base-sec-dir",
        type=Path,
        default=project_root / "outputs" / "sec_q4_fix2_candidate_combo_output_latest",
    )
    parser.add_argument(
        "--financial-metric-source",
        action="append",
        default=None,
        help="Metric-specific preferred SEC package in the form metric=/abs/or/relative/path.",
    )
    parser.add_argument(
        "--eps-fallback-sec-dir",
        type=Path,
        default=project_root / "outputs" / "sec_overlay_fix2_output",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=project_root / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "outputs" / "sec_kpi_hybrid_output_latest",
    )
    return parser.parse_args()


def _parse_metric_source_args(values: list[str] | None, *, project_root: Path) -> dict[str, Path]:
    raw_values = values if values is not None else [f"{metric}={path}" for metric, path in DEFAULT_FINANCIAL_METRIC_SOURCES]
    mapping: dict[str, Path] = {}
    for value in raw_values:
        if "=" not in value:
            raise ValueError(f"Invalid metric source '{value}'. Expected metric=path.")
        metric, raw_path = value.split("=", 1)
        metric_name = metric.strip()
        source_path = Path(raw_path.strip()).expanduser()
        if not source_path.is_absolute():
            source_path = (project_root / source_path).resolve()
        mapping[metric_name] = source_path
    return mapping


def _split_financials_by_metric(frame: pl.DataFrame, metrics: list[str]) -> tuple[pl.DataFrame, pl.DataFrame]:
    if frame.is_empty():
        return frame, frame
    selected = frame.filter(pl.col("metric").is_in(metrics))
    remaining = frame.filter(pl.col("metric").is_in(metrics).not_())
    return selected, remaining


def _combine_financial_metric(
    *,
    base_financials: pl.DataFrame,
    base_lineage: pl.DataFrame,
    preferred_financials: pl.DataFrame,
    preferred_lineage: pl.DataFrame,
    metric: str,
    preferred_origin: str,
    fallback_origin: str,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    preferred_metric = preferred_financials.filter(pl.col("metric") == metric)
    preferred_lineage_metric = preferred_lineage.filter(pl.col("metric") == metric)
    base_metric = base_financials.filter(pl.col("metric") == metric)
    base_lineage_metric = base_lineage.filter(pl.col("metric") == metric)
    merged_financials, merged_lineage, _ = merge_financials(
        primary_consolidated=preferred_metric,
        secondary_consolidated=base_metric,
        primary_lineage=preferred_lineage_metric,
        secondary_lineage=base_lineage_metric,
        primary_origin=preferred_origin,
        secondary_origin=fallback_origin,
    )
    return merged_financials, merged_lineage


def _select_earnings_upgrade_rows(primary: pl.DataFrame, secondary: pl.DataFrame) -> pl.DataFrame:
    if secondary.is_empty():
        return secondary
    key_cols = ["ticker", "fiscal_year", "fiscal_period"]
    primary_non_null = primary.filter(
        pl.col("fiscal_period").is_in(QUARTER_PERIODS) & pl.col("epsActual").is_not_null()
    ).select(key_cols)
    return secondary.filter(
        pl.col("fiscal_period").is_in(QUARTER_PERIODS) & pl.col("epsActual").is_not_null()
    ).join(primary_non_null, on=key_cols, how="anti")


def merge_earnings_prefer_non_null_actuals(
    *,
    primary_consolidated: pl.DataFrame,
    secondary_consolidated: pl.DataFrame,
    primary_lineage: pl.DataFrame,
    secondary_lineage: pl.DataFrame,
    primary_origin: str = "primary_snapshot",
    secondary_origin: str = "secondary_candidate",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    upgrade_rows = _select_earnings_upgrade_rows(primary_consolidated, secondary_consolidated)
    if upgrade_rows.is_empty():
        audit = (
            primary_consolidated.group_by("overlay_origin")
            .agg(pl.len().alias("rows"))
            .sort("overlay_origin")
            if "overlay_origin" in primary_consolidated.columns and not primary_consolidated.is_empty()
            else pl.DataFrame({"overlay_origin": [], "rows": []}, schema={"overlay_origin": pl.String, "rows": pl.UInt32})
        )
        return primary_consolidated, primary_lineage, audit

    key_cols = ["ticker", "fiscal_year", "fiscal_period"]
    upgrade_keys = upgrade_rows.select(key_cols).unique()
    kept_primary = primary_consolidated.join(upgrade_keys, on=key_cols, how="anti")
    kept_lineage = primary_lineage.join(upgrade_keys, on=key_cols, how="anti")
    upgrade_lineage = secondary_lineage.join(upgrade_keys, on=key_cols, how="semi")
    merged_consolidated = pl.concat([kept_primary, upgrade_rows], how="diagonal_relaxed").sort(["ticker", "period_end"])
    merged_lineage = pl.concat([kept_lineage, upgrade_lineage], how="diagonal_relaxed").sort(["ticker", "period_end"])
    audit = (
        merged_consolidated.group_by("overlay_origin")
        .agg(pl.len().alias("rows"))
        .sort("overlay_origin")
    )
    return merged_consolidated, merged_lineage, audit


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    base_dir = args.base_sec_dir.resolve()
    output_dir = args.output_dir.resolve()
    reference_data_dir = args.reference_data_dir.resolve()
    history_root = output_dir.parent / "history" / "output"

    metric_source_map = _parse_metric_source_args(args.financial_metric_source, project_root=project_root)

    base = _load_lineage_frames(base_dir)
    merged_general = base["general"]
    merged_general_lineage = base["general_lineage"]
    merged_financials = base["financials"]
    merged_financials_lineage = base["financials_lineage"]
    merged_earnings = base["earnings"]
    merged_earnings_lineage = base["earnings_lineage"]
    merged_earnings_long = base["earnings_long"]

    financial_metric_audit_rows: list[dict[str, object]] = []
    for metric, source_dir in metric_source_map.items():
        preferred = _load_lineage_frames(source_dir)
        _, remaining_financials = _split_financials_by_metric(merged_financials, [metric])
        _, remaining_lineage = _split_financials_by_metric(merged_financials_lineage, [metric])
        preferred_origin = f"preferred_metric:{metric}:{source_dir.name}"
        fallback_origin = f"fallback_metric:{metric}:{base_dir.name}"
        selected_financials, selected_lineage = _combine_financial_metric(
            base_financials=merged_financials,
            base_lineage=merged_financials_lineage,
            preferred_financials=preferred["financials"],
            preferred_lineage=preferred["financials_lineage"],
            metric=metric,
            preferred_origin=preferred_origin,
            fallback_origin=fallback_origin,
        )
        merged_financials = pl.concat([remaining_financials, selected_financials], how="diagonal_relaxed").sort(
            ["ticker", "statement", "metric", "date"]
        )
        merged_financials_lineage = pl.concat([remaining_lineage, selected_lineage], how="diagonal_relaxed").sort(
            ["ticker", "statement", "metric", "date"]
        )
        metric_rows = (
            selected_lineage.group_by(["statement", "metric", "overlay_origin"])
            .agg(pl.len().alias("rows"))
            .sort(["statement", "metric", "overlay_origin"])
        )
        financial_metric_audit_rows.extend(metric_rows.to_dicts())

    eps_fallback = _load_lineage_frames(args.eps_fallback_sec_dir.resolve())
    merged_earnings, merged_earnings_lineage, earnings_overlay_audit = merge_earnings_prefer_non_null_actuals(
        primary_consolidated=merged_earnings,
        secondary_consolidated=eps_fallback["earnings"],
        primary_lineage=merged_earnings_lineage,
        secondary_lineage=eps_fallback["earnings_lineage"],
        primary_origin=f"primary_eps:{base_dir.name}",
        secondary_origin=f"fallback_eps:{args.eps_fallback_sec_dir.resolve().name}",
    )
    merged_earnings_long = merge_earnings_long(
        primary_long=merged_earnings_long,
        secondary_long=eps_fallback["earnings_long"],
        primary_origin=f"primary_eps:{base_dir.name}",
        secondary_origin=f"fallback_eps:{args.eps_fallback_sec_dir.resolve().name}",
    )

    staging_dir = output_dir.parent / "_staging_sec_metric_hybrid_package"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    legacy_paths = export_legacy_compatible_fundamental_outputs(
        general_reference=merged_general,
        consolidated_financials=merged_financials,
        consolidated_lineage=merged_financials_lineage,
        earnings_frame=merged_earnings,
        reference_data_dir=reference_data_dir,
        output_dir=staging_dir,
        align_shares_with_earnings_semantics=False,
    )

    manifest = {
        "generated_at": utc_now_iso(),
        "output_dir": str(output_dir),
        "scope": "sec_only_metric_hybrid",
        "base_sec_dir": str(base_dir),
        "financial_metric_sources": {metric: str(path) for metric, path in metric_source_map.items()},
        "eps_fallback_sec_dir": str(args.eps_fallback_sec_dir.resolve()),
        "merge_policy": {
            "financials": "per metric, prefer the selected package by quarter key and backfill missing quarters from the base package",
            "earnings": "prefer the base package; replace only quarters where fallback provides a non-null epsActual and the base does not",
            "earnings_long": "prefer the base package; backfill missing EPS rows from the fallback package",
            "general": "use the base package",
        },
        "summary": {
            "general_rows": merged_general.height,
            "financial_rows": merged_financials.height,
            "earnings_rows": merged_earnings.height,
            "financial_tickers": merged_financials.get_column("ticker").n_unique() if not merged_financials.is_empty() else 0,
            "earnings_tickers": merged_earnings.get_column("ticker").n_unique() if not merged_earnings.is_empty() else 0,
        },
    }
    source_summary = (
        merged_financials.group_by(["statement", "selected_source"])
        .agg(pl.len().alias("rows"))
        .sort(["statement", "selected_source"])
    )
    published = publish_sec_output_package(
        output_dir=output_dir,
        history_root=history_root,
        legacy_paths=legacy_paths,
        constituents_source_path=reference_data_dir / "SP500_Constituents.csv",
        general_reference=merged_general,
        general_reference_lineage=merged_general_lineage,
        consolidated_financials=merged_financials,
        consolidated_lineage=merged_financials_lineage,
        source_summary=source_summary,
        earnings_consolidated=merged_earnings,
        earnings_lineage=merged_earnings_lineage,
        earnings_long=merged_earnings_long,
        manifest=manifest,
    )

    lineage_dir = output_dir / "lineage"
    if financial_metric_audit_rows:
        pl.DataFrame(financial_metric_audit_rows).write_parquet(lineage_dir / "financials_sec_metric_hybrid_audit.parquet")
    else:
        pl.DataFrame(
            schema={"statement": pl.String, "metric": pl.String, "overlay_origin": pl.String, "rows": pl.UInt32}
        ).write_parquet(lineage_dir / "financials_sec_metric_hybrid_audit.parquet")
    earnings_overlay_audit.write_parquet(lineage_dir / "earnings_sec_metric_hybrid_audit.parquet")

    print(f"SEC metric hybrid package written to: {output_dir}")
    if published is not None:
        print(f"Previous output snapshot: {published}")


if __name__ == "__main__":
    main()
