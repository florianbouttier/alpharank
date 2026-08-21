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
publish_sec_output_package = _OUTPUT_PACKAGE_MODULE.publish_sec_output_package


QUARTER_PERIODS = ("Q1", "Q2", "Q3", "Q4")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Build a SEC-only overlay package from a primary SEC snapshot plus one or more candidate gap-fill snapshots."
    )
    parser.add_argument("--primary-sec-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument("--secondary-sec-dir", type=Path, action="append", required=True)
    parser.add_argument("--reference-data-dir", type=Path, default=project_root / "data")
    parser.add_argument("--output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    return parser.parse_args()


def _with_overlay_origin(frame: pl.DataFrame, origin: str) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    if "overlay_origin" in frame.columns:
        return frame
    return frame.with_columns(pl.lit(origin).alias("overlay_origin"))


def _merge_quartered_frames(
    *,
    primary: pl.DataFrame,
    secondary: pl.DataFrame,
    key_cols: list[str],
    nonquarter_date_col: str,
) -> pl.DataFrame:
    if primary.is_empty():
        return secondary
    if secondary.is_empty():
        return primary

    quarter_primary = primary.filter(pl.col(key_cols[-1]).is_in(QUARTER_PERIODS))
    quarter_secondary = secondary.filter(pl.col(key_cols[-1]).is_in(QUARTER_PERIODS))
    nonquarter_primary = primary.filter(pl.col(key_cols[-1]).is_in(QUARTER_PERIODS).not_())
    nonquarter_secondary = secondary.filter(pl.col(key_cols[-1]).is_in(QUARTER_PERIODS).not_())
    date_identity_cols = key_cols[:-2] + [nonquarter_date_col]

    filled_quarters = pl.concat(
        [
            quarter_primary,
            quarter_secondary
            .join(quarter_primary.select(key_cols).unique(), on=key_cols, how="anti")
            .join(
                quarter_primary.select(date_identity_cols).unique(),
                on=date_identity_cols,
                how="anti",
            ),
        ],
        how="diagonal_relaxed",
    )
    filled_nonquarters = pl.concat(
        [
            nonquarter_primary,
            nonquarter_secondary.join(
                nonquarter_primary.select([col for col in key_cols if col != key_cols[-1]] + [nonquarter_date_col]).unique(),
                on=[col for col in key_cols if col != key_cols[-1]] + [nonquarter_date_col],
                how="anti",
            ),
        ],
        how="diagonal_relaxed",
    )
    return pl.concat([filled_quarters, filled_nonquarters], how="diagonal_relaxed")


def merge_financials(
    *,
    primary_consolidated: pl.DataFrame,
    secondary_consolidated: pl.DataFrame,
    primary_lineage: pl.DataFrame,
    secondary_lineage: pl.DataFrame,
    primary_origin: str = "primary_snapshot",
    secondary_origin: str = "secondary_candidate",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    key_cols = ["ticker", "statement", "metric", "selected_fiscal_year", "selected_fiscal_period"]
    merged_consolidated = _merge_quartered_frames(
        primary=_with_overlay_origin(primary_consolidated, primary_origin),
        secondary=_with_overlay_origin(secondary_consolidated, secondary_origin),
        key_cols=key_cols,
        nonquarter_date_col="date",
    ).sort(["ticker", "statement", "metric", "date"])
    merged_lineage = _merge_quartered_frames(
        primary=_with_overlay_origin(primary_lineage, primary_origin),
        secondary=_with_overlay_origin(secondary_lineage, secondary_origin),
        key_cols=key_cols,
        nonquarter_date_col="date",
    ).sort(["ticker", "statement", "metric", "date"])
    overlay_audit = (
        merged_lineage.group_by(["statement", "metric", "overlay_origin"])
        .agg(pl.len().alias("rows"))
        .sort(["statement", "metric", "overlay_origin"])
    )
    return merged_consolidated, merged_lineage, overlay_audit


def merge_earnings(
    *,
    primary_consolidated: pl.DataFrame,
    secondary_consolidated: pl.DataFrame,
    primary_lineage: pl.DataFrame,
    secondary_lineage: pl.DataFrame,
    primary_origin: str = "primary_snapshot",
    secondary_origin: str = "secondary_candidate",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    key_cols = ["ticker", "fiscal_year", "fiscal_period"]
    merged_consolidated = _merge_quartered_frames(
        primary=_with_overlay_origin(primary_consolidated, primary_origin),
        secondary=_with_overlay_origin(secondary_consolidated, secondary_origin),
        key_cols=key_cols,
        nonquarter_date_col="period_end",
    ).sort(["ticker", "period_end"])
    merged_lineage = _merge_quartered_frames(
        primary=_with_overlay_origin(primary_lineage, primary_origin),
        secondary=_with_overlay_origin(secondary_lineage, secondary_origin),
        key_cols=key_cols,
        nonquarter_date_col="period_end",
    ).sort(["ticker", "period_end"])
    overlay_audit = (
        merged_lineage.group_by(["overlay_origin"])
        .agg(pl.len().alias("rows"))
        .sort("overlay_origin")
    )
    return merged_consolidated, merged_lineage, overlay_audit


def merge_general(
    *,
    primary_general: pl.DataFrame,
    secondary_general: pl.DataFrame,
    primary_lineage: pl.DataFrame,
    secondary_lineage: pl.DataFrame,
    primary_origin: str = "primary_snapshot",
    secondary_origin: str = "secondary_candidate",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    merged_general = pl.concat(
        [
            _with_overlay_origin(primary_general, primary_origin),
            _with_overlay_origin(secondary_general, secondary_origin).join(
                primary_general.select("ticker").unique(),
                on="ticker",
                how="anti",
            ),
        ],
        how="diagonal_relaxed",
    ).sort("ticker")
    merged_lineage = pl.concat(
        [
            _with_overlay_origin(primary_lineage, primary_origin),
            _with_overlay_origin(secondary_lineage, secondary_origin).join(
                primary_lineage.select("ticker").unique(),
                on="ticker",
                how="anti",
            ),
        ],
        how="diagonal_relaxed",
    ).sort("ticker")
    return merged_general, merged_lineage


def merge_earnings_long(
    *,
    primary_long: pl.DataFrame,
    secondary_long: pl.DataFrame,
    primary_origin: str = "primary_snapshot",
    secondary_origin: str = "secondary_candidate",
) -> pl.DataFrame:
    required_cols = {"selected_fiscal_year", "selected_fiscal_period"}
    if not required_cols.issubset(primary_long.columns) or not required_cols.issubset(secondary_long.columns):
        primary = _with_overlay_origin(primary_long, primary_origin)
        secondary = _with_overlay_origin(secondary_long, secondary_origin)
        return pl.concat(
            [
                primary,
                secondary.join(primary.select(["ticker", "metric", "date"]).unique(), on=["ticker", "metric", "date"], how="anti"),
            ],
            how="diagonal_relaxed",
        ).sort(["ticker", "date", "metric", "filing_date"])

    key_cols = ["ticker", "metric", "selected_fiscal_year", "selected_fiscal_period"]
    return _merge_quartered_frames(
        primary=_with_overlay_origin(primary_long, primary_origin),
        secondary=_with_overlay_origin(secondary_long, secondary_origin),
        key_cols=key_cols,
        nonquarter_date_col="date",
    ).sort(["ticker", "date", "metric", "filing_date"])


def _load_lineage_frames(sec_dir: Path) -> dict[str, pl.DataFrame]:
    lineage_dir = sec_dir / "lineage"
    return {
        "general": pl.read_parquet(lineage_dir / "general_reference.parquet"),
        "general_lineage": pl.read_parquet(lineage_dir / "general_reference_lineage.parquet"),
        "financials": pl.read_parquet(lineage_dir / "financials_sec_consolidated.parquet"),
        "financials_lineage": pl.read_parquet(lineage_dir / "financials_sec_lineage.parquet"),
        "source_summary": pl.read_parquet(lineage_dir / "financials_sec_source_summary.parquet"),
        "earnings": pl.read_parquet(lineage_dir / "earnings_sec_consolidated.parquet"),
        "earnings_lineage": pl.read_parquet(lineage_dir / "earnings_sec_lineage.parquet"),
        "earnings_long": pl.read_parquet(lineage_dir / "earnings_sec_long.parquet"),
    }


def main() -> None:
    args = _parse_args()
    primary_dir = args.primary_sec_dir.resolve()
    secondary_dirs = [path.resolve() for path in args.secondary_sec_dir]
    reference_data_dir = args.reference_data_dir.resolve()
    output_dir = args.output_dir.resolve()
    history_root = output_dir.parent / "history" / "output"

    merged = _load_lineage_frames(primary_dir)
    for secondary_dir in secondary_dirs:
        secondary = _load_lineage_frames(secondary_dir)
        secondary_origin = f"secondary_candidate:{secondary_dir.name}"
        merged_general, merged_general_lineage = merge_general(
            primary_general=merged["general"],
            secondary_general=secondary["general"],
            primary_lineage=merged["general_lineage"],
            secondary_lineage=secondary["general_lineage"],
            secondary_origin=secondary_origin,
        )
        merged_financials, merged_financials_lineage, _ = merge_financials(
            primary_consolidated=merged["financials"],
            secondary_consolidated=secondary["financials"],
            primary_lineage=merged["financials_lineage"],
            secondary_lineage=secondary["financials_lineage"],
            secondary_origin=secondary_origin,
        )
        merged_earnings, merged_earnings_lineage, _ = merge_earnings(
            primary_consolidated=merged["earnings"],
            secondary_consolidated=secondary["earnings"],
            primary_lineage=merged["earnings_lineage"],
            secondary_lineage=secondary["earnings_lineage"],
            secondary_origin=secondary_origin,
        )
        merged_earnings_long = merge_earnings_long(
            primary_long=merged["earnings_long"],
            secondary_long=secondary["earnings_long"],
            secondary_origin=secondary_origin,
        )
        merged = {
            **merged,
            "general": merged_general,
            "general_lineage": merged_general_lineage,
            "financials": merged_financials,
            "financials_lineage": merged_financials_lineage,
            "earnings": merged_earnings,
            "earnings_lineage": merged_earnings_lineage,
            "earnings_long": merged_earnings_long,
        }

    merged_general = merged["general"]
    merged_general_lineage = merged["general_lineage"]
    merged_financials = merged["financials"]
    merged_financials_lineage = merged["financials_lineage"]
    merged_earnings = merged["earnings"]
    merged_earnings_lineage = merged["earnings_lineage"]
    merged_earnings_long = merged["earnings_long"]
    financial_overlay_audit = (
        merged_financials_lineage.group_by(["statement", "metric", "overlay_origin"])
        .agg(pl.len().alias("rows"))
        .sort(["statement", "metric", "overlay_origin"])
    )
    earnings_overlay_audit = (
        merged_earnings_lineage.group_by(["overlay_origin"])
        .agg(pl.len().alias("rows"))
        .sort("overlay_origin")
    )

    staging_dir = output_dir.parent / "_staging_sec_overlay_package"
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
        "scope": "sec_only_overlay",
        "source_policy": "SEC only",
        "primary_sec_dir": str(primary_dir),
        "secondary_sec_dirs": [str(path) for path in secondary_dirs],
        "merge_policy": {
            "financials": "prefer primary snapshot by ticker/statement/metric/fiscal_year/fiscal_period; fill only missing quarters from secondary candidates in the given order",
            "earnings": "prefer primary snapshot by ticker/fiscal_year/fiscal_period; fill only missing quarters from secondary candidates in the given order",
            "earnings_long": "prefer primary snapshot by ticker/fiscal_year/fiscal_period/metric; fill only missing rows from secondary candidates in the given order",
            "general": "prefer primary snapshot by ticker; fill only missing tickers from secondary candidates in the given order",
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
    financial_overlay_audit.write_parquet(lineage_dir / "financials_sec_overlay_audit.parquet")
    earnings_overlay_audit.write_parquet(lineage_dir / "earnings_sec_overlay_audit.parquet")

    print(f"SEC overlay package written to: {output_dir}")
    if published is not None:
        print(f"Previous output snapshot: {published}")


if __name__ == "__main__":
    main()
