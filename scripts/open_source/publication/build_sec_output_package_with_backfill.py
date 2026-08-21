#!/usr/bin/env python3
"""Build SEC-only output package with EODHD legacy backfill and spinoff backfill.

This script extends ``build_sec_output_package.py`` by:
1. Normalizing tickers for SEC lookup (BRK.B -> BRK-B)
2. Backfilling CIK-legacy tickers from EODHD legacy data (marked as non-GAAP)
3. Backfilling spinoffs from parent SEC data (remains GAAP)

IMPORTANT: EODHD backfill is NOT GAAP. It is explicitly tagged in lineage.
Users who require strict GAAP-only data should filter out rows where
source == 'eodhd_legacy_backfill'.
"""
from __future__ import annotations

from pathlib import Path
import shutil

import polars as pl

from alpharank.data.ingestion.backfill import (
    BackfillConfig,
    apply_financial_backfills,
    normalize_sec_ticker,
)
from alpharank.data.open_source.legacy_export import export_legacy_compatible_fundamental_outputs
from alpharank.data.sources.sec_only import (
    build_sec_only_earnings,
    build_sec_only_financials,
    build_sec_only_general_reference_from_raw_lineage,
)
from alpharank.data.ingestion.storage import utc_now_iso, write_json
from alpharank.data.lineage.output_history import snapshot_output_directory


def main(
    *,
    raw_source_dir: str | Path | None = None,
    eodhd_data_dir: str | Path | None = None,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
    enable_eodhd_backfill: bool = True,
    enable_spinoff_backfill: bool = True,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    resolved_raw_source_dir = (
        Path(raw_source_dir).expanduser().resolve()
        if raw_source_dir
        else project_root / "data" / "open_source" / "official" / "raw"
    )
    resolved_eodhd_dir = (
        Path(eodhd_data_dir).expanduser().resolve()
        if eodhd_data_dir
        else project_root / "data" / "eodhd" / "output"
    )
    resolved_reference_data_dir = (
        Path(reference_data_dir).expanduser().resolve()
        if reference_data_dir
        else project_root / "data"
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else project_root / "data" / "sec" / "output"
    )
    history_root = resolved_output_dir.parent / "history" / "output"

    print(f"Loading raw data from: {resolved_raw_source_dir}")
    print(f"Loading EODHD legacy from: {resolved_eodhd_dir}")
    print(f"Backfill config: EODHD={enable_eodhd_backfill}, spinoffs={enable_spinoff_backfill}")

    # ------------------------------------------------------------------
    # Load raw SEC data
    # ------------------------------------------------------------------
    sec_companyfacts = pl.read_parquet(resolved_raw_source_dir / "financials_sec_companyfacts.parquet")
    sec_filing = pl.read_parquet(resolved_raw_source_dir / "financials_sec_filing.parquet")
    sec_calendar = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_calendar.parquet")
    sec_actuals = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_actuals.parquet")
    general_reference_lineage_raw = pl.read_parquet(resolved_raw_source_dir / "general_reference_lineage.parquet")

    # ------------------------------------------------------------------
    # Load EODHD legacy data for backfill
    # ------------------------------------------------------------------
    eodhd_financials = _load_eodhd_as_backfill_source(resolved_eodhd_dir)

    # ------------------------------------------------------------------
    # Apply backfills BEFORE consolidation
    # ------------------------------------------------------------------
    backfill_config = BackfillConfig(
        eodhd_enabled=enable_eodhd_backfill,
        spinoff_enabled=enable_spinoff_backfill,
        ticker_normalization_enabled=True,
    )

    backfilled_financials, backfill_lineage, backfill_audit = apply_financial_backfills(
        sec_financials=sec_companyfacts,
        sec_companyfacts_raw=sec_companyfacts,
        eodhd_financials=eodhd_financials,
        config=backfill_config,
    )

    print(f"\nBackfill applied:")
    print(f"  Rows added: {backfilled_financials.height - sec_companyfacts.height}")
    print(f"  Tickers backfilled: {backfill_audit['ticker'].n_unique() if not backfill_audit.is_empty() else 0}")
    if not backfill_audit.is_empty():
        for row in backfill_audit.to_dicts():
            print(f"    {row['ticker']}: +{row['rows_added']} rows ({row['backfill_type']})")

    # ------------------------------------------------------------------
    # Consolidate with SEC filing (as secondary source)
    # ------------------------------------------------------------------
    consolidated_financials, consolidated_lineage, source_summary = build_sec_only_financials(
        sec_companyfacts=backfilled_financials,
        sec_filing=sec_filing,
    )

    # Merge backfill lineage into consolidated lineage
    if not backfill_lineage.is_empty():
        consolidated_lineage = consolidated_lineage.join(
            backfill_lineage,
            on=["ticker", "metric", "date"],
            how="left",
        ).with_columns(
            [
                pl.when(pl.col("backfill_source").is_not_null())
                .then(pl.col("backfill_source"))
                .otherwise(pl.col("source"))
                .alias("source"),
                pl.when(pl.col("backfill_source").is_not_null())
                .then(pl.concat_str([pl.col("source_label"), pl.lit(" | backfilled"), pl.col("backfill_reason")], separator=" "))
                .otherwise(pl.col("source_label"))
                .alias("source_label"),
            ]
        )

    earnings_consolidated, earnings_lineage, earnings_long = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
    )
    general_reference, general_reference_lineage = build_sec_only_general_reference_from_raw_lineage(
        general_reference_lineage_raw
    )

    # ------------------------------------------------------------------
    # Build legacy-compatible outputs
    # ------------------------------------------------------------------
    staging_dir = resolved_output_dir.parent / "_staging_sec_output_package"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    legacy_paths = export_legacy_compatible_fundamental_outputs(
        general_reference=general_reference,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        earnings_frame=earnings_consolidated,
        reference_data_dir=resolved_reference_data_dir,
        output_dir=staging_dir,
        align_shares_with_earnings_semantics=False,
    )

    # ------------------------------------------------------------------
    # Manifest
    # ------------------------------------------------------------------
    manifest = {
        "generated_at": utc_now_iso(),
        "output_dir": str(resolved_output_dir),
        "raw_source_dir": str(resolved_raw_source_dir),
        "reference_data_dir": str(resolved_reference_data_dir),
        "scope": "sec_only_fundamentals_with_backfill",
        "backfill_policy": {
            "eodhd_enabled": enable_eodhd_backfill,
            "spinoff_enabled": enable_spinoff_backfill,
            "ticker_normalization_enabled": True,
        },
        "backfill_audit": backfill_audit.to_dicts() if not backfill_audit.is_empty() else [],
        "dataset_policy": {
            "financials": "sec_companyfacts -> sec_filing -> eodhd_legacy_backfill (non-GAAP) -> sec_spinoff_parent",
            "earnings_calendar": "sec_submissions",
            "earnings_actual": "sec_companyfacts -> sec_filing",
            "earnings_estimate": None,
            "earnings_surprise_percent": None,
            "general_reference": "sec_mapping + sec_sic",
            "prices": None,
        },
        "semantics": {
            "reported_metrics": [
                "revenue", "gross_profit", "operating_income", "net_income",
                "total_assets", "total_liabilities", "stockholders_equity",
                "cash_and_equivalents", "operating_cash_flow", "capital_expenditures",
                "outstanding_shares", "epsActual",
            ],
            "derived_from_sec": ["free_cash_flow"],
            "missing_by_design": ["epsEstimate", "surprisePercent", "US_Finalprice.parquet", "SP500Price.parquet"],
            "backfill_sources": {
                "eodhd_legacy_backfill": "NOT GAAP. Used for CIK-legacy tickers before SEC start date.",
                "sec_spinoff_parent": "GAAP. Copied from parent company SEC filings before spinoff date.",
            },
        },
        "summary": {
            "general_rows": general_reference.height,
            "financial_rows": consolidated_financials.height,
            "earnings_rows": earnings_consolidated.height,
            "financial_tickers": consolidated_financials.get_column("ticker").n_unique() if not consolidated_financials.is_empty() else 0,
            "earnings_tickers": earnings_consolidated.get_column("ticker").n_unique() if not earnings_consolidated.is_empty() else 0,
            "financial_date_min": consolidated_financials.get_column("date").min() if not consolidated_financials.is_empty() else None,
            "financial_date_max": consolidated_financials.get_column("date").max() if not consolidated_financials.is_empty() else None,
            "backfill_rows_added": backfilled_financials.height - sec_companyfacts.height,
        },
    }

    published = publish_sec_output_package(
        output_dir=resolved_output_dir,
        history_root=history_root,
        legacy_paths=legacy_paths,
        constituents_source_path=resolved_reference_data_dir / "SP500_Constituents.csv",
        general_reference=general_reference,
        general_reference_lineage=general_reference_lineage,
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_consolidated=earnings_consolidated,
        earnings_lineage=earnings_lineage,
        earnings_long=earnings_long,
        backfill_audit=backfill_audit,
        manifest=manifest,
    )

    print(f"\nSEC-only output package written to: {resolved_output_dir}")
    print(f"Backfill audit written to: {resolved_output_dir / 'lineage' / 'backfill_audit.parquet'}")
    if published is not None:
        print(f"Previous output snapshot: {published}")


def _load_eodhd_as_backfill_source(eodhd_dir: Path) -> pl.DataFrame:
    """Load EODHD legacy data and normalize to the backfill schema.

    EODHD files are in wide format (one row per ticker+date with metric columns).
    We pivot them to long format for the backfill logic.
    """
    frames: list[pl.DataFrame] = []

    # Map EODHD file -> statement -> metric -> column name
    eodhd_files = {
        "US_Income_statement.parquet": ("income_statement", {
            "totalRevenue": "revenue",
            "netIncome": "net_income",
            "grossProfit": "gross_profit",
            "operatingIncome": "operating_income",
        }),
        "US_Balance_sheet.parquet": ("balance_sheet", {
            "totalAssets": "total_assets",
            "totalLiab": "total_liabilities",
            "totalStockholderEquity": "stockholders_equity",
            "cashAndEquivalents": "cash_and_equivalents",
        }),
        "US_Cash_flow.parquet": ("cash_flow", {
            "totalCashFromOperatingActivities": "operating_cash_flow",
            "capitalExpenditures": "capital_expenditures",
            "freeCashFlow": "free_cash_flow",
        }),
    }

    for file_name, (statement, metric_map) in eodhd_files.items():
        path = eodhd_dir / file_name
        if not path.exists():
            print(f"  Warning: EODHD file not found: {path}")
            continue

        df = pl.read_parquet(path)
        if df.is_empty():
            continue

        # Select only the columns we need
        cols = ["ticker", "date"] + list(metric_map.keys())
        available_cols = [c for c in cols if c in df.columns]
        if len(available_cols) <= 2:
            continue

        df = df.select(available_cols)

        # Pivot to long format
        id_vars = ["ticker", "date"]
        value_vars = [c for c in available_cols if c not in id_vars]

        for eodhd_col, metric_name in metric_map.items():
            if eodhd_col not in df.columns:
                continue
            sub = df.select([
                pl.col("ticker"),
                pl.col("date"),
                pl.col(eodhd_col).cast(pl.Float64, strict=False).alias("value"),
            ]).filter(pl.col("value").is_not_null())

            if sub.is_empty():
                continue

            sub = sub.with_columns([
                pl.lit(statement).alias("statement"),
                pl.lit(metric_name).alias("metric"),
                pl.lit("eodhd_legacy").alias("source"),
                pl.lit(eodhd_col).alias("source_label"),
            ])
            frames.append(sub)

    if not frames:
        return _empty_backfill_frame()

    return pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date"])


def _empty_backfill_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "date": pl.String,
            "value": pl.Float64,
            "statement": pl.String,
            "metric": pl.String,
            "source": pl.String,
            "source_label": pl.String,
        }
    )


def publish_sec_output_package(
    *,
    output_dir: Path,
    history_root: Path,
    legacy_paths: dict[str, Path],
    constituents_source_path: Path,
    general_reference: pl.DataFrame,
    general_reference_lineage: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    source_summary: pl.DataFrame,
    earnings_consolidated: pl.DataFrame,
    earnings_lineage: pl.DataFrame,
    earnings_long: pl.DataFrame,
    backfill_audit: pl.DataFrame,
    manifest: dict[str, object],
) -> Path | None:
    snapshot_dir = snapshot_output_directory(
        output_dir,
        history_root=history_root,
        snapshot_prefix="sec_output",
        metadata=manifest,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    lineage_dir = output_dir / "lineage"
    lineage_dir.mkdir(parents=True, exist_ok=True)

    allowed_output_files = set(legacy_paths) | {"SP500_Constituents.csv", "README.md"}
    for existing in output_dir.iterdir():
        if existing.name == "lineage" or existing.name in allowed_output_files:
            continue
        if existing.is_file():
            existing.unlink()

    for file_name, source_path in legacy_paths.items():
        shutil.copy2(source_path, output_dir / file_name)
    shutil.copy2(constituents_source_path, output_dir / "SP500_Constituents.csv")
    _write_readme(output_dir / "README.md")

    lineage_outputs = {
        "general_reference.parquet": general_reference,
        "general_reference_lineage.parquet": general_reference_lineage,
        "financials_sec_consolidated.parquet": consolidated_financials,
        "financials_sec_lineage.parquet": consolidated_lineage,
        "financials_sec_source_summary.parquet": source_summary,
        "earnings_sec_consolidated.parquet": earnings_consolidated,
        "earnings_sec_lineage.parquet": earnings_lineage,
        "earnings_sec_long.parquet": earnings_long,
        "backfill_audit.parquet": backfill_audit,
    }
    allowed_lineage_files = set(lineage_outputs) | {"manifest.json"}
    for existing in lineage_dir.iterdir():
        if existing.name in allowed_lineage_files:
            continue
        if existing.is_file():
            existing.unlink()
    for file_name, frame in lineage_outputs.items():
        frame.write_parquet(lineage_dir / file_name)
    write_json(lineage_dir / "manifest.json", manifest)
    return snapshot_dir


def _write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# SEC Output Package (with Backfill)",
                "",
                "Official SEC fundamentals package with backfill for CIK-legacy tickers.",
                "",
                "## Sources",
                "- `sec_companyfacts` / `sec_filing`: GAAP/SEC data (primary)",
                "- `eodhd_legacy_backfill`: EODHD legacy data for CIK-legacy tickers before SEC start date (NOT GAAP)",
                "- `sec_spinoff_parent`: Parent company SEC data copied for spinoffs (GAAP)",
                "",
                "## Filtering GAAP-only data",
                "To get strictly GAAP data, filter out rows where `source == 'eodhd_legacy_backfill'`.",
                "",
                "- `lineage/manifest.json` is the official policy and provenance entrypoint.",
                "- `lineage/backfill_audit.parquet` details which tickers were backfilled and why.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
