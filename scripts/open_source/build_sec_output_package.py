#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import polars as pl

from alpharank.data.open_source.legacy_export import export_legacy_compatible_fundamental_outputs
from alpharank.data.open_source.sec_mapping import load_sec_historical_ticker_bridge
from alpharank.data.open_source.sec_only import (
    build_sec_only_earnings,
    build_sec_only_financials,
    build_sec_only_general_reference_from_raw_lineage,
)
from alpharank.data.open_source.storage import utc_now_iso, write_json
from alpharank.data.output_history import snapshot_output_directory


def main(
    *,
    raw_source_dir: str | Path | None = None,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    resolved_raw_source_dir = (
        Path(raw_source_dir).expanduser().resolve()
        if raw_source_dir
        else project_root / "data" / "open_source" / "official" / "raw"
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

    sec_companyfacts = pl.read_parquet(resolved_raw_source_dir / "financials_sec_companyfacts.parquet")
    sec_filing = pl.read_parquet(resolved_raw_source_dir / "financials_sec_filing.parquet")
    sec_calendar = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_calendar.parquet")
    sec_actuals = pl.read_parquet(resolved_raw_source_dir / "earnings_sec_actuals.parquet")
    general_reference_lineage_raw = pl.read_parquet(resolved_raw_source_dir / "general_reference_lineage.parquet")
    historical_bridge = load_sec_historical_ticker_bridge(resolved_reference_data_dir)

    sec_companyfacts = _filter_frame_by_historical_windows(
        sec_companyfacts,
        bridge=historical_bridge,
        ticker_col="ticker",
        date_col="date",
    )
    sec_companyfacts = _filter_frame_by_bridge_cik(
        sec_companyfacts,
        bridge=historical_bridge,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_filing = _filter_frame_by_historical_windows(
        sec_filing,
        bridge=historical_bridge,
        ticker_col="ticker",
        date_col="date",
    )
    sec_filing = _filter_frame_by_bridge_cik(
        sec_filing,
        bridge=historical_bridge,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_calendar = _filter_frame_by_historical_windows(
        sec_calendar,
        bridge=historical_bridge,
        ticker_col="ticker",
        date_col="period_end",
    )
    sec_calendar = _filter_frame_by_bridge_cik(
        sec_calendar,
        bridge=historical_bridge,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    sec_actuals = _filter_frame_by_historical_windows(
        sec_actuals,
        bridge=historical_bridge,
        ticker_col="ticker",
        date_col="period_end",
    )
    sec_actuals = _filter_frame_by_bridge_cik(
        sec_actuals,
        bridge=historical_bridge,
        ticker_col="ticker",
        accession_col="accession_number",
    )
    general_reference_lineage_raw = _override_general_reference_lineage_from_bridge(
        general_reference_lineage_raw,
        bridge=historical_bridge,
    )

    consolidated_financials, consolidated_lineage, source_summary = build_sec_only_financials(
        sec_companyfacts=sec_companyfacts,
        sec_filing=sec_filing,
    )
    earnings_consolidated, earnings_lineage, earnings_long = build_sec_only_earnings(
        sec_calendar=sec_calendar,
        sec_actuals=sec_actuals,
        sec_financials=consolidated_financials,
    )
    general_reference, general_reference_lineage = build_sec_only_general_reference_from_raw_lineage(
        general_reference_lineage_raw
    )

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

    manifest = {
        "generated_at": utc_now_iso(),
        "output_dir": str(resolved_output_dir),
        "raw_source_dir": str(resolved_raw_source_dir),
        "reference_data_dir": str(resolved_reference_data_dir),
        "scope": "sec_only_fundamentals",
        "dataset_policy": {
            "financials": "sec_companyfacts -> sec_filing",
            "earnings_calendar": "sec_submissions",
            "earnings_actual": "sec_companyfacts -> sec_filing -> sec_derived_eps(net_income / outstanding_shares)",
            "earnings_estimate": None,
            "earnings_surprise_percent": None,
            "general_reference": "sec_mapping + sec_sic",
            "prices": None,
        },
        "semantics": {
            "reported_metrics": [
                "revenue",
                "gross_profit",
                "operating_income",
                "net_income",
                "total_assets",
                "total_liabilities",
                "stockholders_equity",
                "cash_and_equivalents",
                "operating_cash_flow",
                "capital_expenditures",
                "outstanding_shares",
                "epsActual",
            ],
            "derived_from_sec": ["free_cash_flow", "epsActual (fallback only when SEC published EPS is missing)"],
            "missing_by_design": ["epsEstimate", "surprisePercent", "US_Finalprice.parquet", "SP500Price.parquet"],
        },
        "summary": {
            "general_rows": general_reference.height,
            "financial_rows": consolidated_financials.height,
            "earnings_rows": earnings_consolidated.height,
            "financial_tickers": consolidated_financials.get_column("ticker").n_unique() if not consolidated_financials.is_empty() else 0,
            "earnings_tickers": earnings_consolidated.get_column("ticker").n_unique() if not earnings_consolidated.is_empty() else 0,
            "financial_date_min": consolidated_financials.get_column("date").min() if not consolidated_financials.is_empty() else None,
            "financial_date_max": consolidated_financials.get_column("date").max() if not consolidated_financials.is_empty() else None,
            "earnings_period_end_min": earnings_consolidated.get_column("period_end").min() if not earnings_consolidated.is_empty() else None,
            "earnings_period_end_max": earnings_consolidated.get_column("period_end").max() if not earnings_consolidated.is_empty() else None,
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
        manifest=manifest,
    )

    print(f"SEC-only output package written to: {resolved_output_dir}")
    print("Exact-name outputs:")
    for path in sorted(resolved_output_dir.glob("*")):
        if path.is_file():
            print(f"  - {path.name}")
    print(f"Lineage directory: {resolved_output_dir / 'lineage'}")
    if published is not None:
        print(f"Previous output snapshot: {published}")


def _parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Build and publish the strict SEC-only fundamentals package.")
    parser.add_argument(
        "--raw-source-dir",
        type=Path,
        default=project_root / "data" / "open_source" / "official" / "raw",
    )
    parser.add_argument(
        "--reference-data-dir",
        type=Path,
        default=project_root / "data",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "data" / "sec" / "output",
    )
    return parser.parse_args()


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


def _filter_frame_by_historical_windows(
    frame: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
    ticker_col: str,
    date_col: str,
) -> pl.DataFrame:
    if frame.is_empty() or bridge.is_empty() or ticker_col not in frame.columns or date_col not in frame.columns:
        return frame
    windows = (
        bridge.select(
            [
                (pl.col("ticker") + pl.lit(".US")).alias(ticker_col),
                pl.col("start_date").cast(pl.Utf8),
                pl.col("end_date").cast(pl.Utf8),
            ]
        )
        .with_columns(
            [
                pl.col("start_date").str.strptime(pl.Date, strict=False).alias("_start_dt"),
                pl.col("end_date").str.strptime(pl.Date, strict=False).alias("_end_dt"),
            ]
        )
        .drop(["start_date", "end_date"])
    )
    return (
        frame.with_columns(pl.col(date_col).str.strptime(pl.Date, strict=False).alias("_row_dt"))
        .join(windows, on=ticker_col, how="left")
        .filter(
            pl.col("_row_dt").is_null()
            | (
                (pl.col("_start_dt").is_null() | (pl.col("_row_dt") >= pl.col("_start_dt")))
                & (pl.col("_end_dt").is_null() | (pl.col("_row_dt") <= pl.col("_end_dt")))
            )
        )
        .drop(["_row_dt", "_start_dt", "_end_dt"])
    )


def _override_general_reference_lineage_from_bridge(
    raw_lineage: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
) -> pl.DataFrame:
    if raw_lineage.is_empty() or bridge.is_empty() or "ticker" not in raw_lineage.columns:
        return raw_lineage
    overrides = bridge.select(
        [
            (pl.col("ticker") + pl.lit(".US")).alias("ticker"),
            pl.col("name").alias("_bridge_name"),
            pl.col("exchange").alias("_bridge_exchange"),
            pl.col("cik").alias("_bridge_cik"),
        ]
    )
    result = raw_lineage.join(overrides, on="ticker", how="left")
    updates: list[pl.Expr] = []
    if "name" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_name"), pl.col("name")]).alias("name"))
    if "exchange" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_exchange"), pl.col("exchange")]).alias("exchange"))
    if "cik" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_cik"), pl.col("cik")]).alias("cik"))
    if "sec_name" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_name"), pl.col("sec_name")]).alias("sec_name"))
    if "sec_exchange" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_exchange"), pl.col("sec_exchange")]).alias("sec_exchange"))
    if "sec_cik" in result.columns:
        updates.append(pl.coalesce([pl.col("_bridge_cik"), pl.col("sec_cik")]).alias("sec_cik"))
    if updates:
        result = result.with_columns(updates)
    return result.drop([column for column in ["_bridge_name", "_bridge_exchange", "_bridge_cik"] if column in result.columns])


def _filter_frame_by_bridge_cik(
    frame: pl.DataFrame,
    *,
    bridge: pl.DataFrame,
    ticker_col: str,
    accession_col: str,
) -> pl.DataFrame:
    if frame.is_empty() or bridge.is_empty() or ticker_col not in frame.columns or accession_col not in frame.columns:
        return frame
    expected = bridge.select(
        [
            (pl.col("ticker") + pl.lit(".US")).alias(ticker_col),
            pl.col("cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("_bridge_cik"),
        ]
    ).unique(subset=[ticker_col], keep="first")
    return (
        frame.join(expected, on=ticker_col, how="left")
        .with_columns(pl.col(accession_col).cast(pl.Utf8).str.extract(r"(\d{10})").alias("_accession_cik"))
        .filter(
            pl.col("_bridge_cik").is_null()
            | pl.col("_accession_cik").is_null()
            | (pl.col("_bridge_cik") == pl.col("_accession_cik"))
        )
        .drop(["_bridge_cik", "_accession_cik"])
    )


def _write_readme(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# SEC Output Package",
                "",
                "Official strict SEC-only fundamentals package.",
                "",
                "- Exact-name legacy files are published only for fundamentals and earnings.",
                "- No price files are published here because SEC is not a market price source.",
                "- `lineage/manifest.json` is the official policy and provenance entrypoint.",
                "- `free_cash_flow` is derived from SEC cash-flow components when available.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = _parse_args()
    main(
        raw_source_dir=args.raw_source_dir,
        reference_data_dir=args.reference_data_dir,
        output_dir=args.output_dir,
    )
