#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.legacy_export import export_legacy_compatible_outputs
from alpharank.data.open_source.publishing import publish_open_source_output_package
from alpharank.data.open_source.storage import utc_now_iso


def main(
    *,
    price_source_dir: str | Path | None = None,
    financial_source_dir: str | Path | None = None,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    resolved_reference_data_dir = Path(reference_data_dir).expanduser().resolve() if reference_data_dir else project_root / "data"
    resolved_price_source_dir = (
        Path(price_source_dir).expanduser().resolve()
        if price_source_dir
        else project_root / "data" / "open_source" / "audit" / "price_transition_20050101"
    )
    resolved_financial_source_dir = (
        Path(financial_source_dir).expanduser().resolve()
        if financial_source_dir
        else project_root / "data" / "open_source" / "audit" / "sp500_2025"
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else project_root / "data" / "open_source" / "output"
    )

    clean_prices = pl.read_parquet(resolved_price_source_dir / "US_Finalprice.parquet")
    benchmark_prices = pl.read_parquet(resolved_price_source_dir / "SP500Price.parquet")
    general_reference = pl.read_parquet(resolved_financial_source_dir / "general_reference.parquet")
    earnings_frame = pl.read_parquet(resolved_financial_source_dir / "earnings_yfinance.parquet")
    earnings_long_frame = pl.read_parquet(resolved_financial_source_dir / "earnings_yfinance_long.parquet")
    consolidated_financials = pl.read_parquet(resolved_financial_source_dir / "financials_open_source_consolidated.parquet")
    consolidated_lineage = pl.read_parquet(resolved_financial_source_dir / "financials_open_source_lineage.parquet")
    source_summary = pl.read_parquet(resolved_financial_source_dir / "financials_open_source_source_summary.parquet")

    staging_dir = resolved_output_dir.parent / "_staging_output_package"
    legacy_paths = export_legacy_compatible_outputs(
        clean_prices=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference.select(["ticker", "name", "exchange", "cik", "source"]),
        consolidated_financials=consolidated_financials,
        earnings_frame=earnings_frame,
        reference_data_dir=resolved_reference_data_dir,
        output_dir=staging_dir,
    )
    published = publish_open_source_output_package(
        output_dir=resolved_output_dir,
        legacy_paths=legacy_paths,
        constituents_source_path=resolved_reference_data_dir / "SP500_Constituents.csv",
        prices_frame=clean_prices,
        benchmark_prices=benchmark_prices,
        general_reference=general_reference.select(["ticker", "name", "exchange", "cik", "source"]),
        consolidated_financials=consolidated_financials,
        consolidated_lineage=consolidated_lineage,
        source_summary=source_summary,
        earnings_frame=earnings_frame,
        earnings_long_frame=earnings_long_frame,
        manifest={
            "generated_at": utc_now_iso(),
            "output_dir": str(resolved_output_dir),
            "price_source_dir": str(resolved_price_source_dir),
            "financial_source_dir": str(resolved_financial_source_dir),
            "reference_data_dir": str(resolved_reference_data_dir),
        },
        history_root=project_root / "data" / "open_source" / "history" / "output",
    )

    print(f"Open-source output package written to: {resolved_output_dir}")
    print("Exact-name outputs:")
    for path in sorted(resolved_output_dir.glob("*.parquet")):
        print(f"  - {path.name}")
    print("  - SP500_Constituents.csv")
    print("Lineage directory:")
    print(f"  - {resolved_output_dir / 'lineage'}")
    if published.snapshot_dir is not None:
        print(f"Previous output snapshot: {published.snapshot_dir}")
    print(f"Published files: {len(published.published_paths)}")


if __name__ == "__main__":
    main()
