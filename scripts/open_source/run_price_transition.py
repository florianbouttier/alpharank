#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alpharank.data.open_source import run_open_source_price_transition


def main(
    *,
    start_date: str = "2005-01-01",
    end_date: str | None = None,
    tickers: tuple[str, ...] | list[str] | None = None,
    threshold_pct: float = 0.5,
    reference_data_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    resolved_reference_data_dir = (
        Path(reference_data_dir).expanduser().resolve() if reference_data_dir else project_root / "data"
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else project_root / "data" / "open_source" / "audit" / f"price_transition_{start_date.replace('-', '')}"
    )

    result = run_open_source_price_transition(
        reference_data_dir=resolved_reference_data_dir,
        output_dir=resolved_output_dir,
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        threshold_pct=threshold_pct,
    )

    print(f"Open-source price transition written to: {result.output_dir}")
    print(f"Date range: {result.start_date} -> {result.end_date}")
    print(f"Tickers: {result.ticker_count}")
    print(f"Yahoo price rows: {result.yahoo_price_rows}")
    print(f"EODHD reference rows: {result.eodhd_price_rows}")
    print(f"Price alignment rows: {result.price_alignment_rows}")


if __name__ == "__main__":
    main()
