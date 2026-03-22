#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from alpharank.backtest import BacktestDataSource
from run_backtest import default_config, run


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    source = BacktestDataSource.custom(
        name="open_source_prices_2005_transition",
        data_dir=project_root / "data",
        final_price_path=project_root / "data" / "open_source" / "price_transition_20050101" / "US_Finalprice.parquet",
        sp500_price_path=project_root / "data" / "open_source" / "price_transition_20050101" / "SP500Price.parquet",
    )
    config = source.apply(default_config())
    artifacts = run(config)
    print("\n=== Backtest Completed (Open-Source Prices) ===")
    print(f"Run dir: {artifacts.output_paths['run_dir']}")
    print(f"Using source: {source.name}")
    print(f"Using prices: {config.final_price_path}")
    print(f"Using SP500 proxy: {config.sp500_price_path}")


if __name__ == "__main__":
    main()
