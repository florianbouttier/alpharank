#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

from run_backtest import default_config, run


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    price_dir = project_root / "data" / "open_source" / "price_transition_20050101"
    config = default_config(
        final_price_path=price_dir / "US_Finalprice.parquet",
        sp500_price_path=price_dir / "SP500Price.parquet",
    )
    artifacts = run(config)
    print("\n=== Backtest Completed (Open-Source Prices) ===")
    print(f"Run dir: {artifacts.output_paths['run_dir']}")
    print(f"Using prices: {config.final_price_path}")
    print(f"Using SP500 proxy: {config.sp500_price_path}")


if __name__ == "__main__":
    main()
