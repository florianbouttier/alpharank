#!/usr/bin/env python3
from __future__ import annotations

from alpharank.backtest import BacktestDataSource
from run_backtest import default_config, run


# Edit this one line in Python, then run the script.
DATA_SOURCE = BacktestDataSource.open_source_live()


def main() -> None:
    config = DATA_SOURCE.apply(default_config())
    artifacts = run(config)
    print("\n=== Backtest Completed ===")
    print(f"Data source: {DATA_SOURCE.name}")
    print(f"Data dir: {config.data_dir}")
    print(f"Final price path: {config.final_price_path}")
    print(f"SP500 price path: {config.sp500_price_path}")
    print(f"Run dir: {artifacts.output_paths['run_dir']}")


if __name__ == "__main__":
    main()
