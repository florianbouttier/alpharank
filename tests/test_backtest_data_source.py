from __future__ import annotations

from pathlib import Path

from alpharank.backtest.config import BacktestConfig
from alpharank.backtest.data_source import BacktestDataSource


def test_open_source_live_data_source_applies_legacy_compatible_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    live_dir = project_root / "data" / "open_source" / "live"
    config = BacktestConfig(data_dir=project_root / "data", output_dir=project_root / "outputs")

    source = BacktestDataSource.open_source_live(project_root=project_root, live_dir=live_dir)
    applied = source.apply(config)

    assert applied.data_dir == live_dir / "clean" / "legacy_compatible"
    assert applied.final_price_path is None
    assert applied.sp500_price_path is None


def test_open_source_prices_only_data_source_keeps_financial_base_and_overrides_prices(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    live_dir = data_dir / "open_source" / "live"
    config = BacktestConfig(data_dir=data_dir, output_dir=project_root / "outputs")

    source = BacktestDataSource.open_source_prices_only(project_root=project_root, data_dir=data_dir, live_dir=live_dir)
    applied = source.apply(config)

    assert applied.data_dir == data_dir
    assert applied.final_price_path == live_dir / "clean" / "legacy_compatible" / "US_Finalprice.parquet"
    assert applied.sp500_price_path == live_dir / "clean" / "legacy_compatible" / "SP500Price.parquet"
