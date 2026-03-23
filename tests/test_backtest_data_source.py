from __future__ import annotations

from pathlib import Path

from alpharank.backtest.config import BacktestConfig
from alpharank.backtest.data_source import BacktestDataSource


def test_eodhd_data_source_points_to_mirrored_output_folder(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    config = BacktestConfig(data_dir=project_root / "data", output_dir=project_root / "outputs")

    source = BacktestDataSource.eodhd(project_root=project_root)
    applied = source.apply(config)

    assert applied.data_dir == project_root / "data" / "eodhd" / "output"


def test_open_source_live_data_source_applies_legacy_compatible_dir(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    output_dir = project_root / "data" / "open_source" / "output"
    config = BacktestConfig(data_dir=project_root / "data", output_dir=project_root / "outputs")

    source = BacktestDataSource.open_source_official(project_root=project_root, official_dir=output_dir)
    applied = source.apply(config)

    assert applied.data_dir == output_dir
    assert applied.final_price_path is None
    assert applied.sp500_price_path is None


def test_open_source_prices_only_data_source_keeps_financial_base_and_overrides_prices(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    data_dir = project_root / "data"
    output_dir = data_dir / "open_source" / "output"
    config = BacktestConfig(data_dir=data_dir, output_dir=project_root / "outputs")

    source = BacktestDataSource.open_source_prices_only(project_root=project_root, data_dir=data_dir, official_dir=output_dir)
    applied = source.apply(config)

    assert applied.data_dir == data_dir
    assert applied.final_price_path == output_dir / "US_Finalprice.parquet"
    assert applied.sp500_price_path == output_dir / "SP500Price.parquet"
