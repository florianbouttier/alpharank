from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

from pytest import MonkeyPatch
from scripts.run_backtest import default_config

from alpharank.backtest import data_source as data_source_module
from alpharank.backtest.config import BacktestConfig
from alpharank.backtest.data_source import BacktestDataSource
from alpharank.backtest.time_folds import build_backtest_windows


def test_canonical_backtest_source_resolves_model_input_mart(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "repo"
    mart_dir = project_root / "data" / "warehouse" / "mart" / "input_01"

    def resolve(pointer_path: Path, *, warehouse_root: Path):
        assert pointer_path == (
            project_root / "data" / "model_inputs" / "manifests" / "latest.json"
        )
        assert warehouse_root == project_root / "data" / "warehouse"
        return SimpleNamespace(mart_dir=mart_dir)

    monkeypatch.setattr(data_source_module, "resolve_mart_model_input", resolve)

    source = BacktestDataSource.canonical_mart(project_root=project_root)

    assert source.name == "canonical_mart"
    assert source.data_dir == mart_dir


def test_eodhd_data_source_points_to_mirrored_output_folder(tmp_path: Path) -> None:
    project_root = tmp_path / "repo"
    config = BacktestConfig(data_dir=project_root / "data", output_dir=project_root / "outputs")

    source = BacktestDataSource.eodhd(project_root=project_root)
    applied = source.apply(config)

    assert applied.data_dir == project_root / "data" / "eodhd" / "output"


def test_generic_boosting_defaults_to_chronological_validation(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    monkeypatch.setattr(
        data_source_module,
        "resolve_mart_model_input",
        lambda *args, **kwargs: SimpleNamespace(mart_dir=tmp_path),
    )

    library_config = BacktestConfig(data_dir=tmp_path, output_dir=tmp_path)
    cli_config = default_config()
    cpcv_config = default_config(fold_strategy="cpcv")

    assert library_config.fold_strategy == cli_config.fold_strategy == "walk_forward"
    assert library_config.validation_scope == cli_config.validation_scope == "chronological_research"
    assert cpcv_config.validation_scope == "cpcv_diagnostic"
    months = [date(year, month, 1) for year in range(2010, 2017) for month in range(1, 13)]
    windows = build_backtest_windows(months, cli_config)
    assert windows
    assert all(
        max(window.train_months) < min(window.val_months)
        and max(window.val_months) < min(window.test_months)
        for window in windows
    )
    first_cpcv_fold = build_backtest_windows(months, cpcv_config)[0]
    assert min(first_cpcv_fold.train_months) > max(first_cpcv_fold.test_months)


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
