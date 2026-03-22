from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from alpharank.backtest.config import BacktestConfig


@dataclass(frozen=True)
class BacktestDataSource:
    name: str
    data_dir: Path
    final_price_path: Path | None = None
    sp500_price_path: Path | None = None

    def apply(self, config: BacktestConfig) -> BacktestConfig:
        return replace(
            config,
            data_dir=self.data_dir,
            final_price_path=self.final_price_path,
            sp500_price_path=self.sp500_price_path,
        )

    @classmethod
    def eodhd(cls, *, project_root: Path | None = None) -> BacktestDataSource:
        root = _project_root(project_root)
        return cls(
            name="eodhd",
            data_dir=root / "data",
        )

    @classmethod
    def open_source_live(cls, *, project_root: Path | None = None, live_dir: Path | None = None) -> BacktestDataSource:
        root = _project_root(project_root)
        base_live_dir = live_dir or (root / "data" / "open_source" / "live")
        return cls(
            name="open_source_live",
            data_dir=base_live_dir / "clean" / "legacy_compatible",
        )

    @classmethod
    def open_source_prices_only(
        cls,
        *,
        project_root: Path | None = None,
        data_dir: Path | None = None,
        live_dir: Path | None = None,
    ) -> BacktestDataSource:
        root = _project_root(project_root)
        base_data_dir = data_dir or (root / "data")
        base_live_dir = live_dir or (root / "data" / "open_source" / "live")
        legacy_dir = base_live_dir / "clean" / "legacy_compatible"
        return cls(
            name="open_source_prices_only",
            data_dir=base_data_dir,
            final_price_path=legacy_dir / "US_Finalprice.parquet",
            sp500_price_path=legacy_dir / "SP500Price.parquet",
        )

    @classmethod
    def custom(
        cls,
        *,
        name: str,
        data_dir: Path,
        final_price_path: Path | None = None,
        sp500_price_path: Path | None = None,
    ) -> BacktestDataSource:
        return cls(
            name=name,
            data_dir=data_dir,
            final_price_path=final_price_path,
            sp500_price_path=sp500_price_path,
        )


def _project_root(project_root: Path | None) -> Path:
    if project_root is not None:
        return project_root.expanduser().resolve()
    return Path(__file__).resolve().parents[3]
