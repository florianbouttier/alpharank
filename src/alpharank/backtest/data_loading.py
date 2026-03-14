from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import polars as pl


@dataclass
class RawDataBundle:
    final_price: pl.DataFrame
    income_statement: pl.DataFrame
    balance_sheet: pl.DataFrame
    cash_flow: pl.DataFrame
    earnings: pl.DataFrame
    constituents: pl.DataFrame
    sp500_price: pl.DataFrame


def _resolve_dataset_dir(data_dir: Path) -> Path:
    us_dir = data_dir / "US"
    return us_dir if us_dir.exists() else data_dir


def _read_parquet_with_fallbacks(base_dir: Path, candidates: Iterable[str]) -> pl.DataFrame:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return pl.read_parquet(candidate)
    raise FileNotFoundError(f"Could not locate parquet file in {base_dir} among: {list(candidates)}")


def _read_csv_with_fallbacks(base_dir: Path, candidates: Iterable[str]) -> pl.DataFrame:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return pl.read_csv(candidate, try_parse_dates=True)
    raise FileNotFoundError(f"Could not locate csv file in {base_dir} among: {list(candidates)}")


def load_raw_data(
    data_dir: Path,
    *,
    final_price_path: Path | None = None,
    sp500_price_path: Path | None = None,
) -> RawDataBundle:
    base_dir = _resolve_dataset_dir(data_dir)
    final_price = pl.read_parquet(final_price_path) if final_price_path is not None else _read_parquet_with_fallbacks(
        base_dir,
        ["US_Finalprice.parquet", "US_finalprice.parquet"],
    )
    sp500_price = pl.read_parquet(sp500_price_path) if sp500_price_path is not None else _read_parquet_with_fallbacks(
        base_dir,
        ["SP500Price.parquet", "SP500_price.parquet"],
    )

    return RawDataBundle(
        final_price=final_price,
        income_statement=_read_parquet_with_fallbacks(
            base_dir,
            ["US_Income_statement.parquet", "US_income_statement.parquet"],
        ),
        balance_sheet=_read_parquet_with_fallbacks(
            base_dir,
            ["US_Balance_sheet.parquet", "US_balance_sheet.parquet"],
        ),
        cash_flow=_read_parquet_with_fallbacks(
            base_dir,
            ["US_Cash_flow.parquet", "US_cash_flow.parquet"],
        ),
        earnings=_read_parquet_with_fallbacks(
            base_dir,
            ["US_Earnings.parquet", "US_earnings.parquet"],
        ),
        constituents=_read_csv_with_fallbacks(
            base_dir,
            ["SP500_Constituents.csv", "sp500_constituents.csv"],
        ),
        sp500_price=sp500_price,
    )


def find_existing_column(df: pl.DataFrame, candidates: Iterable[str]) -> str | None:
    column_map: Dict[str, str] = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        found = column_map.get(candidate.lower())
        if found is not None:
            return found
    return None
