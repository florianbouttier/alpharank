from __future__ import annotations

from dataclasses import dataclass, field
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
    source_paths: Dict[str, Path] = field(default_factory=dict)


def _resolve_dataset_dir(data_dir: Path) -> Path:
    us_dir = data_dir / "US"
    return us_dir if us_dir.exists() else data_dir


def _resolve_existing_path(base_dir: Path, candidates: Iterable[str]) -> Path:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not locate parquet file in {base_dir} among: {list(candidates)}")


def _read_parquet_with_fallbacks(base_dir: Path, candidates: Iterable[str]) -> tuple[pl.DataFrame, Path]:
    path = _resolve_existing_path(base_dir, candidates)
    return pl.read_parquet(path), path


def _read_csv_with_fallbacks(base_dir: Path, candidates: Iterable[str]) -> tuple[pl.DataFrame, Path]:
    for name in candidates:
        candidate = base_dir / name
        if candidate.exists():
            return pl.read_csv(candidate, try_parse_dates=True), candidate
    raise FileNotFoundError(f"Could not locate csv file in {base_dir} among: {list(candidates)}")


def load_raw_data(
    data_dir: Path,
    *,
    final_price_path: Path | None = None,
    sp500_price_path: Path | None = None,
) -> RawDataBundle:
    base_dir = _resolve_dataset_dir(data_dir)
    if final_price_path is not None:
        final_price_source = Path(final_price_path)
        final_price = pl.read_parquet(final_price_source)
    else:
        final_price, final_price_source = _read_parquet_with_fallbacks(
            base_dir,
            ["US_Finalprice.parquet", "US_finalprice.parquet"],
        )

    if sp500_price_path is not None:
        sp500_price_source = Path(sp500_price_path)
        sp500_price = pl.read_parquet(sp500_price_source)
    else:
        sp500_price, sp500_price_source = _read_parquet_with_fallbacks(
            base_dir,
            ["SP500Price.parquet", "SP500_price.parquet"],
        )

    income_statement, income_statement_source = _read_parquet_with_fallbacks(
        base_dir,
        ["US_Income_statement.parquet", "US_income_statement.parquet"],
    )
    balance_sheet, balance_sheet_source = _read_parquet_with_fallbacks(
        base_dir,
        ["US_Balance_sheet.parquet", "US_balance_sheet.parquet"],
    )
    cash_flow, cash_flow_source = _read_parquet_with_fallbacks(
        base_dir,
        ["US_Cash_flow.parquet", "US_cash_flow.parquet"],
    )
    earnings, earnings_source = _read_parquet_with_fallbacks(
        base_dir,
        ["US_Earnings.parquet", "US_earnings.parquet"],
    )
    constituents, constituents_source = _read_csv_with_fallbacks(
        base_dir,
        ["SP500_Constituents.csv", "sp500_constituents.csv"],
    )

    return RawDataBundle(
        final_price=final_price,
        income_statement=income_statement,
        balance_sheet=balance_sheet,
        cash_flow=cash_flow,
        earnings=earnings,
        constituents=constituents,
        sp500_price=sp500_price,
        source_paths={
            "final_price": final_price_source,
            "income_statement": income_statement_source,
            "balance_sheet": balance_sheet_source,
            "cash_flow": cash_flow_source,
            "earnings": earnings_source,
            "sp500_constituents": constituents_source,
            "sp500_price": sp500_price_source,
        },
    )


def find_existing_column(df: pl.DataFrame, candidates: Iterable[str]) -> str | None:
    column_map: Dict[str, str] = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        found = column_map.get(candidate.lower())
        if found is not None:
            return found
    return None
