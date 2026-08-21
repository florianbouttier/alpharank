from __future__ import annotations

from pathlib import Path

import polars as pl


def _write_minimal_legacy_reference(reference_dir: Path, *, ticker: str, code: str, name: str) -> None:
    pl.DataFrame(
        {
            "ticker": [ticker],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "US_Finalprice.parquet")
    pl.DataFrame(
        {
            "ticker": ["SPY.US"],
            "date": ["2025-01-01"],
            "adjusted_close": [1.0],
            "close": [1.0],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "volume": [10.0],
        }
    ).write_parquet(reference_dir / "SP500Price.parquet")
    pl.DataFrame(
        {
            "Code": [code],
            "Name": [name],
            "Exchange": ["NASDAQ"],
            "CurrencyCode": ["USD"],
            "CurrencySymbol": ["$"],
            "CIK": ["0000000000"],
            "Sector": ["Technology"],
            "Industry": [""],
        }
    ).write_parquet(reference_dir / "US_General.parquet")
    pl.DataFrame({"ticker": [ticker], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "totalRevenue": [100.0]}).write_parquet(
        reference_dir / "US_Income_statement.parquet"
    )
    pl.DataFrame(
        {
            "ticker": [ticker],
            "date": ["2025-03-31"],
            "filing_date": ["2025-05-01"],
            "commonStockSharesOutstanding": ["0.0"],
            "totalAssets": [500.0],
        }
    ).write_parquet(reference_dir / "US_Balance_sheet.parquet")
    pl.DataFrame({"ticker": [ticker], "date": ["2025-03-31"], "filing_date": ["2025-05-01"], "freeCashFlow": [50.0]}).write_parquet(
        reference_dir / "US_Cash_flow.parquet"
    )
    pl.DataFrame(
        {"ticker": [ticker], "date": ["2025-03-31"], "dateFormatted": ["2025-03-31"], "sharesMln": [10.0], "shares": [10_000_000.0]}
    ).write_parquet(reference_dir / "US_share.parquet")
    pl.DataFrame(
        {
            "ticker": [ticker],
            "beforeAfterMarket": [""],
            "currency": [""],
            "date": ["2025-03-31"],
            "epsActual": [1.5],
            "epsDifference": [0.1],
            "epsEstimate": [1.4],
            "reportDate": ["2025-05-01"],
            "surprisePercent": [7.0],
        }
    ).write_parquet(reference_dir / "US_Earnings.parquet")
