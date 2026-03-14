from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).parent.parent / "src"))

from alpharank.backtest.data_loading import RawDataBundle
from alpharank.backtest.pipeline import _apply_data_quality_ticker_exclusions


def test_apply_data_quality_ticker_exclusions_matches_legacy_list() -> None:
    raw = RawDataBundle(
        final_price=pl.DataFrame({"ticker": ["AAA.US", "SII.US"], "date": [date(2020, 1, 31), date(2020, 1, 31)]}),
        income_statement=pl.DataFrame({"ticker": ["CBE.US", "AAA.US"], "date": [date(2020, 1, 31), date(2020, 1, 31)]}),
        balance_sheet=pl.DataFrame({"ticker": ["TIE.US", "AAA.US"], "date": [date(2020, 1, 31), date(2020, 1, 31)]}),
        cash_flow=pl.DataFrame({"ticker": ["AAA.US", "SII.US"], "date": [date(2020, 1, 31), date(2020, 1, 31)]}),
        earnings=pl.DataFrame({"ticker": ["AAA.US", "CBE.US"], "date": [date(2020, 1, 31), date(2020, 1, 31)]}),
        constituents=pl.DataFrame({"Ticker": ["AAA", "TIE"], "Date": [date(2020, 1, 1), date(2020, 1, 1)]}),
        sp500_price=pl.DataFrame({"date": [date(2020, 1, 31)], "close": [100.0]}),
    )

    filtered = _apply_data_quality_ticker_exclusions(raw, ("SII.US", "CBE.US", "TIE.US"))

    assert filtered.final_price.get_column("ticker").to_list() == ["AAA.US"]
    assert filtered.income_statement.get_column("ticker").to_list() == ["AAA.US"]
    assert filtered.balance_sheet.get_column("ticker").to_list() == ["AAA.US"]
    assert filtered.cash_flow.get_column("ticker").to_list() == ["AAA.US"]
    assert filtered.earnings.get_column("ticker").to_list() == ["AAA.US"]
    assert filtered.constituents.get_column("Ticker").to_list() == ["AAA"]
    assert filtered.sp500_price.height == 1
