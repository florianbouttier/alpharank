from __future__ import annotations

import json

import polars as pl

from alpharank.data.ticker_integrity import (
    exclude_tickers_from_frame,
    load_ticker_exclusion_registry,
)


def test_registry_returns_only_full_trajectory_exclusions(tmp_path) -> None:
    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "registry_id": "test",
                "entries": [
                    {"ticker": "bad.us", "decision": "exclude_all_dates"},
                    {"ticker": "review.us", "decision": "review"},
                ],
            }
        ),
        encoding="utf-8",
    )

    registry = load_ticker_exclusion_registry(path)

    assert registry.excluded_tickers == ("BAD.US",)


def test_default_registry_is_the_reviewed_ten_ticker_quarantine() -> None:
    registry = load_ticker_exclusion_registry()

    assert registry.registry_id == "historical_ticker_exclusions_v1"
    assert registry.excluded_tickers == (
        "SII.US",
        "CBE.US",
        "TIE.US",
        "CPWR.US",
        "BMC.US",
        "COL.US",
        "GR.US",
        "EP.US",
        "SW.US",
        "HAR.US",
    )


def test_exclusion_applies_to_complete_ticker_trajectory() -> None:
    frame = pl.DataFrame(
        {
            "ticker": ["BAD.US", "GOOD.US", "BAD.US"],
            "date": ["2010-01-01", "2010-01-01", "2025-01-01"],
        }
    )

    filtered = exclude_tickers_from_frame(frame, ("bad.us",))

    assert filtered["ticker"].to_list() == ["GOOD.US"]


def test_constituent_symbols_are_normalized_before_exclusion() -> None:
    frame = pl.DataFrame({"Ticker": ["BRK.B", "EP", "AAPL"]})

    filtered = exclude_tickers_from_frame(
        frame,
        ("BRK-B.US", "EP.US"),
        ticker_column="Ticker",
    )

    assert filtered["Ticker"].to_list() == ["AAPL"]
