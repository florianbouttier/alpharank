from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from alpharank.replay.sector_allocation import build_sector_champion_allocation


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 12, tzinfo=timezone.utc)


def _predictions() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "decision_month": [date(2026, 4, 1)] * 7,
            "ticker": ["MISS", "TECH", "HEALTH", "ENERGY", "FIN", "IND", "TECH2"],
            "score": [0.99, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40],
            "future_return_1m": [0.01] * 7,
            "benchmark_future_return_1m": [0.005] * 7,
        }
    ).with_columns(pl.col("decision_month").cast(pl.Date))


def _history() -> pl.DataFrame:
    sectors = {
        "TECH": "Technology",
        "HEALTH": "Healthcare",
        "ENERGY": "Energy",
        "FIN": "Financial Services",
        "IND": "Industrials",
        "TECH2": "Technology",
    }
    return pl.DataFrame(
        {
            "ticker": list(sectors),
            "Sector": list(sectors.values()),
            "observed_at": [_at(2026, 4, 15)] * len(sectors),
            "effective_at": [_at(2026, 4, 15)] * len(sectors),
            "classification_id": [f"sector-{ticker}" for ticker in sectors],
            "source_url": ["https://example.test/sector"] * len(sectors),
            "confidence": ["provider_reported"] * len(sectors),
        }
    )


def _market_prices() -> pl.DataFrame:
    return pl.DataFrame({"date": ["2026-04-29", "2026-04-30"]})


def test_sector_champions_exclude_unknown_and_select_one_name_per_sector() -> None:
    result = build_sector_champion_allocation(
        _predictions(),
        _history(),
        _market_prices(),
        strategy="Boosting Top 5 | 1 per sector",
    )

    assert result.holdings["ticker"].to_list() == ["TECH", "HEALTH", "ENERGY", "FIN", "IND"]
    assert result.holdings["Sector"].n_unique() == 5
    assert result.holdings["target_weight"].to_list() == [0.2] * 5
    assert result.audit["missing_sector_tickers"].to_list() == [["MISS"]]
    assert result.audit["status"].to_list() == ["selected"]


def test_future_sector_mutation_does_not_change_past_selection() -> None:
    baseline = build_sector_champion_allocation(
        _predictions(),
        _history(),
        _market_prices(),
        strategy="sector",
    )
    future = pl.concat(
        [
            _history(),
            pl.DataFrame(
                {
                    "ticker": ["TECH"],
                    "Sector": ["Utilities"],
                    "observed_at": [_at(2026, 5, 2)],
                    "effective_at": [_at(2026, 5, 2)],
                    "classification_id": ["sector-tech-future"],
                    "source_url": ["https://example.test/future"],
                    "confidence": ["provider_reported"],
                }
            ),
        ],
        how="vertical",
    )
    mutated = build_sector_champion_allocation(
        _predictions(),
        future,
        _market_prices(),
        strategy="sector",
    )

    assert mutated.holdings.select("ticker", "Sector").equals(
        baseline.holdings.select("ticker", "Sector")
    )


def test_sector_champions_block_when_five_sectors_are_not_known() -> None:
    result = build_sector_champion_allocation(
        _predictions(),
        _history().filter(pl.col("Sector") != "Industrials"),
        _market_prices(),
        strategy="sector",
    )

    assert result.holdings.is_empty()
    assert result.audit["status"].to_list() == ["blocked_insufficient_distinct_sectors"]
