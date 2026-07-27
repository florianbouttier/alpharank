from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.data.open_source.constituents import refresh_monthly_constituents


def test_refresh_monthly_constituents_obeys_effective_month_and_ticker_change() -> None:
    source = pl.DataFrame(
        {
            "Date": [date(2026, 4, 1), date(2026, 4, 1)],
            "Ticker": ["OLD", "KEEP"],
            "Name": ["Old Co", "Keep Co"],
        }
    )
    registry = {
        "base_month": "2026-04-01",
        "events": [
            {
                "effective_date": "2026-05-07",
                "source_url": "https://example.test/add",
                "operations": [
                    {"action": "add", "ticker": "NEW", "name": "New Co"},
                    {"action": "remove", "ticker": "OLD"},
                ],
            },
            {
                "effective_date": "2026-06-01",
                "source_url": "https://example.test/rename",
                "operations": [
                    {
                        "action": "ticker_change",
                        "ticker": "KEEP",
                        "new_ticker": "KEPT",
                        "name": "Kept Co",
                    }
                ],
            },
        ],
    }

    result = refresh_monthly_constituents(
        source,
        registry=registry,
        target_month=date(2026, 7, 1),
    )

    april = set(result.frame.filter(pl.col("Date") == date(2026, 4, 1))["Ticker"])
    may = set(result.frame.filter(pl.col("Date") == date(2026, 5, 1))["Ticker"])
    june = set(result.frame.filter(pl.col("Date") == date(2026, 6, 1))["Ticker"])
    july = set(result.frame.filter(pl.col("Date") == date(2026, 7, 1))["Ticker"])
    assert april == {"OLD", "KEEP"}
    assert may == {"OLD", "KEEP"}
    assert june == {"NEW", "KEPT"}
    assert july == {"NEW", "KEPT"}
    assert len(result.operation_audit) == 3


def test_refresh_monthly_constituents_requires_explicit_noop_permission() -> None:
    source = pl.DataFrame(
        {
            "Date": [date(2026, 4, 1)],
            "Ticker": ["KEEP"],
            "Name": ["Keep Co"],
        }
    )
    registry = {
        "base_month": "2026-04-01",
        "events": [
            {
                "effective_date": "2026-04-09",
                "source_url": "https://example.test",
                "operations": [
                    {
                        "action": "add",
                        "ticker": "KEEP",
                        "name": "Keep Co",
                        "allow_existing": True,
                    }
                ],
            }
        ],
    }
    result = refresh_monthly_constituents(
        source,
        registry=registry,
        target_month=date(2026, 5, 1),
    )
    assert result.operation_audit[0]["status"] == "inherited_snapshot_already_applied"
