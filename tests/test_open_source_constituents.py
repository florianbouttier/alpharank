from __future__ import annotations

from datetime import date, datetime, timedelta, timezone

import polars as pl

from alpharank.data.open_source.constituents import (
    membership_at_decision_time,
    refresh_monthly_constituents,
    resolve_constituent_snapshot_duplicates,
)


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
    assert may == {"NEW", "KEEP"}
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


def test_membership_effective_at_decision_time() -> None:
    source = pl.DataFrame(
        {
            "Date": [date(2026, 4, 1)] * 3,
            "Ticker": ["CTRA", "POOL", "EA"],
            "Name": ["Coterra", "Pool", "Electronic Arts"],
        }
    )
    registry = {
        "base_month": "2026-04-01",
        "events": [
            {
                "effective_date": "2026-05-07",
                "source_url": "https://example.test/veev",
                "operations": [
                    {"action": "add", "ticker": "VEEV", "name": "Veeva"},
                    {"action": "remove", "ticker": "CTRA"},
                ],
            },
            {
                "effective_date": "2026-06-22",
                "source_url": "https://example.test/mrvl-flex",
                "operations": [
                    {"action": "add", "ticker": "MRVL", "name": "Marvell"},
                    {"action": "add", "ticker": "FLEX", "name": "Flex"},
                    {"action": "remove", "ticker": "POOL"},
                ],
            },
            {
                "effective_date": "2026-08-05",
                "source_url": "https://example.test/ferg",
                "operations": [
                    {"action": "add", "ticker": "FERG", "name": "Ferguson"},
                    {"action": "remove", "ticker": "EA"},
                ],
            },
        ],
    }
    effective_times = [
        datetime(2026, 5, 7, 4, 0, tzinfo=timezone.utc),
        datetime(2026, 6, 22, 4, 0, tzinfo=timezone.utc),
        datetime(2026, 8, 5, 4, 0, tzinfo=timezone.utc),
    ]
    decisions = [
        value + offset
        for value in effective_times
        for offset in (-timedelta(microseconds=1), timedelta(0))
    ]

    membership = membership_at_decision_time(
        source,
        registry=registry,
        decision_times=decisions,
    )

    def names(at: datetime) -> set[str]:
        return set(
            membership.filter(pl.col("decision_at") == at)["ticker"].to_list()
        )

    assert "VEEV" not in names(decisions[0])
    assert "VEEV" in names(decisions[1]) and "CTRA" not in names(decisions[1])
    assert {"MRVL", "FLEX"}.isdisjoint(names(decisions[2]))
    assert {"MRVL", "FLEX"}.issubset(names(decisions[3]))
    assert "EA" in names(decisions[4]) and "FERG" not in names(decisions[4])
    assert "EA" not in names(decisions[5]) and "FERG" in names(decisions[5])


def test_constituent_snapshot_has_unique_key() -> None:
    duplicate_groups = 214
    rows: list[dict[str, object]] = []
    for index in range(duplicate_groups):
        ticker = f"T{index:03d}"
        rows.extend(
            [
                {"Date": date(2020, 1, 1), "Ticker": ticker, "Name": "Zulu"},
                {"Date": date(2020, 1, 1), "Ticker": ticker, "Name": "Alpha"},
                {"Date": date(2020, 1, 1), "Ticker": ticker, "Name": "Zulu"},
            ]
        )

    result = resolve_constituent_snapshot_duplicates(pl.DataFrame(rows))

    assert result.frame.height == duplicate_groups
    assert (
        result.frame.group_by(["Date", "Ticker"])
        .len()
        .filter(pl.col("len") > 1)
        .is_empty()
    )
    assert len(result.audit) == duplicate_groups
    assert {row["selected_name"] for row in result.audit} == {"Zulu"}
    assert all(row["resolution_rule"] for row in result.audit)
