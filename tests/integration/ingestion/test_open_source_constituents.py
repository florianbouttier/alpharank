from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from alpharank.data.sources.constituents import (
    membership_at_decision_time,
    refresh_monthly_constituents,
    resolve_constituent_snapshot_duplicates,
)


def _lineaged_event(
    event_id: str,
    effective_date: str,
    source_url: str,
    operations: list[dict[str, Any]],
) -> dict[str, Any]:
    effective_day = date.fromisoformat(effective_date)
    observed_day = effective_day - timedelta(days=1)
    return {
        "event_id": event_id,
        "observed_at": f"{observed_day.isoformat()}T23:59:59-04:00",
        "effective_at": f"{effective_date}T00:00:00-04:00",
        "effective_date": effective_date,
        "source_url": source_url,
        "confidence": "high",
        "operations": operations,
    }


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
            _lineaged_event(
                "test-20260507-add",
                "2026-05-07",
                "https://example.test/add",
                [
                    {"action": "add", "ticker": "NEW", "name": "New Co"},
                    {"action": "remove", "ticker": "OLD"},
                ],
            ),
            _lineaged_event(
                "test-20260601-rename",
                "2026-06-01",
                "https://example.test/rename",
                [
                    {
                        "action": "ticker_change",
                        "ticker": "KEEP",
                        "new_ticker": "KEPT",
                        "name": "Kept Co",
                    }
                ],
            ),
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


def test_august_registry_separates_avb_from_vivmark_successor() -> None:
    project_root = Path(__file__).resolve().parents[3]
    required_tickers = [
        "CASY",
        "CTRA",
        "BK",
        "EPAM",
        "POOL",
        "CPB",
        "SATS",
        "CAG",
        "HON",
        "EA",
        "AVB",
        "EQR",
    ]
    tickers = required_tickers + [f"TEST{index:03d}" for index in range(491)]
    source = pl.DataFrame(
        {
            "Date": [date(2026, 4, 1)] * len(tickers),
            "Ticker": tickers,
            "Name": [f"Name {ticker}" for ticker in tickers],
        }
    )
    registry = json.loads(
        (
            project_root
            / "configs"
            / "data_quality"
            / "sp500_constituent_changes_2026.json"
        ).read_text(encoding="utf-8")
    )

    result = refresh_monthly_constituents(
        source, registry=registry, target_month=date(2026, 8, 1)
    )
    august = set(
        result.frame.filter(pl.col("Date") == date(2026, 8, 1))["Ticker"]
    )

    assert {"RDDT", "VMRK", "FERG"}.issubset(august)
    assert {"AVB", "EQR", "EA"}.isdisjoint(august)
    assert result.frame.filter(pl.col("Date") == date(2026, 8, 1)).height == 503


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
            _lineaged_event(
                "test-20260409-noop",
                "2026-04-09",
                "https://example.test/noop",
                [
                    {
                        "action": "add",
                        "ticker": "KEEP",
                        "name": "Keep Co",
                        "allow_existing": True,
                    }
                ],
            )
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
            _lineaged_event(
                "test-20260507-veev",
                "2026-05-07",
                "https://example.test/veev",
                [
                    {"action": "add", "ticker": "VEEV", "name": "Veeva"},
                    {"action": "remove", "ticker": "CTRA"},
                ],
            ),
            _lineaged_event(
                "test-20260622-mrvl-flex",
                "2026-06-22",
                "https://example.test/mrvl-flex",
                [
                    {"action": "add", "ticker": "MRVL", "name": "Marvell"},
                    {"action": "add", "ticker": "FLEX", "name": "Flex"},
                    {"action": "remove", "ticker": "POOL"},
                ],
            ),
            _lineaged_event(
                "test-20260805-ferg",
                "2026-08-05",
                "https://example.test/ferg",
                [
                    {"action": "add", "ticker": "FERG", "name": "Ferguson"},
                    {"action": "remove", "ticker": "EA"},
                ],
            ),
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


def test_membership_event_lineage_is_complete() -> None:
    source = pl.DataFrame(
        {"Date": [date(2026, 4, 1)], "Ticker": ["OLD"], "Name": ["Old Co"]}
    )
    event = _lineaged_event(
        "test-20260507-new-old",
        "2026-05-07",
        "https://example.test/new-old",
        [
            {"action": "add", "ticker": "NEW", "name": "New Co"},
            {"action": "remove", "ticker": "OLD"},
        ],
    )
    registry = {"base_month": "2026-04-01", "events": [event]}

    result = refresh_monthly_constituents(
        source, registry=registry, target_month=date(2026, 5, 1)
    )

    assert len(result.operation_audit) == 2
    assert all(
        {
            "event_id",
            "source_url",
            "observed_at",
            "effective_at",
            "effective_date",
            "confidence",
        }
        <= row.keys()
        for row in result.operation_audit
    )
    incomplete_event = dict(event)
    incomplete_event.pop("observed_at")
    with pytest.raises(ValueError, match="missing observed_at"):
        membership_at_decision_time(
            source,
            registry={"base_month": "2026-04-01", "events": [incomplete_event]},
            decision_times=[datetime(2026, 5, 31, tzinfo=timezone.utc)],
        )


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
