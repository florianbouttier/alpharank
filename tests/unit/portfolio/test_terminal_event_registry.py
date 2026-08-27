from __future__ import annotations

import json
from copy import deepcopy
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.terminal_event_registry import (
    DEFAULT_TERMINAL_EVENT_REGISTRY,
    load_terminal_event_registry,
)
from alpharank.portfolio.terminal_returns import (
    resolve_provisional_terminal_shareholder_returns,
    resolve_terminal_shareholder_returns,
)


def test_terminal_event_registry_is_complete_and_fail_closed() -> None:
    registry = load_terminal_event_registry()

    assert registry.path == DEFAULT_TERMINAL_EVENT_REGISTRY.resolve()
    assert len(registry.sha256) == 64
    assert len(registry.events) == 9
    assert sum(len(event["source_documents"]) for event in registry.events) == 13

    terminal = registry.terminal_consideration_events(
        price_vintage_id="prices-v2"
    )
    assert terminal.height == 8
    assert terminal["terminal_event_id"].n_unique() == 8
    assert terminal["price_vintage_id"].unique().to_list() == ["prices-v2"]

    kraft = terminal.filter(terminal["ticker"] == "KRFT.US").row(0, named=True)
    assert kraft["event_type"] == "stock_merger"
    assert kraft["successor_ticker"] == "KHC.US"
    assert kraft["exchange_ratio"] == 1.0
    assert kraft["distribution_per_share"] == pytest.approx(17.05)

    express_scripts = terminal.filter(terminal["ticker"] == "ESRX.US").row(
        0, named=True
    )
    assert express_scripts["cash_per_share"] == 48.75
    assert express_scripts["successor_ticker"] == "CI.US"
    assert express_scripts["exchange_ratio"] == 0.2434

    avalonbay = terminal.filter(terminal["ticker"] == "AVB.US").row(0, named=True)
    assert avalonbay["successor_ticker"] == "VMRK.US"
    assert avalonbay["exchange_ratio"] == 2.793
    assert avalonbay["cash_per_share"] == 0.0

    blocks = registry.pre_execution_blocks()
    assert blocks.height == 1
    frc = blocks.row(0, named=True)
    assert frc["ticker"] == "FRC.US"
    assert frc["effective_date"] == date(2023, 5, 1)
    assert frc["known_at"] == datetime(
        2023, 5, 1, 7, 26, tzinfo=timezone.utc
    )
    assert frc["entry_allowed"] is False

    entry_blocks = registry.terminal_entry_blocks()
    assert entry_blocks.height == 9
    expected_months = {
        "KRFT.US": date(2015, 8, 1),
        "HSP.US": date(2015, 10, 1),
        "WFM.US": date(2017, 9, 1),
        "ESRX.US": date(2019, 1, 1),
        "SCG.US": date(2019, 1, 1),
        "NFX.US": date(2019, 3, 1),
        "NLSN.US": date(2022, 11, 1),
        "FRC.US": date(2023, 5, 1),
        "AVB.US": date(2026, 9, 1),
    }
    assert {
        row["ticker"]: row["blocked_from_holding_month"]
        for row in entry_blocks.to_dicts()
    } == expected_months


def test_reviewed_registry_projects_to_terminal_return_contract() -> None:
    registry = load_terminal_event_registry()
    holdings = pl.DataFrame(
        {
            "ticker": [
                "KRFT.US",
                "HSP.US",
                "WFM.US",
                "ESRX.US",
                "NFX.US",
                "NLSN.US",
            ],
            "holding_month": [
                date(2015, 7, 1),
                date(2015, 9, 1),
                date(2017, 8, 1),
                date(2018, 12, 1),
                date(2019, 2, 1),
                date(2022, 10, 1),
            ],
            "realized_return": [None] * 6,
            "last_close": [85.49, 89.97, 41.80, 101.51, 18.38, 27.86],
        }
    )
    successor_prices = pl.DataFrame(
        {
            "ticker": ["KHC.US", "CI.US", "ECA.US"],
            "holding_month": [
                date(2015, 7, 1),
                date(2018, 12, 1),
                date(2019, 2, 1),
            ],
            "price_asof_date": [
                date(2015, 7, 31),
                date(2018, 12, 31),
                date(2019, 2, 28),
            ],
            "holding_end_price": [79.47, 189.92, 7.25],
            "price_vintage_id": ["prices-v2"] * 3,
        }
    )

    result = resolve_terminal_shareholder_returns(
        holdings,
        terminal_events=registry.terminal_consideration_events(
            price_vintage_id="prices-v2"
        ),
        successor_prices=successor_prices,
        price_vintage_id="prices-v2",
    )

    expected_values = [
        79.47 + 17.05,
        90.0,
        42.0,
        48.75 + 0.2434 * 189.92,
        2.6719 * 7.25,
        28.0,
    ]
    expected_returns = [
        value / start - 1.0
        for value, start in zip(
            expected_values,
            [85.49, 89.97, 41.80, 101.51, 18.38, 27.86],
            strict=True,
        )
    ]
    assert result.holdings["realized_return"].to_list() == pytest.approx(
        expected_returns
    )
    assert result.report["resolved_terminal_returns"] == 6


def test_reviewed_registry_replaces_only_provisional_terminal_returns() -> None:
    registry = load_terminal_event_registry()
    holdings = pl.DataFrame(
        {
            "strategy": ["Legacy", "Legacy"],
            "ticker": ["HSP.US", "OTHER.US"],
            "holding_month": [date(2015, 9, 1), date(2015, 9, 1)],
            "realized_return": [-0.001, 0.10],
            "return_resolution": [
                "provisional_last_observation",
                "observed_market_next_open_to_month_end",
            ],
            "return_resolution_reason": [
                "ticker_price_series_ended_before_market_month_end",
                None,
            ],
            "manual_review_status": [
                "pending_manual_terminal_event_review",
                None,
            ],
            "execution_price_unadjusted": [89.97, 100.0],
            "holding_return_end_at": [
                datetime(2015, 9, 2, 20, tzinfo=timezone.utc),
                datetime(2015, 9, 30, 20, tzinfo=timezone.utc),
            ],
            "scheduled_holding_end_at": [
                datetime(2015, 9, 30, 20, tzinfo=timezone.utc),
                datetime(2015, 9, 30, 20, tzinfo=timezone.utc),
            ],
        }
    )

    result = resolve_provisional_terminal_shareholder_returns(
        holdings,
        terminal_events=registry.terminal_consideration_events(
            price_vintage_id="prices-v2"
        ),
        price_vintage_id="prices-v2",
    )

    hsp = result.holdings.filter(pl.col("ticker") == "HSP.US").row(0, named=True)
    other = result.holdings.filter(pl.col("ticker") == "OTHER.US").row(
        0, named=True
    )
    assert hsp["realized_return"] == pytest.approx(90.0 / 89.97 - 1.0)
    assert hsp["return_resolution"] == "resolved_terminal_event"
    assert hsp["manual_review_status"] == "reviewed_terminal_event_resolved"
    assert hsp["holding_return_end_at"] == hsp["scheduled_holding_end_at"]
    assert other["realized_return"] == 0.10
    assert other["return_resolution"] == (
        "observed_market_next_open_to_month_end"
    )
    assert result.report["reviewed_provisional_rows"] == 1


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda payload: payload["events"][1]["portfolio_resolution"].update(
                {"cash_per_share": None}
            ),
            "requires positive cash",
        ),
        (
            lambda payload: payload["events"][4]["portfolio_resolution"].update(
                {"successor_ticker": None}
            ),
            "successor_ticker must be a non-empty string",
        ),
        (
            lambda payload: payload["events"][7]["portfolio_resolution"].update(
                {"allow_entry": True}
            ),
            "must reject the fill",
        ),
        (
            lambda payload: payload["events"][7].update(
                {"known_at": "2023-05-01T10:00:00-04:00"}
            ),
            "was not known before open",
        ),
        (
            lambda payload: payload["events"][0]["source_documents"][0].update(
                {"sha256": "not-a-hash"}
            ),
            "invalid SHA-256",
        ),
    ],
)
def test_terminal_event_registry_rejects_ambiguous_records(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    payload = deepcopy(
        json.loads(DEFAULT_TERMINAL_EVENT_REGISTRY.read_text(encoding="utf-8"))
    )
    mutation(payload)
    path = tmp_path / "terminal_events.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_terminal_event_registry(path)
