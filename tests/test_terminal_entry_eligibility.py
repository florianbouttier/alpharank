from datetime import date

import polars as pl
import pytest

from alpharank.data.terminal_eligibility import apply_terminal_entry_gate
from alpharank.portfolio.terminal_event_registry import load_terminal_event_registry


def test_terminal_entry_gate_blocks_only_economically_impossible_entries() -> None:
    candidates = pl.DataFrame(
        {
            "year_month": [
                date(2018, 12, 1),
                date(2019, 1, 1),
                date(2023, 4, 1),
                date(2023, 5, 1),
                date(2023, 5, 1),
            ],
            "ticker": ["ESRX.US", "ESRX.US", "FRC.US", "FRC.US", "AAPL.US"],
            "score": [0.9, 0.95, 0.8, 0.85, 0.7],
        }
    )

    result = apply_terminal_entry_gate(
        candidates,
        load_terminal_event_registry().terminal_entry_blocks(),
    )

    assert result.eligible.select("year_month", "ticker").to_dicts() == [
        {"year_month": date(2018, 12, 1), "ticker": "ESRX.US"},
        {"year_month": date(2023, 4, 1), "ticker": "FRC.US"},
        {"year_month": date(2023, 5, 1), "ticker": "AAPL.US"},
    ]
    assert result.blocked.select(
        "year_month", "ticker", "terminal_event_id"
    ).to_dicts() == [
        {
            "year_month": date(2019, 1, 1),
            "ticker": "ESRX.US",
            "terminal_event_id": "ESRX-2018-12-20-CI",
        },
        {
            "year_month": date(2023, 5, 1),
            "ticker": "FRC.US",
            "terminal_event_id": "FRC-2023-05-01-FDIC",
        },
    ]


def test_terminal_entry_gate_fails_on_duplicate_ticker_rules() -> None:
    candidates = pl.DataFrame(
        {"year_month": [date(2019, 1, 1)], "ticker": ["ESRX.US"]}
    )
    blocks = load_terminal_event_registry().terminal_entry_blocks()
    blocks = pl.concat([blocks, blocks.filter(pl.col("ticker") == "ESRX.US")])

    with pytest.raises(ValueError, match="Multiple terminal entry blocks"):
        apply_terminal_entry_gate(candidates, blocks)
