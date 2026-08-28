from datetime import date

import polars as pl
import pytest

from alpharank.data.terminal_eligibility import (
    apply_terminal_entry_gate,
    apply_terminal_entry_gate_to_decisions,
)
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
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


def test_entry_only_events_block_the_four_no_sec_impossible_fills() -> None:
    candidates = pl.DataFrame(
        {
            "year_month": [
                date(2010, 3, 1),
                date(2019, 10, 1),
                date(2022, 11, 1),
                date(2023, 1, 1),
            ],
            "ticker": ["RX.US", "TSS.US", "TWTR.US", "ABMD.US"],
        }
    )

    result = apply_terminal_entry_gate(
        candidates,
        load_terminal_event_registry().terminal_entry_blocks(),
    )

    assert result.eligible.is_empty()
    assert result.blocked.select("ticker", "entry_block_rule").to_dicts() == [
        {
            "ticker": "RX.US",
            "entry_block_rule": "completed_terminal_event_blocks_post_event_entry",
        },
        {
            "ticker": "TSS.US",
            "entry_block_rule": "completed_terminal_event_blocks_post_event_entry",
        },
        {
            "ticker": "TWTR.US",
            "entry_block_rule": "completed_terminal_event_blocks_post_event_entry",
        },
        {
            "ticker": "ABMD.US",
            "entry_block_rule": "completed_terminal_event_blocks_post_event_entry",
        },
    ]


def test_decision_gate_blocks_terminal_entries_before_ranking() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [
                date(2018, 12, 1),
                date(2018, 12, 1),
                date(2023, 5, 1),
                date(2023, 5, 1),
            ],
            "ticker": ["SCG.US", "AAPL.US", "FRC.US", "MSFT.US"],
            "score": [0.99, 0.80, 0.98, 0.79],
            "future_return_1m": [0.0, 0.10, 0.0, 0.05],
            "benchmark_future_return_1m": [0.02, 0.02, 0.01, 0.01],
        }
    )

    result = apply_terminal_entry_gate_to_decisions(
        predictions,
        load_terminal_event_registry().terminal_entry_blocks(),
    )

    assert result.eligible.select(
        "decision_month", "holding_month", "ticker"
    ).to_dicts() == [
        {
            "decision_month": date(2018, 12, 1),
            "holding_month": date(2019, 1, 1),
            "ticker": "AAPL.US",
        },
        {
            "decision_month": date(2023, 5, 1),
            "holding_month": date(2023, 6, 1),
            "ticker": "MSFT.US",
        },
    ]
    assert result.blocked.select(
        "decision_month", "holding_month", "ticker", "terminal_event_id"
    ).to_dicts() == [
        {
            "decision_month": date(2018, 12, 1),
            "holding_month": date(2019, 1, 1),
            "ticker": "SCG.US",
            "terminal_event_id": "SCG-2019-01-01-D",
        },
        {
            "decision_month": date(2023, 5, 1),
            "holding_month": date(2023, 6, 1),
            "ticker": "FRC.US",
            "terminal_event_id": "FRC-2023-05-01-FDIC",
        },
    ]
    holdings = boosting_predictions_to_holdings(
        result.eligible.drop("holding_month"),
        strategy="Boosting Top 1",
        top_n=1,
    )
    assert holdings.select("holding_month", "ticker").to_dicts() == [
        {"holding_month": date(2019, 1, 1), "ticker": "AAPL.US"},
        {"holding_month": date(2023, 6, 1), "ticker": "MSFT.US"},
    ]
