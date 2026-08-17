from __future__ import annotations

from datetime import datetime, timezone

import polars as pl
import pytest

from alpharank.data.sector_history import resolve_point_in_time_sectors


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 20, tzinfo=timezone.utc)


def test_sector_used_was_known_at_decision_date() -> None:
    decisions = pl.DataFrame(
        {
            "decision_at": [_at(2020, 1, 31), _at(2020, 3, 31)],
            "ticker": ["AAA", "AAA"],
        }
    )
    history = pl.DataFrame(
        {
            "ticker": ["AAA", "AAA"],
            "Sector": ["Technology", "Health Care"],
            "effective_at": [_at(2020, 1, 1), _at(2020, 2, 1)],
            "observed_at": [_at(2019, 12, 15), _at(2020, 1, 20)],
            "classification_id": ["aaa-202001-tech", "aaa-202002-health"],
            "source_url": [
                "https://example.test/aaa-tech",
                "https://example.test/aaa-health",
            ],
            "confidence": ["high", "high"],
        }
    )

    resolved = resolve_point_in_time_sectors(decisions, history)

    assert resolved["Sector"].to_list() == ["Technology", "Health Care"]
    assert resolved["sector_constraint_enabled"].to_list() == [True, True]
    assert (
        resolved["sector_known_at_selected"] <= resolved["decision_at"]
    ).all()

    future_mutation = history.with_columns(
        pl.when(pl.col("effective_at") == _at(2020, 2, 1))
        .then(pl.lit("Industrials"))
        .otherwise(pl.col("Sector"))
        .alias("Sector")
    )
    mutated = resolve_point_in_time_sectors(decisions, future_mutation)
    assert mutated["Sector"][0] == resolved["Sector"][0]

    uncovered_decisions = pl.DataFrame(
        {
            "decision_at": [_at(2020, 4, 30), _at(2020, 4, 30)],
            "ticker": ["AAA", "BBB"],
        }
    )
    uncovered = resolve_point_in_time_sectors(uncovered_decisions, history)
    assert uncovered["sector_constraint_enabled"].to_list() == [False, False]
    assert uncovered["missing_point_in_time_sector_count"].to_list() == [1, 1]
    assert set(uncovered["sector_constraint_reason"]) == {
        "disabled_missing_point_in_time_sector"
    }

    with pytest.raises(ValueError, match="Point-in-time sector history is missing"):
        resolve_point_in_time_sectors(
            decisions,
            pl.DataFrame({"ticker": ["AAA"], "Sector": ["Technology"]}),
        )
