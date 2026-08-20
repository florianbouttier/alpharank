from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from alpharank.data.fundamental_coverage import apply_missing_fundamentals_policy


POLICY = {
    "policy_id": "sec-only-exclude-ex-ante-v1",
    "required_source": "SEC",
    "missing_action": "exclude_ex_ante",
    "fallback_sources": [],
}


def _at(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 20, tzinfo=timezone.utc)


def test_missing_fundamentals_policy_is_ex_ante() -> None:
    candidates = pl.DataFrame(
        {
            "decision_at": [
                _at(2020, 1, 31),
                _at(2020, 1, 31),
                _at(2021, 1, 31),
                _at(2021, 1, 31),
            ],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
            "future_return_1m": [0.10, None, -0.20, 0.30],
            "survived_later": [True, False, True, True],
        }
    )
    sec_availability = pl.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "available_at": [_at(2019, 12, 15), _at(2020, 6, 1)],
            "fundamental_set_id": ["aaa-sec-2019q3", "bbb-sec-2020q1"],
            "source": ["SEC", "SEC"],
        }
    )

    reference = apply_missing_fundamentals_policy(
        candidates, sec_availability, policy=POLICY
    )
    mutated_future = candidates.with_columns(
        (pl.col("future_return_1m").fill_null(0.0) * -999).alias(
            "future_return_1m"
        ),
        (~pl.col("survived_later")).alias("survived_later"),
    )
    candidate = apply_missing_fundamentals_policy(
        mutated_future, sec_availability, policy=POLICY
    )

    selected = ["decision_at", "ticker", "fundamentals_eligible"]
    assert candidate.annotated.select(selected).equals(
        reference.annotated.select(selected)
    )
    assert reference.eligible.select("decision_at", "ticker").rows() == [
        (_at(2020, 1, 31), "AAA"),
        (_at(2021, 1, 31), "AAA"),
        (_at(2021, 1, 31), "BBB"),
    ]
    assert reference.coverage_by_year.select(
        "decision_year",
        "candidate_count",
        "sec_available_count",
        "missing_sec_count",
        "missing_sec_ticker_count",
        "sec_coverage_rate",
    ).rows() == [
        (2020, 2, 1, 1, 1, 0.5),
        (2021, 2, 2, 0, 0, 1.0),
    ]
