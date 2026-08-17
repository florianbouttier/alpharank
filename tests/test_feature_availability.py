from __future__ import annotations

from datetime import datetime, timezone

import polars as pl

from alpharank.data.feature_availability import (
    materialize_feature_availability,
    select_features_at_decisions,
)


POLICY = {
    "policy_id": "sec-filing-availability-v1",
    "timezone": "America/New_York",
    "date_only_assumption": "23:59:59",
    "operational_delay_hours": 24,
    "require_filing_version_id": True,
}


def test_feature_availability_precedes_decision() -> None:
    filings = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "feature_name": ["net_income_ttm", "net_income_ttm"],
            "value": [10.0, 20.0],
            "filing_date": ["2020-01-15", "2020-03-15"],
            "accepted_at": ["2020-01-15T16:30:00-05:00", None],
            "filing_version_id": ["accession-v1", "accession-v2"],
        }
    )
    decisions = pl.DataFrame(
        {
            "decision_at": [
                datetime(2020, 1, 31, 20, tzinfo=timezone.utc),
                datetime(2020, 4, 30, 20, tzinfo=timezone.utc),
            ],
            "ticker": ["AAA.US", "AAA.US"],
            "feature_name": ["net_income_ttm", "net_income_ttm"],
        }
    )
    available = materialize_feature_availability(filings, policy=POLICY)
    reference = select_features_at_decisions(decisions, available)

    assert reference["value"].to_list() == [10.0, 20.0]
    assert (
        reference["available_at_selected"] <= reference["decision_at"]
    ).all()
    assert available["availability_basis"].to_list() == [
        "sec_acceptance_timestamp",
        "filing_date_end_of_day_fallback",
    ]

    mutated_future = available.with_columns(
        pl.when(pl.col("filing_version_id") == "accession-v2")
        .then(pl.lit(9999.0))
        .otherwise(pl.col("value"))
        .alias("value")
    )
    candidate = select_features_at_decisions(decisions, mutated_future)
    assert candidate["value"][0] == reference["value"][0]
