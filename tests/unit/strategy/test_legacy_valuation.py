from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.replay.common_strategy import build_native_boosting_holdings
from alpharank.strategy.legacy_valuation import (
    LEGACY_PE_MARKET_CAP_POLICY_ID,
    NO_SEC_FUNDAMENTALS_POLICY_ID,
    build_legacy_selection_universe,
    classify_legacy_valuation_eligibility,
    filter_predictions_to_legacy_valuation_universe,
)


def test_no_sec_policy_uses_only_price_and_membership_keys() -> None:
    monthly_return = pl.DataFrame(
        {
            "ticker": ["A.US", "B.US", "C.US"],
            "year_month": [date(2020, 1, 1)] * 3,
            "monthly_return": [0.01, 0.02, 0.03],
        }
    )
    membership = pl.DataFrame(
        {
            "ticker": ["A.US", "C.US", "D.US"],
            "year_month": [date(2020, 1, 1)] * 3,
        }
    )

    selected = build_legacy_selection_universe(
        policy_id=NO_SEC_FUNDAMENTALS_POLICY_ID,
        monthly_return=monthly_return,
        historical_membership=membership,
    )

    assert selected.to_dicts() == [
        {"ticker": "A.US", "year_month": date(2020, 1, 1)},
        {"ticker": "C.US", "year_month": date(2020, 1, 1)},
    ]


def test_pe_policy_requires_all_fundamental_frames() -> None:
    monthly_return = pl.DataFrame({"ticker": ["A.US"], "year_month": [date(2020, 1, 1)]})
    membership = monthly_return.clone()

    with pytest.raises(ValueError, match="requires fundamental frames"):
        build_legacy_selection_universe(
            policy_id=LEGACY_PE_MARKET_CAP_POLICY_ID,
            monthly_return=monthly_return,
            historical_membership=membership,
        )


def test_selection_universe_rejects_unknown_policy() -> None:
    empty = pl.DataFrame(schema={"ticker": pl.String, "year_month": pl.Date})

    with pytest.raises(ValueError, match="Unsupported Legacy fundamental"):
        build_legacy_selection_universe(
            policy_id="implicit_fallback",
            monthly_return=empty,
            historical_membership=empty,
        )


def test_registry_assigns_one_causal_reason_per_candidate() -> None:
    month = date(2020, 1, 1)
    candidates = pl.DataFrame(
        {
            "ticker": ["NONE.US", "MISS.US", "LOSS.US", "EXP.US", "OK.US"],
            "decision_month": [month] * 5,
        }
    )
    valuation = pl.DataFrame(
        {
            "ticker": ["LOSS.US", "EXP.US", "OK.US"],
            "decision_month": [month] * 3,
            "pe": [-2.0, 100.0, 12.0],
            "market_cap": [1.0, 1.0, 1.0],
        }
    )

    registry = classify_legacy_valuation_eligibility(
        candidates=candidates,
        valuation=valuation,
        sec_tickers={"MISS.US", "LOSS.US", "EXP.US", "OK.US"},
    ).sort("ticker")

    assert dict(zip(registry["ticker"], registry["eligibility_reason"], strict=True)) == {
        "EXP.US": "pe_at_least_100",
        "LOSS.US": "pe_nonpositive",
        "MISS.US": "missing_point_in_time_pe",
        "NONE.US": "no_sec_source_rows",
        "OK.US": "eligible",
    }
    assert registry.filter(pl.col("legacy_valuation_eligible"))["ticker"].to_list() == ["OK.US"]


def test_matched_universe_filters_before_ranking() -> None:
    month = date(2020, 1, 1)
    predictions = pl.DataFrame(
        {
            "ticker": ["A.US", "B.US"],
            "decision_month": [month, month],
            "score": [0.9, 0.8],
            "future_return_1m": [0.01, 0.02],
            "benchmark_future_return_1m": [0.005, 0.005],
        }
    )
    registry = pl.DataFrame(
        {
            "ticker": ["A.US", "B.US"],
            "decision_month": [month, month],
            "legacy_valuation_eligible": [False, True],
        }
    )

    filtered = filter_predictions_to_legacy_valuation_universe(predictions, registry)
    selected = build_native_boosting_holdings(filtered, top_n_values=(1,))

    assert filtered["ticker"].to_list() == ["B.US"]
    assert selected["ticker"].to_list() == ["B.US"]


def test_registry_rejects_ambiguous_ticker_month_valuation() -> None:
    month = date(2020, 1, 1)
    candidates = pl.DataFrame({"ticker": ["A.US"], "decision_month": [month]})
    valuation = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US"],
            "decision_month": [month, month],
            "pe": [10.0, 11.0],
            "market_cap": [1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="duplicate ticker-month"):
        classify_legacy_valuation_eligibility(
            candidates=candidates,
            valuation=valuation,
            sec_tickers={"A.US"},
        )
