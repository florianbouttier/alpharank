from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.replay.common_strategy import build_native_boosting_holdings
from alpharank.strategy.legacy_valuation import (
    classify_legacy_valuation_eligibility,
    filter_predictions_to_legacy_valuation_universe,
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
