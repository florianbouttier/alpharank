from __future__ import annotations

from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings
from alpharank.portfolio.adapters.legacy import legacy_detailed_to_holdings
from alpharank.portfolio.allocation import portfolio_turnover
from alpharank.portfolio.comparison import align_return_series
from alpharank.portfolio.contracts import validate_holdings
from alpharank.portfolio.performance import (
    advanced_performance_statistics,
    annual_returns,
    legacy_report_statistics,
)
from alpharank.portfolio.lineage import (
    compare_input_hashes,
    compare_ticker_exclusions,
    ticker_exclusions_from_manifest,
)
from alpharank.data.price_eligibility import (
    price_eligibility_policy_from_manifest,
)
from alpharank.portfolio.simulation import simulate_weighted_portfolio


def test_shared_simulator_uses_decision_t_and_holding_t_plus_one() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["alpha", "alpha"],
            "decision_month": [date(2020, 1, 1)] * 2,
            "holding_month": [date(2020, 2, 1)] * 2,
            "ticker": ["A", "B"],
            "target_weight": [0.6, 0.4],
            "realized_return": [0.10, -0.05],
            "benchmark_return": [0.02, 0.02],
            "sector": ["Tech", "Health"],
        }
    )
    monthly = simulate_weighted_portfolio(holdings, transaction_cost_bps=10.0)
    assert monthly["gross_return"][0] == pytest.approx(0.04)
    assert monthly["turnover"][0] == pytest.approx(1.0)
    assert monthly["net_return"][0] == pytest.approx(0.039)
    assert monthly["active_return"][0] == pytest.approx(0.019)
    assert monthly["relative_return"][0] == pytest.approx(1.039 / 1.02 - 1.0)


def test_contract_rejects_same_month_lookahead() -> None:
    invalid = pl.DataFrame(
        {
            "strategy": ["alpha"],
            "decision_month": [date(2020, 1, 1)],
            "holding_month": [date(2020, 1, 1)],
            "ticker": ["A"],
            "target_weight": [1.0],
            "realized_return": [0.10],
            "benchmark_return": [0.02],
        }
    )
    with pytest.raises(ValueError, match="decision_month"):
        validate_holdings(invalid)


def test_missing_legacy_return_is_renormalized_for_performance() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["legacy", "legacy"],
            "decision_month": [date(2020, 1, 1)] * 2,
            "holding_month": [date(2020, 2, 1)] * 2,
            "ticker": ["A", "B"],
            "target_weight": [0.75, 0.25],
            "realized_return": [0.10, None],
            "benchmark_return": [0.02, 0.02],
        }
    )
    monthly = simulate_weighted_portfolio(holdings)
    assert monthly["gross_return"][0] == pytest.approx(0.10)
    assert monthly["n_positions"][0] == 2


def test_legacy_adapter_and_boosting_adapter_share_the_same_contract() -> None:
    benchmark = pl.DataFrame(
        {"year_month": [date(2020, 2, 1)], "monthly_return": [0.02]}
    )
    legacy = legacy_detailed_to_holdings(
        pl.DataFrame(
            {
                "year_month": [date(2020, 2, 1)] * 2,
                "ticker": ["A", "B"],
                "dr": [0.10, -0.05],
                "weight_normalized": [0.5, 0.5],
            }
        ),
        strategy="same",
        benchmark_monthly=benchmark,
    )
    boosting = boosting_predictions_to_holdings(
        pl.DataFrame(
            {
                "decision_month": [date(2020, 1, 1)] * 2,
                "ticker": ["A", "B"],
                "score": [0.9, 0.8],
                "future_return_1m": [0.10, -0.05],
                "benchmark_future_return_1m": [0.02, 0.02],
            }
        ),
        strategy="same",
        top_n=2,
    )
    legacy_monthly = simulate_weighted_portfolio(legacy)
    boosting_monthly = simulate_weighted_portfolio(
        boosting.select(legacy.columns)
    )
    assert legacy_monthly["net_return"][0] == pytest.approx(
        boosting_monthly["net_return"][0]
    )


def test_boosting_selection_ignores_future_return_availability() -> None:
    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
            "future_return_1m": [0.25, 0.10, -0.05],
            "benchmark_future_return_1m": [0.02, 0.02, 0.02],
        }
    )
    missing_top_return = predictions.with_columns(
        pl.when(pl.col("ticker") == "A")
        .then(None)
        .otherwise(pl.col(column))
        .alias(column)
        for column in ("future_return_1m", "benchmark_future_return_1m")
    )

    complete = boosting_predictions_to_holdings(
        predictions,
        strategy="boosting",
        top_n=2,
    )
    with_missing_return = boosting_predictions_to_holdings(
        missing_top_return,
        strategy="boosting",
        top_n=2,
    )

    assert complete["ticker"].to_list() == ["A", "B"]
    assert with_missing_return["ticker"].to_list() == ["A", "B"]
    assert with_missing_return["realized_return"].to_list() == [None, 0.10]
    assert with_missing_return["benchmark_return"].to_list() == [None, 0.02]


def test_turnover_and_period_alignment_are_explicit() -> None:
    assert portfolio_turnover({"A": 0.5, "B": 0.5}, {"A": 0.5, "C": 0.5}) == pytest.approx(0.5)
    first = pl.DataFrame(
        {
            "holding_month": [date(2020, 1, 1), date(2020, 2, 1)],
            "net_return": [0.01, 0.02],
        }
    )
    second = pl.DataFrame(
        {
            "holding_month": [date(2020, 2, 1), date(2020, 3, 1)],
            "net_return": [0.03, 0.04],
        }
    )
    aligned = align_return_series({"alpha": first, "legacy": second})
    assert aligned.to_dicts() == [
        {"holding_month": date(2020, 2, 1), "alpha": 0.02, "legacy": 0.03}
    ]


def test_common_performance_excludes_partial_years_from_worst_year() -> None:
    months = [date(2019, 12, 1)] + [date(2020, month, 1) for month in range(1, 13)]
    returns = np.asarray([-0.9] + [0.01] * 12)
    metrics = legacy_report_statistics(returns, holding_months=months)
    yearly = annual_returns(returns, holding_months=months)
    assert metrics["worst_full_calendar_year"] == 2020
    assert yearly.filter(pl.col("year") == 2019)["is_full_calendar_year"][0] is False


def test_advanced_statistics_use_the_same_canonical_base() -> None:
    returns = np.asarray([0.10, -0.05, 0.02, 0.03])
    benchmark = np.asarray([0.04, -0.02, 0.01, 0.01])
    advanced = advanced_performance_statistics(
        returns,
        benchmark_returns=benchmark,
    )
    base = legacy_report_statistics(
        returns,
        holding_months=[date(2020, month, 1) for month in range(1, 5)],
    )
    assert advanced["cagr"] == pytest.approx(base["cagr"])
    assert advanced["sharpe"] == pytest.approx(base["sharpe"])
    assert advanced["sortino"] > advanced["sharpe"]
    assert advanced["benchmark_hit_rate"] == pytest.approx(0.75)


def test_comparison_lineage_rejects_distinct_data_snapshots() -> None:
    report = compare_input_hashes(
        {"final_price": "price-a", "sp500_price": "spy"},
        {"final_price": "price-b", "sp500_price": "spy"},
    )
    assert report["passed"] is False
    assert report["differing_keys"] == ["final_price"]


def test_comparison_lineage_requires_every_declared_input() -> None:
    report = compare_input_hashes(
        {"final_price": "price", "sp500_price": "spy"},
        {"final_price": "price"},
    )
    assert report["passed"] is False
    assert report["missing_right"] == ["sp500_price"]


def test_comparison_lineage_rejects_distinct_ticker_quarantines() -> None:
    report = compare_ticker_exclusions(
        ("SII.US", "CBE.US", "TIE.US"),
        ("SII.US", "CBE.US", "TIE.US", "SW.US"),
    )
    assert report["passed"] is False
    assert report["missing_left"] == ["SW.US"]
    assert report["missing_right"] == []


def test_ticker_exclusions_are_read_from_both_manifest_shapes() -> None:
    legacy = {"run_config": {"excluded_tickers": ["sii.us", "CBE.US"]}}
    boosting = {"config": {"excluded_tickers": ["CBE.US", "SII.US"]}}
    assert ticker_exclusions_from_manifest(legacy) == ("CBE.US", "SII.US")
    assert ticker_exclusions_from_manifest(boosting) == ("CBE.US", "SII.US")


def test_price_eligibility_policy_is_read_from_both_manifest_shapes() -> None:
    policy = {
        "price_eligibility_policy_id": "monthly_price_eligibility_v1",
        "minimum_monthly_price_observations": 10,
        "minimum_monthly_median_dollar_volume": 1_000_000.0,
        "maximum_monthly_ohlc_violation_rate": 0.05,
    }
    assert price_eligibility_policy_from_manifest({"run_config": policy}) == (
        price_eligibility_policy_from_manifest({"config": policy})
    )
