from __future__ import annotations

from datetime import date, datetime, timezone

import polars as pl

from alpharank.backtest.features import (
    compute_monthly_stock_prices,
    compute_technical_features,
)
from alpharank.backtest.fundamentals import _asof_join_monthly
from alpharank.data.contracts.point_in_time import join_point_in_time_attributes
from alpharank.portfolio.adapters.boosting import boosting_predictions_to_holdings


def _utc(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, 21, 0, tzinfo=timezone.utc)


def test_future_mutations_do_not_change_past_decisions() -> None:
    cutoff = date(2020, 2, 1)

    predictions = pl.DataFrame(
        {
            "decision_month": [date(2020, 1, 1)] * 3,
            "ticker": ["A", "B", "C"],
            "score": [0.9, 0.8, 0.7],
            "future_return_1m": [0.1, 0.2, 0.3],
            "benchmark_future_return_1m": [0.01, 0.01, 0.01],
        }
    )
    target_mutation = predictions.with_columns(
        pl.when(pl.col("ticker") == "A")
        .then(-0.99)
        .otherwise(pl.col("future_return_1m"))
        .alias("future_return_1m")
    )
    target_reference = boosting_predictions_to_holdings(
        predictions, strategy="qa", top_n=2
    ).select("decision_month", "ticker", "target_weight", "selection_rank")
    target_candidate = boosting_predictions_to_holdings(
        target_mutation, strategy="qa", top_n=2
    ).select("decision_month", "ticker", "target_weight", "selection_rank")
    assert target_candidate.equals(target_reference)

    price_dates = [date(2017 + index // 12, index % 12 + 1, 20) for index in range(40)]
    prices = pl.DataFrame(
        {
            "ticker": ["A"] * len(price_dates),
            "date": price_dates,
            "adjusted_close": [100.0 + index for index in range(len(price_dates))],
        }
    )
    future_price_mutation = prices.with_columns(
        pl.when(pl.col("date") == max(price_dates))
        .then(pl.col("adjusted_close") * 100.0)
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )
    technical_reference = compute_technical_features(
        compute_monthly_stock_prices(prices)
    ).filter(pl.col("year_month") <= cutoff)
    technical_candidate = compute_technical_features(
        compute_monthly_stock_prices(future_price_mutation)
    ).filter(pl.col("year_month") <= cutoff)
    assert technical_candidate.equals(technical_reference, null_equal=True)

    decisions = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "decision_at": [_utc(2020, 1, 31), _utc(2020, 2, 28)],
        }
    )
    membership = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "effective_at": [_utc(2019, 1, 1), _utc(2020, 3, 1)],
            "is_member": [True, False],
        }
    )
    membership_mutation = membership.with_columns(
        pl.when(pl.col("effective_at") == _utc(2020, 3, 1))
        .then(True)
        .otherwise(pl.col("is_member"))
        .alias("is_member")
    )
    membership_reference = join_point_in_time_attributes(
        decisions,
        membership,
        entity_column="ticker",
        decision_time_column="decision_at",
        effective_time_column="effective_at",
        attribute_columns=("is_member",),
    )
    membership_candidate = join_point_in_time_attributes(
        decisions,
        membership_mutation,
        entity_column="ticker",
        decision_time_column="decision_at",
        effective_time_column="effective_at",
        attribute_columns=("is_member",),
    )
    assert membership_candidate.equals(membership_reference)

    sectors = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "effective_at": [_utc(2019, 1, 1), _utc(2020, 3, 1)],
            "sector": ["Technology", "Industrials"],
        }
    )
    sector_mutation = sectors.with_columns(
        pl.when(pl.col("effective_at") == _utc(2020, 3, 1))
        .then(pl.lit("Financials"))
        .otherwise(pl.col("sector"))
        .alias("sector")
    )
    sector_reference = join_point_in_time_attributes(
        decisions,
        sectors,
        entity_column="ticker",
        decision_time_column="decision_at",
        effective_time_column="effective_at",
        attribute_columns=("sector",),
    )
    sector_candidate = join_point_in_time_attributes(
        decisions,
        sector_mutation,
        entity_column="ticker",
        decision_time_column="decision_at",
        effective_time_column="effective_at",
        attribute_columns=("sector",),
    )
    assert sector_candidate.equals(sector_reference)

    monthly = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "date": [date(2020, 1, 31), date(2020, 2, 28)],
        }
    )
    filings = pl.DataFrame(
        {
            "ticker": ["A", "A"],
            "report_date": [date(2019, 12, 15), date(2020, 3, 15)],
            "revenue_ttm": [100.0, 200.0],
        }
    )
    future_filing_mutation = filings.with_columns(
        pl.when(pl.col("report_date") == date(2020, 3, 15))
        .then(999_999.0)
        .otherwise(pl.col("revenue_ttm"))
        .alias("revenue_ttm")
    )
    filing_reference = _asof_join_monthly(monthly, filings)
    filing_candidate = _asof_join_monthly(monthly, future_filing_mutation)
    assert filing_candidate.equals(filing_reference, null_equal=True)
