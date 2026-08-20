from __future__ import annotations

from datetime import date

import polars as pl

from scripts.experiments.render_start_year_performance import (
    build_calendar_returns,
    build_start_year_performance,
)


def _monthly(strategy: str, months: list[date], returns: list[float]) -> pl.DataFrame:
    decisions = [
        date(value.year - 1, 12, 1)
        if value.month == 1
        else date(value.year, value.month - 1, 1)
        for value in months
    ]
    return pl.DataFrame(
        {
            "strategy": [strategy] * len(months),
            "decision_month": decisions,
            "holding_month": months,
            "net_return": returns,
        }
    )


def test_start_year_report_discloses_partial_boosting_history() -> None:
    legacy_months = [date(2010, 2, 1), date(2011, 1, 1), date(2011, 8, 1), date(2011, 9, 1)]
    common_months = legacy_months[-2:]
    legacy = pl.concat(
        [
            _monthly("Combined_Frequency", legacy_months, [0.01] * 4),
            _monthly("SPY total return", legacy_months, [0.005] * 4),
        ]
    )
    common = pl.concat(
        [
            _monthly("Boosting Top 5", common_months, [0.02, 0.02]),
            _monthly("Boosting Top 10", common_months, [0.015, 0.015]),
            _monthly("Legacy", common_months, [0.01, 0.01]),
            _monthly("SPY total return", common_months, [0.005, 0.005]),
        ]
    )
    result = build_start_year_performance(common, legacy, first_year=2010)
    boosting_2010 = result.filter(
        (pl.col("requested_start_year") == 2010)
        & (pl.col("strategy") == "Boosting Top 5")
    ).row(0, named=True)
    legacy_2010 = result.filter(
        (pl.col("requested_start_year") == 2010)
        & (pl.col("strategy") == "Legacy")
    ).row(0, named=True)
    assert boosting_2010["coverage"] == "partial_from_2011-08"
    assert legacy_2010["coverage"] == "partial_from_2010-02"


def test_calendar_report_keeps_partial_year_status() -> None:
    months = [date(2011, 8, 1), date(2011, 9, 1)]
    common = pl.concat(
        [_monthly(strategy, months, [0.01, 0.02]) for strategy in ("Boosting Top 5", "Boosting Top 10", "Legacy", "SPY total return")]
    )
    legacy = pl.concat(
        [_monthly(strategy, months, [0.01, 0.02]) for strategy in ("Combined_Frequency", "SPY total return")]
    )
    annual = build_calendar_returns(common, legacy, first_year=2010)
    assert annual["is_full_calendar_year"].to_list() == [False] * 4
