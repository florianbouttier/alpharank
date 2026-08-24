from datetime import date

import numpy as np
import polars as pl
import pytest

from alpharank.multihorizon.trading import (
    legacy_report_statistics,
    summarize_monthly_backtest,
)


def test_legacy_report_statistics_uses_cagr_sharpe_and_full_years() -> None:
    months = [date(2020, month, 1) for month in range(1, 13)]
    months += [date(2021, month, 1) for month in range(1, 13)]
    returns = np.array([0.01] * 12 + [-0.01] * 12)

    metrics = legacy_report_statistics(
        returns,
        holding_months=months,
        risk_free_rate=0.02,
    )

    expected_cagr = np.prod(1.0 + returns) ** 0.5 - 1.0
    expected_volatility = np.std(returns, ddof=1) * np.sqrt(12.0)
    assert metrics["cagr"] == pytest.approx(expected_cagr)
    assert metrics["sharpe"] == pytest.approx(
        (expected_cagr - 0.02) / expected_volatility
    )
    assert metrics["worst_full_calendar_year"] == 2021
    assert metrics["worst_full_calendar_year_return"] == pytest.approx(
        0.99**12 - 1.0
    )
    assert metrics["full_calendar_years"] == 2


def test_legacy_report_statistics_excludes_partial_boundary_years() -> None:
    months = [date(2019, 12, 1)]
    months += [date(2020, month, 1) for month in range(1, 13)]
    months += [date(2021, 1, 1)]
    returns = np.array([-0.90] + [0.01] * 12 + [-0.80])

    metrics = legacy_report_statistics(returns, holding_months=months)

    assert metrics["worst_full_calendar_year"] == 2020
    assert metrics["worst_full_calendar_year_return"] == pytest.approx(
        1.01**12 - 1.0
    )
    assert metrics["full_calendar_years"] == 1


def test_legacy_report_statistics_rejects_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="same length"):
        legacy_report_statistics(
            np.array([0.01, 0.02]),
            holding_months=[date(2020, 1, 1)],
        )


def test_monthly_summary_computes_tracking_error() -> None:
    monthly = pl.DataFrame(
        {
            "decision_month": [date(2024, 1, 1), date(2024, 2, 1)],
            "holding_month": [date(2024, 2, 1), date(2024, 3, 1)],
            "gross_return": [0.03, 0.01],
            "net_return": [0.02, 0.00],
            "benchmark_return": [0.01, 0.01],
            "turnover": [1.0, 0.5],
            "transaction_cost": [0.01, 0.01],
        }
    )

    summary = summarize_monthly_backtest(monthly)

    assert summary["annualized_tracking_error"] == pytest.approx(
        np.std([0.01, -0.01], ddof=1) * np.sqrt(12.0)
    )
