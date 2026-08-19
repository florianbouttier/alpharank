from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.portfolio.execution import (
    LEGACY_NEXT_SESSION_OPEN,
    apply_next_session_open_holding_returns,
    build_execution_sensitivity_report,
    build_monthly_execution_orders,
    write_execution_sensitivity_report,
)


def test_next_session_open_returns_are_causal_and_adjusted() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["Legacy"],
            "decision_month": [date(2024, 1, 1)],
            "holding_month": [date(2024, 2, 1)],
            "ticker": ["A.US"],
            "target_weight": [1.0],
            "realized_return": [999.0],
            "benchmark_return": [0.0],
        }
    )
    prices = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US", "A.US"],
            "date": [date(2024, 1, 31), date(2024, 2, 1), date(2024, 2, 29)],
            "open": [95.0, 100.0, 118.0],
            "close": [100.0, 110.0, 120.0],
            "adjusted_close": [50.0, 55.0, 60.0],
        }
    )

    resolved = apply_next_session_open_holding_returns(holdings, prices)

    assert resolved["realized_return"].item() == pytest.approx(0.2)
    assert resolved["execution_policy_id"].item() == "next_session_open_v1"
    assert resolved["execution_price_unadjusted"].item() == 100.0
    assert resolved["execution_price_adjusted"].item() == 50.0
    assert resolved["signal_cutoff_at"].item() < resolved["execution_at"].item()
    assert (
        resolved["execution_at"].item()
        < resolved["first_return_observation_at"].item()
        <= resolved["holding_return_end_at"].item()
    )


def test_early_final_quote_is_kept_and_flagged_for_manual_review() -> None:
    holdings = pl.DataFrame(
        {
            "strategy": ["Boosting"],
            "decision_month": [date(2024, 1, 1)],
            "holding_month": [date(2024, 2, 1)],
            "ticker": ["EXIT.US"],
            "target_weight": [1.0],
            "realized_return": [None],
            "benchmark_return": [0.0],
        }
    )
    prices = pl.DataFrame(
        {
            "ticker": ["EXIT.US", "EXIT.US", "EXIT.US", "MARKET.US"],
            "date": [
                date(2024, 1, 31),
                date(2024, 2, 1),
                date(2024, 2, 11),
                date(2024, 2, 29),
            ],
            "open": [100.0, 100.0, 90.0, 100.0],
            "close": [100.0, 99.0, 90.0, 100.0],
            "adjusted_close": [100.0, 99.0, 90.0, 100.0],
        }
    )

    resolved = apply_next_session_open_holding_returns(holdings, prices)

    assert resolved["realized_return"].item() == pytest.approx(-0.10)
    assert resolved["holding_return_end_at"].item().date() == date(2024, 2, 11)
    assert resolved["scheduled_holding_end_at"].item().date() == date(2024, 2, 29)
    assert resolved["return_resolution"].item() == "provisional_last_observation"
    assert resolved["manual_review_status"].item() == (
        "pending_manual_terminal_event_review"
    )


def test_order_price_occurs_after_signal_cutoff(tmp_path: Path) -> None:
    holdings = pl.DataFrame(
        {
            "portfolio_model": ["Combined_Frequency"],
            "year_month": [date(2025, 2, 1)],
            "ticker": ["AAA.US"],
        }
    )
    prices = pl.DataFrame(
        {
            "ticker": ["AAA.US", "AAA.US"],
            "date": ["2025-01-31", "2025-02-03"],
            "open": [99.0, 101.0],
            "close": [100.0, 103.0],
            "vwap": [99.5, 102.0],
        }
    )

    orders = build_monthly_execution_orders(holdings, prices)
    assert orders["signal_cutoff_at"][0] == datetime(
        2025, 1, 31, 21, 0, tzinfo=timezone.utc
    )
    report = build_execution_sensitivity_report(orders, prices)
    canonical = report.filter(pl.col("is_canonical")).row(0, named=True)
    close_reference = report.filter(
        pl.col("scenario") == "signal_close_reference"
    ).row(0, named=True)
    vwap = report.filter(pl.col("scenario") == "observed_session_vwap").row(
        0, named=True
    )

    assert canonical["execution_policy_id"] == "next_session_open_v1"
    assert canonical["price"] == 101.0
    assert canonical["execution_at"] > canonical["signal_cutoff_at"]
    assert canonical["execution_after_signal_cutoff"] is True
    assert close_reference["price"] == 100.0
    assert close_reference["execution_after_signal_cutoff"] is False
    assert close_reference["status"] == "reference_only_not_after_signal"
    assert vwap["price"] == 102.0
    assert vwap["execution_after_signal_cutoff"] is True

    manifest = write_execution_sensitivity_report(report, tmp_path)
    assert manifest["execution_policy"] == LEGACY_NEXT_SESSION_OPEN.to_manifest()
    assert manifest["scenario_count"] == 3
    assert (tmp_path / "legacy_execution_sensitivity.parquet").is_file()
    persisted = json.loads(
        (tmp_path / "legacy_execution_policy.json").read_text(encoding="utf-8")
    )
    assert persisted["row_count"] == 3


def test_historical_execution_report_logs_terminal_unavailability(tmp_path: Path) -> None:
    holdings = pl.DataFrame(
        {
            "portfolio_model": ["Combined_Equal"],
            "year_month": [date(2019, 1, 1)],
            "ticker": ["ESRX.US"],
        }
    )
    prices = pl.DataFrame(
        {
            "ticker": ["ESRX.US"],
            "date": ["2018-12-21"],
            "open": [95.0],
            "close": [96.0],
        }
    )
    orders = build_monthly_execution_orders(holdings, prices)

    with pytest.raises(RuntimeError, match="Canonical execution is unavailable"):
        build_execution_sensitivity_report(orders, prices)

    report = build_execution_sensitivity_report(
        orders,
        prices,
        require_canonical_available=False,
    )
    manifest = write_execution_sensitivity_report(
        report,
        tmp_path,
        require_canonical_available=False,
    )

    canonical = report.filter(pl.col("scenario") == "next_session_open")
    assert canonical["status"].item() == "unavailable_no_future_open"
    assert manifest["canonical_unavailable_count"] == 1
    assert manifest["validation_status"] == (
        "historical_compatibility_with_logged_unavailable"
    )
