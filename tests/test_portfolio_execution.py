from __future__ import annotations

from datetime import date, datetime, timezone
import json
from pathlib import Path

import polars as pl

from alpharank.portfolio.execution import (
    LEGACY_NEXT_SESSION_OPEN,
    build_execution_sensitivity_report,
    build_monthly_execution_orders,
    write_execution_sensitivity_report,
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
            "date": [date(2025, 1, 31), date(2025, 2, 3)],
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
