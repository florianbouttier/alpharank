from __future__ import annotations

from datetime import date

import polars as pl

from alpharank.replay.reconciliation import reconcile_economic_frames


def test_v1_v2_reconciliation_is_complete() -> None:
    v1_holdings = _holdings("A", 1.0)
    v2_holdings = _holdings("B", 1.0)
    v1_monthly = _monthly(0.10, 0.0, 1.0)
    v2_monthly = _monthly(0.08, 0.001, 1.0)

    frames = reconcile_economic_frames(
        v1_holdings=v1_holdings,
        v1_monthly=v1_monthly,
        v2_holdings=v2_holdings,
        v2_monthly=v2_monthly,
    )

    assert frames["selection"].filter(pl.col("status") != "unchanged").height == 2
    assert frames["monthly"].height == 1
    row = frames["monthly"].row(0, named=True)
    assert row["status"] == "divergent_explained"
    assert "UNI-001" in row["cause_codes"]
    assert "LEG-003" in row["cause_codes"]
    assert "SIM-003" in row["cause_codes"]
    assert frames["metrics"]["delta_cagr"].item() != 0.0


def test_v1_v2_reconciliation_names_reviewed_terminal_causes() -> None:
    v1_holdings = pl.concat([_holdings("FRC.US", 1.0), _holdings("HSP.US", 1.0)])
    v2_holdings = _holdings("HSP.US", 1.0)
    v1_monthly = _monthly(0.01, 0.0, 1.0)
    v2_monthly = _monthly(0.02, 0.0, 1.0)
    frc_key = (
        "Legacy",
        date(2024, 1, 1),
        date(2024, 2, 1),
        "FRC.US",
    )

    frames = reconcile_economic_frames(
        v1_holdings=v1_holdings,
        v1_monthly=v1_monthly,
        v2_holdings=v2_holdings,
        v2_monthly=v2_monthly,
        reviewed_terminal_months={("Legacy", date(2024, 2, 1))},
        reviewed_pre_execution_blocks={frc_key},
    )

    monthly = frames["monthly"].row(0, named=True)
    removed = frames["selection"].filter(pl.col("ticker") == "FRC.US").row(
        0, named=True
    )
    assert "RUN-012" in monthly["cause_codes"]
    assert "RUN-011" in monthly["cause_codes"]
    assert removed["cause_codes"] == "RUN-011 reviewed pre-execution suspension"


def _holdings(ticker: str, weight: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["Legacy"],
            "decision_month": [date(2024, 1, 1)],
            "holding_month": [date(2024, 2, 1)],
            "ticker": [ticker],
            "target_weight": [weight],
        }
    )


def _monthly(net: float, cost: float, turnover: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "strategy": ["Legacy"],
            "decision_month": [date(2024, 1, 1)],
            "holding_month": [date(2024, 2, 1)],
            "gross_return": [net + cost],
            "turnover": [turnover],
            "transaction_cost": [cost],
            "net_return": [net],
            "benchmark_return": [0.05],
        }
    )
