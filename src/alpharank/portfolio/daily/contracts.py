"""Typed outputs for daily research valuation; all returns are float fractions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

import polars as pl


@dataclass(frozen=True, slots=True)
class DailyLedgerResult:
    """Daily closes and their exhaustive coverage/monthly reconciliation evidence."""

    ledger: pl.DataFrame
    monthly_reconciliation: pl.DataFrame
    gaps: pl.DataFrame
    coverage: dict[str, int]


class DailyPoint(TypedDict):
    date: str
    wealth: float | None
    drawdown: float | None
    relative_wealth: float | None


class DailySeriesView(TypedDict):
    points: list[DailyPoint]
    metrics: dict[str, float | None]
    coverage: dict[str, int]
    annual: list[dict[str, float | int | bool | None]]
    cumulative: list[dict[str, float | int | None]]


class DailyPeriodView(TypedDict):
    requested_start: str
    requested_end: str
    start_date: str
    end_date: str
    anchor_date: str
    frequency: str
    periods_per_year: int
    series: dict[str, DailySeriesView]
