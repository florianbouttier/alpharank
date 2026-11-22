"""Explain unavailable observations and propose an explicitly separate monthly source."""

from __future__ import annotations

from calendar import monthrange
from datetime import date, timedelta
from typing import Any, TypedDict

import polars as pl

from alpharank.reporting.strategy_analysis.dataset import AnalysisDataset


class MonthlyFallbackWindow(TypedDict):
    available: bool
    source: str
    start: str | None
    end: str | None
    period_changed: bool
    reason: str


class PriceGapRow(TypedDict):
    strategy: str
    ticker: str
    holding_month: str
    first_missing_session: str
    last_missing_session: str
    missing_sessions: int
    target_weight: float
    dates: list[str]
    marks: list[dict[str, str | float]]


def require_complete_months(start: date, end: date) -> None:
    """Reject a frequency change that would silently replace the selected bounds."""
    if start.day != 1 or end.day != monthrange(end.year, end.month)[1]:
        raise ValueError(
            "La référence mensuelle exige des mois complets : début le 1er, fin le dernier "
            "jour du mois. Choisissez explicitement ces bornes ; aucun arrondi n'est appliqué."
        )


def monthly_fallback_window(start: date, end: date, *, enabled: bool) -> MonthlyFallbackWindow:
    """Suggest only whole calendar months contained inside, never outside, the selection."""
    first = start.replace(day=1)
    if start.day != 1:
        first = (first + timedelta(days=32)).replace(day=1)
    last = end
    if end.day != monthrange(end.year, end.month)[1]:
        last = end.replace(day=1) - timedelta(days=1)
    available = enabled and first <= last
    return {
        "available": available,
        "source": "monthly_reference",
        "start": first.isoformat() if available else None,
        "end": last.isoformat() if available else None,
        "period_changed": available and (first != start or last != end),
        "reason": "Référence mensuelle distincte, sans cours quotidien imputé."
        if available
        else "Aucun mois complet inclus dans cette fenêtre, ou source mensuelle non disponible.",
    }


def window_price_gaps(
    dataset: AnalysisDataset, *, start: date, end: date
) -> tuple[bool, list[PriceGapRow]]:
    """Join captured ticker/session gaps to the sealed holdings, never a mutable lookup."""
    if dataset.history is not None:
        gaps = dataset.history.gaps
    else:
        captured = dataset.payload.get("analysis", {}).get("price_gaps")
        if captured is None:
            return False, []
        gaps = pl.from_dicts(
            captured,
            schema={
                "ticker": pl.String,
                "valuation_date": pl.String,
                "holding_month": pl.String,
                "mark_date": pl.String,
                "mark_value": pl.Float64,
            },
        ).with_columns(
            pl.col("valuation_date").str.to_date(),
            pl.col("holding_month").str.to_date(),
            pl.col("mark_date").str.to_date(),
        )
    gaps = gaps.filter(pl.col("valuation_date").is_between(start, end))
    if gaps.is_empty():
        return True, []
    holdings = (
        pl.from_dicts(dataset.payload["holdings"], infer_schema_length=None)
        .select("strategy", "ticker", "holding_month", "target_weight")
        .with_columns(pl.col("holding_month").str.to_date())
    )
    if holdings.select("strategy", "ticker", "holding_month").is_duplicated().any():
        raise ValueError("Coverage audit requires unique sealed strategy/ticker/month holdings.")
    groups = (
        gaps.sort("valuation_date")
        .group_by("ticker", "holding_month")
        .agg(
            pl.col("valuation_date"),
            pl.struct("valuation_date", "mark_date", "mark_value").alias("marks"),
        )
    )
    matched = groups.join(holdings, on=["ticker", "holding_month"], how="inner", validate="1:m")
    rows: list[PriceGapRow] = []
    for row in matched.sort("strategy", "holding_month", "ticker").iter_rows(named=True):
        days = row["valuation_date"]
        rows.append(
            {
                "strategy": row["strategy"],
                "ticker": row["ticker"],
                "holding_month": row["holding_month"].isoformat(),
                "first_missing_session": days[0].isoformat(),
                "last_missing_session": days[-1].isoformat(),
                "missing_sessions": len(days),
                "target_weight": float(row["target_weight"]),
                "dates": [day.isoformat() for day in days],
                "marks": [
                    {
                        "date": mark["valuation_date"].isoformat(),
                        "mark_date": mark["mark_date"].isoformat(),
                        "mark_value": mark["mark_value"],
                    }
                    for mark in row["marks"]
                    if mark["mark_date"] is not None
                ],
            }
        )
    return True, rows


def window_valuation_reconciliation(
    dataset: AnalysisDataset, *, start: date, end: date
) -> list[dict[str, Any]]:
    """Expose captured changed endpoints, never a future month's closing result."""
    if dataset.history is None:
        return []
    endpoints = dataset.history.ledger.group_by("strategy", "holding_month").agg(
        pl.col("valuation_date").max().alias("endpoint")
    )
    audit = dataset.history.monthly_reconciliation.join(
        endpoints, on=["strategy", "holding_month"], validate="1:1"
    ).filter(pl.col("endpoint").is_between(start, end) & (pl.col("status") == "estimated_endpoint"))
    return [
        {key: value.isoformat() if isinstance(value, date) else value for key, value in row.items()}
        for row in audit.sort("strategy", "holding_month").iter_rows(named=True)
    ]
