"""Explain provider price revisions without promoting them into validated history."""

from __future__ import annotations

from datetime import date, timedelta

import polars as pl

from alpharank.data.prices.contracts import PriceGatePolicy
from alpharank.data.prices.gates import PriceGateResult

PRICE_REVISION_DIAGNOSTIC_CONTRACT = "provider_price_revision_diagnostic_v1"


def build_price_revision_diagnostic(
    *,
    provider_gate: PriceGateResult,
    previous_prices: pl.DataFrame,
    provider_prices: pl.DataFrame,
    expected_through: str,
    policy: PriceGatePolicy,
) -> dict[str, object]:
    """Classify observable revision patterns and state the defensible reason."""

    cutoff = date.fromisoformat(expected_through) - timedelta(
        days=policy.recent_mutable_calendar_days
    )
    material = provider_gate.daily_return_revisions.filter(
        (pl.col("date") < cutoff)
        & (pl.col("absolute_daily_return_difference") > policy.historical_return_revision_threshold)
    )
    ratios = _price_level_ratios(previous_prices, provider_prices)
    ticker_reports = [
        _diagnose_ticker(ticker, revisions, ratios.filter(pl.col("ticker") == ticker))
        for key, revisions in material.partition_by("ticker", as_dict=True).items()
        for ticker in [key[0] if isinstance(key, tuple) else key]
    ]
    return {
        "contract": PRICE_REVISION_DIAGNOSTIC_CONTRACT,
        "provider": "Yahoo Finance via yfinance",
        "material_historical_revision_count": material.height,
        "material_historical_revision_ticker_count": len(ticker_reports),
        "observable_reason": (
            "The provider changed already-published adjusted-close daily returns "
            "without an approved sourced correction overlay."
            if material.height
            else "No material historical daily-return revision was observed."
        ),
        "root_cause_status": "provider_adjustment_method_not_externally_proven"
        if material.height
        else "not_applicable",
        "publication_action": (
            "retain_previous_validated_returns_and_append_only_new_provider_returns"
        ),
        "ticker_diagnostics": sorted(ticker_reports, key=lambda item: str(item["ticker"])),
    }


def _price_level_ratios(previous: pl.DataFrame, provider: pl.DataFrame) -> pl.DataFrame:
    old = _normalized_prices(previous).rename({"adjusted_close": "previous_adjusted_close"})
    current = _normalized_prices(provider).rename({"adjusted_close": "provider_adjusted_close"})
    return old.join(current, on=["ticker", "date"], how="inner").with_columns(
        (pl.col("provider_adjusted_close") / pl.col("previous_adjusted_close")).alias(
            "price_level_ratio"
        )
    )


def _diagnose_ticker(
    ticker: str,
    revisions: pl.DataFrame,
    ratios: pl.DataFrame,
) -> dict[str, object]:
    ordered = revisions.sort("date")
    differences = ordered.get_column("daily_return_difference")
    large_moves = ordered.filter(
        (pl.col("previous_daily_return").abs().fill_null(0.0) > 0.2)
        | (pl.col("candidate_daily_return").abs().fill_null(0.0) > 0.2)
    ).height
    alternating = (
        differences.len() > 1
        and differences.sign().diff().abs().fill_null(0).max() == 2
        and large_moves > 1
    )
    ratio_min = ratios.select(pl.col("price_level_ratio").min()).item() if ratios.height else None
    ratio_max = ratios.select(pl.col("price_level_ratio").max()).item() if ratios.height else None
    ratio_spread = (float(ratio_max) - float(ratio_min)) if ratio_min and ratio_max else None
    if alternating:
        pattern = "alternating_adjusted_close_discontinuities"
    elif ratio_spread is not None and ratio_spread <= 1e-10:
        pattern = "uniform_price_level_rescale"
    else:
        pattern = "provider_recalculated_adjusted_history"
    return {
        "ticker": ticker,
        "pattern": pattern,
        "revision_count": ordered.height,
        "first_revision_date": str(ordered.select(pl.col("date").min()).item()),
        "last_revision_date": str(ordered.select(pl.col("date").max()).item()),
        "maximum_absolute_daily_return_difference": ordered.select(
            pl.col("absolute_daily_return_difference").max()
        ).item(),
        "large_move_revision_count": large_moves,
        "price_level_ratio_min": ratio_min,
        "price_level_ratio_max": ratio_max,
        "price_level_ratio_spread": ratio_spread,
        "reason_for_non_replacement": (
            "Historical daily returns differ from the validated vintage and no "
            "approved correction evidence authorizes replacement."
        ),
    }


def _normalized_prices(frame: pl.DataFrame) -> pl.DataFrame:
    date_expression = (
        pl.col("date").str.to_date(strict=False)
        if frame.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    return frame.select(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        date_expression.alias("date"),
        pl.col("adjusted_close").cast(pl.Float64, strict=False),
    )
