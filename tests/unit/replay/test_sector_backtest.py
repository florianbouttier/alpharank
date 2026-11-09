from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.replay.sector_backtest import (
    REFERENCE_STRATEGIES,
    SECTOR_NATIVE,
    SECTOR_TREND,
    build_sector_backtest,
)


def test_build_sector_backtest_excludes_only_trailing_pending_month() -> None:
    native = _sector_holdings(SECTOR_NATIVE)
    trend = _sector_holdings(SECTOR_TREND)

    result = build_sector_backtest(native, trend, _reference_monthly())

    sector_monthly = result.monthly_returns.filter(pl.col("strategy") == SECTOR_NATIVE)
    assert sector_monthly["decision_month"].to_list() == [date(2024, 1, 1), date(2024, 2, 1)]
    assert result.holdings.height == 20
    assert sector_monthly["maximum_sector_weight"].to_list() == pytest.approx([0.2, 0.2])


def test_build_sector_backtest_fails_on_partial_mature_return() -> None:
    native = _sector_holdings(SECTOR_NATIVE).with_columns(
        pl.when((pl.col("decision_month") == date(2024, 2, 1)) & (pl.col("ticker") == "T0"))
        .then(None)
        .otherwise(pl.col("realized_return"))
        .alias("realized_return")
    )

    with pytest.raises(ValueError, match="Partially missing portfolio month"):
        build_sector_backtest(native, _sector_holdings(SECTOR_TREND), _reference_monthly())


def test_build_sector_backtest_fails_on_non_trailing_pending_month() -> None:
    native = _sector_holdings(SECTOR_NATIVE).with_columns(
        pl.when(pl.col("decision_month") == date(2024, 1, 1))
        .then(None)
        .otherwise(pl.col("realized_return"))
        .alias("realized_return"),
        pl.when(pl.col("decision_month") == date(2024, 1, 1))
        .then(None)
        .otherwise(pl.col("benchmark_return"))
        .alias("benchmark_return"),
        pl.when(pl.col("decision_month") == date(2024, 1, 1))
        .then(pl.lit("horizon_pending"))
        .otherwise(pl.col("target_status_1m"))
        .alias("target_status_1m"),
    )

    with pytest.raises(ValueError, match="Non-trailing pending month"):
        build_sector_backtest(native, _sector_holdings(SECTOR_TREND), _reference_monthly())


def _sector_holdings(strategy: str) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for decision_month, holding_month, realized in (
        (date(2024, 1, 1), date(2024, 2, 1), 0.02),
        (date(2024, 2, 1), date(2024, 3, 1), -0.01),
        (date(2024, 3, 1), date(2024, 4, 1), None),
    ):
        for rank in range(5):
            rows.append(
                {
                    "strategy": strategy,
                    "decision_month": decision_month,
                    "holding_month": holding_month,
                    "ticker": f"T{rank}",
                    "target_weight": 0.2,
                    "realized_return": realized,
                    "benchmark_return": (
                        None
                        if realized is None
                        else (0.01 if decision_month == date(2024, 1, 1) else -0.005)
                    ),
                    "target_status_1m": "horizon_pending" if realized is None else "evaluable",
                    "Sector": f"S{rank}",
                }
            )
    return pl.DataFrame(rows)


def _reference_monthly() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for strategy in REFERENCE_STRATEGIES:
        for decision_month, holding_month, net_return, benchmark_return in (
            (date(2024, 1, 1), date(2024, 2, 1), 0.01, 0.01),
            (date(2024, 2, 1), date(2024, 3, 1), 0.0, -0.005),
        ):
            rows.append(
                {
                    "strategy": strategy,
                    "decision_month": decision_month,
                    "holding_month": holding_month,
                    "gross_return": net_return,
                    "turnover": 0.0,
                    "transaction_cost": 0.0,
                    "net_return": net_return,
                    "benchmark_return": benchmark_return,
                    "active_return": net_return - benchmark_return,
                    "relative_return": (1.0 + net_return) / (1.0 + benchmark_return) - 1.0,
                    "n_positions": 5,
                    "maximum_position_weight": 0.2,
                    "maximum_sector_weight": 0.0,
                    "sector_count": 0,
                }
            )
    return pl.DataFrame(rows)
