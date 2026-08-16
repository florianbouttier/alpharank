from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.prices.composition import (
    compose_hybrid_price_history,
    roll_forward_validated_price_history,
)
from alpharank.data.prices.contracts import ADJUSTMENT_POLICY_VERSION, PRICE_LINEAGE_COLUMNS


def _lineage(
    ticker: str,
    dates: list[str],
    adjusted: list[float],
    *,
    source: str,
    vintage: str,
    closes: list[float] | None = None,
) -> pl.DataFrame:
    closes = closes or adjusted
    size = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": [100.0] * size,
            "adjusted_close": adjusted,
            "ticker": [ticker] * size,
            "source": [source] * size,
            "dataset": [f"prices_{source}"] * size,
            "ingestion_run_id": [vintage] * size,
            "ingested_at": ["2026-08-15T00:00:00Z"] * size,
            "source_vintage_id": [vintage] * size,
            "return_source_vintage_id": [vintage] * size,
            "adjustment_policy_version": [ADJUSTMENT_POLICY_VERSION] * size,
            "adjustment_bridge_factor": [1.0] * size,
            "eodhd_seed_sha256": ["seed"] * size,
            "correction_overlay_id": [None] * size,
        }
    ).select(PRICE_LINEAGE_COLUMNS)


def test_active_ticker_uses_only_one_fresh_yahoo_vintage() -> None:
    seed = _lineage("A.US", ["2020-01-02"], [10.0], source="eodhd_frozen_history", vintage="seed")
    yahoo = _lineage(
        "A.US",
        ["2020-01-02", "2020-01-03"],
        [9.9, 10.9],
        source="yfinance",
        vintage="run_2",
    )

    result = compose_hybrid_price_history(
        eodhd_seed=seed,
        active_yahoo_vintage=yahoo,
        retained_open_history=None,
        active_tickers=["A"],
    )

    assert result.prices.height == 2
    assert result.lineage["source"].unique().to_list() == ["yfinance"]
    assert result.lineage["source_vintage_id"].unique().to_list() == ["run_2"]
    assert result.composition_report["missing_active_yahoo_tickers"] == []


def test_inactive_tail_extends_seed_with_same_vintage_daily_return() -> None:
    seed = _lineage(
        "OLD.US",
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05"],
        [90.0, 91.0, 92.0, 93.0, 94.0],
        source="eodhd_frozen_history",
        vintage="seed",
        closes=[100.0, 101.0, 102.0, 103.0, 104.0],
    )
    retained = _lineage(
        "OLD.US",
        ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05", "2020-01-06"],
        [81.0, 81.9, 82.8, 83.7, 84.6, 85.5],
        source="yfinance",
        vintage="full_old",
        closes=[100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
    )

    result = compose_hybrid_price_history(
        eodhd_seed=seed,
        active_yahoo_vintage=pl.DataFrame(),
        retained_open_history=retained,
        active_tickers=[],
    )

    assert result.prices.height == 6
    assert result.prices.sort("date")["adjusted_close"].to_list() == pytest.approx(
        [90.0, 91.0, 92.0, 93.0, 94.0, 95.0]
    )
    assert result.composition_report["bridged_inactive_tickers"] == 1
    assert result.composition_report["unresolved_inactive_tails"] == []


def test_inactive_tail_after_long_symbol_gap_is_not_attached() -> None:
    seed = _lineage(
        "OLD.US",
        ["2011-04-07", "2011-04-08"],
        [49.0, 50.0],
        source="eodhd_frozen_history",
        vintage="seed",
    )
    retained = _lineage(
        "OLD.US",
        ["2026-04-23", "2026-04-24"],
        [19.0, 20.0],
        source="yfinance",
        vintage="reused_symbol",
    )

    result = compose_hybrid_price_history(
        eodhd_seed=seed,
        active_yahoo_vintage=pl.DataFrame(),
        retained_open_history=retained,
        active_tickers=[],
    )

    assert result.prices.height == 2
    assert result.composition_report["bridged_inactive_tickers"] == 0
    assert result.composition_report["unresolved_inactive_tails"][0]["reason"] == (
        "tail_starts_after_symbol_gap"
    )


def test_roll_forward_preserves_inactive_and_replaces_active() -> None:
    previous = pl.concat(
        [
            _lineage("A.US", ["2026-08-12"], [10.0], source="yfinance", vintage="old"),
            _lineage(
                "OLD.US",
                ["2020-01-02"],
                [20.0],
                source="eodhd_frozen_history",
                vintage="seed",
            ),
        ]
    )
    fresh = _lineage(
        "A.US",
        ["2026-08-12", "2026-08-14"],
        [10.5, 11.0],
        source="yfinance",
        vintage="fresh",
    )

    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=fresh,
        active_tickers=["A"],
    )

    assert result.lineage.filter(pl.col("ticker") == "OLD.US").equals(
        previous.filter(pl.col("ticker") == "OLD.US"), null_equal=True
    )
    assert result.lineage.filter(pl.col("ticker") == "A.US")[
        "source_vintage_id"
    ].unique().to_list() == ["fresh"]


def test_roll_forward_preserves_confirmed_terminal_active_ticker() -> None:
    previous = pl.concat(
        [
            _lineage("A.US", ["2026-08-12"], [10.0], source="yfinance", vintage="old"),
            _lineage("EA.US", ["2026-08-10"], [209.7], source="yfinance", vintage="old"),
        ]
    )
    fresh = _lineage(
        "A.US", ["2026-08-14"], [11.0], source="yfinance", vintage="fresh"
    )

    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=fresh,
        active_tickers=["A", "EA"],
        preserved_terminal_tickers=["EA"],
    )

    assert result.lineage.filter(pl.col("ticker") == "EA.US").equals(
        previous.filter(pl.col("ticker") == "EA.US"), null_equal=True
    )
    assert result.composition_report["preserved_terminal_tickers"] == ["EA.US"]
