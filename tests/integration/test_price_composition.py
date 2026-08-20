from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.prices.composition import (
    compose_hybrid_price_history,
    roll_forward_validated_price_history,
)
from alpharank.data.prices.contracts import ADJUSTMENT_POLICY_VERSION, PRICE_LINEAGE_COLUMNS
from alpharank.data.prices.history import (
    build_persistent_price_history_registry,
    persistent_history_summary,
)


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


def test_active_reused_symbol_cannot_inherit_the_old_security_history() -> None:
    registry = pl.DataFrame(
        {
            "source_ticker": ["SNDK", "SNDK"],
            "canonical_ticker": ["SNDK_OLD", "SNDK"],
            "security_id": ["old", "new"],
            "issuer_cik": ["0001000180", "0002023554"],
            "valid_from": ["2005-01-01", "2025-02-24"],
            "valid_to": ["2016-05-12", None],
            "identity_status": ["historical", "current"],
            "evidence": ["fixture-old", "fixture-new"],
        }
    )
    previous = pl.concat(
        [
            _lineage(
                "SNDK.US",
                ["2016-05-12"],
                [75.0],
                source="eodhd_frozen_history",
                vintage="seed",
            ),
            _lineage(
                "SNDK.US",
                ["2025-02-20", "2025-02-24"],
                [48.0, 50.0],
                source="yfinance",
                vintage="old",
            ),
        ]
    )
    fresh = _lineage(
        "SNDK.US",
        ["2025-02-20", "2025-02-24", "2025-02-25"],
        [48.0, 50.0, 51.0],
        source="yfinance",
        vintage="fresh",
    )

    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=fresh,
        active_tickers=["SNDK"],
        active_resolution_vintage_id="fresh",
        security_identity_registry=registry,
    )

    assert result.prices.select("ticker", "date").to_dicts() == [
        {"ticker": "SNDK.US", "date": "2025-02-24"},
        {"ticker": "SNDK.US", "date": "2025-02-25"},
        {"ticker": "SNDK_OLD.US", "date": "2016-05-12"},
    ]
    identity = result.composition_report["security_identity"]
    assert identity["previous_validated_lineage"]["rejected_rows"] == 1
    assert identity["active_yahoo_vintage"]["rejected_rows"] == 1


def test_roll_forward_preserves_inactive_and_replaces_active() -> None:
    previous = pl.concat(
        [
            _lineage("A.US", ["2026-08-12"], [10.0], source="yfinance", vintage="old"),
            _lineage(
                "CI.US",
                ["2026-07-01", "2026-07-02"],
                [30.0, 31.0],
                source="yfinance",
                vintage="first-published",
            ),
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
    assert result.lineage.filter(pl.col("ticker") == "CI.US").equals(
        previous.filter(pl.col("ticker") == "CI.US"), null_equal=True
    )
    assert result.lineage.filter(pl.col("ticker") == "A.US")[
        "source_vintage_id"
    ].unique().to_list() == ["fresh"]
    assert result.composition_report["preserved_open_source_only_tickers"] == 1

    registry = build_persistent_price_history_registry(
        result.lineage,
        active_tickers=["A"],
    )
    ci = registry.filter(pl.col("ticker") == "CI.US").row(0, named=True)
    assert ci["persistence_class"] == "inactive_open_source_only"
    assert ci["has_eodhd_seed"] is False
    assert ci["has_open_source_history"] is True
    summary = persistent_history_summary(registry)
    assert summary["non_eodhd_persisted_tickers"] == ["CI.US"]


def test_roll_forward_accepts_only_exact_audited_prior_active_keys() -> None:
    previous = _lineage(
        "A.US",
        ["2026-08-12", "2026-08-13"],
        [10.0, 10.5],
        source="yfinance",
        vintage="old",
    )
    current = _lineage(
        "A.US",
        ["2026-08-13", "2026-08-14"],
        [10.6, 11.0],
        source="yfinance",
        vintage="fresh",
    )
    carried = previous.filter(pl.col("date") == "2026-08-12")

    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=pl.concat([carried, current]),
        active_tickers=["A"],
        active_resolution_vintage_id="fresh",
    )

    assert result.lineage.height == 3
    assert result.composition_report["active_yahoo_vintage_id"] == "fresh"
    assert result.composition_report["audited_carried_active_rows"] == 1
    assert result.composition_report["audited_carried_active_tickers"] == 1

    changed_carried = carried.with_columns(pl.lit(9.9).alias("adjusted_close"))
    with pytest.raises(RuntimeError, match="preceding validated lineage"):
        roll_forward_validated_price_history(
            previous_validated_lineage=previous,
            active_yahoo_vintage=pl.concat([changed_carried, current]),
            active_tickers=["A"],
            active_resolution_vintage_id="fresh",
        )


def test_roll_forward_preserves_confirmed_terminal_active_ticker() -> None:
    previous = pl.concat(
        [
            _lineage("A.US", ["2026-08-12"], [10.0], source="yfinance", vintage="old"),
            _lineage("EA.US", ["2026-08-10"], [209.7], source="yfinance", vintage="old"),
        ]
    )
    fresh = _lineage("A.US", ["2026-08-14"], [11.0], source="yfinance", vintage="fresh")

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
    registry = build_persistent_price_history_registry(
        result.lineage,
        active_tickers=["A", "EA"],
        preserved_terminal_tickers=["EA"],
    )
    assert persistent_history_summary(registry)["non_eodhd_persisted_tickers"] == ["EA.US"]
