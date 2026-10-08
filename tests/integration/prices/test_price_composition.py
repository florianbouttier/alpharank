from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from alpharank.data.prices import (
    PRICE_MATURITY_BRIDGE_POLICY_ID,
    resolve_price_refresh_universe,
)
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


def test_roll_forward_refreshes_recent_leaver_without_reactivating_it() -> None:
    previous = pl.concat(
        [
            _lineage("A.US", ["2026-08-31"], [10.0], source="yfinance", vintage="old"),
            _lineage("TTD.US", ["2026-08-31"], [50.0], source="yfinance", vintage="old"),
        ]
    )
    fresh = pl.concat(
        [
            _lineage(
                "A.US",
                ["2026-08-31", "2026-09-30"],
                [10.0, 11.0],
                source="yfinance",
                vintage="fresh",
            ),
            _lineage(
                "TTD.US",
                ["2026-08-31", "2026-09-30"],
                [50.0, 45.0],
                source="yfinance",
                vintage="fresh",
            ),
        ]
    )

    result = roll_forward_validated_price_history(
        previous_validated_lineage=previous,
        active_yahoo_vintage=fresh,
        active_tickers=["A"],
        maturity_bridge_tickers=["TTD"],
        active_resolution_vintage_id="fresh",
    )

    assert result.lineage.group_by("ticker").agg(pl.col("date").max()).sort(
        "ticker"
    ).to_dicts() == [
        {"ticker": "A.US", "date": "2026-09-30"},
        {"ticker": "TTD.US", "date": "2026-09-30"},
    ]
    assert result.composition_report["active_ticker_count"] == 1
    assert result.composition_report["refreshable_active_ticker_count"] == 1
    assert result.composition_report["refreshable_price_ticker_count"] == 2
    assert result.composition_report["maturity_bridge_tickers"] == ["TTD.US"]

    registry = build_persistent_price_history_registry(
        result.lineage,
        active_tickers=["A"],
        maturity_bridge_tickers=["TTD"],
    )
    ttd = registry.filter(pl.col("ticker") == "TTD.US").row(0, named=True)
    assert ttd["current_active"] is False
    assert ttd["maturity_bridge"] is True
    assert ttd["persistence_class"] == "maturity_bridge_refreshed"


def test_refresh_universe_bridges_only_tracked_recent_leavers(tmp_path: Path) -> None:
    registry_path = tmp_path / "constituent_changes.json"
    _write_constituent_registry(registry_path)

    result = resolve_price_refresh_universe(
        current_tickers=("A", "BE"),
        tracked_tickers=("A", "BE", "TTD", "BLDR", "BK", "CTVA", "OLD", "FUT"),
        registry_path=registry_path,
        expected_through="2026-10-08",
    )

    assert result.current_tickers == ("A", "BE")
    assert result.maturity_bridge_tickers == ("BK", "BLDR", "CTVA", "TTD")
    assert result.refresh_tickers == ("A", "BE", "BK", "BLDR", "CTVA", "TTD")
    assert result.manifest() == {
        "policy_id": PRICE_MATURITY_BRIDGE_POLICY_ID,
        "window_start": "2026-09-01",
        "expected_through": "2026-10-08",
        "current_ticker_count": 2,
        "maturity_bridge_ticker_count": 4,
        "maturity_bridge_tickers": ["BK", "BLDR", "CTVA", "TTD"],
        "refresh_ticker_count": 6,
        "semantics": (
            "Refresh current members and index leavers effective in the current or "
            "preceding calendar month so the last formed one-month portfolio matures."
        ),
    }


def test_refresh_universe_allows_new_current_member_without_prior_history(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "constituent_changes.json"
    _write_constituent_registry(registry_path)

    result = resolve_price_refresh_universe(
        current_tickers=("A", "NEW"),
        tracked_tickers=("A",),
        registry_path=registry_path,
        expected_through="2026-10-08",
    )

    assert result.current_tickers == ("A", "NEW")
    assert result.refresh_tickers == ("A", "NEW")


def _write_constituent_registry(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "index": "S&P 500",
                "events": [
                    {
                        "effective_date": "2026-08-31",
                        "operations": [{"action": "remove", "ticker": "OLD"}],
                    },
                    {
                        "effective_date": "2026-09-21",
                        "operations": [
                            {"action": "remove", "ticker": "TTD"},
                            {"action": "remove", "ticker": "BLDR"},
                            {"action": "add", "ticker": "BE"},
                        ],
                    },
                    {
                        "effective_date": "2026-09-28",
                        "operations": [
                            {
                                "action": "ticker_change",
                                "ticker": "BK",
                                "new_ticker": "BNY",
                            }
                        ],
                    },
                    {
                        "effective_date": "2026-10-06",
                        "operations": [{"action": "remove", "ticker": "CTVA"}],
                    },
                    {
                        "effective_date": "2026-10-10",
                        "operations": [{"action": "remove", "ticker": "FUT"}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
