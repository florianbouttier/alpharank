from __future__ import annotations

from datetime import date

import polars as pl
import pytest

from alpharank.data.open_source.price_quality import (
    assert_no_extreme_adjusted_price_moves,
    build_split_detection_prices,
    find_extreme_adjusted_price_moves,
    load_reviewed_extreme_price_moves,
    repair_confirmed_split_discontinuities,
)
from alpharank.data.ingestion.storage import merge_upsert_frames


def test_price_quality_flags_partial_split_scale() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["MNST.US"] * 4,
            "date": [
                date(2026, 7, 30),
                date(2026, 7, 31),
                date(2026, 8, 7),
                date(2026, 8, 10),
            ],
            "adjusted_close": [97.65, 48.19, 45.18, 91.43],
        }
    )

    findings = find_extreme_adjusted_price_moves(
        prices,
        event_since="2026-07-31",
        tickers=["MNST.US"],
    )

    assert findings["date"].to_list() == [date(2026, 7, 31), date(2026, 8, 10)]
    with pytest.raises(RuntimeError, match="discontinuities"):
        assert_no_extreme_adjusted_price_moves(
            prices,
            event_since="2026-07-31",
            tickers=["MNST.US"],
        )


def test_price_quality_ignores_moves_before_refresh_window() -> None:
    prices = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US", "A.US"],
            "date": [date(2020, 1, 1), date(2020, 1, 2), date(2026, 8, 10)],
            "adjusted_close": [100.0, 50.0, 52.0],
        }
    )
    findings = find_extreme_adjusted_price_moves(prices, event_since="2026-08-01")
    assert findings.is_empty()


def test_price_quality_accepts_only_exact_bounded_reviewed_move(tmp_path) -> None:
    registry_path = tmp_path / "reviewed_moves.json"
    registry_path.write_text(
        """{
  "registry_id": "reviewed_extreme_price_moves_test_v1",
  "events": [{
    "review_id": "mrna-20260819-phase3",
    "ticker": "MRNA",
    "date": "2026-08-19",
    "prior_adjusted_close_min": 62.95,
    "prior_adjusted_close_max": 62.97,
    "adjusted_close_min": 174.37,
    "adjusted_close_max": 174.39,
    "one_day_return_min": 1.769,
    "one_day_return_max": 1.771,
    "known_at": "2026-08-19T22:03:28Z",
    "reason": "Independent market reporting confirms a real news-driven move.",
    "source_urls": ["https://example.com/market-evidence"]
  }]
}
""",
        encoding="utf-8",
    )
    reviewed_moves, manifest = load_reviewed_extreme_price_moves(registry_path)
    prices = pl.DataFrame(
        {
            "ticker": ["MRNA.US", "MRNA.US"],
            "date": [date(2026, 8, 18), date(2026, 8, 19)],
            "adjusted_close": [62.96, 174.38],
        }
    )

    reviewed = assert_no_extreme_adjusted_price_moves(
        prices,
        event_since="2026-08-19",
        reviewed_moves=reviewed_moves,
    )

    assert manifest["registry_id"] == "reviewed_extreme_price_moves_test_v1"
    assert len(manifest["sha256"]) == 64
    assert reviewed.select("ticker", "date", "review_id").to_dicts() == [
        {
            "ticker": "MRNA.US",
            "date": date(2026, 8, 19),
            "review_id": "mrna-20260819-phase3",
        }
    ]

    with pytest.raises(RuntimeError, match="require review"):
        assert_no_extreme_adjusted_price_moves(
            prices.with_columns(
                pl.when(pl.col("date") == date(2026, 8, 19))
                .then(pl.lit(180.0))
                .otherwise(pl.col("adjusted_close"))
                .alias("adjusted_close")
            ),
            event_since="2026-08-19",
            reviewed_moves=reviewed_moves,
        )


def test_full_refresh_split_detection_cannot_be_masked_by_old_adjusted_vintage() -> None:
    existing = pl.DataFrame(
        {
            "ticker": ["MNST.US", "MNST.US"],
            "date": ["2026-08-07", "2026-08-11"],
            "adjusted_close": [45.18, 45.53],
        }
    )
    fresh = pl.DataFrame(
        {
            "ticker": ["MNST.US", "MNST.US"],
            "date": ["2026-08-07", "2026-08-11"],
            "adjusted_close": [90.36, 45.53],
        }
    )

    detection = build_split_detection_prices(
        existing_prices=existing,
        fresh_prices=fresh,
        full_history_refresh=True,
    )
    findings = find_extreme_adjusted_price_moves(
        detection,
        event_since="2026-08-09",
    )

    assert findings.height == 1
    assert findings["date"].item() == date(2026, 8, 11)


def test_confirmed_split_repair_back_adjusts_only_pre_event_delta() -> None:
    delta = pl.DataFrame(
        {
            "ticker": ["MNST.US", "MNST.US"],
            "date": ["2026-08-10", "2026-08-11"],
            "open": [90.0, 45.0],
            "high": [92.0, 46.0],
            "low": [89.0, 44.0],
            "close": [91.43, 45.53],
            "adjusted_close": [91.43, 45.53],
            "volume": [5_000_000.0, 11_000_000.0],
        }
    )
    findings = find_extreme_adjusted_price_moves(delta, event_since="2026-08-11")
    splits = pl.DataFrame(
        {
            "ticker": ["MNST.US"],
            "date": ["2026-08-11"],
            "split_ratio": [2.0],
            "source": ["yahoo_actions"],
        }
    )

    repaired, repairs = repair_confirmed_split_discontinuities(
        delta,
        findings=findings,
        splits=splits,
    )

    assert repairs[0]["split_ratio"] == 2.0
    assert repaired["adjusted_close"].to_list() == pytest.approx([45.715, 45.53])
    assert repaired["volume"].to_list() == pytest.approx([10_000_000.0, 11_000_000.0])
    assert_no_extreme_adjusted_price_moves(repaired, event_since="2026-08-11")


def test_unconfirmed_split_does_not_repair_jump() -> None:
    delta = pl.DataFrame(
        {
            "ticker": ["A.US", "A.US"],
            "date": ["2026-08-10", "2026-08-11"],
            "adjusted_close": [100.0, 50.0],
        }
    )
    findings = find_extreme_adjusted_price_moves(delta, event_since="2026-08-11")

    repaired, repairs = repair_confirmed_split_discontinuities(
        delta,
        findings=findings,
        splits=pl.DataFrame(
            schema={"ticker": pl.String, "date": pl.String, "split_ratio": pl.Float64}
        ),
    )

    assert repairs == []
    assert repaired.equals(delta)


def test_merge_upsert_frames_has_no_persistence_side_effect() -> None:
    existing = pl.DataFrame(
        {"ticker": ["A.US"], "date": [date(2026, 8, 7)], "value": [1.0], "seq": [1]}
    )
    delta = pl.DataFrame(
        {"ticker": ["A.US"], "date": [date(2026, 8, 7)], "value": [2.0], "seq": [2]}
    )
    merged = merge_upsert_frames(
        existing,
        delta,
        key_cols=["ticker", "date"],
        order_cols=["seq"],
    )
    assert existing["value"].to_list() == [1.0]
    assert merged["value"].to_list() == [2.0]
