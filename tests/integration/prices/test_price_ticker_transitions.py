from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.prices.contracts import (
    ADJUSTMENT_POLICY_VERSION,
    PRICE_LINEAGE_COLUMNS,
)
from alpharank.data.prices.ticker_transitions import (
    apply_price_ticker_transition_overlay,
)


def _lineage(ticker: str, dates: list[str], prices: list[float]) -> pl.DataFrame:
    size = len(dates)
    return pl.DataFrame(
        {
            "date": dates,
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": [100.0] * size,
            "adjusted_close": prices,
            "ticker": [ticker] * size,
            "source": ["yfinance"] * size,
            "dataset": ["prices_yfinance"] * size,
            "ingestion_run_id": ["run_1"] * size,
            "ingested_at": ["2026-06-22T00:00:00Z"] * size,
            "source_vintage_id": ["run_1"] * size,
            "return_source_vintage_id": ["run_1"] * size,
            "adjustment_policy_version": [ADJUSTMENT_POLICY_VERSION] * size,
            "adjustment_bridge_factor": [1.0] * size,
            "eodhd_seed_sha256": ["seed"] * size,
            "correction_overlay_id": [None] * size,
        }
    ).select(PRICE_LINEAGE_COLUMNS)


def _registry() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "transition_id": ["issuer_old_from_new_2026"],
            "security_id": ["sec-cik-0000000001"],
            "issuer_cik": ["0000000001"],
            "provider_ticker": ["NEW.US"],
            "target_ticker": ["OLD.US"],
            "validated_anchor_date": ["2026-04-24"],
            "copy_from": ["2026-04-25"],
            "copy_through": ["2026-05-31"],
            "model_ticker_effective_from": ["2026-06-01"],
            "official_ticker_effective_from": ["2026-06-24"],
            "conversion_ratio": [1.0],
            "required_overlap_rows": [3],
            "maximum_overlap_return_delta": [1e-12],
            "maximum_anchor_relative_delta": [1e-12],
            "evidence_known_at": ["2026-06-22T00:00:00Z"],
            "evidence_url": ["https://example.test/official"],
            "evidence_statement": ["Same security and unchanged CUSIP."],
        }
    )


def _same_security_lineage() -> pl.DataFrame:
    overlap_dates = [
        "2026-04-17",
        "2026-04-20",
        "2026-04-21",
        "2026-04-22",
        "2026-04-23",
        "2026-04-24",
    ]
    overlap_prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0]
    return pl.concat(
        [
            _lineage("OLD.US", overlap_dates, overlap_prices),
            _lineage(
                "NEW.US",
                ["2010-01-04", *overlap_dates, "2026-04-27", "2026-04-28"],
                [50.0, *overlap_prices, 115.5, 127.05],
            ),
        ]
    )


def test_transition_adds_only_missing_dates_from_provider_returns() -> None:
    previous = _same_security_lineage().sort(["ticker", "date"])

    result = apply_price_ticker_transition_overlay(previous, registry=_registry())

    added = result.lineage.filter(pl.col("correction_overlay_id").is_not_null())
    assert added.select("ticker", "date").to_dicts() == [
        {"ticker": "OLD.US", "date": "2026-04-27"},
        {"ticker": "OLD.US", "date": "2026-04-28"},
    ]
    assert added["adjusted_close"].to_list() == pytest.approx([115.5, 127.05])
    assert result.report["previous_rows_changed"] == 0
    assert result.report["manual_price_values"] == 0
    assert result.audit["provider_ticker"].unique().to_list() == ["NEW.US"]
    assert result.lineage.filter(pl.col("date") == "2010-01-04")["ticker"].to_list() == ["NEW.US"]


def test_transition_is_idempotent() -> None:
    first = apply_price_ticker_transition_overlay(
        _same_security_lineage(),
        registry=_registry(),
    )

    second = apply_price_ticker_transition_overlay(first.lineage, registry=_registry())

    assert second.lineage.equals(first.lineage, null_equal=True)
    assert second.report["added_rows"] == 0
    assert second.report["transitions"][0]["status"] == "already_present"


def test_transition_rejects_different_overlap_returns() -> None:
    lineage = _same_security_lineage().with_columns(
        pl.when((pl.col("ticker") == "NEW.US") & (pl.col("date") == "2026-04-23"))
        .then(pl.lit(120.0))
        .otherwise(pl.col("adjusted_close"))
        .alias("adjusted_close")
    )

    with pytest.raises(RuntimeError, match="overlap returns disagree"):
        apply_price_ticker_transition_overlay(lineage, registry=_registry())
