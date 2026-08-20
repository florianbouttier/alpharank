from __future__ import annotations

import polars as pl

from alpharank.data.definitive import select_definitive_observations


def _staged(rows: list[dict[str, object]]) -> pl.DataFrame:
    return pl.DataFrame(rows).select(
        pl.col("ticker").cast(pl.String),
        pl.col("date").cast(pl.String),
        pl.col("value").cast(pl.Float64),
        pl.col("source_name").cast(pl.String),
        pl.lit("daily_prices").alias("dataset_name"),
        pl.col("receipt_id").cast(pl.String),
        pl.col("payload_sha256").cast(pl.String),
        pl.col("retrieved_at").cast(pl.String),
    )


def _select(staged: pl.DataFrame, *, cutoff: str = "2026-08-20T12:00:00+00:00"):
    return select_definitive_observations(
        staged,
        business_key=("ticker", "date"),
        value_column="value",
        source_priority=("eodhd", "yfinance"),
        rule_id="daily_price_priority_v1",
        knowledge_cutoff=cutoff,
    )


def _row(
    source_name: str,
    value: float | None,
    *,
    receipt_id: str,
    retrieved_at: str,
) -> dict[str, object]:
    return {
        "ticker": "AAPL.US",
        "date": "2026-08-19",
        "value": value,
        "source_name": source_name,
        "receipt_id": receipt_id,
        "payload_sha256": ("a" if source_name == "eodhd" else "b") * 64,
        "retrieved_at": retrieved_at,
    }


def test_def_selects_priority_and_records_decision_provenance() -> None:
    result = _select(
        _staged(
            [
                _row(
                    "yfinance",
                    231.0,
                    receipt_id="yf_01",
                    retrieved_at="2026-08-20T10:00:00+00:00",
                ),
                _row(
                    "eodhd",
                    230.0,
                    receipt_id="eodhd_01",
                    retrieved_at="2026-08-20T09:00:00+00:00",
                ),
            ]
        )
    )

    assert result.frame.select("source_name", "value", "selection_reason").to_dicts() == [
        {
            "source_name": "eodhd",
            "value": 230.0,
            "selection_reason": "highest_priority_source_known_at_cutoff",
        }
    ]
    assert result.decisions.select(
        "selection_rule_id",
        "knowledge_cutoff",
        "selected_receipt_id",
        "selected_payload_sha256",
        "known_sources",
    ).to_dicts() == [
        {
            "selection_rule_id": "daily_price_priority_v1",
            "knowledge_cutoff": "2026-08-20T12:00:00+00:00",
            "selected_receipt_id": "eodhd_01",
            "selected_payload_sha256": "a" * 64,
            "known_sources": "eodhd | yfinance",
        }
    ]


def test_def_ignores_information_retrieved_after_cutoff() -> None:
    result = _select(
        _staged(
            [
                _row(
                    "yfinance",
                    231.0,
                    receipt_id="yf_known",
                    retrieved_at="2026-08-20T10:00:00+00:00",
                ),
                _row(
                    "eodhd",
                    999.0,
                    receipt_id="eodhd_future",
                    retrieved_at="2026-08-20T13:00:00+00:00",
                ),
            ]
        )
    )

    assert result.frame["value"].to_list() == [231.0]
    assert result.decisions.select(
        "selection_reason", "excluded_after_cutoff_count"
    ).to_dicts() == [
        {
            "selection_reason": "only_source_known_at_cutoff",
            "excluded_after_cutoff_count": 1,
        }
    ]


def test_def_falls_back_only_when_preferred_value_is_null_and_keeps_zero() -> None:
    fallback = _select(
        _staged(
            [
                _row(
                    "eodhd",
                    None,
                    receipt_id="eodhd_null",
                    retrieved_at="2026-08-20T09:00:00+00:00",
                ),
                _row(
                    "yfinance",
                    231.0,
                    receipt_id="yf_01",
                    retrieved_at="2026-08-20T10:00:00+00:00",
                ),
            ]
        )
    )
    observed_zero = _select(
        _staged(
            [
                _row(
                    "eodhd",
                    0.0,
                    receipt_id="eodhd_zero",
                    retrieved_at="2026-08-20T09:00:00+00:00",
                ),
                _row(
                    "yfinance",
                    231.0,
                    receipt_id="yf_01",
                    retrieved_at="2026-08-20T10:00:00+00:00",
                ),
            ]
        )
    )

    assert fallback.frame.select("source_name", "selection_reason").row(0) == (
        "yfinance",
        "preferred_source_missing_value_fallback",
    )
    assert observed_zero.frame.select("source_name", "value").row(0) == (
        "eodhd",
        0.0,
    )


def test_def_records_unresolved_key_when_no_value_is_observed() -> None:
    result = _select(
        _staged(
            [
                _row(
                    "eodhd",
                    None,
                    receipt_id="eodhd_null",
                    retrieved_at="2026-08-20T09:00:00+00:00",
                ),
                _row(
                    "yfinance",
                    None,
                    receipt_id="yf_null",
                    retrieved_at="2026-08-20T10:00:00+00:00",
                ),
            ]
        )
    )

    assert result.frame.is_empty()
    assert result.selected_key_count == 0
    assert result.unresolved_key_count == 1
    assert result.decisions["selection_reason"].to_list() == [
        "unresolved_no_observed_value"
    ]
