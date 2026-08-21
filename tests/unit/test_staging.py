from __future__ import annotations

import polars as pl
import pytest

from alpharank.data.warehouse.staging import normalize_staging_observations

VALUE_SCHEMA = {
    "ticker": pl.String,
    "date": pl.String,
    "adjusted_close": pl.Float64,
}
PAYLOAD_A = "a" * 64
PAYLOAD_B = "b" * 64


def _observation(
    *,
    source_name: str,
    receipt_id: str,
    payload_sha256: str,
    adjusted_close: float,
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ticker": ["AAPL.US"],
            "date": ["2026-08-20"],
            "adjusted_close": [adjusted_close],
            "source_name": [source_name],
            "dataset_name": ["daily_prices"],
            "receipt_id": [receipt_id],
            "payload_sha256": [payload_sha256],
            "retrieved_at": ["2026-08-20T10:00:00+00:00"],
        }
    )


def _stage(*frames: pl.DataFrame) -> pl.DataFrame:
    return normalize_staging_observations(
        frames,
        business_key=("ticker", "date"),
        value_schema=VALUE_SCHEMA,
    )


def test_stg_keeps_conflicting_provider_observations_separate() -> None:
    staged = _stage(
        _observation(
            source_name="eodhd",
            receipt_id="eodhd_01",
            payload_sha256=PAYLOAD_A,
            adjusted_close=230.0,
        ),
        _observation(
            source_name="yfinance",
            receipt_id="yfinance_01",
            payload_sha256=PAYLOAD_B,
            adjusted_close=231.0,
        ),
    )

    assert staged.height == 2
    assert staged.select("source_name", "adjusted_close").to_dicts() == [
        {"source_name": "eodhd", "adjusted_close": 230.0},
        {"source_name": "yfinance", "adjusted_close": 231.0},
    ]


def test_stg_result_is_independent_of_provider_input_order() -> None:
    eodhd = _observation(
        source_name="eodhd",
        receipt_id="eodhd_01",
        payload_sha256=PAYLOAD_A,
        adjusted_close=230.0,
    )
    yfinance = _observation(
        source_name="yfinance",
        receipt_id="yfinance_01",
        payload_sha256=PAYLOAD_B,
        adjusted_close=231.0,
    )

    assert _stage(eodhd, yfinance).equals(_stage(yfinance, eodhd))


@pytest.mark.parametrize(
    "selection_column",
    ["selected_source", "source_priority", "fallback_used", "selection_reason"],
)
def test_stg_rejects_source_selection_columns(selection_column: str) -> None:
    observation = _observation(
        source_name="yfinance",
        receipt_id="yfinance_01",
        payload_sha256=PAYLOAD_A,
        adjusted_close=231.0,
    ).with_columns(pl.lit("forbidden").alias(selection_column))

    with pytest.raises(ValueError, match="Source selection is forbidden in STG"):
        _stage(observation)


def test_stg_rejects_duplicate_rows_from_one_receipt() -> None:
    observation = _observation(
        source_name="yfinance",
        receipt_id="yfinance_01",
        payload_sha256=PAYLOAD_A,
        adjusted_close=231.0,
    )

    with pytest.raises(ValueError, match="duplicate provider observation"):
        _stage(observation, observation)
