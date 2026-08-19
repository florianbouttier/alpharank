from __future__ import annotations

import polars as pl

from alpharank.data.open_source.definitive_prices import (
    bootstrap_definitive_prices,
    build_definitive_prices,
    stage_yahoo_prices,
)


def _raw(rows: list[tuple[str, str, float | None]]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "ticker": ticker,
                "date": date,
                "open": adjusted_close,
                "high": adjusted_close,
                "low": adjusted_close,
                "close": adjusted_close,
                "volume": 1000.0,
                "adjusted_close": adjusted_close,
            }
            for ticker, date, adjusted_close in rows
        ],
        schema={
            "ticker": pl.String,
            "date": pl.String,
            "open": pl.Float64,
            "high": pl.Float64,
            "low": pl.Float64,
            "close": pl.Float64,
            "volume": pl.Float64,
            "adjusted_close": pl.Float64,
        },
    )


def test_def_uses_current_valid_row_and_preserves_exact_previous_key_when_missing() -> None:
    previous = stage_yahoo_prices(
        _raw(
            [
                ("AAPL.US", "2026-08-18", 230.0),
                ("AAPL.US", "2026-08-19", 232.0),
            ]
        ),
        run_id="raw01",
        observed_at="2026-08-19T02:15:00Z",
    )
    current = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-19", 233.0)]),
        run_id="raw02",
        observed_at="2026-08-20T02:15:00Z",
    )
    result = build_definitive_prices(
        staged_current=current,
        previous_definitive=previous,
        requested_tickers=("AAPL",),
    )

    assert result.frame.select("date", "adjusted_close", "ingestion_run_id").to_dicts() == [
        {"date": "2026-08-18", "adjusted_close": 230.0, "ingestion_run_id": "raw01"},
        {"date": "2026-08-19", "adjusted_close": 233.0, "ingestion_run_id": "raw02"},
    ]
    assert result.audit.select("date", "selection_reason").to_dicts() == [
        {"date": "2026-08-18", "selection_reason": "carried_forward_missing_current_raw"}
    ]


def test_def_carries_same_key_when_current_adjusted_close_is_null() -> None:
    previous = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-18", 230.0)]),
        run_id="raw01",
        observed_at="2026-08-19T02:15:00Z",
    )
    current = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-18", None)]),
        run_id="raw02",
        observed_at="2026-08-20T02:15:00Z",
    )
    result = build_definitive_prices(
        staged_current=current,
        previous_definitive=previous,
        requested_tickers=("AAPL",),
    )

    assert result.frame["adjusted_close"].to_list() == [230.0]
    assert result.frame["ingestion_run_id"].to_list() == ["raw01"]
    assert result.audit["selection_reason"].to_list() == [
        "carried_forward_invalid_current_raw"
    ]


def test_def_never_carries_a_price_to_a_different_date() -> None:
    previous = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-18", 230.0)]),
        run_id="raw01",
        observed_at="2026-08-19T02:15:00Z",
    )
    current = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-19", None)]),
        run_id="raw02",
        observed_at="2026-08-20T02:15:00Z",
    )
    result = build_definitive_prices(
        staged_current=current,
        previous_definitive=previous,
        requested_tickers=("AAPL",),
    )

    assert result.frame.select("date", "adjusted_close").to_dicts() == [
        {"date": "2026-08-18", "adjusted_close": 230.0}
    ]
    assert result.unresolved_row_count == 1
    assert result.audit.filter(pl.col("date") == "2026-08-19")[
        "selection_reason"
    ].item() == "unresolved_invalid_current_raw"


def test_bootstrap_def_keeps_existing_raw_origin() -> None:
    previous = stage_yahoo_prices(
        _raw([("AAPL.US", "2026-08-18", 230.0)]),
        run_id="raw01",
        observed_at="2026-08-19T02:15:00Z",
    )
    assert bootstrap_definitive_prices(previous)["ingestion_run_id"].to_list() == ["raw01"]
