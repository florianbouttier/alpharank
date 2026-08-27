"""Validate constituent, vintage, terminal-event and benchmark price inputs."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl


def latest_constituents(path: Path) -> tuple[str, ...]:
    frame = pl.read_csv(path, infer_schema_length=0)
    date_column = "Date" if "Date" in frame.columns else "date"
    ticker_column = "Ticker" if "Ticker" in frame.columns else "ticker"
    latest = frame.select(pl.col(date_column).max()).item()
    return tuple(
        frame.filter(pl.col(date_column) == latest)
        .get_column(ticker_column)
        .drop_nulls()
        .cast(pl.String)
        .str.to_uppercase()
        .unique()
        .sort()
        .to_list()
    )


def resolve_active_resolution_vintage_id(*, run_id: str, fresh_yahoo: pl.DataFrame) -> str:
    """Bind audited carried rows to the full-ingestion run that selected them."""

    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise RuntimeError("Full-ingestion manifest does not declare a run_id")
    vintage_column = _vintage_column(fresh_yahoo)
    observed = {
        str(value)
        for value in fresh_yahoo.get_column(vintage_column).drop_nulls().unique().to_list()
    }
    if normalized_run_id not in observed:
        raise RuntimeError(
            "Fresh Yahoo vintage has no observation from the full-ingestion run; "
            f"run_id={normalized_run_id}"
        )
    return normalized_run_id


def _vintage_column(frame: pl.DataFrame) -> str:
    if "source_vintage_id" in frame.columns:
        return "source_vintage_id"
    if "ingestion_run_id" in frame.columns:
        return "ingestion_run_id"
    raise RuntimeError("Fresh Yahoo vintage does not carry a source or ingestion run id")


def validated_terminal_tickers(
    *, requested: tuple[str, ...], registry_path: Path, expected_through: str
) -> tuple[str, ...]:
    normalized = {str(ticker).upper().removesuffix(".US") for ticker in requested}
    if not normalized:
        return ()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    confirmed = {
        str(operation["ticker"]).upper().removesuffix(".US")
        for event in registry.get("events", [])
        if str(event.get("effective_date", "")) <= expected_through
        for operation in event.get("operations", [])
        if operation.get("action") == "remove"
    }
    unconfirmed = sorted(normalized - confirmed)
    if unconfirmed:
        raise RuntimeError(f"Terminal price preservation lacks confirmation: {unconfirmed}")
    return tuple(sorted(f"{ticker}.US" for ticker in normalized))


def refreshable_active_tickers(
    active_tickers: tuple[str, ...], terminal_tickers: tuple[str, ...]
) -> tuple[str, ...]:
    terminal = set(terminal_tickers)
    return tuple(
        ticker
        for ticker in active_tickers
        if f"{ticker.upper().removesuffix('.US')}.US" not in terminal
    )


def prepare_benchmark_prices(path: Path, *, expected_run_id: str | None) -> pl.DataFrame:
    frame = pl.read_parquet(path)
    if expected_run_id is not None:
        if "ingestion_run_id" not in frame.columns:
            raise RuntimeError("Acquisition benchmark has no ingestion run id")
        observed = set(frame.get_column("ingestion_run_id").drop_nulls().cast(pl.String).unique())
        if observed != {expected_run_id}:
            raise RuntimeError(f"Acquisition benchmark is not run-bound: {sorted(observed)}")
    columns = ["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"]
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"Benchmark price payload is missing columns: {missing}")
    return frame.select(columns).sort(["ticker", "date"])
