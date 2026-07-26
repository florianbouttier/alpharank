from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl


DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_quality"
    / "historical_ticker_exclusions_v1.json"
)


@dataclass(frozen=True)
class TickerExclusionRegistry:
    """Versioned, dataset-scoped ticker quarantine."""

    registry_id: str
    path: Path
    excluded_tickers: tuple[str, ...]
    payload: dict[str, Any]


def normalize_tickers(tickers: Iterable[str]) -> tuple[str, ...]:
    """Normalize ticker exclusions while preserving their declared order."""

    return tuple(
        dict.fromkeys(
            str(ticker).strip().upper()
            for ticker in tickers
            if str(ticker).strip()
        )
    )


def load_ticker_exclusion_registry(
    path: str | Path = DEFAULT_HISTORICAL_TICKER_EXCLUSION_REGISTRY,
) -> TickerExclusionRegistry:
    """Load a versioned registry and return all full-trajectory exclusions."""

    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported ticker exclusion schema: {payload.get('schema_version')!r}")
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Ticker exclusion registry must contain a non-empty entries list.")
    excluded = normalize_tickers(
        entry["ticker"]
        for entry in entries
        if entry.get("decision") == "exclude_all_dates"
    )
    if not excluded:
        raise ValueError("Ticker exclusion registry contains no exclude_all_dates decision.")
    return TickerExclusionRegistry(
        registry_id=str(payload["registry_id"]),
        path=resolved,
        excluded_tickers=excluded,
        payload=payload,
    )


def exclude_tickers_from_frame(
    frame: pl.DataFrame,
    excluded_tickers: Iterable[str],
    *,
    ticker_column: str = "ticker",
) -> pl.DataFrame:
    """Remove excluded symbols from every row of a frame."""

    normalized = normalize_tickers(excluded_tickers)
    if frame.is_empty() or not normalized or ticker_column not in frame.columns:
        return frame
    ticker = pl.col(ticker_column).cast(pl.Utf8).str.to_uppercase()
    if ticker_column == "Ticker":
        ticker = ticker.str.replace_all(r"\.", "-") + pl.lit(".US")
    return frame.filter(~ticker.is_in(normalized))
