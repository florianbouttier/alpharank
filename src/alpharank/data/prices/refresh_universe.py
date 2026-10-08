"""Resolve the temporary price refresh bridge for recent index leavers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Mapping, Sequence

PRICE_MATURITY_BRIDGE_POLICY_ID = "recent_constituent_exit_price_bridge_v1"


@dataclass(frozen=True, slots=True)
class PriceRefreshUniverse:
    """Current members plus recent leavers needed for one-month return maturity."""

    current_tickers: tuple[str, ...]
    maturity_bridge_tickers: tuple[str, ...]
    refresh_tickers: tuple[str, ...]
    window_start: date
    expected_through: date

    def manifest(self) -> dict[str, object]:
        """Return the explicit, serializable refresh-universe contract."""

        return {
            "policy_id": PRICE_MATURITY_BRIDGE_POLICY_ID,
            "window_start": self.window_start.isoformat(),
            "expected_through": self.expected_through.isoformat(),
            "current_ticker_count": len(self.current_tickers),
            "maturity_bridge_ticker_count": len(self.maturity_bridge_tickers),
            "maturity_bridge_tickers": list(self.maturity_bridge_tickers),
            "refresh_ticker_count": len(self.refresh_tickers),
            "semantics": (
                "Refresh current members and index leavers effective in the current or "
                "preceding calendar month so the last formed one-month portfolio matures."
            ),
        }


def resolve_price_refresh_universe(
    *,
    current_tickers: Sequence[str],
    tracked_tickers: Sequence[str],
    registry_path: Path,
    expected_through: str | date,
) -> PriceRefreshUniverse:
    """Resolve current members and recent official leavers without inventing continuity."""

    as_of = (
        expected_through
        if isinstance(expected_through, date)
        else date.fromisoformat(expected_through)
    )
    window_start = _previous_month_start(as_of)
    current = {_normalize_ticker(ticker) for ticker in current_tickers}
    tracked = {_normalize_ticker(ticker) for ticker in tracked_tickers}
    registry = _load_registry(registry_path)
    recent_leavers = _recent_leavers(
        registry,
        window_start=window_start,
        expected_through=as_of,
    )
    maturity_bridge = tuple(sorted((recent_leavers & tracked) - current))
    return PriceRefreshUniverse(
        current_tickers=tuple(sorted(current)),
        maturity_bridge_tickers=maturity_bridge,
        refresh_tickers=tuple(sorted(current | set(maturity_bridge))),
        window_start=window_start,
        expected_through=as_of,
    )


def canonical_price_refresh_tickers(
    *,
    current_tickers: Sequence[str],
    maturity_bridge_tickers: Sequence[str],
) -> tuple[str, ...]:
    """Return canonical market symbols for every provider price refresh."""

    tickers = {
        *(_canonical_ticker(ticker) for ticker in current_tickers),
        *(_canonical_ticker(ticker) for ticker in maturity_bridge_tickers),
    }
    return tuple(sorted(tickers))


def _recent_leavers(
    registry: Mapping[str, object],
    *,
    window_start: date,
    expected_through: date,
) -> set[str]:
    events = registry.get("events")
    if not isinstance(events, list):
        raise ValueError("The constituent registry requires an events list")
    tickers: set[str] = set()
    for event in events:
        if not isinstance(event, Mapping):
            raise ValueError("Every constituent event must be an object")
        effective_date = date.fromisoformat(str(event.get("effective_date") or ""))
        if not window_start <= effective_date <= expected_through:
            continue
        operations = event.get("operations")
        if not isinstance(operations, list):
            raise ValueError("Every constituent event requires an operations list")
        for operation in operations:
            if not isinstance(operation, Mapping):
                raise ValueError("Every constituent operation must be an object")
            if operation.get("action") not in {"remove", "ticker_change"}:
                continue
            ticker = str(operation.get("ticker") or "").strip()
            if not ticker:
                raise ValueError("A removal operation requires a ticker")
            tickers.add(_normalize_ticker(ticker))
    return tickers


def _load_registry(path: Path) -> Mapping[str, object]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(registry, Mapping) or registry.get("index") != "S&P 500":
        raise ValueError("The constituent registry must target the S&P 500")
    return registry


def _previous_month_start(value: date) -> date:
    return (value.replace(day=1) - timedelta(days=1)).replace(day=1)


def _normalize_ticker(ticker: str) -> str:
    return str(ticker).strip().upper().removesuffix(".US")


def _canonical_ticker(ticker: str) -> str:
    return f"{_normalize_ticker(ticker)}.US"
