"""Resolve realized shareholder returns for terminal security events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Any

import polars as pl


TERMINAL_EVENT_TYPES = frozenset(
    {"cash_merger", "stock_merger", "bankruptcy", "delisting", "ticker_change"}
)
TERMINAL_EVENT_COLUMNS = (
    "terminal_event_id",
    "event_type",
    "ticker",
    "successor_ticker",
    "effective_date",
    "known_at",
    "price_vintage_id",
    "cash_per_share",
    "recovery_per_share",
    "exchange_ratio",
    "distribution_per_share",
    "source",
    "source_url",
)
SUCCESSOR_PRICE_COLUMNS = (
    "ticker",
    "holding_month",
    "price_asof_date",
    "holding_end_price",
    "price_vintage_id",
)


@dataclass(frozen=True)
class TerminalReturnResult:
    """Holdings with resolved returns and an event-level audit summary."""

    holdings: pl.DataFrame
    report: dict[str, Any]


def resolve_terminal_shareholder_returns(
    holdings: pl.DataFrame,
    *,
    terminal_events: pl.DataFrame,
    price_vintage_id: str,
    successor_prices: pl.DataFrame | None = None,
    realized_return_column: str = "realized_return",
    starting_price_column: str = "last_close",
) -> TerminalReturnResult:
    """Resolve missing holding returns without conditioning portfolio selection.

    Existing market returns are never replaced. A missing return is resolved
    only when one terminal event occurs in the holding month and its complete
    consideration can be valued from the declared immutable price vintage.
    Bankruptcy and delisting recovery must be explicit, including zero.
    """

    required_holdings = {"ticker", "holding_month", realized_return_column}
    missing_holdings = required_holdings - set(holdings.columns)
    if missing_holdings:
        raise ValueError(
            f"Terminal return holdings are missing columns: {sorted(missing_holdings)}"
        )
    vintage_id = str(price_vintage_id).strip()
    if not vintage_id:
        raise ValueError("Terminal return resolution requires a price vintage id.")
    if holdings.is_empty():
        return TerminalReturnResult(
            holdings,
            {
                "terminal_return_policy_version": 1,
                "price_vintage_id": vintage_id,
                "observed_returns": 0,
                "resolved_terminal_returns": 0,
                "unresolved_missing_returns": 0,
                "terminal_event_ids": [],
            },
        )

    events = _normalize_terminal_events(terminal_events, price_vintage_id=vintage_id)
    prices = _normalize_successor_prices(
        successor_prices, price_vintage_id=vintage_id
    )
    event_lookup: dict[tuple[str, object], dict[str, Any]] = {}
    for event in events.to_dicts():
        key = (event["ticker"], event["effective_date"].replace(day=1))
        if key in event_lookup:
            raise ValueError(
                "Multiple terminal events match one ticker/holding month: "
                f"ticker={key[0]}, holding_month={key[1]}."
            )
        event_lookup[key] = event

    price_lookup: dict[tuple[str, object], dict[str, Any]] = {}
    for price in prices.to_dicts():
        key = (price["ticker"], price["holding_month"])
        if key in price_lookup:
            raise ValueError(
                "Multiple successor prices match one ticker/holding month: "
                f"ticker={key[0]}, holding_month={key[1]}."
            )
        price_lookup[key] = price

    normalized = holdings.with_row_index("_terminal_row_id").with_columns(
        pl.col("ticker")
        .cast(pl.String)
        .map_elements(_normalize_ticker, return_dtype=pl.String)
        .alias("_terminal_ticker"),
        pl.col("holding_month").cast(pl.Date, strict=False).alias("_terminal_month"),
        pl.col(realized_return_column)
        .cast(pl.Float64, strict=False)
        .alias("_terminal_observed_return"),
    )
    normalized = normalized.with_columns(
        pl.col("_terminal_month").dt.truncate("1mo")
    )
    missing_return_rows = normalized.filter(
        pl.col("_terminal_observed_return").is_finite().fill_null(False).not_()
    )
    if not missing_return_rows.is_empty() and starting_price_column not in holdings.columns:
        raise ValueError(
            "Terminal return resolution requires starting price column "
            f"{starting_price_column!r} for missing returns."
        )

    resolution_rows: list[dict[str, Any]] = []
    for row in normalized.to_dicts():
        observed = row["_terminal_observed_return"]
        if observed is not None and math.isfinite(float(observed)):
            resolution_rows.append(
                _resolution_row(
                    row_id=row["_terminal_row_id"],
                    realized_return=float(observed),
                    status="observed_market_return",
                )
            )
            continue

        ticker = row["_terminal_ticker"]
        holding_month = row["_terminal_month"]
        event = event_lookup.get((ticker, holding_month))
        if event is None:
            resolution_rows.append(
                _resolution_row(
                    row_id=row["_terminal_row_id"],
                    realized_return=None,
                    status="unresolved_missing_return",
                )
            )
            continue

        start_price = _finite_positive(
            row.get(starting_price_column),
            field=starting_price_column,
            event_id=event["terminal_event_id"],
        )
        terminal_value, successor_price = _terminal_value(
            event,
            holding_month=holding_month,
            price_lookup=price_lookup,
        )
        resolved_return = terminal_value / start_price - 1.0
        resolution_rows.append(
            _resolution_row(
                row_id=row["_terminal_row_id"],
                realized_return=resolved_return,
                status="resolved_terminal_event",
                event=event,
                terminal_value=terminal_value,
                successor_price=successor_price,
            )
        )

    resolutions = pl.DataFrame(resolution_rows, infer_schema_length=None)
    result = (
        normalized.join(resolutions, on="_terminal_row_id", how="left")
        .drop(realized_return_column)
        .rename({"_resolved_return": realized_return_column})
        .drop(
            "_terminal_row_id",
            "_terminal_ticker",
            "_terminal_month",
            "_terminal_observed_return",
        )
    )
    report = {
        "terminal_return_policy_version": 1,
        "price_vintage_id": vintage_id,
        "observed_returns": result.filter(
            pl.col("return_resolution") == "observed_market_return"
        ).height,
        "resolved_terminal_returns": result.filter(
            pl.col("return_resolution") == "resolved_terminal_event"
        ).height,
        "unresolved_missing_returns": result.filter(
            pl.col("return_resolution") == "unresolved_missing_return"
        ).height,
        "terminal_event_ids": sorted(
            result.get_column("terminal_event_id").drop_nulls().unique().to_list()
        ),
    }
    return TerminalReturnResult(result, report)


def _normalize_terminal_events(
    frame: pl.DataFrame, *, price_vintage_id: str
) -> pl.DataFrame:
    missing = set(TERMINAL_EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Terminal events are missing columns: {sorted(missing)}")
    events = frame.select(TERMINAL_EVENT_COLUMNS).with_columns(
        pl.col("terminal_event_id").cast(pl.String),
        pl.col("event_type").cast(pl.String),
        pl.col("ticker")
        .cast(pl.String)
        .map_elements(_normalize_ticker, return_dtype=pl.String),
        pl.col("successor_ticker")
        .cast(pl.String)
        .map_elements(_normalize_optional_ticker, return_dtype=pl.String),
        pl.col("effective_date").cast(pl.Date, strict=False),
        pl.col("known_at").map_elements(
            _parse_datetime, return_dtype=pl.Datetime(time_zone="UTC")
        ),
        pl.col("price_vintage_id").cast(pl.String),
        pl.col("cash_per_share").cast(pl.Float64, strict=False),
        pl.col("recovery_per_share").cast(pl.Float64, strict=False),
        pl.col("exchange_ratio").cast(pl.Float64, strict=False),
        pl.col("distribution_per_share").cast(pl.Float64, strict=False),
        pl.col("source").cast(pl.String),
        pl.col("source_url").cast(pl.String),
    )
    invalid_types = sorted(
        set(events.get_column("event_type").to_list()) - TERMINAL_EVENT_TYPES
    )
    if invalid_types:
        raise ValueError(f"Unsupported terminal event types: {invalid_types}")
    required_non_null = (
        "terminal_event_id",
        "event_type",
        "ticker",
        "effective_date",
        "known_at",
        "price_vintage_id",
        "source",
        "source_url",
    )
    for column in required_non_null:
        if events.select(pl.col(column).is_null().any()).item():
            raise ValueError(f"Terminal event field {column!r} cannot be null.")
    if events.select(pl.col("terminal_event_id").n_unique()).item() != events.height:
        raise ValueError("Terminal event identifiers must be unique.")
    mismatched = events.filter(pl.col("price_vintage_id") != price_vintage_id)
    if not mismatched.is_empty():
        raise ValueError(
            "Terminal event price vintage does not match the requested package: "
            f"expected={price_vintage_id!r}."
        )
    return events.sort(["ticker", "effective_date", "terminal_event_id"])


def _normalize_successor_prices(
    frame: pl.DataFrame | None, *, price_vintage_id: str
) -> pl.DataFrame:
    if frame is None or frame.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "holding_month": pl.Date,
                "price_asof_date": pl.Date,
                "holding_end_price": pl.Float64,
                "price_vintage_id": pl.String,
            }
        )
    missing = set(SUCCESSOR_PRICE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Successor prices are missing columns: {sorted(missing)}")
    prices = frame.select(SUCCESSOR_PRICE_COLUMNS).with_columns(
        pl.col("ticker")
        .cast(pl.String)
        .map_elements(_normalize_ticker, return_dtype=pl.String),
        pl.col("holding_month").cast(pl.Date, strict=False).dt.truncate("1mo"),
        pl.col("price_asof_date").cast(pl.Date, strict=False),
        pl.col("holding_end_price").cast(pl.Float64, strict=False),
        pl.col("price_vintage_id").cast(pl.String),
    )
    if prices.filter(pl.col("price_vintage_id") != price_vintage_id).height:
        raise ValueError(
            "Successor price vintage does not match the terminal event package."
        )
    if prices.filter(
        pl.col("price_asof_date").dt.truncate("1mo") != pl.col("holding_month")
    ).height:
        raise ValueError("Successor price must be observed inside the holding month.")
    return prices.sort(["ticker", "holding_month", "price_asof_date"])


def _terminal_value(
    event: dict[str, Any],
    *,
    holding_month: object,
    price_lookup: dict[tuple[str, object], dict[str, Any]],
) -> tuple[float, float | None]:
    event_id = event["terminal_event_id"]
    event_type = event["event_type"]
    distribution = _finite_non_negative(
        event["distribution_per_share"],
        field="distribution_per_share",
        event_id=event_id,
        default=0.0,
    )
    if event_type == "cash_merger":
        cash = _finite_non_negative(
            event["cash_per_share"], field="cash_per_share", event_id=event_id
        )
        return cash + distribution, None
    if event_type in {"bankruptcy", "delisting"}:
        recovery = _finite_non_negative(
            event["recovery_per_share"],
            field="recovery_per_share",
            event_id=event_id,
        )
        return recovery + distribution, None
    if event_type in {"stock_merger", "ticker_change"}:
        successor = event["successor_ticker"]
        if successor is None:
            raise ValueError(
                f"Terminal event {event_id!r} requires a successor ticker."
            )
        ratio = _finite_positive(
            event["exchange_ratio"], field="exchange_ratio", event_id=event_id
        )
        price = price_lookup.get((successor, holding_month))
        if price is None:
            raise ValueError(
                f"Terminal event {event_id!r} lacks a successor end price."
            )
        if price["price_asof_date"] < event["effective_date"]:
            raise ValueError(
                f"Terminal event {event_id!r} successor price predates the event."
            )
        end_price = _finite_positive(
            price["holding_end_price"],
            field="holding_end_price",
            event_id=event_id,
        )
        cash = _finite_non_negative(
            event["cash_per_share"],
            field="cash_per_share",
            event_id=event_id,
            default=0.0,
        )
        return cash + distribution + ratio * end_price, end_price
    raise AssertionError(f"Unhandled terminal event type: {event_type}")


def _resolution_row(
    *,
    row_id: int,
    realized_return: float | None,
    status: str,
    event: dict[str, Any] | None = None,
    terminal_value: float | None = None,
    successor_price: float | None = None,
) -> dict[str, Any]:
    return {
        "_terminal_row_id": row_id,
        "_resolved_return": realized_return,
        "return_resolution": status,
        "terminal_event_id": event["terminal_event_id"] if event else None,
        "terminal_event_type": event["event_type"] if event else None,
        "terminal_effective_date": event["effective_date"] if event else None,
        "terminal_event_known_at": event["known_at"] if event else None,
        "terminal_event_source": event["source"] if event else None,
        "terminal_event_source_url": event["source_url"] if event else None,
        "terminal_price_vintage_id": event["price_vintage_id"] if event else None,
        "terminal_value_per_share": terminal_value,
        "terminal_successor_ticker": event["successor_ticker"] if event else None,
        "terminal_successor_end_price": successor_price,
    }


def _finite_positive(value: object, *, field: str, event_id: str) -> float:
    numeric = _finite_non_negative(value, field=field, event_id=event_id)
    if numeric <= 0.0:
        raise ValueError(f"Terminal event {event_id!r} requires positive {field}.")
    return numeric


def _finite_non_negative(
    value: object,
    *,
    field: str,
    event_id: str,
    default: float | None = None,
) -> float:
    if value is None:
        if default is not None:
            return default
        raise ValueError(f"Terminal event {event_id!r} requires explicit {field}.")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(
            f"Terminal event {event_id!r} requires finite non-negative {field}."
        )
    return numeric


def _normalize_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    if not ticker:
        raise ValueError("Terminal event ticker cannot be empty.")
    return ticker if ticker.endswith(".US") else f"{ticker}.US"


def _normalize_optional_ticker(value: object) -> str | None:
    if value is None:
        return None
    return _normalize_ticker(value)


def _parse_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid terminal event known_at: {value!r}.") from exc
    else:
        raise ValueError(f"Invalid terminal event known_at: {value!r}.")
    if parsed.tzinfo is None:
        raise ValueError("Terminal event known_at must include an explicit timezone.")
    return parsed.astimezone(timezone.utc)
