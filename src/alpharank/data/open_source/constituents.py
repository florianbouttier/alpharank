from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import polars as pl


CONSTITUENT_COLUMNS = ("Date", "Ticker", "Name")
INDEX_EVENT_TIMEZONE = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ConstituentRefreshResult:
    frame: pl.DataFrame
    operation_audit: tuple[dict[str, Any], ...]
    monthly_summary: tuple[dict[str, Any], ...]
    base_month: date
    target_month: date


def load_constituent_change_registry(path: Path) -> dict[str, Any]:
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("index") != "S&P 500":
        raise ValueError("The registry must explicitly target the S&P 500.")
    if not registry.get("base_month") or not isinstance(registry.get("events"), list):
        raise ValueError("The registry requires base_month and events.")
    return registry


def refresh_monthly_constituents(
    frame: pl.DataFrame,
    *,
    registry: Mapping[str, Any],
    target_month: date,
) -> ConstituentRefreshResult:
    normalized = _normalize_constituents(frame)
    base_month = date.fromisoformat(str(registry["base_month"]))
    target_month = target_month.replace(day=1)
    if target_month < base_month:
        raise ValueError(f"target_month {target_month} is before registry base_month {base_month}.")

    base = normalized.filter(pl.col("Date") == base_month)
    if base.is_empty():
        raise ValueError(f"No constituent snapshot exists for base_month={base_month}.")
    if base.select(pl.col("Ticker").n_unique()).item() != base.height:
        raise ValueError(f"Duplicate tickers exist in base_month={base_month}.")

    members = {
        row["Ticker"]: row["Name"]
        for row in base.select(["Ticker", "Name"]).sort("Ticker").to_dicts()
    }
    events = sorted(registry["events"], key=lambda item: str(item["effective_date"]))
    months = _month_starts_after(base_month, target_month)
    audit: list[dict[str, Any]] = []
    base_events = [
        event
        for event in events
        if base_month
        <= date.fromisoformat(str(event["effective_date"]))
        <= _month_end(base_month)
    ]
    for event in base_events:
        for operation in event.get("operations", []):
            audit.append(
                _apply_operation(
                    members,
                    operation=operation,
                    effective_date=str(event["effective_date"]),
                    source_url=str(event["source_url"]),
                    snapshot_month=base_month,
                )
            )
    snapshots = [
        pl.DataFrame(
            {
                "Date": [base_month] * len(members),
                "Ticker": sorted(members),
                "Name": [members[ticker] for ticker in sorted(members)],
            },
            schema={"Date": pl.Date, "Ticker": pl.String, "Name": pl.String},
        )
    ]
    summary: list[dict[str, Any]] = [
        {
            "month": base_month.isoformat(),
            "constituent_count": len(members),
            "event_count": len(base_events),
            "status": "base_reconciled_to_month_end",
        }
    ]
    for month in months:
        month_end = _month_end(month)
        month_events = [
            event
            for event in events
            if month
            <= date.fromisoformat(str(event["effective_date"]))
            <= month_end
        ]
        for event in month_events:
            for operation in event.get("operations", []):
                audit.append(
                    _apply_operation(
                        members,
                        operation=operation,
                        effective_date=str(event["effective_date"]),
                        source_url=str(event["source_url"]),
                        snapshot_month=month,
                    )
                )
        snapshots.append(
            pl.DataFrame(
                {
                    "Date": [month] * len(members),
                    "Ticker": sorted(members),
                    "Name": [members[ticker] for ticker in sorted(members)],
                },
                schema={"Date": pl.Date, "Ticker": pl.String, "Name": pl.String},
            )
        )
        summary.append(
            {
                "month": month.isoformat(),
                "constituent_count": len(members),
                "event_count": len(month_events),
                "status": "reconstructed_from_official_events",
            }
        )

    historical = (
        frame.select(CONSTITUENT_COLUMNS)
        .with_columns(
            pl.col("Date").cast(pl.Date, strict=False),
            pl.col("Ticker").cast(pl.String),
            pl.col("Name").cast(pl.String),
        )
        .filter(pl.col("Date").is_null() | (pl.col("Date") < base_month))
    )
    refreshed = (
        pl.concat([historical, *snapshots], how="vertical")
        .sort(["Date", "Ticker"])
    )
    return ConstituentRefreshResult(
        frame=refreshed,
        operation_audit=tuple(audit),
        monthly_summary=tuple(summary),
        base_month=base_month,
        target_month=target_month,
    )


def membership_at_decision_time(
    frame: pl.DataFrame,
    *,
    registry: Mapping[str, Any],
    decision_times: Sequence[datetime],
) -> pl.DataFrame:
    """Replay exact effective membership for timezone-aware decisions."""

    normalized = _normalize_constituents(frame)
    base_month = date.fromisoformat(str(registry["base_month"]))
    base = normalized.filter(pl.col("Date") == base_month)
    if base.is_empty():
        raise ValueError(f"No constituent snapshot exists for base_month={base_month}.")
    if base.select(pl.col("Ticker").n_unique()).item() != base.height:
        raise ValueError(f"Duplicate tickers exist in base_month={base_month}.")
    normalized_decisions = sorted(_require_aware(value) for value in decision_times)
    events = sorted(registry["events"], key=_event_effective_at)
    members = {
        row["Ticker"]: row["Name"]
        for row in base.select("Ticker", "Name").to_dicts()
    }
    rows: list[dict[str, Any]] = []
    event_index = 0
    for decision_at in normalized_decisions:
        while event_index < len(events) and _event_effective_at(
            events[event_index]
        ) <= decision_at:
            event = events[event_index]
            for operation in event.get("operations", []):
                _apply_operation(
                    members,
                    operation=operation,
                    effective_date=str(event["effective_date"]),
                    source_url=str(event["source_url"]),
                    snapshot_month=decision_at.date().replace(day=1),
                )
            event_index += 1
        rows.extend(
            {
                "decision_at": decision_at,
                "ticker": ticker,
                "name": members[ticker],
                "is_member": True,
            }
            for ticker in sorted(members)
        )
    return pl.DataFrame(rows, infer_schema_length=None).sort(
        ["decision_at", "ticker"]
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_constituent_price_coverage(
    clean_prices: pl.DataFrame,
    *,
    constituents_path: Path,
) -> tuple[dict[str, object], pl.DataFrame]:
    constituents = pl.read_csv(constituents_path, try_parse_dates=True)
    current_month = constituents.select(pl.col("Date").cast(pl.Date).max()).item()
    current = (
        constituents.filter(pl.col("Date").cast(pl.Date) == current_month)
        .select(
            (pl.col("Ticker").cast(pl.String) + pl.lit(".US")).alias("ticker")
        )
        .unique()
        .sort("ticker")
    )
    latest_prices = (
        clean_prices.filter(pl.col("adjusted_close").is_not_null())
        .group_by("ticker")
        .agg(pl.col("date").cast(pl.Date).max().alias("max_price_date"))
    )
    coverage = current.join(latest_prices, on="ticker", how="left").sort(
        ["max_price_date", "ticker"]
    )
    missing_count = coverage.filter(pl.col("max_price_date").is_null()).height
    date_distribution = (
        coverage.group_by("max_price_date")
        .len(name="ticker_count")
        .sort("max_price_date")
    )
    non_null_dates = coverage.filter(pl.col("max_price_date").is_not_null())
    latest_common_date = (
        non_null_dates.select(pl.col("max_price_date").min()).item()
        if non_null_dates.height == coverage.height and coverage.height
        else None
    )
    latest_any_date = (
        non_null_dates.select(pl.col("max_price_date").max()).item()
        if not non_null_dates.is_empty()
        else None
    )
    summary: dict[str, object] = {
        "constituent_month": current_month.isoformat(),
        "member_count": coverage.height,
        "missing_price_count": missing_count,
        "latest_common_price_date": (
            latest_common_date.isoformat() if latest_common_date else None
        ),
        "latest_any_price_date": (
            latest_any_date.isoformat() if latest_any_date else None
        ),
        "max_date_distribution": _json_safe_rows(date_distribution),
    }
    return summary, coverage


def _normalize_constituents(frame: pl.DataFrame) -> pl.DataFrame:
    missing = set(CONSTITUENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Constituent frame is missing columns: {sorted(missing)}")
    return (
        frame.select(CONSTITUENT_COLUMNS)
        .with_columns(
            pl.col("Date").cast(pl.Date, strict=False),
            pl.col("Ticker").cast(pl.String).str.strip_chars(),
            pl.col("Name").cast(pl.String).str.strip_chars(),
        )
        .filter(
            pl.col("Date").is_not_null()
            & pl.col("Ticker").is_not_null()
            & (pl.col("Ticker") != "")
        )
    )


def _json_safe_rows(frame: pl.DataFrame) -> list[dict[str, object]]:
    return [
        {
            key: value.isoformat() if hasattr(value, "isoformat") else value
            for key, value in row.items()
        }
        for row in frame.to_dicts()
    ]


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("Membership decision times must include a timezone.")
    return value.astimezone(timezone.utc)


def _event_effective_at(event: Mapping[str, Any]) -> datetime:
    explicit = event.get("effective_at")
    if explicit is not None:
        parsed = datetime.fromisoformat(str(explicit).replace("Z", "+00:00"))
        return _require_aware(parsed)
    effective_date = date.fromisoformat(str(event["effective_date"]))
    return datetime.combine(
        effective_date,
        time.min,
        tzinfo=INDEX_EVENT_TIMEZONE,
    ).astimezone(timezone.utc)


def _month_end(month: date) -> date:
    next_month = date(
        month.year + (month.month == 12),
        1 if month.month == 12 else month.month + 1,
        1,
    )
    return date.fromordinal(next_month.toordinal() - 1)


def _month_starts_after(start: date, end: date) -> list[date]:
    months: list[date] = []
    cursor = start
    while cursor < end:
        cursor = date(cursor.year + (cursor.month == 12), 1 if cursor.month == 12 else cursor.month + 1, 1)
        months.append(cursor)
    return months


def _apply_operation(
    members: dict[str, str],
    *,
    operation: Mapping[str, Any],
    effective_date: str,
    source_url: str,
    snapshot_month: date,
) -> dict[str, Any]:
    action = str(operation["action"])
    ticker = str(operation["ticker"])
    status = "applied"
    new_ticker: str | None = None

    if action == "add":
        if ticker in members:
            if not operation.get("allow_existing"):
                raise ValueError(f"Cannot add existing constituent {ticker} on {effective_date}.")
            status = "inherited_snapshot_already_applied"
        else:
            members[ticker] = str(operation["name"])
    elif action == "remove":
        if ticker not in members:
            if not operation.get("allow_missing"):
                raise ValueError(f"Cannot remove missing constituent {ticker} on {effective_date}.")
            status = "inherited_snapshot_already_applied"
        else:
            del members[ticker]
    elif action == "ticker_change":
        new_ticker = str(operation["new_ticker"])
        if ticker not in members:
            raise ValueError(f"Cannot rename missing ticker {ticker} on {effective_date}.")
        if new_ticker in members:
            raise ValueError(f"Cannot rename {ticker} to existing ticker {new_ticker}.")
        prior_name = members.pop(ticker)
        members[new_ticker] = str(operation.get("name") or prior_name)
    elif action == "rename":
        if ticker not in members:
            raise ValueError(f"Cannot rename missing constituent {ticker} on {effective_date}.")
        members[ticker] = str(operation["name"])
    else:
        raise ValueError(f"Unsupported constituent operation: {action}")

    return {
        "effective_date": effective_date,
        "snapshot_month": snapshot_month.isoformat(),
        "action": action,
        "ticker": ticker,
        "new_ticker": new_ticker,
        "name": operation.get("name"),
        "status": status,
        "note": operation.get("note"),
        "source_url": source_url,
    }
