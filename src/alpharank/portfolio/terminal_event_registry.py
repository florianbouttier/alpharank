"""Load the reviewed shareholder-event registry and fail closed on ambiguity."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date, datetime, time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import polars as pl

DEFAULT_TERMINAL_EVENT_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_quality"
    / "terminal_shareholder_events_v2.json"
)
REGISTRY_ID = "terminal_shareholder_events_v2"
SUPPORTED_REGISTRY_IDS = {
    "terminal_shareholder_events_v1",
    REGISTRY_ID,
}
TERMINAL_EVENT_TYPES = {
    "cash_merger",
    "stock_merger",
    "stock_and_cash_merger",
}
PRE_EXECUTION_EVENT_TYPES = {"receivership_and_pre_open_suspension"}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class TerminalEventRegistry:
    """Validated source registry plus deterministic runtime projections."""

    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def events(self) -> list[dict[str, Any]]:
        return list(self.payload["events"])

    def terminal_consideration_events(
        self, *, price_vintage_id: str
    ) -> pl.DataFrame:
        """Project reviewed considerations onto the shared terminal-return schema."""

        vintage = str(price_vintage_id).strip()
        if not vintage:
            raise ValueError("Terminal events require a non-empty price vintage id.")
        rows: list[dict[str, Any]] = []
        for event in self.events:
            resolution = event["portfolio_resolution"]
            if resolution["mode"] != "terminal_consideration":
                continue
            event_type = event["event_type"]
            runtime_type = (
                "stock_merger"
                if event_type == "stock_and_cash_merger"
                else event_type
            )
            primary_source = event["source_documents"][0]
            rows.append(
                {
                    "terminal_event_id": event["event_id"],
                    "event_type": runtime_type,
                    "ticker": event["ticker"],
                    "successor_ticker": resolution["successor_ticker"],
                    "effective_date": _parse_date(
                        event["effective_date"], field="effective_date"
                    ),
                    "known_at": _parse_datetime(
                        event["known_at"], field="known_at"
                    ),
                    "price_vintage_id": vintage,
                    "cash_per_share": resolution["cash_per_share"],
                    "recovery_per_share": resolution["recovery_per_share"],
                    "exchange_ratio": resolution["exchange_ratio"],
                    "distribution_per_share": sum(
                        float(item["amount_per_share"])
                        for item in resolution["distributions"]
                    ),
                    "source": (
                        f"{self.payload['registry_id']}:{primary_source['source_id']}"
                    ),
                    "source_url": primary_source["url"],
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            ["ticker", "effective_date"]
        )

    def pre_execution_blocks(self) -> pl.DataFrame:
        """Return events that forbid a regular-session fill after public notice."""

        rows = []
        for event in self.events:
            resolution = event["portfolio_resolution"]
            if resolution["mode"] != "pre_execution_trading_suspension":
                continue
            rows.append(
                {
                    "terminal_event_id": event["event_id"],
                    "ticker": event["ticker"],
                    "effective_date": _parse_date(
                        event["effective_date"], field="effective_date"
                    ),
                    "known_at": _parse_datetime(
                        event["known_at"], field="known_at"
                    ),
                    "last_primary_trading_date": _parse_date(
                        event["last_primary_trading_date"],
                        field="last_primary_trading_date",
                    ),
                    "entry_allowed": False,
                    "reason": event["event_type"],
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            ["ticker", "effective_date"]
        )

    def terminal_entry_blocks(self) -> pl.DataFrame:
        """Return the first holding month in which each event forbids entry.

        A merger or acquisition that occurs during an already-open holding
        month is resolved through shareholder consideration. It blocks new
        entry from the following month. When the final primary-market session
        was already in the preceding month, the security cannot be bought in
        the effective month and is blocked immediately. A pre-open trading
        suspension follows the same effective-month rule.
        """

        rows: list[dict[str, Any]] = []
        for event in self.events:
            resolution = event["portfolio_resolution"]
            effective_date = _parse_date(
                event["effective_date"], field="effective_date"
            )
            last_primary_trading_date = _parse_date(
                event["last_primary_trading_date"],
                field="last_primary_trading_date",
            )
            effective_month = effective_date.replace(day=1)
            if resolution["mode"] == "pre_execution_trading_suspension":
                blocked_from_holding_month = effective_month
                rule = "pre_open_suspension_blocks_effective_month"
            elif resolution["mode"] == "post_terminal_entry_block":
                blocked_from_holding_month = (
                    effective_month
                    if last_primary_trading_date < effective_month
                    else _next_month(effective_month)
                )
                rule = "completed_terminal_event_blocks_post_event_entry"
            elif last_primary_trading_date < effective_month:
                blocked_from_holding_month = effective_month
                rule = "no_primary_session_blocks_effective_month"
            else:
                blocked_from_holding_month = _next_month(effective_month)
                rule = "terminal_consideration_blocks_following_month"
            rows.append(
                {
                    "terminal_event_id": event["event_id"],
                    "ticker": event["ticker"],
                    "effective_date": effective_date,
                    "known_at": _parse_datetime(
                        event["known_at"], field="known_at"
                    ),
                    "blocked_from_holding_month": blocked_from_holding_month,
                    "entry_allowed": False,
                    "entry_block_rule": rule,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            ["ticker", "blocked_from_holding_month"]
        )


def load_terminal_event_registry(
    path: Path = DEFAULT_TERMINAL_EVENT_REGISTRY,
) -> TerminalEventRegistry:
    """Read and validate the reviewed JSON without consulting mutable web state."""

    resolved = path.resolve()
    raw = resolved.read_bytes()
    try:
        source_payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Terminal event registry is not valid JSON: {error}") from error
    payload = _resolve_registry_extension(resolved, source_payload)
    _validate_registry(payload)
    return TerminalEventRegistry(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )


def _resolve_registry_extension(path: Path, payload: object) -> object:
    if not isinstance(payload, dict) or "extends_registry_path" not in payload:
        return payload
    base_name = payload.get("extends_registry_path")
    expected_id = payload.get("extends_registry_id")
    expected_sha256 = payload.get("extends_registry_sha256")
    if not isinstance(base_name, str) or Path(base_name).name != base_name:
        raise ValueError("Terminal event registry extension must name a sibling file.")
    base_path = path.parent / base_name
    base_raw = base_path.read_bytes()
    observed_sha256 = hashlib.sha256(base_raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("Terminal event registry extension hash does not match its base.")
    try:
        base_payload = json.loads(base_raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Base terminal event registry is not valid JSON: {error}") from error
    _validate_registry(base_payload)
    if base_payload.get("registry_id") != expected_id:
        raise ValueError("Terminal event registry extension id does not match its base.")
    merged = {**base_payload, **payload}
    merged["events"] = [*base_payload["events"], *payload.get("events", [])]
    return merged


def _next_month(value: date) -> date:
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def verify_terminal_event_source_hashes(
    registry: TerminalEventRegistry,
    *,
    timeout_seconds: float = 60.0,
    attempts: int = 3,
) -> list[dict[str, str]]:
    """Refetch primary evidence and reject any body that differs from review."""

    if timeout_seconds <= 0.0 or attempts <= 0:
        raise ValueError("Remote source verification requires positive limits.")
    results: list[dict[str, str]] = []
    for event in registry.events:
        for source in event["source_documents"]:
            request = Request(
                source["url"],
                headers={
                    "User-Agent": "AlphaRank methodology audit contact@example.com"
                },
            )
            last_error: Exception | None = None
            for _ in range(attempts):
                try:
                    with urlopen(  # noqa: S310
                        request, timeout=timeout_seconds
                    ) as response:
                        body = response.read()
                    break
                except (OSError, TimeoutError, URLError) as error:
                    last_error = error
            else:
                raise RuntimeError(
                    "Could not refetch terminal event source after "
                    f"{attempts} attempts: source_id={source['source_id']!r}."
                ) from last_error
            observed = hashlib.sha256(body).hexdigest()
            expected = source["sha256"]
            if observed != expected:
                raise RuntimeError(
                    "Terminal event source hash drifted: "
                    f"source_id={source['source_id']!r}, expected={expected}, "
                    f"observed={observed}."
                )
            results.append(
                {
                    "event_id": event["event_id"],
                    "source_id": source["source_id"],
                    "sha256": observed,
                }
            )
    return results


def _validate_registry(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Terminal event registry root must be an object.")
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported terminal event registry schema version.")
    if payload.get("registry_id") not in SUPPORTED_REGISTRY_IDS:
        raise ValueError("Unexpected terminal event registry id.")
    if payload.get("status") != "reviewed_selected_portfolio_events":
        raise ValueError("Terminal event registry is not in reviewed status.")
    _parse_datetime(payload.get("retrieved_at"), field="retrieved_at")
    events = payload.get("events")
    if not isinstance(events, list) or not events:
        raise ValueError("Terminal event registry must contain events.")

    event_ids: set[str] = set()
    ticker_dates: set[tuple[str, date]] = set()
    source_ids: set[str] = set()
    for event in events:
        _validate_event(
            event,
            event_ids=event_ids,
            ticker_dates=ticker_dates,
            source_ids=source_ids,
        )


def _validate_event(
    event: object,
    *,
    event_ids: set[str],
    ticker_dates: set[tuple[str, date]],
    source_ids: set[str],
) -> None:
    if not isinstance(event, dict):
        raise ValueError("Every terminal event must be an object.")
    required = {
        "event_id",
        "ticker",
        "issuer_name",
        "event_type",
        "effective_date",
        "known_at",
        "known_at_basis",
        "last_primary_trading_date",
        "successor_first_trading_date",
        "portfolio_resolution",
        "source_documents",
        "review_note",
    }
    missing = sorted(required - set(event))
    if missing:
        raise ValueError(f"Terminal event lacks fields: {missing}")
    event_id = _non_empty_string(event["event_id"], field="event_id")
    if event_id in event_ids:
        raise ValueError(f"Duplicate terminal event id: {event_id}")
    event_ids.add(event_id)
    ticker = _validate_ticker(event["ticker"], field="ticker")
    effective = _parse_date(event["effective_date"], field="effective_date")
    key = (ticker, effective)
    if key in ticker_dates:
        raise ValueError(f"Duplicate terminal event ticker/date: {key}")
    ticker_dates.add(key)
    known_at = _parse_datetime(event["known_at"], field="known_at")
    last_trade = _parse_date(
        event["last_primary_trading_date"], field="last_primary_trading_date"
    )
    if last_trade > effective:
        raise ValueError(f"Terminal event {event_id} trades after its effective date.")
    successor_first = event["successor_first_trading_date"]
    if successor_first is not None:
        successor_first_date = _parse_date(
            successor_first, field="successor_first_trading_date"
        )
        if successor_first_date < effective:
            raise ValueError(
                f"Terminal event {event_id} successor trades before effectiveness."
            )
    _non_empty_string(event["issuer_name"], field="issuer_name")
    _non_empty_string(event["known_at_basis"], field="known_at_basis")
    _non_empty_string(event["review_note"], field="review_note")
    event_type = _non_empty_string(event["event_type"], field="event_type")
    resolution = event["portfolio_resolution"]
    if not isinstance(resolution, dict):
        raise ValueError(f"Terminal event {event_id} resolution must be an object.")
    _validate_event_resolution(
        event_id,
        event_type,
        resolution,
        effective=effective,
        known_at=known_at,
        last_trade=last_trade,
    )
    _validate_sources(
        event_id,
        event["source_documents"],
        known_at=known_at,
        source_ids=source_ids,
    )


def _validate_event_resolution(
    event_id: str,
    event_type: str,
    resolution: dict[str, Any],
    *,
    effective: date,
    known_at: datetime,
    last_trade: date,
) -> None:
    if resolution.get("mode") == "post_terminal_entry_block":
        _validate_post_terminal_entry_block(event_id, resolution)
    elif event_type in TERMINAL_EVENT_TYPES:
        _validate_terminal_consideration(event_id, event_type, resolution)
    elif event_type in PRE_EXECUTION_EVENT_TYPES:
        _validate_pre_execution_block(
            event_id,
            resolution,
            effective=effective,
            known_at=known_at,
            last_trade=last_trade,
        )
    else:
        raise ValueError(f"Unsupported terminal event type: {event_type}")


def _validate_terminal_consideration(
    event_id: str, event_type: str, resolution: dict[str, Any]
) -> None:
    _require_resolution_fields(event_id, resolution)
    if resolution["mode"] != "terminal_consideration":
        raise ValueError(f"Terminal event {event_id} has an incompatible mode.")
    if resolution["allow_entry"] is not True:
        raise ValueError(f"Terminal event {event_id} must preserve a valid entry.")
    if resolution["shareholder_consideration_status"] != "complete":
        raise ValueError(f"Terminal event {event_id} consideration is incomplete.")
    cash = _optional_non_negative(
        resolution["cash_per_share"], field="cash_per_share", event_id=event_id
    )
    recovery = resolution["recovery_per_share"]
    if recovery is not None:
        raise ValueError(f"Merger event {event_id} cannot declare recovery_per_share.")
    successor = resolution["successor_ticker"]
    ratio = resolution["exchange_ratio"]
    needs_successor = resolution["requires_successor_holding_end_price"]
    if event_type == "cash_merger":
        if cash is None or cash <= 0.0:
            raise ValueError(f"Cash merger {event_id} requires positive cash.")
        if successor is not None or ratio is not None or needs_successor is not False:
            raise ValueError(f"Cash merger {event_id} cannot require a successor.")
    else:
        _validate_ticker(successor, field="successor_ticker")
        ratio_value = _positive(ratio, field="exchange_ratio", event_id=event_id)
        if ratio_value <= 0.0 or needs_successor is not True:
            raise ValueError(f"Stock merger {event_id} requires successor valuation.")
        if cash is None:
            raise ValueError(f"Stock merger {event_id} requires explicit cash, including zero.")
        if event_type == "stock_and_cash_merger" and cash <= 0.0:
            distributions = resolution.get("distributions", [])
            if not any(float(item.get("amount_per_share", 0.0)) > 0 for item in distributions):
                raise ValueError(f"Mixed merger {event_id} lacks its cash component.")
    distributions = resolution["distributions"]
    if not isinstance(distributions, list):
        raise ValueError(f"Terminal event {event_id} distributions must be a list.")
    for distribution in distributions:
        if not isinstance(distribution, dict):
            raise ValueError(f"Terminal event {event_id} has an invalid distribution.")
        _non_empty_string(distribution.get("kind"), field="distribution.kind")
        _non_negative(
            distribution.get("amount_per_share"),
            field="distribution.amount_per_share",
            event_id=event_id,
        )
        record = distribution.get("record_date")
        payment = distribution.get("payment_date")
        record_date = _parse_date(record, field="distribution.record_date")
        if payment is not None and _parse_date(
            payment, field="distribution.payment_date"
        ) < record_date:
            raise ValueError(f"Terminal event {event_id} pays before record date.")


def _validate_pre_execution_block(
    event_id: str,
    resolution: dict[str, Any],
    *,
    effective: date,
    known_at: datetime,
    last_trade: date,
) -> None:
    _require_resolution_fields(event_id, resolution)
    if resolution["mode"] != "pre_execution_trading_suspension":
        raise ValueError(f"Pre-execution event {event_id} has an incompatible mode.")
    if resolution["allow_entry"] is not False:
        raise ValueError(f"Pre-execution event {event_id} must reject the fill.")
    if resolution["shareholder_consideration_status"] != "not_applicable_no_execution":
        raise ValueError(f"Pre-execution event {event_id} cannot invent consideration.")
    consideration_fields = (
        "cash_per_share",
        "successor_ticker",
        "exchange_ratio",
        "recovery_per_share",
    )
    if any(resolution[field] is not None for field in consideration_fields):
        raise ValueError(f"Pre-execution event {event_id} contains consideration.")
    if resolution["requires_successor_holding_end_price"] is not False:
        raise ValueError(f"Pre-execution event {event_id} cannot require a successor.")
    if resolution["distributions"] != []:
        raise ValueError(f"Pre-execution event {event_id} cannot contain distributions.")
    regular_open = datetime.combine(effective, time(9, 30), tzinfo=NEW_YORK)
    if known_at.astimezone(NEW_YORK) >= regular_open:
        raise ValueError(f"Pre-execution event {event_id} was not known before open.")
    if last_trade >= effective:
        raise ValueError(f"Pre-execution event {event_id} lacks a prior final session.")


def _validate_post_terminal_entry_block(
    event_id: str,
    resolution: dict[str, Any],
) -> None:
    _require_resolution_fields(event_id, resolution)
    if resolution["allow_entry"] is not False:
        raise ValueError(f"Post-terminal event {event_id} must reject later entry.")
    if resolution["shareholder_consideration_status"] != "not_evaluated_entry_only":
        raise ValueError(f"Post-terminal event {event_id} must remain entry-only.")
    consideration_fields = (
        "cash_per_share",
        "successor_ticker",
        "exchange_ratio",
        "recovery_per_share",
    )
    if any(resolution[field] is not None for field in consideration_fields):
        raise ValueError(f"Post-terminal event {event_id} contains consideration.")
    if resolution["requires_successor_holding_end_price"] is not False:
        raise ValueError(f"Post-terminal event {event_id} cannot require a successor.")
    if resolution["distributions"] != []:
        raise ValueError(f"Post-terminal event {event_id} cannot contain distributions.")


def _validate_sources(
    event_id: str,
    sources: object,
    *,
    known_at: datetime,
    source_ids: set[str],
) -> None:
    if not isinstance(sources, list) or not sources:
        raise ValueError(f"Terminal event {event_id} requires primary evidence.")
    published_times: list[datetime] = []
    for source in sources:
        if not isinstance(source, dict):
            raise ValueError(f"Terminal event {event_id} source must be an object.")
        required = {
            "source_id",
            "authority",
            "document_type",
            "url",
            "published_or_accepted_at",
            "retrieved_at",
            "sha256",
            "supports",
        }
        missing = sorted(required - set(source))
        if missing:
            raise ValueError(f"Terminal event {event_id} source lacks fields: {missing}")
        source_id = _non_empty_string(source["source_id"], field="source_id")
        if source_id in source_ids:
            raise ValueError(f"Duplicate terminal source id: {source_id}")
        source_ids.add(source_id)
        _non_empty_string(source["authority"], field="source.authority")
        _non_empty_string(source["document_type"], field="source.document_type")
        url = _non_empty_string(source["url"], field="source.url")
        if not url.startswith("https://"):
            raise ValueError(f"Terminal source {source_id} must use HTTPS.")
        published_times.append(
            _parse_datetime(
                source["published_or_accepted_at"],
                field="source.published_or_accepted_at",
            )
        )
        _parse_datetime(source["retrieved_at"], field="source.retrieved_at")
        digest = _non_empty_string(source["sha256"], field="source.sha256")
        if SHA256_PATTERN.fullmatch(digest) is None:
            raise ValueError(f"Terminal source {source_id} has an invalid SHA-256.")
        supports = source["supports"]
        if not isinstance(supports, list) or not supports:
            raise ValueError(f"Terminal source {source_id} must state supported facts.")
        for fact in supports:
            _non_empty_string(fact, field="source.supports")
    if all(value != known_at for value in published_times):
        raise ValueError(
            f"Terminal event {event_id} known_at is not anchored to a source timestamp."
        )


def _require_resolution_fields(event_id: str, resolution: dict[str, Any]) -> None:
    required = {
        "mode",
        "allow_entry",
        "cash_per_share",
        "successor_ticker",
        "exchange_ratio",
        "recovery_per_share",
        "requires_successor_holding_end_price",
        "distributions",
        "shareholder_consideration_status",
    }
    missing = sorted(required - set(resolution))
    if missing:
        raise ValueError(f"Terminal event {event_id} resolution lacks fields: {missing}")


def _validate_ticker(value: object, *, field: str) -> str:
    ticker = _non_empty_string(value, field=field)
    if ticker != ticker.upper() or not ticker.endswith(".US"):
        raise ValueError(f"{field} must be an uppercase normalized .US ticker.")
    return ticker


def _parse_date(value: object, *, field: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO date string.")
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO date string.") from error


def _parse_datetime(value: object, *, field: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be an ISO timestamp string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{field} must be an ISO timestamp string.") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must include a timezone.")
    return parsed


def _non_empty_string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _optional_non_negative(
    value: object, *, field: str, event_id: str
) -> float | None:
    if value is None:
        return None
    return _non_negative(value, field=field, event_id=event_id)


def _non_negative(value: object, *, field: str, event_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Terminal event {event_id} has invalid {field}.")
    numeric = float(value)
    if numeric < 0.0 or numeric != numeric or numeric in {float("inf"), float("-inf")}:
        raise ValueError(f"Terminal event {event_id} has invalid {field}.")
    return numeric


def _positive(value: object, *, field: str, event_id: str) -> float:
    numeric = _non_negative(value, field=field, event_id=event_id)
    if numeric <= 0.0:
        raise ValueError(f"Terminal event {event_id} requires positive {field}.")
    return numeric
