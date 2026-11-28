"""Load reviewed successor prices used to value terminal considerations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

import polars as pl

from alpharank.portfolio.terminal_event_registry import TerminalEventRegistry


DEFAULT_TERMINAL_PRICE_REGISTRY = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "data_quality"
    / "terminal_successor_prices_v1.json"
)
REGISTRY_ID = "terminal_successor_prices_v1"
DEFAULT_SUCCESSOR_OPENING_REGISTRY = DEFAULT_TERMINAL_PRICE_REGISTRY.with_name(
    "terminal_successor_openings_v1.json"
)
REVIEWED_SUCCESSOR_OPENING_POLICY_ID = "reviewed_terminal_successor_openings_v1"
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SOURCE_KINDS = {"sealed_snapshot", "sec_issuer_filing"}


@dataclass(frozen=True)
class TerminalPriceRegistry:
    """Validated registry plus deterministic projection to the return contract."""

    path: Path
    sha256: str
    payload: dict[str, Any]

    @property
    def price_vintage_id(self) -> str:
        return str(self.payload["price_vintage_id"])

    @property
    def observations(self) -> list[dict[str, Any]]:
        return list(self.payload["observations"])

    def successor_prices(
        self,
        daily_prices: pl.DataFrame,
        *,
        snapshot_id: str,
        composition_id: str,
        price_artifact_sha256: str,
    ) -> pl.DataFrame:
        """Build successor prices after matching the sealed snapshot identity."""

        base = self.payload["base_snapshot"]
        observed_identity = {
            "snapshot_id": snapshot_id,
            "composition_id": composition_id,
            "price_artifact_sha256": price_artifact_sha256,
        }
        expected_identity = {
            "snapshot_id": base["snapshot_id"],
            "composition_id": base["composition_id"],
            "price_artifact_sha256": base["price_artifact_sha256"],
        }
        if observed_identity != expected_identity:
            raise ValueError(
                "Terminal successor prices do not match the sealed base snapshot: "
                f"expected={expected_identity!r}, observed={observed_identity!r}."
            )
        required_prices = {"ticker", "date", base["price_column"]}
        missing = sorted(required_prices - set(daily_prices.columns))
        if missing:
            raise ValueError(f"Successor-price source lacks columns: {missing}")
        normalized = daily_prices.select(
            pl.col("ticker").cast(pl.String),
            pl.col("date").cast(pl.String).str.to_date(strict=False),
            pl.col(base["price_column"])
            .cast(pl.Float64, strict=False)
            .alias("source_price"),
        )
        rows: list[dict[str, Any]] = []
        for observation in self.observations:
            price = float(observation["holding_end_price"])
            if observation["source_kind"] == "sealed_snapshot":
                source_row = normalized.filter(
                    (pl.col("ticker") == observation["ticker"])
                    & (pl.col("date") == _parse_date(observation["price_asof_date"]))
                )
                if source_row.height != 1:
                    raise ValueError(
                        "Sealed successor price must match exactly one source row: "
                        f"observation_id={observation['observation_id']!r}."
                    )
                source_price = source_row["source_price"][0]
                if source_price is None or not math.isclose(
                    float(source_price), price, rel_tol=0.0, abs_tol=1e-12
                ):
                    raise ValueError(
                        "Sealed successor price differs from its reviewed value: "
                        f"observation_id={observation['observation_id']!r}."
                    )
            rows.append(
                {
                    "ticker": observation["ticker"],
                    "holding_month": _parse_date(observation["holding_month"]),
                    "price_asof_date": _parse_date(observation["price_asof_date"]),
                    "holding_end_price": price,
                    "price_vintage_id": self.price_vintage_id,
                    "price_observation_id": observation["observation_id"],
                    "price_source_kind": observation["source_kind"],
                    "price_source_reference": observation["source_reference"],
                    "price_registry_sha256": self.sha256,
                }
            )
        return pl.DataFrame(rows, infer_schema_length=None).sort(
            ["holding_month", "ticker"]
        )


def load_terminal_price_registry(
    path: Path = DEFAULT_TERMINAL_PRICE_REGISTRY,
) -> TerminalPriceRegistry:
    """Load the immutable reviewed registry without consulting remote state."""

    resolved = path.resolve()
    raw = resolved.read_bytes()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"Terminal price registry is not valid JSON: {error}") from error
    _validate_registry(payload)
    return TerminalPriceRegistry(
        path=resolved,
        sha256=hashlib.sha256(raw).hexdigest(),
        payload=payload,
    )


def load_reviewed_successor_openings(
    events: TerminalEventRegistry,
    path: Path = DEFAULT_SUCCESSOR_OPENING_REGISTRY,
    *,
    expected_sha256: str | None = None,
) -> pl.DataFrame:
    """Read the reviewed opening keys and bind them to their complete event terms."""
    raw = path.read_bytes()
    registry_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and registry_hash != expected_sha256:
        raise ValueError("Reviewed successor opening registry SHA-256 differs from its recorded source.")
    payload = json.loads(raw)
    if (
        not isinstance(payload, dict)
        or payload.get("policy_id") != REVIEWED_SUCCESSOR_OPENING_POLICY_ID
        or payload.get("price_basis") != "raw_usd_per_contemporaneous_successor_share"
        or not isinstance(payload.get("observations"), list)
        or not payload["observations"]
    ):
        raise ValueError("Invalid reviewed terminal successor opening registry.")
    events_by_id = {event["event_id"]: event for event in events.events}
    reviewed: list[dict[str, object]] = []
    for row in payload["observations"]:
        if not isinstance(row, dict):
            raise ValueError("A reviewed successor opening must be an object.")
        event_id = _non_empty(row.get("event_id"), field="event_id")
        event = events_by_id.get(event_id)
        event_bytes = json.dumps(event, sort_keys=True, separators=(",", ":")).encode("utf-8")
        if event is None or hashlib.sha256(event_bytes).hexdigest() != row.get("event_sha256"):
            raise ValueError(f"Reviewed successor event terms changed: {event_id}.")
        terms = event["portfolio_resolution"]
        exit_date = _parse_date(row.get("exit_date"))
        if _parse_date(event["successor_first_trading_date"]) > exit_date:
            raise ValueError(f"Reviewed opening precedes first successor trade: {event_id}.")
        reviewed.append(
            {
                "exit_event_id": event_id,
                "exit_date": exit_date,
                "successor_ticker": _ticker(terms.get("successor_ticker")),
                "successor_ratio": _numeric_price(terms.get("exchange_ratio"), event_id),
                "reviewed_successor_open": _numeric_price(row.get("open"), event_id),
                "reviewed_successor_volume": _numeric_price(row.get("volume"), event_id),
                "successor_opening_registry_sha256": registry_hash,
            }
        )
    result = pl.DataFrame(reviewed, infer_schema_length=None)
    if result.select("exit_event_id", "exit_date").is_duplicated().any() or result.filter(
        pl.any_horizontal(
            ~pl.col(column).is_finite() | (pl.col(column) <= 0.0)
            for column in ("successor_ratio", "reviewed_successor_open", "reviewed_successor_volume")
        )
    ).height:
        raise ValueError("Invalid or duplicate reviewed successor opening.")
    return result


def verify_terminal_price_source_hashes(
    registry: TerminalPriceRegistry,
    *,
    timeout_seconds: float = 60.0,
    attempts: int = 3,
) -> list[dict[str, str]]:
    """Refetch external price evidence and reject source-body drift."""

    if timeout_seconds <= 0.0 or attempts <= 0:
        raise ValueError("Remote source verification requires positive limits.")
    results: list[dict[str, str]] = []
    for observation in registry.observations:
        for source in observation.get("source_documents", []):
            request = Request(
                source["url"],
                headers={
                    "User-Agent": "AlphaRank methodology audit contact@example.com"
                },
            )
            last_error: Exception | None = None
            for _ in range(attempts):
                try:
                    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310
                        body = response.read()
                    break
                except (OSError, TimeoutError, URLError) as error:
                    last_error = error
            else:
                raise RuntimeError(
                    "Could not refetch terminal price source after "
                    f"{attempts} attempts: source_id={source['source_id']!r}."
                ) from last_error
            observed = hashlib.sha256(body).hexdigest()
            if observed != source["sha256"]:
                raise RuntimeError(
                    "Terminal price source hash drifted: "
                    f"source_id={source['source_id']!r}, "
                    f"expected={source['sha256']}, observed={observed}."
                )
            results.append(
                {
                    "observation_id": observation["observation_id"],
                    "source_id": source["source_id"],
                    "sha256": observed,
                }
            )
    return results


def _validate_registry(payload: object) -> None:
    if not isinstance(payload, dict):
        raise ValueError("Terminal price registry root must be an object.")
    if payload.get("schema_version") != 1 or payload.get("registry_id") != REGISTRY_ID:
        raise ValueError("Unsupported terminal price registry identity.")
    if payload.get("status") != "reviewed_selected_portfolio_successor_prices":
        raise ValueError("Terminal price registry is not reviewed.")
    _parse_datetime(payload.get("retrieved_at"))
    _non_empty(payload.get("price_vintage_id"), field="price_vintage_id")
    base = payload.get("base_snapshot")
    if not isinstance(base, dict):
        raise ValueError("Terminal price registry requires a base snapshot.")
    for field in ("snapshot_id", "composition_id", "price_artifact", "price_column", "price_basis"):
        _non_empty(base.get(field), field=f"base_snapshot.{field}")
    for field in ("composition_id", "price_artifact_sha256"):
        if not SHA256_PATTERN.fullmatch(str(base.get(field, ""))):
            raise ValueError(f"Invalid SHA-256 in base_snapshot.{field}.")
    observations = payload.get("observations")
    if not isinstance(observations, list) or not observations:
        raise ValueError("Terminal price registry requires observations.")
    ids: set[str] = set()
    keys: set[tuple[str, date]] = set()
    source_ids: set[str] = set()
    for observation in observations:
        if not isinstance(observation, dict):
            raise ValueError("Every terminal price observation must be an object.")
        observation_id = _non_empty(
            observation.get("observation_id"), field="observation_id"
        )
        if observation_id in ids:
            raise ValueError(f"Duplicate terminal price observation: {observation_id}")
        ids.add(observation_id)
        ticker = _ticker(observation.get("ticker"))
        holding_month = _parse_date(observation.get("holding_month"))
        price_date = _parse_date(observation.get("price_asof_date"))
        if holding_month.day != 1 or (
            holding_month.year,
            holding_month.month,
        ) != (price_date.year, price_date.month):
            raise ValueError(f"Observation {observation_id} is outside its holding month.")
        key = (ticker, holding_month)
        if key in keys:
            raise ValueError(f"Duplicate successor price key: {key}")
        keys.add(key)
        price = _numeric_price(observation.get("holding_end_price"), observation_id)
        if not math.isfinite(price) or price <= 0.0:
            raise ValueError(f"Observation {observation_id} requires a positive price.")
        if observation.get("currency") != "USD":
            raise ValueError(f"Observation {observation_id} must be denominated in USD.")
        source_kind = observation.get("source_kind")
        if source_kind not in SOURCE_KINDS:
            raise ValueError(f"Observation {observation_id} has unsupported source kind.")
        _non_empty(observation.get("source_reference"), field="source_reference")
        _non_empty(observation.get("review_note"), field="review_note")
        documents = observation.get("source_documents", [])
        if source_kind == "sec_issuer_filing" and not documents:
            raise ValueError(f"Observation {observation_id} requires source documents.")
        for source in documents:
            source_id = _non_empty(source.get("source_id"), field="source_id")
            if source_id in source_ids:
                raise ValueError(f"Duplicate terminal price source id: {source_id}")
            source_ids.add(source_id)
            for field in ("authority", "document_type", "url"):
                _non_empty(source.get(field), field=field)
            _parse_datetime(source.get("published_or_accepted_at"))
            _parse_datetime(source.get("retrieved_at"))
            if not SHA256_PATTERN.fullmatch(str(source.get("sha256", ""))):
                raise ValueError(f"Source {source_id} has invalid SHA-256.")
            supports = source.get("supports")
            if not isinstance(supports, list) or not supports:
                raise ValueError(f"Source {source_id} must state supported facts.")


def _parse_date(value: object) -> date:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return date.fromisoformat(str(value))
    except ValueError as error:
        raise ValueError(f"Invalid terminal successor price date: {value!r}.") from error


def _parse_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"Invalid terminal successor timestamp: {value!r}.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"Invalid terminal successor timestamp: {value!r}.") from error
    if parsed.tzinfo is None:
        raise ValueError("Terminal successor timestamps require an explicit timezone.")
    return parsed.astimezone(timezone.utc)


def _ticker(value: object) -> str:
    ticker = _non_empty(value, field="ticker").upper()
    if not ticker.endswith(".US"):
        raise ValueError(f"Terminal successor ticker must end in .US: {ticker!r}.")
    return ticker


def _numeric_price(value: object, observation_id: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Observation {observation_id} requires a numeric price.")
    return float(value)


def _non_empty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Terminal price field {field} must be a non-empty string.")
    return value.strip()
