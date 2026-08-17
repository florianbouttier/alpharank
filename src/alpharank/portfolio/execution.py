"""Versioned, causal execution conventions for Legacy portfolio orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl


NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class ExecutionPolicy:
    identifier: str
    canonical_scenario: str
    signal_cutoff_rule: str
    execution_rule: str

    def to_manifest(self) -> dict[str, str]:
        return asdict(self)


LEGACY_NEXT_SESSION_OPEN = ExecutionPolicy(
    identifier="next_session_open_v1",
    canonical_scenario="next_session_open",
    signal_cutoff_rule="decision timestamp is the final information boundary",
    execution_rule="first observed regular-session open strictly after signal cutoff",
)

EXECUTION_SCENARIOS = (
    "signal_close_reference",
    "next_session_open",
    "observed_session_vwap",
)


def build_monthly_execution_orders(
    holdings: pl.DataFrame,
    daily_prices: pl.DataFrame,
) -> pl.DataFrame:
    """Derive each signal cutoff from the last observed session before holding."""

    required_holdings = {"portfolio_model", "year_month", "ticker"}
    missing = sorted(required_holdings - set(holdings.columns))
    if missing:
        raise ValueError(f"Legacy holdings are missing execution fields: {missing}")
    price_dates: dict[str, list[date]] = {}
    for row in daily_prices.select("ticker", "date").unique().to_dicts():
        value = row["date"]
        if isinstance(value, datetime):
            value = value.date()
        if isinstance(value, date):
            price_dates.setdefault(str(row["ticker"]), []).append(value)
    for values in price_dates.values():
        values.sort()

    rows: list[dict[str, object]] = []
    keys = holdings.select(
        "portfolio_model",
        pl.col("year_month").cast(pl.Date, strict=False),
        "ticker",
    ).unique()
    for row in keys.to_dicts():
        holding_month = row["year_month"]
        candidates = [
            value
            for value in price_dates.get(str(row["ticker"]), [])
            if value < holding_month
        ]
        if not candidates:
            raise RuntimeError(
                f"No pre-holding signal session for {row['ticker']} {holding_month}"
            )
        signal_date = candidates[-1]
        cutoff = datetime.combine(signal_date, time(16, 0), tzinfo=NEW_YORK)
        rows.append(
            {
                "order_id": (
                    f"{row['portfolio_model']}|{holding_month}|{row['ticker']}"
                ),
                "ticker": row["ticker"],
                "signal_cutoff_at": cutoff.astimezone(ZoneInfo("UTC")),
            }
        )
    return pl.from_dicts(rows).sort("order_id")


def build_execution_sensitivity_report(
    orders: pl.DataFrame,
    daily_prices: pl.DataFrame,
    *,
    policy: ExecutionPolicy = LEGACY_NEXT_SESSION_OPEN,
) -> pl.DataFrame:
    """Build mandatory close/open/VWAP rows without inventing unavailable prices."""

    required_orders = {"order_id", "ticker", "signal_cutoff_at"}
    required_prices = {"ticker", "date", "open", "close"}
    missing_orders = sorted(required_orders - set(orders.columns))
    missing_prices = sorted(required_prices - set(daily_prices.columns))
    if missing_orders:
        raise ValueError(f"Orders are missing execution fields: {missing_orders}")
    if missing_prices:
        raise ValueError(f"Prices are missing execution fields: {missing_prices}")

    by_ticker: dict[str, list[dict[str, object]]] = {}
    for row in daily_prices.sort(["ticker", "date"]).to_dicts():
        by_ticker.setdefault(str(row["ticker"]), []).append(row)

    output: list[dict[str, object]] = []
    for order in orders.to_dicts():
        cutoff = order["signal_cutoff_at"]
        if not isinstance(cutoff, datetime) or cutoff.tzinfo is None:
            raise ValueError("signal_cutoff_at must contain timezone-aware datetimes.")
        ticker = str(order["ticker"])
        sessions = [_session_record(row) for row in by_ticker.get(ticker, [])]
        prior = [row for row in sessions if row["close_at"] <= cutoff]
        future = [row for row in sessions if row["open_at"] > cutoff]
        signal_close = prior[-1] if prior else None
        next_session = future[0] if future else None
        vwap_session = next(
            (row for row in future if row["vwap"] is not None),
            None,
        )
        output.extend(
            [
                _scenario_row(
                    order,
                    policy,
                    scenario="signal_close_reference",
                    price=(signal_close or {}).get("close"),
                    execution_at=(signal_close or {}).get("close_at"),
                    status="reference_only_not_after_signal",
                ),
                _scenario_row(
                    order,
                    policy,
                    scenario="next_session_open",
                    price=(next_session or {}).get("open"),
                    execution_at=(next_session or {}).get("open_at"),
                    status="available" if next_session else "unavailable_no_future_open",
                ),
                _scenario_row(
                    order,
                    policy,
                    scenario="observed_session_vwap",
                    price=(vwap_session or {}).get("vwap"),
                    execution_at=(vwap_session or {}).get("close_at"),
                    status="available" if vwap_session else "unavailable_no_observed_vwap",
                ),
            ]
        )
    report = pl.from_dicts(output).with_columns(
        pl.col("signal_cutoff_at").cast(pl.Datetime(time_zone="UTC")),
        pl.col("execution_at").cast(pl.Datetime(time_zone="UTC")),
        pl.col("price").cast(pl.Float64),
    )
    validate_execution_sensitivity_report(report, policy=policy)
    return report.sort(["order_id", "scenario"])


def validate_execution_sensitivity_report(
    report: pl.DataFrame,
    *,
    policy: ExecutionPolicy = LEGACY_NEXT_SESSION_OPEN,
) -> None:
    """Require every scenario and a causal canonical execution for every order."""

    expected = set(EXECUTION_SCENARIOS)
    for order_key, group in report.partition_by("order_id", as_dict=True).items():
        scenarios = set(group["scenario"].to_list())
        if scenarios != expected:
            raise RuntimeError(
                f"Execution sensitivity is incomplete for {order_key}: {scenarios}"
            )
    canonical = report.filter(pl.col("scenario") == policy.canonical_scenario)
    invalid = canonical.filter(
        (pl.col("status") != "available")
        | pl.col("execution_at").is_null()
        | (pl.col("execution_at") <= pl.col("signal_cutoff_at"))
    )
    if canonical.is_empty() or not invalid.is_empty():
        raise RuntimeError("Canonical execution is unavailable or not after the signal.")


def write_execution_sensitivity_report(
    report: pl.DataFrame,
    output_dir: Path,
    *,
    policy: ExecutionPolicy = LEGACY_NEXT_SESSION_OPEN,
) -> dict[str, object]:
    """Persist the mandatory sensitivity table and its versioned policy."""

    validate_execution_sensitivity_report(report, policy=policy)
    output_dir.mkdir(parents=True, exist_ok=True)
    report.write_parquet(output_dir / "legacy_execution_sensitivity.parquet")
    manifest = {
        "execution_policy": policy.to_manifest(),
        "scenario_count": len(EXECUTION_SCENARIOS),
        "order_count": report["order_id"].n_unique(),
        "row_count": report.height,
    }
    (output_dir / "legacy_execution_policy.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def _session_record(row: dict[str, object]) -> dict[str, object]:
    session_date = row["date"]
    if isinstance(session_date, datetime):
        session_date = session_date.date()
    if not isinstance(session_date, date):
        raise ValueError("Execution price dates must be valid dates.")
    open_at = datetime.combine(session_date, time(9, 30), tzinfo=NEW_YORK)
    close_at = datetime.combine(session_date, time(16, 0), tzinfo=NEW_YORK)
    return {
        **row,
        "open_at": open_at.astimezone(ZoneInfo("UTC")),
        "close_at": close_at.astimezone(ZoneInfo("UTC")),
        "vwap": row.get("vwap"),
    }


def _scenario_row(
    order: dict[str, object],
    policy: ExecutionPolicy,
    *,
    scenario: str,
    price: object,
    execution_at: object,
    status: str,
) -> dict[str, object]:
    cutoff = order["signal_cutoff_at"]
    return {
        "order_id": order["order_id"],
        "ticker": order["ticker"],
        "signal_cutoff_at": cutoff,
        "scenario": scenario,
        "price": price,
        "execution_at": execution_at,
        "execution_after_signal_cutoff": (
            bool(execution_at > cutoff) if execution_at is not None else False
        ),
        "status": status,
        "is_canonical": scenario == policy.canonical_scenario,
        "execution_policy_id": policy.identifier,
    }
