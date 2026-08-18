"""Versioned, causal execution conventions for Legacy portfolio orders."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
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
    normalized_dates = daily_prices.select(
        pl.col("ticker").cast(pl.String),
        pl.col("date").cast(pl.Date, strict=False),
    ).filter(pl.col("date").is_not_null())
    for row in normalized_dates.unique().to_dicts():
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
    normalized_prices = daily_prices.with_columns(
        pl.col("ticker").cast(pl.String),
        pl.col("date").cast(pl.Date, strict=False),
    ).filter(pl.col("date").is_not_null())
    for row in normalized_prices.sort(["ticker", "date"]).to_dicts():
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


def apply_next_session_open_holding_returns(
    holdings: pl.DataFrame,
    daily_prices: pl.DataFrame,
    *,
    policy: ExecutionPolicy = LEGACY_NEXT_SESSION_OPEN,
) -> pl.DataFrame:
    """Replace monthly close returns with adjusted-open to month-end returns.

    The entry is the first observed regular-session open strictly after the
    decision cutoff. The exit is the last observed adjusted close inside the
    holding month. Adjustment factors are applied to the entry open so splits
    and distributions remain represented without using future availability to
    select a holding.
    """

    required_holdings = {
        "strategy",
        "decision_month",
        "holding_month",
        "ticker",
        "realized_return",
    }
    required_prices = {"ticker", "date", "open", "close", "adjusted_close"}
    missing_holdings = sorted(required_holdings - set(holdings.columns))
    missing_prices = sorted(required_prices - set(daily_prices.columns))
    if missing_holdings:
        raise ValueError(f"Holdings are missing execution fields: {missing_holdings}")
    if missing_prices:
        raise ValueError(f"Prices are missing execution fields: {missing_prices}")

    normalized_prices = (
        daily_prices.select(
            pl.col("ticker").cast(pl.String),
            pl.col("date").cast(pl.Date, strict=False),
            pl.col("open").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("adjusted_close").cast(pl.Float64),
        )
        .filter(
            pl.col("date").is_not_null()
            & pl.col("open").is_finite()
            & (pl.col("open") > 0.0)
            & pl.col("close").is_finite()
            & (pl.col("close") > 0.0)
            & pl.col("adjusted_close").is_finite()
            & (pl.col("adjusted_close") > 0.0)
        )
        .with_columns(
            (
                pl.col("open") * pl.col("adjusted_close") / pl.col("close")
            ).alias("adjusted_open")
        )
        .sort(["ticker", "date"])
    )
    prices_by_ticker = {
        str(key[0] if isinstance(key, tuple) else key): frame
        for key, frame in normalized_prices.partition_by("ticker", as_dict=True).items()
    }
    market_month_ends = {
        row["holding_month"]: row["scheduled_end_date"]
        for row in normalized_prices.with_columns(
            pl.col("date").dt.truncate("1mo").alias("holding_month")
        )
        .group_by("holding_month")
        .agg(pl.col("date").max().alias("scheduled_end_date"))
        .to_dicts()
    }
    keys = holdings.select(
        "ticker",
        pl.col("decision_month").cast(pl.Date),
        pl.col("holding_month").cast(pl.Date),
    ).unique()
    rows: list[dict[str, object]] = []
    for row in keys.sort(["holding_month", "ticker"]).to_dicts():
        ticker = str(row["ticker"])
        decision_month = row["decision_month"]
        holding_month = row["holding_month"]
        ticker_prices = prices_by_ticker.get(ticker)
        if ticker_prices is None:
            raise RuntimeError(f"No execution price history for {ticker}")
        before_holding = ticker_prices.filter(pl.col("date") < holding_month)
        inside_holding = ticker_prices.filter(
            pl.col("date").dt.truncate("1mo") == holding_month
        )
        if before_holding.is_empty() or inside_holding.is_empty():
            raise RuntimeError(
                f"Incomplete next-open holding window for {ticker} {holding_month}"
            )
        signal_date = before_holding["date"][-1]
        entry = inside_holding.row(0, named=True)
        ending = inside_holding.row(-1, named=True)
        scheduled_end_date = market_month_ends[holding_month]
        is_partial_observation = ending["date"] < scheduled_end_date
        signal_at = datetime.combine(signal_date, time(16, 0), tzinfo=NEW_YORK)
        execution_at = datetime.combine(entry["date"], time(9, 30), tzinfo=NEW_YORK)
        end_at = datetime.combine(ending["date"], time(16, 0), tzinfo=NEW_YORK)
        scheduled_end_at = datetime.combine(
            scheduled_end_date, time(16, 0), tzinfo=NEW_YORK
        )
        if execution_at <= signal_at or end_at <= execution_at:
            raise RuntimeError(
                f"Invalid next-open holding chronology for {ticker} {holding_month}"
            )
        rows.append(
            {
                "ticker": ticker,
                "decision_month": decision_month,
                "holding_month": holding_month,
                "realized_return_next_open": (
                    float(ending["adjusted_close"])
                    / float(entry["adjusted_open"])
                    - 1.0
                ),
                "feature_max_asof_at": signal_at.astimezone(ZoneInfo("UTC")),
                "signal_cutoff_at": signal_at.astimezone(ZoneInfo("UTC")),
                "execution_at": execution_at.astimezone(ZoneInfo("UTC")),
                "first_return_observation_at": (
                    execution_at + timedelta(microseconds=1)
                ).astimezone(ZoneInfo("UTC")),
                "holding_return_end_at": end_at.astimezone(ZoneInfo("UTC")),
                "scheduled_holding_end_at": scheduled_end_at.astimezone(
                    ZoneInfo("UTC")
                ),
                "holding_observation_gap_calendar_days": (
                    scheduled_end_date - ending["date"]
                ).days,
                "execution_policy_id": policy.identifier,
                "return_resolution": (
                    "provisional_last_observation"
                    if is_partial_observation
                    else "observed_market_next_open_to_month_end"
                ),
                "return_resolution_reason": (
                    "ticker_price_series_ended_before_market_month_end"
                    if is_partial_observation
                    else None
                ),
                "manual_review_status": (
                    "pending_manual_terminal_event_review"
                    if is_partial_observation
                    else None
                ),
            }
        )
    resolved = pl.from_dicts(rows)
    return (
        holdings.drop("realized_return")
        .join(
            resolved,
            on=["ticker", "decision_month", "holding_month"],
            how="left",
            validate="m:1",
        )
        .rename({"realized_return_next_open": "realized_return"})
    )


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
