from __future__ import annotations

from datetime import date
import hashlib
import json
from pathlib import Path
from typing import Any

import polars as pl


EXTREME_ADJUSTED_RETURN_THRESHOLD = 0.40
SPLIT_RATIO_RELATIVE_TOLERANCE = 0.15
REVIEWED_MOVE_BOUND_COLUMNS = (
    "prior_adjusted_close_min",
    "prior_adjusted_close_max",
    "adjusted_close_min",
    "adjusted_close_max",
    "one_day_return_min",
    "one_day_return_max",
)


def build_split_detection_prices(
    *,
    existing_prices: pl.DataFrame,
    fresh_prices: pl.DataFrame,
    full_history_refresh: bool,
) -> pl.DataFrame:
    """Return one coherent source vintage for corporate-action detection."""

    required = ["ticker", "date", "adjusted_close"]
    if full_history_refresh:
        return fresh_prices.select(
            [column for column in PRICE_QUALITY_COLUMNS if column in fresh_prices.columns]
        ).sort(["ticker", "date"])
    existing = existing_prices.with_columns(pl.lit(0).alias("_fresh_priority"))
    fresh = fresh_prices.with_columns(pl.lit(1).alias("_fresh_priority"))
    return (
        pl.concat([existing, fresh], how="diagonal_relaxed")
        .sort(["ticker", "date", "_fresh_priority"])
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
        .drop("_fresh_priority")
        .sort(["ticker", "date"])
    )


PRICE_QUALITY_COLUMNS = (
    "ticker",
    "date",
    "adjusted_close",
    "close",
    "open",
    "high",
    "low",
    "volume",
)


def find_extreme_adjusted_price_moves(
    prices: pl.DataFrame,
    *,
    event_since: date | str,
    tickers: tuple[str, ...] | list[str] | None = None,
    threshold: float = EXTREME_ADJUSTED_RETURN_THRESHOLD,
) -> pl.DataFrame:
    """Identify recent adjusted-close jumps that require explicit review."""
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be strictly between 0 and 1")
    required = {"ticker", "date", "adjusted_close"}
    missing = required - set(prices.columns)
    if missing:
        raise ValueError(f"Missing price quality columns: {sorted(missing)}")

    normalized_tickers = (
        [str(ticker).upper() for ticker in tickers]
        if tickers is not None
        else None
    )
    selected = prices.select(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
        pl.col("adjusted_close").cast(pl.Float64, strict=False),
    ).filter(pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") > 0))
    if normalized_tickers is not None:
        selected = selected.filter(pl.col("ticker").is_in(normalized_tickers))

    return (
        selected.sort(["ticker", "date"])
        .with_columns(
            pl.col("adjusted_close").shift(1).over("ticker").alias("prior_adjusted_close")
        )
        .with_columns(
            (
                pl.col("adjusted_close") / pl.col("prior_adjusted_close") - 1.0
            ).alias("one_day_return")
        )
        .filter(
            (pl.col("date") >= pl.lit(event_since).cast(pl.Date))
            & (pl.col("one_day_return").abs() >= threshold)
        )
        .select(
            "ticker",
            "date",
            "prior_adjusted_close",
            "adjusted_close",
            "one_day_return",
        )
        .sort(["date", "ticker"])
    )


def assert_no_extreme_adjusted_price_moves(
    prices: pl.DataFrame,
    *,
    event_since: date | str,
    tickers: tuple[str, ...] | list[str] | None = None,
    threshold: float = EXTREME_ADJUSTED_RETURN_THRESHOLD,
    reviewed_moves: pl.DataFrame | None = None,
) -> pl.DataFrame:
    findings = find_extreme_adjusted_price_moves(
        prices,
        event_since=event_since,
        tickers=tickers,
        threshold=threshold,
    )
    unreviewed, reviewed = split_reviewed_extreme_price_moves(
        findings,
        reviewed_moves=reviewed_moves,
    )
    if not unreviewed.is_empty():
        examples = unreviewed.head(10).to_dicts()
        raise RuntimeError(
            "Recent adjusted-close discontinuities require review before publication: "
            f"count={unreviewed.height}, threshold={threshold:.0%}, examples={examples}"
        )
    return reviewed


def load_reviewed_extreme_price_moves(
    path: Path,
) -> tuple[pl.DataFrame, dict[str, Any]]:
    """Load a hash-bound registry of real market moves reviewed outside prices."""

    payload_bytes = path.read_bytes()
    payload = json.loads(payload_bytes)
    if not isinstance(payload, dict) or not str(payload.get("registry_id", "")).strip():
        raise ValueError("Reviewed price-move registry requires registry_id")
    events = payload.get("events")
    if not isinstance(events, list):
        raise ValueError("Reviewed price-move registry requires an events list")
    required = {
        "review_id",
        "ticker",
        "date",
        "known_at",
        "reason",
        "source_urls",
        *REVIEWED_MOVE_BOUND_COLUMNS,
    }
    normalized: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Reviewed price-move events must be JSON objects")
        missing = sorted(required - set(event))
        if missing:
            raise ValueError(f"Reviewed price-move event is missing fields: {missing}")
        source_urls = event["source_urls"]
        if not isinstance(source_urls, list) or not source_urls or not all(
            str(url).startswith("https://") for url in source_urls
        ):
            raise ValueError("Reviewed price-move source_urls must be non-empty HTTPS URLs")
        row = {
            **event,
            "review_id": str(event["review_id"]).strip(),
            "ticker": f"{str(event['ticker']).upper().removesuffix('.US')}.US",
            "date": event["date"],
            "known_at": str(event["known_at"]).strip(),
            "reason": str(event["reason"]).strip(),
            "source_urls": [str(url) for url in source_urls],
        }
        if not row["review_id"] or not row["known_at"] or not row["reason"]:
            raise ValueError("Reviewed price-move identity, known_at, and reason are required")
        for lower, upper in (
            ("prior_adjusted_close_min", "prior_adjusted_close_max"),
            ("adjusted_close_min", "adjusted_close_max"),
            ("one_day_return_min", "one_day_return_max"),
        ):
            row[lower] = float(row[lower])
            row[upper] = float(row[upper])
            if row[lower] > row[upper]:
                raise ValueError(f"Reviewed price-move bounds are inverted: {lower}/{upper}")
        normalized.append(row)
    frame = (
        pl.DataFrame(normalized)
        .with_columns(pl.col("date").cast(pl.Date, strict=False))
        .sort(["ticker", "date"])
        if normalized
        else _empty_reviewed_move_frame()
    )
    if frame.select(pl.struct(["ticker", "date"]).is_duplicated().sum()).item():
        raise ValueError("Reviewed price-move registry has duplicate ticker/date keys")
    return frame, {
        "registry_id": payload["registry_id"],
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload_bytes).hexdigest(),
        "event_count": frame.height,
    }


def split_reviewed_extreme_price_moves(
    findings: pl.DataFrame,
    *,
    reviewed_moves: pl.DataFrame | None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Separate exact, bounded reviewed moves from findings that still block."""

    if findings.is_empty() or reviewed_moves is None or reviewed_moves.is_empty():
        return findings, pl.DataFrame(schema={**findings.schema, "review_id": pl.String})
    required = {"review_id", "ticker", "date", *REVIEWED_MOVE_BOUND_COLUMNS}
    missing = sorted(required - set(reviewed_moves.columns))
    if missing:
        raise ValueError(f"Reviewed price-move frame is missing columns: {missing}")
    registry = reviewed_moves.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date, strict=False),
    )
    if registry.select(pl.struct(["ticker", "date"]).is_duplicated().sum()).item():
        raise ValueError("Reviewed price-move frame has duplicate ticker/date keys")
    joined = findings.join(registry, on=["ticker", "date"], how="left")
    within_bounds = (
        pl.col("review_id").is_not_null()
        & pl.col("prior_adjusted_close").is_between(
            pl.col("prior_adjusted_close_min"),
            pl.col("prior_adjusted_close_max"),
            closed="both",
        )
        & pl.col("adjusted_close").is_between(
            pl.col("adjusted_close_min"),
            pl.col("adjusted_close_max"),
            closed="both",
        )
        & pl.col("one_day_return").is_between(
            pl.col("one_day_return_min"),
            pl.col("one_day_return_max"),
            closed="both",
        )
    )
    finding_columns = findings.columns
    reviewed_columns = [
        *finding_columns,
        "review_id",
        "known_at",
        "reason",
        "source_urls",
    ]
    reviewed = joined.filter(within_bounds).select(reviewed_columns)
    unreviewed = joined.filter(~within_bounds).select(finding_columns)
    return unreviewed, reviewed


def _empty_reviewed_move_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "review_id": pl.String,
            "ticker": pl.String,
            "date": pl.Date,
            "known_at": pl.String,
            "reason": pl.String,
            "source_urls": pl.List(pl.String),
            **{column: pl.Float64 for column in REVIEWED_MOVE_BOUND_COLUMNS},
        }
    )


def repair_confirmed_split_discontinuities(
    prices_delta: pl.DataFrame,
    *,
    findings: pl.DataFrame,
    splits: pl.DataFrame,
    relative_tolerance: float = SPLIT_RATIO_RELATIVE_TOLERANCE,
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    """Back-adjust a fresh source delta only when its split event explains the jump."""

    repaired = prices_delta
    repairs: list[dict[str, object]] = []
    if repaired.is_empty() or findings.is_empty() or splits.is_empty():
        return repaired, repairs

    split_rows = splits.with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("date").cast(pl.Date),
        pl.col("split_ratio").cast(pl.Float64, strict=False),
    )
    for finding in findings.iter_rows(named=True):
        ticker = str(finding["ticker"]).upper()
        event_date = finding["date"]
        observed_multiplier = float(finding["adjusted_close"]) / float(finding["prior_adjusted_close"])
        event = split_rows.filter(
            (pl.col("ticker") == ticker) & (pl.col("date") == pl.lit(event_date))
        )
        if event.height != 1:
            continue
        ratio = float(event.get_column("split_ratio").item())
        expected_multiplier = 1.0 / ratio
        relative_error = abs(observed_multiplier / expected_multiplier - 1.0)
        if ratio <= 0.0 or relative_error > relative_tolerance:
            continue

        before_event = (pl.col("ticker").cast(pl.String).str.to_uppercase() == ticker) & (
            pl.col("date").cast(pl.Date) < pl.lit(event_date)
        )
        price_columns = [
            column
            for column in ("open", "high", "low", "close", "adjusted_close")
            if column in repaired.columns
        ]
        expressions = [
            pl.when(before_event)
            .then(pl.col(column) / ratio)
            .otherwise(pl.col(column))
            .alias(column)
            for column in price_columns
        ]
        if "volume" in repaired.columns:
            expressions.append(
                pl.when(before_event)
                .then(pl.col("volume") * ratio)
                .otherwise(pl.col("volume"))
                .alias("volume")
            )
        repaired = repaired.with_columns(expressions)
        repairs.append(
            {
                "ticker": ticker,
                "date": str(event_date),
                "split_ratio": ratio,
                "observed_multiplier": observed_multiplier,
                "expected_multiplier": expected_multiplier,
                "relative_error": relative_error,
                "source": str(event.get_column("source").item()) if "source" in event.columns else None,
            }
        )
    return repaired, repairs
