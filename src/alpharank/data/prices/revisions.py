"""Immutable, knowledge-dated price revision packages.

This module does not mutate a published price vintage.  It compares a proposed
replacement with the published lineage, requires every historical value change
to be covered by reviewed evidence, and returns a new package plus its complete
row-level diff.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
from typing import Any

import polars as pl

from alpharank.data.prices.contracts import PRICE_LINEAGE_COLUMNS, PRICE_VALUE_COLUMNS


PRICE_REVISION_TYPES = frozenset(
    {"stock_split", "cash_dividend", "vendor_correction"}
)
PRICE_REVISION_EVENT_COLUMNS = (
    "revision_id",
    "revision_type",
    "ticker",
    "effective_date",
    "known_at",
    "affected_from",
    "affected_through",
    "source",
    "source_url",
    "reason",
)
_KEY_COLUMNS = ("ticker", "date")
_REVISED_VALUE_COLUMNS = tuple(
    column for column in PRICE_VALUE_COLUMNS if column not in _KEY_COLUMNS
)


@dataclass(frozen=True)
class PriceRevisionPackage:
    """A new immutable price vintage and the evidence needed to audit it."""

    prices: pl.DataFrame
    lineage: pl.DataFrame
    revision_diff: pl.DataFrame
    report: dict[str, Any]


def build_price_revision_package(
    *,
    previous_lineage: pl.DataFrame,
    candidate_lineage: pl.DataFrame,
    revision_events: pl.DataFrame,
    previous_vintage_id: str,
    new_vintage_id: str,
    package_known_at: str | datetime,
) -> PriceRevisionPackage:
    """Validate historical corrections and materialize a distinct price vintage.

    New date keys are allowed because a new vintage may extend history. Removed
    keys are rejected. Changes on an existing key require exactly one reviewed
    split, dividend, or vendor-correction event whose evidence was known by the
    package timestamp and whose affected range covers the row.
    """

    old_id = str(previous_vintage_id).strip()
    new_id = str(new_vintage_id).strip()
    if not old_id or not new_id:
        raise ValueError("Price package vintage identifiers must be non-empty.")
    if old_id == new_id:
        raise ValueError(
            "Historical price revisions require a new package vintage; "
            "the canonical vintage cannot be overwritten."
        )

    known_at = _parse_datetime(package_known_at, field="package_known_at")
    previous = _normalize_lineage(previous_lineage, label="previous lineage")
    candidate = _normalize_lineage(candidate_lineage, label="candidate lineage")
    events = _normalize_revision_events(revision_events)

    previous_keys = previous.select(_KEY_COLUMNS)
    candidate_keys = candidate.select(_KEY_COLUMNS)
    removed = previous_keys.join(candidate_keys, on=_KEY_COLUMNS, how="anti")
    if not removed.is_empty():
        raise ValueError(
            "A revised price vintage cannot remove published ticker/date keys; "
            f"removed_rows={removed.height}."
        )
    added = candidate_keys.join(previous_keys, on=_KEY_COLUMNS, how="anti")
    diff = _build_revision_diff(previous, candidate)
    if diff.is_empty():
        raise ValueError(
            "A price revision package must contain at least one historical value change."
        )

    event_rows = events.to_dicts()
    assignments: list[dict[str, Any]] = []
    for row in diff.select(_KEY_COLUMNS).to_dicts():
        matches = [
            event
            for event in event_rows
            if event["ticker"] == row["ticker"]
            and event["affected_from"] <= row["date"] <= event["affected_through"]
        ]
        if not matches:
            raise ValueError(
                "Historical price change lacks reviewed revision evidence: "
                f"ticker={row['ticker']}, date={row['date']}."
            )
        if len(matches) != 1:
            raise ValueError(
                "Historical price change matches multiple revision events: "
                f"ticker={row['ticker']}, date={row['date']}."
            )
        event = matches[0]
        if event["known_at"] > known_at:
            raise ValueError(
                "Price revision evidence was not known at package creation: "
                f"revision_id={event['revision_id']}."
            )
        assignments.append(
            {
                "ticker": row["ticker"],
                "date": row["date"],
                "revision_id": event["revision_id"],
                "revision_type": event["revision_type"],
                "revision_effective_date": event["effective_date"],
                "revision_known_at": event["known_at"],
                "affected_from": event["affected_from"],
                "affected_through": event["affected_through"],
                "revision_source": event["source"],
                "revision_source_url": event["source_url"],
                "revision_reason": event["reason"],
            }
        )

    assignment_frame = pl.DataFrame(assignments).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("revision_effective_date").cast(pl.Date),
        pl.col("revision_known_at").cast(pl.Datetime(time_zone="UTC")),
        pl.col("affected_from").cast(pl.Date),
        pl.col("affected_through").cast(pl.Date),
    )
    diff = diff.join(assignment_frame, on=_KEY_COLUMNS, how="left")
    revised_lineage = (
        candidate.join(
            assignment_frame.select(*_KEY_COLUMNS, "revision_id"),
            on=_KEY_COLUMNS,
            how="left",
        )
        .with_columns(
            pl.coalesce("revision_id", "correction_overlay_id").alias(
                "correction_overlay_id"
            )
        )
        .drop("revision_id")
        .select(PRICE_LINEAGE_COLUMNS)
        .sort(_KEY_COLUMNS)
    )

    previous_sha256 = _frame_sha256(previous)
    candidate_sha256 = _frame_sha256(revised_lineage)
    correction_ids = sorted(
        diff.get_column("revision_id").drop_nulls().unique().to_list()
    )
    correction_types = sorted(
        diff.get_column("revision_type").drop_nulls().unique().to_list()
    )
    report = {
        "price_revision_report_version": 1,
        "previous_vintage_id": old_id,
        "new_vintage_id": new_id,
        "package_known_at": known_at.isoformat().replace("+00:00", "Z"),
        "previous_lineage_sha256": previous_sha256,
        "new_lineage_sha256": candidate_sha256,
        "revision_diff_sha256": _frame_sha256(diff),
        "changed_rows": diff.height,
        "added_rows": added.height,
        "removed_rows": 0,
        "revision_ids": correction_ids,
        "revision_types": correction_types,
        "revision_diff_required": True,
        "passed": True,
    }
    return PriceRevisionPackage(
        prices=revised_lineage.select(PRICE_VALUE_COLUMNS),
        lineage=revised_lineage,
        revision_diff=diff,
        report=report,
    )


def _normalize_lineage(frame: pl.DataFrame, *, label: str) -> pl.DataFrame:
    missing = set(PRICE_LINEAGE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"{label} is missing columns: {sorted(missing)}")
    date_expr = (
        pl.col("date").str.to_date(strict=False)
        if frame.schema.get("date") == pl.String
        else pl.col("date").cast(pl.Date, strict=False)
    )
    normalized = frame.select(PRICE_LINEAGE_COLUMNS).with_columns(
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        date_expr.alias("date"),
    )
    duplicate_count = normalized.height - normalized.select(
        pl.struct(_KEY_COLUMNS).n_unique()
    ).item()
    if duplicate_count:
        raise ValueError(f"{label} has {duplicate_count} duplicate ticker/date keys.")
    if normalized.select(pl.col("date").is_null().any()).item():
        raise ValueError(f"{label} contains an invalid date.")
    return normalized.sort(_KEY_COLUMNS)


def _normalize_revision_events(frame: pl.DataFrame) -> pl.DataFrame:
    missing = set(PRICE_REVISION_EVENT_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Price revision events are missing columns: {sorted(missing)}")
    events = frame.select(PRICE_REVISION_EVENT_COLUMNS).with_columns(
        pl.col("revision_id").cast(pl.String),
        pl.col("revision_type").cast(pl.String),
        pl.col("ticker").cast(pl.String).str.to_uppercase(),
        pl.col("effective_date").cast(pl.Date, strict=False),
        pl.col("affected_from").cast(pl.Date, strict=False),
        pl.col("affected_through").cast(pl.Date, strict=False),
        pl.col("known_at").map_elements(
            lambda value: _parse_datetime(value, field="known_at"),
            return_dtype=pl.Datetime(time_zone="UTC"),
        ),
        pl.col("source").cast(pl.String),
        pl.col("source_url").cast(pl.String),
        pl.col("reason").cast(pl.String),
    )
    if events.is_empty():
        raise ValueError("Historical price changes require revision events.")
    invalid_types = sorted(
        set(events.get_column("revision_type").to_list()) - PRICE_REVISION_TYPES
    )
    if invalid_types:
        raise ValueError(f"Unsupported price revision types: {invalid_types}")
    if events.select(pl.col("revision_id").n_unique()).item() != events.height:
        raise ValueError("Price revision_id values must be unique.")
    required_non_null = [
        "revision_id",
        "ticker",
        "effective_date",
        "known_at",
        "affected_from",
        "affected_through",
        "source",
        "source_url",
        "reason",
    ]
    for column in required_non_null:
        if events.select(pl.col(column).is_null().any()).item():
            raise ValueError(f"Price revision event field {column!r} cannot be null.")
    if events.filter(pl.col("affected_from") > pl.col("affected_through")).height:
        raise ValueError("Price revision affected_from must precede affected_through.")
    return events.sort(["ticker", "affected_from", "revision_id"])


def _build_revision_diff(
    previous: pl.DataFrame, candidate: pl.DataFrame
) -> pl.DataFrame:
    old = previous.select(
        *_KEY_COLUMNS,
        *[
            pl.col(column).alias(f"previous_{column}")
            for column in _REVISED_VALUE_COLUMNS
        ],
    )
    new = candidate.select(
        *_KEY_COLUMNS,
        *[
            pl.col(column).alias(f"candidate_{column}")
            for column in _REVISED_VALUE_COLUMNS
        ],
    )
    comparisons = [
        pl.col(f"previous_{column}")
        .eq_missing(pl.col(f"candidate_{column}"))
        .not_()
        for column in _REVISED_VALUE_COLUMNS
    ]
    changed_columns = pl.concat_list(
        [
            pl.when(comparison)
            .then(pl.lit(column))
            .otherwise(pl.lit(None).cast(pl.String))
            for column, comparison in zip(
                _REVISED_VALUE_COLUMNS, comparisons, strict=True
            )
        ]
    ).list.drop_nulls()
    return (
        old.join(new, on=_KEY_COLUMNS, how="inner")
        .filter(pl.any_horizontal(comparisons))
        .with_columns(changed_columns.alias("changed_columns"))
        .sort(_KEY_COLUMNS)
    )


def _parse_datetime(value: object, *, field: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError(f"Invalid {field}: {value!r}.") from exc
    else:
        raise ValueError(f"Invalid {field}: {value!r}.")
    if parsed.tzinfo is None:
        raise ValueError(f"{field} must include an explicit timezone.")
    return parsed.astimezone(timezone.utc)


def _frame_sha256(frame: pl.DataFrame) -> str:
    buffer = BytesIO()
    frame.write_ipc(buffer, compression="uncompressed")
    return hashlib.sha256(buffer.getvalue()).hexdigest()
