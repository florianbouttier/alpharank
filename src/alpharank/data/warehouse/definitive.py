"""Point-in-time source selection for the canonical DEF layer."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Sequence

import polars as pl

DEF_CONTRACT_ID = "alpharank_definitive_observations_v1"


@dataclass(frozen=True)
class DefinitiveSelectionResult:
    frame: pl.DataFrame
    decisions: pl.DataFrame
    selected_key_count: int
    unresolved_key_count: int


def select_definitive_observations(
    staged: pl.DataFrame,
    *,
    business_key: Sequence[str],
    value_column: str,
    source_priority: Sequence[str],
    rule_id: str,
    knowledge_cutoff: str,
) -> DefinitiveSelectionResult:
    """Select one observed value per key using only candidates known at cutoff."""

    if not business_key:
        raise ValueError("DEF business_key cannot be empty")
    if not source_priority or len(source_priority) != len(set(source_priority)):
        raise ValueError("DEF source_priority must be a non-empty unique sequence")
    if not rule_id.strip():
        raise ValueError("DEF rule_id cannot be empty")
    cutoff = _parse_aware_timestamp(knowledge_cutoff, "knowledge_cutoff")
    required = {
        *business_key,
        value_column,
        "source_name",
        "dataset_name",
        "receipt_id",
        "payload_sha256",
        "retrieved_at",
    }
    missing = sorted(required - set(staged.columns))
    if missing:
        raise ValueError(f"DEF input columns are missing: {missing}")
    unknown_sources = sorted(
        set(staged.get_column("source_name").drop_nulls().to_list())
        - set(source_priority)
    )
    if unknown_sources:
        raise ValueError(f"DEF source priority is incomplete: {unknown_sources}")

    rows_by_key: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in staged.iter_rows(named=True):
        value = row[value_column]
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"DEF rejects non-finite {value_column} values")
        key = tuple(row[column] for column in business_key)
        rows_by_key.setdefault(key, []).append(row)

    priority_rank = {source: rank for rank, source in enumerate(source_priority)}
    selected_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    for key in sorted(rows_by_key, key=_sortable_key):
        candidates = rows_by_key[key]
        known = [
            row
            for row in candidates
            if _parse_aware_timestamp(str(row["retrieved_at"]), "retrieved_at") <= cutoff
        ]
        latest_by_source: dict[str, dict[str, Any]] = {}
        for row in known:
            source = str(row["source_name"])
            current = latest_by_source.get(source)
            if current is None or _receipt_order(row) > _receipt_order(current):
                latest_by_source[source] = row
        ordered = sorted(
            latest_by_source.values(),
            key=lambda row: (
                priority_rank[str(row["source_name"])],
                str(row["source_name"]),
            ),
        )
        selected = next(
            (row for row in ordered if row[value_column] is not None),
            None,
        )
        preferred = ordered[0] if ordered else None
        if selected is None:
            reason = (
                "unresolved_no_observed_value"
                if ordered
                else "unresolved_no_candidate_known_at_cutoff"
            )
        elif len(ordered) == 1:
            reason = "only_source_known_at_cutoff"
        elif selected is preferred:
            reason = "highest_priority_source_known_at_cutoff"
        else:
            reason = "preferred_source_missing_value_fallback"

        decision = {
            **dict(zip(business_key, key)),
            "contract": DEF_CONTRACT_ID,
            "selection_rule_id": rule_id,
            "knowledge_cutoff": knowledge_cutoff,
            "selection_reason": reason,
            "selected_source_name": (
                str(selected["source_name"]) if selected is not None else None
            ),
            "selected_dataset_name": (
                str(selected["dataset_name"]) if selected is not None else None
            ),
            "selected_receipt_id": (
                str(selected["receipt_id"]) if selected is not None else None
            ),
            "selected_payload_sha256": (
                str(selected["payload_sha256"]) if selected is not None else None
            ),
            "selected_retrieved_at": (
                str(selected["retrieved_at"]) if selected is not None else None
            ),
            "selected_value": selected[value_column] if selected is not None else None,
            "known_candidate_count": len(known),
            "known_source_count": len(ordered),
            "known_sources": " | ".join(str(row["source_name"]) for row in ordered),
            "excluded_after_cutoff_count": len(candidates) - len(known),
        }
        decision_rows.append(decision)
        if selected is not None:
            selected_rows.append(
                {
                    **selected,
                    "selection_rule_id": rule_id,
                    "knowledge_cutoff": knowledge_cutoff,
                    "selection_reason": reason,
                }
            )

    frame_schema = {
        **staged.schema,
        "selection_rule_id": pl.String,
        "knowledge_cutoff": pl.String,
        "selection_reason": pl.String,
    }
    decision_schema = {
        **{column: staged.schema[column] for column in business_key},
        "contract": pl.String,
        "selection_rule_id": pl.String,
        "knowledge_cutoff": pl.String,
        "selection_reason": pl.String,
        "selected_source_name": pl.String,
        "selected_dataset_name": pl.String,
        "selected_receipt_id": pl.String,
        "selected_payload_sha256": pl.String,
        "selected_retrieved_at": pl.String,
        "selected_value": staged.schema[value_column],
        "known_candidate_count": pl.Int64,
        "known_source_count": pl.Int64,
        "known_sources": pl.String,
        "excluded_after_cutoff_count": pl.Int64,
    }
    frame = pl.DataFrame(selected_rows, schema=frame_schema, strict=False)
    decisions = pl.DataFrame(decision_rows, schema=decision_schema, strict=False)
    if not frame.is_empty():
        frame = frame.sort(list(business_key))
    if not decisions.is_empty():
        decisions = decisions.sort(list(business_key))
    return DefinitiveSelectionResult(
        frame=frame,
        decisions=decisions,
        selected_key_count=frame.height,
        unresolved_key_count=decisions.height - frame.height,
    )


def _parse_aware_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{label} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None:
        raise ValueError(f"{label} must include a timezone")
    return parsed


def _sortable_key(key: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple("" if value is None else str(value) for value in key)


def _receipt_order(row: dict[str, Any]) -> tuple[datetime, str]:
    return (
        _parse_aware_timestamp(str(row["retrieved_at"]), "retrieved_at"),
        str(row["receipt_id"]),
    )
