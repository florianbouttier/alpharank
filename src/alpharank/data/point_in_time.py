"""Generic fail-closed point-in-time attribute joins."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl


def join_point_in_time_attributes(
    decisions: pl.DataFrame,
    history: pl.DataFrame,
    *,
    entity_column: str,
    decision_time_column: str,
    effective_time_column: str,
    attribute_columns: Sequence[str],
) -> pl.DataFrame:
    """Join only the latest attribute version effective by each decision.

    The selected effective timestamp is retained as
    ``<effective_time_column>_selected`` so callers can audit the causal
    boundary rather than trusting an opaque join.
    """

    required_decisions = {entity_column, decision_time_column}
    required_history = {
        entity_column,
        effective_time_column,
        *attribute_columns,
    }
    missing_decisions = sorted(required_decisions - set(decisions.columns))
    missing_history = sorted(required_history - set(history.columns))
    if missing_decisions:
        raise ValueError(
            "Point-in-time decisions are missing: " + ", ".join(missing_decisions)
        )
    if missing_history:
        raise ValueError(
            "Point-in-time history is missing: " + ", ".join(missing_history)
        )
    if not attribute_columns:
        raise ValueError("At least one point-in-time attribute is required.")

    selected_time = f"{effective_time_column}_selected"
    left = decisions.with_columns(
        pl.col(entity_column).cast(pl.String),
        pl.col(decision_time_column)
        .cast(pl.Datetime(time_zone="UTC"), strict=False)
        .alias(decision_time_column),
    ).sort([entity_column, decision_time_column])
    right = history.select(
        pl.col(entity_column).cast(pl.String),
        pl.col(effective_time_column)
        .cast(pl.Datetime(time_zone="UTC"), strict=False)
        .alias(selected_time),
        *[pl.col(column) for column in attribute_columns],
    )
    duplicate_versions = (
        right.group_by([entity_column, selected_time])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicate_versions.is_empty():
        raise ValueError(
            "Point-in-time history contains duplicate entity/effective-time versions."
        )
    right = right.sort([entity_column, selected_time])
    result = left.join_asof(
        right,
        left_on=decision_time_column,
        right_on=selected_time,
        by=entity_column,
        strategy="backward",
        check_sortedness=False,
    )
    leaked = result.filter(
        pl.col(selected_time).is_not_null()
        & (pl.col(selected_time) > pl.col(decision_time_column))
    )
    if not leaked.is_empty():
        raise RuntimeError("Point-in-time join selected a future attribute version.")
    return result
