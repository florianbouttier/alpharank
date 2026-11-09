"""Point-in-time sector classifications and coverage policy."""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from alpharank.data.contracts.point_in_time import join_point_in_time_attributes

SECTOR_HISTORY_LINEAGE_COLUMNS = (
    "classification_id",
    "source_url",
    "confidence",
    "observed_at",
    "effective_at",
)
IDENTITY_INTERVAL_COLUMNS = ("identity_valid_from", "identity_valid_to")


def resolve_point_in_time_sectors(
    decisions: pl.DataFrame,
    sector_history: pl.DataFrame,
    *,
    ticker_column: str = "ticker",
    decision_time_column: str = "decision_at",
    sector_column: str = "Sector",
    lineage_columns: Sequence[str] = SECTOR_HISTORY_LINEAGE_COLUMNS,
) -> pl.DataFrame:
    """Resolve sectors known and effective at each decision.

    The sector constraint is enabled for a decision timestamp only when every
    candidate has a fully lineaged classification. A static current-sector map
    cannot satisfy this contract and is therefore never used implicitly.
    """

    required_history = {
        ticker_column,
        sector_column,
        "observed_at",
        "effective_at",
        *lineage_columns,
    }
    missing_history = sorted(required_history - set(sector_history.columns))
    if missing_history:
        raise ValueError(
            "Point-in-time sector history is missing: " + ", ".join(missing_history)
        )
    identity_columns = [
        column for column in IDENTITY_INTERVAL_COLUMNS if column in sector_history.columns
    ]
    history = (
        sector_history.select(
            pl.col(ticker_column).cast(pl.String),
            pl.col(sector_column).cast(pl.String),
            pl.col("observed_at").cast(
                pl.Datetime(time_zone="UTC"), strict=False
            ),
            pl.col("effective_at").cast(
                pl.Datetime(time_zone="UTC"), strict=False
            ),
            *[
                pl.col(column)
                for column in lineage_columns
                if column not in {"observed_at", "effective_at"}
            ],
            *[pl.col(column).cast(pl.Date, strict=False) for column in identity_columns],
        )
        .with_columns(
            pl.max_horizontal("observed_at", "effective_at").alias(
                "sector_known_at"
            )
        )
    )
    incomplete_lineage = history.filter(
        pl.any_horizontal(
            pl.col(sector_column).is_null() | (pl.col(sector_column) == ""),
            pl.col("sector_known_at").is_null(),
            *[
                pl.col(column).is_null()
                | (pl.col(column).cast(pl.String, strict=False) == "")
                for column in lineage_columns
                if column not in {"observed_at", "effective_at"}
            ],
        )
    )
    if not incomplete_lineage.is_empty():
        raise ValueError("Point-in-time sector history contains incomplete lineage.")

    attributes = (
        sector_column,
        "observed_at",
        "effective_at",
        *tuple(
            column
            for column in lineage_columns
            if column not in {"observed_at", "effective_at"}
        ),
        *identity_columns,
    )
    resolved = join_point_in_time_attributes(
        decisions,
        history,
        entity_column=ticker_column,
        decision_time_column=decision_time_column,
        effective_time_column="sector_known_at",
        attribute_columns=attributes,
    )
    resolved = _apply_identity_interval(
        resolved,
        decision_time_column=decision_time_column,
        sector_column=sector_column,
        identity_columns=identity_columns,
    )
    future_sector = resolved.filter(
        pl.col(sector_column).is_not_null()
        & (
            (pl.col("observed_at") > pl.col(decision_time_column))
            | (pl.col("effective_at") > pl.col(decision_time_column))
        )
    )
    if not future_sector.is_empty():
        raise RuntimeError("A future sector classification reached a past decision.")

    availability = pl.col(sector_column).is_not_null() & (
        pl.col(sector_column) != ""
    )
    coverage = resolved.group_by(decision_time_column).agg(
        availability.all().alias("sector_constraint_enabled"),
        (~availability).sum().alias("missing_point_in_time_sector_count"),
    )
    return (
        resolved.join(coverage, on=decision_time_column, how="left")
        .with_columns(
            pl.when(pl.col("sector_constraint_enabled"))
            .then(pl.lit("point_in_time_coverage_complete"))
            .otherwise(pl.lit("disabled_missing_point_in_time_sector"))
            .alias("sector_constraint_reason")
        )
        .sort([decision_time_column, ticker_column])
    )


def _apply_identity_interval(
    resolved: pl.DataFrame,
    *,
    decision_time_column: str,
    sector_column: str,
    identity_columns: list[str],
) -> pl.DataFrame:
    if not identity_columns:
        return resolved.with_columns(pl.lit(True).alias("identity_interval_eligible"))
    decision_date = pl.col(decision_time_column).dt.date()
    valid_from = (
        pl.col("identity_valid_from")
        if "identity_valid_from" in identity_columns
        else pl.lit(None).cast(pl.Date)
    )
    valid_to = (
        pl.col("identity_valid_to")
        if "identity_valid_to" in identity_columns
        else pl.lit(None).cast(pl.Date)
    )
    eligible = (valid_from.is_null() | (decision_date >= valid_from)) & (
        valid_to.is_null() | (decision_date <= valid_to)
    )
    return resolved.with_columns(
        eligible.alias("identity_interval_eligible"),
        pl.when(eligible).then(pl.col(sector_column)).otherwise(None).alias(sector_column),
    )
