"""Provider-neutral normalization for the canonical STG layer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import polars as pl

STG_CONTRACT_ID = "alpharank_staging_observations_v1"
STG_PROVENANCE_SCHEMA = {
    "source_name": pl.String,
    "dataset_name": pl.String,
    "receipt_id": pl.String,
    "payload_sha256": pl.String,
    "retrieved_at": pl.String,
}
FORBIDDEN_SOURCE_SELECTION_COLUMNS = frozenset(
    {
        "fallback_used",
        "selected_source",
        "selection_reason",
        "source_priority",
    }
)
PRICE_VALUE_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "adjusted_close",
)
PRICE_METADATA_COLUMNS = (
    "source",
    "dataset",
    "ingestion_run_id",
    "ingested_at",
)
PRICE_COLUMNS = ("date", *PRICE_VALUE_COLUMNS, "ticker", *PRICE_METADATA_COLUMNS)


def normalize_staging_observations(
    frames: Sequence[pl.DataFrame],
    *,
    business_key: Sequence[str],
    value_schema: Mapping[str, Any],
) -> pl.DataFrame:
    """Normalize observations while retaining every provider candidate.

    STG has no priority input and rejects columns that encode a source choice.
    Conflicting providers for the same business key therefore remain separate
    rows, each tied to its RAW receipt and payload hash.
    """

    if not business_key:
        raise ValueError("STG business_key cannot be empty")
    if not value_schema:
        raise ValueError("STG value_schema cannot be empty")
    missing_key_columns = sorted(set(business_key) - set(value_schema))
    if missing_key_columns:
        raise ValueError(f"STG business key is outside value schema: {missing_key_columns}")
    output_schema = {**dict(value_schema), **STG_PROVENANCE_SCHEMA}
    if not frames:
        return pl.DataFrame(schema=output_schema)

    staged_frames: list[pl.DataFrame] = []
    input_row_count = 0
    for frame in frames:
        forbidden = sorted(FORBIDDEN_SOURCE_SELECTION_COLUMNS.intersection(frame.columns))
        if forbidden:
            raise ValueError(f"Source selection is forbidden in STG: {forbidden}")
        missing = sorted(set(output_schema) - set(frame.columns))
        if missing:
            raise ValueError(f"STG observation columns are missing: {missing}")
        input_row_count += frame.height
        staged_frames.append(
            frame.select(
                *(
                    pl.col(column).cast(dtype, strict=True).alias(column)
                    for column, dtype in output_schema.items()
                )
            )
        )

    staged = pl.concat(staged_frames, how="vertical")
    if staged.height != input_row_count:
        raise RuntimeError("STG normalization changed the provider observation count")
    if staged.select(
        pl.any_horizontal(
            pl.col("source_name").is_null(),
            pl.col("dataset_name").is_null(),
            pl.col("receipt_id").is_null(),
            pl.col("payload_sha256").is_null(),
            pl.col("retrieved_at").is_null(),
        ).any()
    ).item():
        raise ValueError("STG provenance cannot be null")
    invalid_hash_count = staged.filter(
        ~pl.col("payload_sha256").str.contains(r"^[0-9a-f]{64}$")
    ).height
    if invalid_hash_count:
        raise ValueError(f"STG contains {invalid_hash_count} invalid payload SHA-256 values")

    observation_key = [*business_key, "source_name", "receipt_id"]
    duplicate_count = staged.select(
        pl.struct(observation_key).is_duplicated().sum()
    ).item()
    if duplicate_count:
        raise ValueError(
            f"STG contains {duplicate_count} duplicate provider observation rows"
        )
    return staged.sort([*business_key, "source_name", "receipt_id"])


def stage_yahoo_prices(
    frame: pl.DataFrame,
    *,
    run_id: str,
    observed_at: str,
    dataset: str = "prices_yfinance",
) -> pl.DataFrame:
    """Normalize one legacy RAW Yahoo observation without choosing or filling."""

    required = {"ticker", "date", *PRICE_VALUE_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Yahoo STG price columns are missing: {missing}")
    staged = frame.select(
        pl.col("date").cast(pl.String, strict=False),
        *(pl.col(column).cast(pl.Float64, strict=False) for column in PRICE_VALUE_COLUMNS),
        pl.col("ticker").cast(pl.String, strict=False).str.to_uppercase(),
    ).with_columns(
        pl.lit("yfinance").alias("source"),
        pl.lit(dataset).alias("dataset"),
        pl.lit(run_id).alias("ingestion_run_id"),
        pl.lit(observed_at).alias("ingested_at"),
    )
    _require_unique_price_keys(staged, layer="STG")
    return staged.sort(["ticker", "date"])


def _require_unique_price_keys(frame: pl.DataFrame, *, layer: str) -> None:
    duplicate_count = frame.select(
        pl.struct(["ticker", "date"]).is_duplicated().sum()
    ).item()
    if duplicate_count:
        raise ValueError(
            f"{layer} price frame contains {duplicate_count} duplicate ticker/date rows"
        )
