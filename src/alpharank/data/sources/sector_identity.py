"""Time-bounded ticker-to-CIK identities for historical sector observations."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from alpharank.data.open_source.sec_mapping import load_sec_historical_ticker_bridge
from alpharank.data.security_identity import load_security_identity_registry

IDENTITY_COLUMNS = (
    "ticker",
    "cik",
    "identity_valid_from",
    "identity_valid_to",
    "identity_source",
    "identity_status",
    "identity_evidence",
    "identity_id",
)
_SECTOR_IDENTITY_OVERRIDES = "sector_identity_overrides_v1.csv"


def build_sector_identity_map(
    requested_tickers: Sequence[str],
    general_reference_lineage: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Resolve replay tickers to issuer CIKs without crossing reviewed intervals."""

    requested = _normalize_requested_tickers(requested_tickers)
    if not requested:
        return _empty_identity_frame(), pl.DataFrame({"ticker": []}, schema={"ticker": pl.String})
    base = _base_identities(requested, general_reference_lineage)
    historical = _historical_identities(requested)
    reviewed = _reviewed_reuse_identities(requested)
    overrides = _sector_identity_overrides(requested)
    bounded_tickers = (
        set(historical["ticker"].to_list())
        | set(reviewed["ticker"].to_list())
        | set(overrides["ticker"].to_list())
    )
    if bounded_tickers:
        base = base.filter(~pl.col("ticker").is_in(sorted(bounded_tickers)))
    identities = pl.concat([base, historical, reviewed, overrides], how="diagonal_relaxed")
    identities = _finalize_identities(identities)
    resolved = set(identities["ticker"].to_list())
    unresolved = pl.DataFrame(
        {"ticker": [ticker for ticker in requested if ticker not in resolved]}
    )
    return identities, unresolved


def _normalize_requested_tickers(tickers: Sequence[str]) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                f"{str(ticker).strip().removesuffix('.US')}.US"
                for ticker in tickers
                if str(ticker).strip()
            }
        )
    )


def _base_identities(
    requested: tuple[str, ...],
    general_reference_lineage: pl.DataFrame,
) -> pl.DataFrame:
    required = {"ticker", "cik"}
    missing = sorted(required - set(general_reference_lineage.columns))
    if missing:
        raise ValueError("General reference lineage is missing: " + ", ".join(missing))
    return (
        general_reference_lineage.select(
            pl.col("ticker").cast(pl.String).str.replace_all("-", "."),
            _normalized_cik("cik"),
        )
        .filter(pl.col("ticker").is_in(requested) & pl.col("cik").is_not_null())
        .unique(subset=["ticker"], keep="last", maintain_order=True)
        .with_columns(
            pl.lit(None).cast(pl.Date).alias("identity_valid_from"),
            pl.lit(None).cast(pl.Date).alias("identity_valid_to"),
            pl.lit("retained_general_reference_lineage").alias("identity_source"),
            pl.lit("undated_same_symbol_identity").alias("identity_status"),
            pl.lit("general_reference_lineage").alias("identity_evidence"),
        )
    )


def _historical_identities(requested: tuple[str, ...]) -> pl.DataFrame:
    bridge = load_sec_historical_ticker_bridge()
    if bridge.is_empty():
        return _empty_identity_frame().drop("identity_id")
    return bridge.select(
        (pl.col("ticker") + pl.lit(".US")).alias("ticker"),
        _normalized_cik("cik"),
        pl.col("start_date").str.to_date(strict=False).alias("identity_valid_from"),
        pl.col("end_date").str.to_date(strict=False).alias("identity_valid_to"),
        pl.col("mapping_source").alias("identity_source"),
        pl.lit("reviewed_historical_bridge").alias("identity_status"),
        pl.col("mapping_source").alias("identity_evidence"),
    ).filter(pl.col("ticker").is_in(requested))


def _reviewed_reuse_identities(requested: tuple[str, ...]) -> pl.DataFrame:
    registry = load_security_identity_registry()
    if registry.is_empty():
        return _empty_identity_frame().drop("identity_id")
    return registry.select(
        (pl.col("canonical_ticker") + pl.lit(".US")).alias("ticker"),
        _normalized_cik("issuer_cik"),
        pl.col("valid_from").str.to_date(strict=False).alias("identity_valid_from"),
        pl.col("valid_to").str.to_date(strict=False).alias("identity_valid_to"),
        pl.lit("security_identity_intervals_v1").alias("identity_source"),
        pl.col("identity_status"),
        pl.col("evidence").alias("identity_evidence"),
    ).filter(pl.col("ticker").is_in(requested))


def _sector_identity_overrides(requested: tuple[str, ...]) -> pl.DataFrame:
    path = Path(__file__).parents[1] / "open_source/reference" / _SECTOR_IDENTITY_OVERRIDES
    frame = pl.read_csv(path, infer_schema_length=0)
    return frame.select(
        (pl.col("ticker") + pl.lit(".US")).alias("ticker"),
        _normalized_cik("cik"),
        pl.col("valid_from").str.to_date(strict=False).alias("identity_valid_from"),
        pl.col("valid_to").str.to_date(strict=False).alias("identity_valid_to"),
        pl.col("identity_source"),
        pl.col("identity_status"),
        pl.col("identity_evidence"),
    ).filter(pl.col("ticker").is_in(requested))


def _finalize_identities(identities: pl.DataFrame) -> pl.DataFrame:
    if identities.is_empty():
        return _empty_identity_frame()
    prepared = identities.with_columns(
        pl.struct(
            "ticker",
            "cik",
            "identity_valid_from",
            "identity_valid_to",
            "identity_source",
        )
        .map_elements(_identity_id, return_dtype=pl.String)
        .alias("identity_id")
    ).select(IDENTITY_COLUMNS)
    _require_valid_identity_intervals(prepared)
    return prepared.sort(["ticker", "identity_valid_from", "cik"], nulls_last=False)


def _require_valid_identity_intervals(identities: pl.DataFrame) -> None:
    invalid = identities.filter(
        pl.col("identity_valid_from").is_not_null()
        & pl.col("identity_valid_to").is_not_null()
        & (pl.col("identity_valid_from") > pl.col("identity_valid_to"))
    )
    if not invalid.is_empty():
        raise ValueError("Sector identity map contains an inverted validity interval.")
    duplicates = identities.group_by("ticker", "cik").len().filter(pl.col("len") > 1)
    if not duplicates.is_empty():
        raise ValueError("Sector identity map contains duplicate ticker/CIK rows.")


def _identity_id(row: dict[str, object]) -> str:
    payload = "|".join(str(row[column]) for column in row)
    return "sector-identity-" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalized_cik(column: str) -> pl.Expr:
    return (
        pl.col(column)
        .cast(pl.String, strict=False)
        .str.extract(r"(\d+)")
        .str.zfill(10)
        .alias("cik")
    )


def _empty_identity_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "cik": pl.String,
            "identity_valid_from": pl.Date,
            "identity_valid_to": pl.Date,
            "identity_source": pl.String,
            "identity_status": pl.String,
            "identity_evidence": pl.String,
            "identity_id": pl.String,
        }
    )
