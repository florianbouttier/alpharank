from __future__ import annotations

from pathlib import Path

import polars as pl


def resolve_sec_company_mapping(
    *,
    requested_tickers: list[str] | tuple[str, ...],
    sec_mapping_all: pl.DataFrame,
    reference_data_dir: Path | None = None,
    existing_general_reference_lineage: pl.DataFrame | None = None,
) -> pl.DataFrame:
    requested = tuple(sorted({str(ticker).replace(".US", "") for ticker in requested_tickers if str(ticker).strip()}))
    if not requested:
        return _empty_mapping_frame()

    live = _prepare_live_sec_mapping(sec_mapping_all).filter(pl.col("ticker").is_in(list(requested)))
    lineage_bridge = _prepare_raw_lineage_bridge(existing_general_reference_lineage).filter(pl.col("ticker").is_in(list(requested)))
    eodhd_bridge = _prepare_eodhd_cik_bridge(reference_data_dir).filter(pl.col("ticker").is_in(list(requested)))

    combined = pl.concat([live, lineage_bridge, eodhd_bridge], how="vertical") if not live.is_empty() or not lineage_bridge.is_empty() or not eodhd_bridge.is_empty() else _empty_mapping_frame()
    if combined.is_empty():
        return combined

    return (
        combined.sort(["ticker", "mapping_priority", "name", "exchange", "cik"], descending=[False, False, False, False, False])
        .unique(subset=["ticker"], keep="first", maintain_order=True)
        .sort("ticker")
    )


def _prepare_live_sec_mapping(sec_mapping_all: pl.DataFrame) -> pl.DataFrame:
    if sec_mapping_all.is_empty():
        return _empty_mapping_frame()
    return (
        sec_mapping_all.select(
            [
                pl.col("ticker").cast(pl.Utf8).alias("ticker"),
                pl.col("name").cast(pl.Utf8).alias("name"),
                pl.col("exchange").cast(pl.Utf8).alias("exchange"),
                pl.col("cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("cik"),
            ]
        )
        .filter(pl.col("ticker").is_not_null() & pl.col("cik").is_not_null())
        .with_columns(
            [
                pl.lit("sec_live_mapping").alias("mapping_source"),
                pl.lit(1).alias("mapping_priority"),
            ]
        )
        .unique(subset=["ticker"], keep="first", maintain_order=True)
    )


def _prepare_raw_lineage_bridge(existing_general_reference_lineage: pl.DataFrame | None) -> pl.DataFrame:
    if existing_general_reference_lineage is None or existing_general_reference_lineage.is_empty():
        return _empty_mapping_frame()
    required = {"ticker", "sec_name", "sec_exchange", "sec_cik"}
    if not required.issubset(existing_general_reference_lineage.columns):
        return _empty_mapping_frame()
    return (
        existing_general_reference_lineage.select(
            [
                pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"),
                pl.col("sec_name").cast(pl.Utf8).alias("name"),
                pl.col("sec_exchange").cast(pl.Utf8).alias("exchange"),
                pl.col("sec_cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("cik"),
            ]
        )
        .filter(pl.col("ticker").is_not_null() & pl.col("cik").is_not_null())
        .with_columns(
            [
                pl.lit("raw_sec_lineage_bridge").alias("mapping_source"),
                pl.lit(2).alias("mapping_priority"),
            ]
        )
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )


def _prepare_eodhd_cik_bridge(reference_data_dir: Path | None) -> pl.DataFrame:
    if reference_data_dir is None:
        return _empty_mapping_frame()
    candidate_paths = (
        reference_data_dir / "eodhd" / "output" / "US_General.parquet",
        reference_data_dir / "US_General.parquet",
    )
    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pl.read_parquet(path)
        required = {"Code", "Name", "Exchange", "CIK"}
        if not required.issubset(frame.columns):
            continue
        return (
            frame.select(
                [
                    pl.col("Code").cast(pl.Utf8).alias("ticker"),
                    pl.col("Name").cast(pl.Utf8).alias("name"),
                    pl.col("Exchange").cast(pl.Utf8).alias("exchange"),
                    pl.col("CIK").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("cik"),
                ]
            )
            .filter(pl.col("ticker").is_not_null() & pl.col("cik").is_not_null())
            .with_columns(
                [
                    pl.lit("eodhd_cik_bridge").alias("mapping_source"),
                    pl.lit(3).alias("mapping_priority"),
                ]
            )
            .unique(subset=["ticker"], keep="last", maintain_order=True)
        )
    return _empty_mapping_frame()


def _empty_mapping_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "name": pl.String,
            "exchange": pl.String,
            "cik": pl.String,
            "mapping_source": pl.String,
            "mapping_priority": pl.Int64,
        }
    )
