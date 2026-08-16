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

    manual_bridge = load_sec_historical_ticker_bridge(reference_data_dir)
    live = _prepare_live_sec_mapping(sec_mapping_all)
    lineage_bridge = _prepare_raw_lineage_bridge(existing_general_reference_lineage)
    eodhd_bridge = _prepare_eodhd_cik_bridge(reference_data_dir)

    combined = (
        pl.concat([manual_bridge, live, lineage_bridge, eodhd_bridge], how="diagonal_relaxed")
        if not manual_bridge.is_empty() or not live.is_empty() or not lineage_bridge.is_empty() or not eodhd_bridge.is_empty()
        else _empty_mapping_frame()
    )
    if combined.is_empty():
        return combined

    combined = _expand_requested_ticker_aliases(combined, requested=requested)

    return (
        combined.sort(
            ["ticker", "ticker_alias_priority", "mapping_priority", "name", "exchange", "cik"],
            descending=[False, False, False, False, False, False],
        )
        .unique(subset=["ticker"], keep="first", maintain_order=True)
        .drop("ticker_alias_priority")
        .sort("ticker")
    )


def _expand_requested_ticker_aliases(
    mapping: pl.DataFrame,
    *,
    requested: tuple[str, ...],
) -> pl.DataFrame:
    alias_rows: list[dict[str, object]] = []
    for ticker in requested:
        alias_rows.append(
            {
                "mapping_ticker": ticker,
                "requested_ticker": ticker,
                "ticker_alias_priority": 0,
            }
        )
        yahoo_alias = ticker.replace(".", "-")
        if yahoo_alias != ticker:
            alias_rows.append(
                {
                    "mapping_ticker": yahoo_alias,
                    "requested_ticker": ticker,
                    "ticker_alias_priority": 1,
                }
            )
    aliases = pl.DataFrame(alias_rows)
    return (
        mapping.rename({"ticker": "mapping_ticker"})
        .join(aliases, on="mapping_ticker", how="inner")
        .drop("mapping_ticker")
        .rename({"requested_ticker": "ticker"})
    )


def load_sec_historical_ticker_bridge(reference_data_dir: Path | None = None) -> pl.DataFrame:
    candidate_paths: list[Path] = []
    if reference_data_dir is not None:
        candidate_paths.append(reference_data_dir / "sec" / "manual_historical_ticker_bridge.csv")
    candidate_paths.append(Path(__file__).with_name("reference") / "sec_historical_ticker_bridge.csv")

    for path in candidate_paths:
        if not path.exists():
            continue
        frame = pl.read_csv(path)
        required = {"ticker", "name", "exchange", "cik"}
        if not required.issubset(frame.columns):
            continue
        prepared = (
            frame.select(
                [
                    pl.col("ticker").cast(pl.Utf8).str.replace(r"\.US$", "").alias("ticker"),
                    pl.col("name").cast(pl.Utf8).alias("name"),
                    pl.col("exchange").cast(pl.Utf8).alias("exchange"),
                    pl.col("cik").cast(pl.Utf8).str.extract(r"(\d+)").str.zfill(10).alias("cik"),
                    *(
                        [pl.col("start_date").cast(pl.Utf8)]
                        if "start_date" in frame.columns
                        else [pl.lit(None).cast(pl.Utf8).alias("start_date")]
                    ),
                    *(
                        [pl.col("end_date").cast(pl.Utf8)]
                        if "end_date" in frame.columns
                        else [pl.lit(None).cast(pl.Utf8).alias("end_date")]
                    ),
                    *(
                        [pl.col("mapping_source").cast(pl.Utf8)]
                        if "mapping_source" in frame.columns
                        else [pl.lit("sec_manual_historical_bridge").alias("mapping_source")]
                    ),
                    *(
                        [pl.col("mapping_priority").cast(pl.Int64, strict=False)]
                        if "mapping_priority" in frame.columns
                        else [pl.lit(0).cast(pl.Int64).alias("mapping_priority")]
                    ),
                ]
            )
            .filter(pl.col("ticker").is_not_null() & pl.col("cik").is_not_null())
            .unique(subset=["ticker"], keep="first", maintain_order=True)
        )
        return _ensure_mapping_columns(prepared)
    return _empty_mapping_frame()


def _prepare_live_sec_mapping(sec_mapping_all: pl.DataFrame) -> pl.DataFrame:
    if sec_mapping_all.is_empty():
        return _empty_mapping_frame()
    return _ensure_mapping_columns(
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
    return _ensure_mapping_columns(
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
        return _ensure_mapping_columns(
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


def _ensure_mapping_columns(frame: pl.DataFrame) -> pl.DataFrame:
    columns = frame.columns
    result = frame
    if "start_date" not in columns:
        result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias("start_date"))
    if "end_date" not in columns:
        result = result.with_columns(pl.lit(None).cast(pl.Utf8).alias("end_date"))
    return result.select(
        [
            "ticker",
            "name",
            "exchange",
            "cik",
            "start_date",
            "end_date",
            "mapping_source",
            "mapping_priority",
        ]
    )


def _empty_mapping_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "name": pl.String,
            "exchange": pl.String,
            "cik": pl.String,
            "start_date": pl.String,
            "end_date": pl.String,
            "mapping_source": pl.String,
            "mapping_priority": pl.Int64,
        }
    )
