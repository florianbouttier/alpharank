from __future__ import annotations

from pathlib import Path

import polars as pl

from alpharank.data.open_source.config import GENERAL_COLUMNS, METRIC_SPECS
from alpharank.data.open_source.storage import coerce_schema


LEGACY_STATEMENT_FILES = {
    "income_statement": "US_Income_statement.parquet",
    "balance_sheet": "US_Balance_sheet.parquet",
    "cash_flow": "US_Cash_flow.parquet",
    "shares": "US_share.parquet",
}


def export_legacy_compatible_outputs(
    *,
    clean_prices: pl.DataFrame,
    benchmark_prices: pl.DataFrame,
    general_reference: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    earnings_frame: pl.DataFrame,
    reference_data_dir: Path,
    output_dir: Path,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: dict[str, Path] = {}

    price_path = output_dir / "US_Finalprice.parquet"
    clean_prices.select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"]).sort(
        ["ticker", "date"]
    ).write_parquet(price_path)
    exported["US_Finalprice.parquet"] = price_path

    benchmark_path = output_dir / "SP500Price.parquet"
    benchmark_prices.select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"]).sort(
        ["ticker", "date"]
    ).write_parquet(benchmark_path)
    exported["SP500Price.parquet"] = benchmark_path

    general_path = output_dir / "US_General.parquet"
    _build_general_legacy_frame(general_reference, reference_data_dir).write_parquet(general_path)
    exported["US_General.parquet"] = general_path

    financial_frames: dict[str, pl.DataFrame] = {}
    for statement in LEGACY_STATEMENT_FILES:
        financial_frames[statement] = _build_financial_legacy_frame(
            statement=statement,
            consolidated_financials=consolidated_financials,
            reference_data_dir=reference_data_dir,
        )

    financial_frames["balance_sheet"] = _merge_balance_shares(
        balance_frame=financial_frames["balance_sheet"],
        shares_frame=financial_frames["shares"],
    )

    for statement, file_name in LEGACY_STATEMENT_FILES.items():
        frame = financial_frames[statement]
        path = output_dir / file_name
        frame.write_parquet(path)
        exported[file_name] = path

    earnings_path = output_dir / "US_Earnings.parquet"
    _build_earnings_legacy_frame(earnings_frame, reference_data_dir).write_parquet(earnings_path)
    exported["US_Earnings.parquet"] = earnings_path
    return exported


def _build_general_legacy_frame(general_reference: pl.DataFrame, reference_data_dir: Path) -> pl.DataFrame:
    reference_schema = pl.read_parquet(reference_data_dir / "US_General.parquet").schema
    if general_reference.is_empty():
        return pl.DataFrame(schema=reference_schema)
    frame = general_reference.select(GENERAL_COLUMNS).with_columns(
        [
            pl.col("ticker").str.replace(r"\.US$", "").alias("Code"),
            pl.col("name").alias("Name"),
            pl.col("exchange").alias("Exchange"),
            pl.lit("USD").alias("CurrencyCode"),
            pl.lit("$").alias("CurrencySymbol"),
            pl.lit("United States").alias("CountryName"),
            pl.lit("US").alias("CountryISO"),
            pl.col("cik").alias("CIK"),
        ]
    )
    return coerce_schema(frame, reference_schema).sort("Code")


def _build_financial_legacy_frame(
    *,
    statement: str,
    consolidated_financials: pl.DataFrame,
    reference_data_dir: Path,
) -> pl.DataFrame:
    reference_path = reference_data_dir / LEGACY_STATEMENT_FILES[statement]
    reference_schema = pl.read_parquet(reference_path).schema
    statement_frame = consolidated_financials.filter(pl.col("statement") == statement)
    if statement_frame.is_empty():
        return pl.DataFrame(schema=reference_schema)

    metric_map = {
        spec.metric: spec.eodhd_column
        for spec in METRIC_SPECS
        if spec.statement == statement and spec.statement != "earnings"
    }
    keyed = (
        statement_frame.filter(pl.col("metric").is_in(list(metric_map)))
        .with_columns(pl.col("metric").replace_strict(metric_map, default=None).alias("legacy_metric"))
        .filter(pl.col("legacy_metric").is_not_null())
    )
    if keyed.is_empty():
        return pl.DataFrame(schema=reference_schema)

    values = (
        keyed.group_by(["ticker", "date", "legacy_metric"])
        .agg(pl.col("value").sort_by("filing_date").last().alias("value"))
        .pivot(index=["ticker", "date"], on="legacy_metric", values="value", aggregate_function="first")
    )
    filing_dates = keyed.group_by(["ticker", "date"]).agg(pl.col("filing_date").drop_nulls().sort().last().alias("filing_date"))
    frame = values.join(filing_dates, on=["ticker", "date"], how="left", coalesce=True)

    if statement == "shares":
        frame = frame.with_columns(
            [
                pl.col("date").alias("dateFormatted"),
                (pl.col("shares") / 1_000_000.0).alias("sharesMln"),
            ]
        )
    return coerce_schema(frame, reference_schema).sort(["ticker", "date"])


def _build_earnings_legacy_frame(earnings_frame: pl.DataFrame, reference_data_dir: Path) -> pl.DataFrame:
    reference_schema = pl.read_parquet(reference_data_dir / "US_Earnings.parquet").schema
    if earnings_frame.is_empty():
        return pl.DataFrame(schema=reference_schema)

    frame = (
        earnings_frame.select(
            [
                pl.col("ticker"),
                pl.coalesce([pl.col("period_end"), pl.col("reportDate")]).alias("date"),
                pl.col("reportDate"),
                pl.col("epsEstimate"),
                pl.col("epsActual"),
                (pl.col("epsActual") - pl.col("epsEstimate")).alias("epsDifference"),
                pl.col("surprisePercent"),
            ]
        )
        .with_columns(
            [
                pl.lit(None).cast(pl.Utf8).alias("beforeAfterMarket"),
                pl.lit(None).cast(pl.Utf8).alias("currency"),
            ]
        )
        .sort(["ticker", "reportDate"])
        .unique(subset=["ticker", "reportDate"], keep="last", maintain_order=True)
    )
    return coerce_schema(frame, reference_schema)


def _merge_balance_shares(*, balance_frame: pl.DataFrame, shares_frame: pl.DataFrame) -> pl.DataFrame:
    if balance_frame.is_empty() or shares_frame.is_empty():
        return balance_frame
    if "commonStockSharesOutstanding" not in balance_frame.columns or "shares" not in shares_frame.columns:
        return balance_frame

    share_lookup = (
        shares_frame.select(
            [
                pl.col("ticker"),
                pl.col("dateFormatted").cast(pl.Utf8).alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares_from_legacy_share"),
            ]
        )
        .unique(subset=["ticker", "date"], keep="last", maintain_order=True)
    )

    joined = balance_frame.join(share_lookup, on=["ticker", "date"], how="left")
    enriched = joined.with_columns(
        pl.coalesce(
            [
                pl.col("commonStockSharesOutstanding").cast(pl.Float64, strict=False),
                pl.col("shares_from_legacy_share"),
            ]
        ).alias("commonStockSharesOutstanding")
    )
    return enriched.drop("shares_from_legacy_share")
