from __future__ import annotations

from typing import Iterable

import polars as pl

from alpharank.data.open_source.config import GENERAL_COLUMNS, GENERAL_CORE_COLUMNS


GENERAL_REFERENCE_LINEAGE_COLUMNS: tuple[str, ...] = (
    *GENERAL_COLUMNS,
    "selected_name_source",
    "selected_exchange_source",
    "yahoo_name",
    "yahoo_exchange",
    "yahoo_sector",
    "yahoo_industry",
    "sec_name",
    "sec_exchange",
    "sec_cik",
    "sec_sic",
    "sec_sic_description",
)


def empty_general_reference_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.String for column in GENERAL_COLUMNS})


def empty_general_reference_lineage_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.String for column in GENERAL_REFERENCE_LINEAGE_COLUMNS})


def build_general_reference(
    *,
    tickers: Iterable[str],
    sec_mapping: pl.DataFrame,
    yahoo_metadata: pl.DataFrame,
    sec_profiles: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    ticker_list = [f"{ticker}.US" if not str(ticker).endswith(".US") else str(ticker) for ticker in tickers]
    base = pl.DataFrame({"ticker": ticker_list}) if ticker_list else empty_general_reference_frame().select(["ticker"])
    if base.is_empty():
        empty = empty_general_reference_frame()
        return empty, empty_general_reference_lineage_frame()

    sec_base = (
        sec_mapping.select(
            [
                (pl.col("ticker").cast(pl.Utf8) + pl.lit(".US")).alias("ticker"),
                pl.col("name").cast(pl.Utf8).alias("sec_name"),
                pl.col("exchange").cast(pl.Utf8).alias("sec_exchange"),
                pl.col("cik").cast(pl.Utf8).str.zfill(10).alias("sec_cik"),
            ]
        )
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    yahoo_base = (
        yahoo_metadata.select(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("name").cast(pl.Utf8).alias("yahoo_name"),
                pl.col("exchange").cast(pl.Utf8).alias("yahoo_exchange"),
                pl.col("sector_raw_value").cast(pl.Utf8).alias("yahoo_sector"),
                pl.col("industry").cast(pl.Utf8).alias("yahoo_industry"),
            ]
        )
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    sec_profile_base = (
        sec_profiles.select(
            [
                pl.col("ticker").cast(pl.Utf8),
                pl.col("sic").cast(pl.Utf8).alias("sec_sic"),
                pl.col("sic_description").cast(pl.Utf8).alias("sec_sic_description"),
            ]
        )
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )

    combined = base.join(sec_base, on="ticker", how="left", coalesce=True)
    combined = combined.join(yahoo_base, on="ticker", how="left", coalesce=True)
    combined = combined.join(sec_profile_base, on="ticker", how="left", coalesce=True)

    sector_expr, sector_source_expr, mapping_rule_expr, sector_raw_expr = _sector_mapping_exprs()
    with_selected = combined.with_columns(
        [
            pl.coalesce([pl.col("yahoo_name"), pl.col("sec_name")]).alias("name"),
            pl.coalesce([pl.col("sec_exchange"), pl.col("yahoo_exchange")]).alias("exchange"),
            pl.col("sec_cik").alias("cik"),
            sector_expr.alias("Sector"),
            pl.coalesce([pl.col("yahoo_industry"), pl.col("sec_sic_description")]).alias("industry"),
            sector_source_expr.alias("sector_source"),
            sector_raw_expr.alias("sector_raw_value"),
            pl.col("sec_sic").alias("sic"),
            pl.col("sec_sic_description").alias("sic_description"),
            mapping_rule_expr.alias("mapping_rule"),
            pl.lit("open_source_general").alias("source"),
            pl.when(pl.col("yahoo_name").is_not_null())
            .then(pl.lit("yfinance"))
            .otherwise(pl.lit("sec_mapping"))
            .alias("selected_name_source"),
            pl.when(pl.col("sec_exchange").is_not_null())
            .then(pl.lit("sec_mapping"))
            .when(pl.col("yahoo_exchange").is_not_null())
            .then(pl.lit("yfinance"))
            .otherwise(pl.lit("unknown"))
            .alias("selected_exchange_source"),
        ]
    )

    general_reference = (
        with_selected.select(list(GENERAL_COLUMNS))
        .unique(subset=["ticker"], keep="last", maintain_order=True)
        .sort("ticker")
    )
    lineage = (
        with_selected.select(list(GENERAL_REFERENCE_LINEAGE_COLUMNS))
        .unique(subset=["ticker"], keep="last", maintain_order=True)
        .sort("ticker")
    )
    return general_reference, lineage


def _sector_mapping_exprs() -> tuple[pl.Expr, pl.Expr, pl.Expr, pl.Expr]:
    yahoo_sector = pl.col("yahoo_sector").cast(pl.Utf8, strict=False)
    yahoo_sector_normalized = (
        pl.when(yahoo_sector.is_in(["Consumer Staples"]))
        .then(pl.lit("Consumer Defensive"))
        .when(yahoo_sector.is_in(["Consumer Discretionary"]))
        .then(pl.lit("Consumer Cyclical"))
        .when(yahoo_sector.is_in(["Financial"]))
        .then(pl.lit("Financial Services"))
        .otherwise(yahoo_sector)
    )
    sic_description = pl.col("sec_sic_description").cast(pl.Utf8, strict=False)
    sec_sector, sec_rule = _sec_sic_sector_exprs(sic_description)

    sector = pl.coalesce([yahoo_sector_normalized, sec_sector, pl.lit("Unknown")])
    sector_source = (
        pl.when(yahoo_sector_normalized.is_not_null())
        .then(pl.lit("yfinance"))
        .when(sec_sector.is_not_null())
        .then(pl.lit("sec_sic"))
        .otherwise(pl.lit("unknown"))
    )
    mapping_rule = (
        pl.when(yahoo_sector_normalized.is_not_null())
        .then(pl.lit("yfinance:sector"))
        .when(sec_rule.is_not_null())
        .then(sec_rule)
        .otherwise(pl.lit("fallback:unknown"))
    )
    sector_raw_value = pl.coalesce([yahoo_sector, sic_description])
    return sector, sector_source, mapping_rule, sector_raw_value


def _sec_sic_sector_exprs(sic_description: pl.Expr) -> tuple[pl.Expr, pl.Expr]:
    desc = sic_description.str.to_lowercase()

    conditions: list[tuple[pl.Expr, str, str]] = [
        (desc.str.contains("real estate|reit|property|properties"), "Real Estate", "sec_sic:real_estate"),
        (desc.str.contains("bank|insurance|finance|capital|investment|broker|asset management|mortgage"), "Financial Services", "sec_sic:financial_services"),
        (desc.str.contains("oil|gas|petroleum|drilling|pipeline|energy"), "Energy", "sec_sic:energy"),
        (desc.str.contains("mining|gold|silver|copper|steel|aluminum|chemical|chemicals|paper|forest|fertilizer|materials"), "Basic Materials", "sec_sic:basic_materials"),
        (desc.str.contains("hospital|medical|pharma|biotech|therapeutic|health|diagnostic|laboratories"), "Healthcare", "sec_sic:healthcare"),
        (desc.str.contains("software|computer|semiconductor|data processing|internet|electronic computers"), "Technology", "sec_sic:technology"),
        (desc.str.contains("telecommunications|telephone|broadcast|media|entertainment|cable|wireless"), "Communication Services", "sec_sic:communication_services"),
        (desc.str.contains("electric|water supply|water|gas production|utility|utilities"), "Utilities", "sec_sic:utilities"),
        (desc.str.contains("food|foods|beverage|beverages|tobacco|grocery|drug stores|household"), "Consumer Defensive", "sec_sic:consumer_defensive"),
        (desc.str.contains("retail|restaurant|restaurants|lodging|hotel|apparel|auto dealers|consumer goods"), "Consumer Cyclical", "sec_sic:consumer_cyclical"),
        (desc.str.contains("transport|trucking|railroad|air freight|aerospace|machinery|industrial|manufacturing|construction|defense"), "Industrials", "sec_sic:industrials"),
    ]

    sector = pl.lit(None).cast(pl.Utf8)
    rule = pl.lit(None).cast(pl.Utf8)
    for condition, sector_value, rule_value in conditions:
        sector = pl.when(condition).then(pl.lit(sector_value)).otherwise(sector)
        rule = pl.when(condition).then(pl.lit(rule_value)).otherwise(rule)
    return sector, rule

