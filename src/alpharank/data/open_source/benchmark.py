from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import polars as pl

from alpharank.data.open_source.config import METRIC_SPECS


def load_eodhd_prices(data_dir: Path, tickers: Iterable[str], year: int) -> pl.DataFrame:
    start = f"{year}-01-01"
    end = f"{year}-12-31"
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    return (
        pl.read_parquet(data_dir / "US_Finalprice.parquet")
        .filter(pl.col("ticker").is_in(ticker_set))
        .filter((pl.col("date") >= pl.lit(start)) & (pl.col("date") <= pl.lit(end)))
        .select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"])
        .sort(["ticker", "date"])
    )


def normalize_eodhd_financials(data_dir: Path, tickers: Iterable[str], year: int) -> pl.DataFrame:
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    frames: list[pl.DataFrame] = []
    parquet_map = {
        "income_statement": "US_Income_statement.parquet",
        "balance_sheet": "US_Balance_sheet.parquet",
        "cash_flow": "US_Cash_flow.parquet",
    }
    for statement, path in parquet_map.items():
        df = pl.read_parquet(data_dir / path).filter(pl.col("ticker").is_in(ticker_set))
        df = df.filter(pl.col("date").str.starts_with(f"{year}"))
        for spec in [spec for spec in METRIC_SPECS if spec.statement == statement]:
            if spec.eodhd_column not in df.columns:
                continue
            frames.append(
                df.select(
                    [
                        pl.col("ticker"),
                        pl.lit(statement).alias("statement"),
                        pl.lit(spec.metric).alias("metric"),
                        pl.col("date"),
                        pl.col("filing_date").cast(pl.Utf8, strict=False).alias("filing_date"),
                        pl.col(spec.eodhd_column).cast(pl.Float64, strict=False).alias("value"),
                        pl.lit("eodhd").alias("source"),
                        pl.lit(spec.eodhd_column).alias("source_label"),
                    ]
                ).filter(pl.col("value").is_not_null())
            )
    return pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date"]) if frames else _empty_financial_frame()


def build_price_alignment(eodhd_prices: pl.DataFrame, yahoo_prices: pl.DataFrame) -> pl.DataFrame:
    joined = eodhd_prices.rename(
        {
            "adjusted_close": "eodhd_adjusted_close",
            "close": "eodhd_close",
            "open": "eodhd_open",
            "high": "eodhd_high",
            "low": "eodhd_low",
            "volume": "eodhd_volume",
        }
    ).join(
        yahoo_prices.rename(
            {
                "adjusted_close": "yahoo_adjusted_close",
                "close": "yahoo_close",
                "open": "yahoo_open",
                "high": "yahoo_high",
                "low": "yahoo_low",
                "volume": "yahoo_volume",
            }
        ),
        on=["ticker", "date"],
        how="full",
        coalesce=True,
    )
    return (
        joined.with_columns(
            [
                pl.when(pl.col("eodhd_adjusted_close").is_not_null() & pl.col("yahoo_adjusted_close").is_not_null())
                .then(pl.lit("matched"))
                .when(pl.col("eodhd_adjusted_close").is_not_null())
                .then(pl.lit("eodhd_only"))
                .otherwise(pl.lit("yahoo_only"))
                .alias("match_status"),
                (pl.col("yahoo_adjusted_close") - pl.col("eodhd_adjusted_close")).alias("adjusted_close_diff"),
            ]
        )
        .with_columns(
            pl.when(pl.col("eodhd_adjusted_close").abs() > 0)
            .then((pl.col("adjusted_close_diff") / pl.col("eodhd_adjusted_close")) * 10_000)
            .otherwise(None)
            .alias("adjusted_close_diff_bps")
        )
        .sort(["ticker", "date"])
    )


def build_financial_alignment(
    eodhd_financials: pl.DataFrame,
    open_financials: pl.DataFrame,
    open_source: str,
) -> pl.DataFrame:
    open_frame = (
        open_financials.filter(pl.col("source") == open_source)
        .rename(
            {
                "value": "open_value",
                "filing_date": "open_filing_date",
                "source_label": "open_source_label",
            }
        )
        .with_columns(
            [
                pl.col("open_filing_date").cast(pl.Utf8, strict=False),
                pl.col("open_source_label").cast(pl.Utf8, strict=False),
            ]
        )
    )
    eodhd_frame = (
        eodhd_financials.rename(
            {
                "value": "eodhd_value",
                "filing_date": "eodhd_filing_date",
                "source_label": "eodhd_source_label",
            }
        )
        .with_columns(
            [
                pl.col("eodhd_filing_date").cast(pl.Utf8, strict=False),
                pl.col("eodhd_source_label").cast(pl.Utf8, strict=False),
            ]
        )
    )
    joined = eodhd_frame.join(
        open_frame.select(
            [
                "ticker",
                "statement",
                "metric",
                "date",
                "open_value",
                "open_filing_date",
                "open_source_label",
            ]
        ),
        on=["ticker", "statement", "metric", "date"],
        how="full",
        coalesce=True,
    )
    return (
        joined.with_columns(
            [
                pl.lit(open_source).alias("open_source"),
                pl.when(pl.col("eodhd_value").is_not_null() & pl.col("open_value").is_not_null())
                .then(pl.lit("matched"))
                .when(pl.col("eodhd_value").is_not_null())
                .then(pl.lit("eodhd_only"))
                .otherwise(pl.lit("open_only"))
                .alias("match_status"),
                (pl.col("open_value") - pl.col("eodhd_value")).alias("value_diff"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("eodhd_value").abs() > 0)
                .then((pl.col("value_diff") / pl.col("eodhd_value")) * 10_000)
                .otherwise(None)
                .alias("value_diff_bps"),
                _date_diff_days_expr("eodhd_filing_date", "open_filing_date").alias("filing_date_diff_days"),
            ]
        )
        .sort(["open_source", "ticker", "statement", "metric", "date"])
    )


def summarize_alignment(
    *,
    tickers: Iterable[str],
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    output_path: Path,
) -> None:
    price_summary = price_alignment.group_by(["ticker", "match_status"]).agg(
        [
            pl.len().alias("rows"),
            pl.col("adjusted_close_diff").abs().max().alias("max_abs_diff"),
            pl.col("adjusted_close_diff_bps").abs().max().alias("max_abs_diff_bps"),
        ]
    )
    financial_summary = financial_alignment.group_by(["open_source", "statement", "metric", "match_status"]).agg(
        [
            pl.len().alias("rows"),
            pl.col("value_diff").abs().max().alias("max_abs_diff"),
            pl.col("value_diff_bps").abs().max().alias("max_abs_diff_bps"),
            pl.col("filing_date_diff_days").abs().max().alias("max_abs_filing_date_diff_days"),
        ]
    )
    payload = {
        "tickers": list(tickers),
        "price_summary": price_summary.to_dicts(),
        "financial_summary": financial_summary.to_dicts(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _date_diff_days_expr(left: str, right: str) -> pl.Expr:
    left_date = pl.col(left).str.strptime(pl.Date, strict=False)
    right_date = pl.col(right).str.strptime(pl.Date, strict=False)
    return pl.when(left_date.is_not_null() & right_date.is_not_null()).then((right_date - left_date).dt.total_days()).otherwise(None)


def _empty_financial_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "ticker": pl.String,
            "statement": pl.String,
            "metric": pl.String,
            "date": pl.String,
            "filing_date": pl.String,
            "value": pl.Float64,
            "source": pl.String,
            "source_label": pl.String,
        }
    )
