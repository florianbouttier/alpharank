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


def load_sp500_tickers_for_year(data_dir: Path, year: int) -> tuple[str, ...]:
    return tuple(
        pl.read_csv(data_dir / "SP500_Constituents.csv", try_parse_dates=True)
        .filter(pl.col("Date").dt.year() == year)
        .filter(pl.col("Ticker").is_not_null() & (pl.col("Ticker") != ""))
        .with_columns(pl.col("Ticker").str.replace_all(r"\.", "-").alias("Ticker"))
        .select("Ticker")
        .unique()
        .sort("Ticker")
        .to_series()
        .to_list()
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


def build_error_summary_tables(
    *,
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    threshold_pct: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    threshold_bps = threshold_pct * 100.0
    price_summary = pl.DataFrame(
        {
            "dataset": ["price"],
            "source": ["yfinance"],
            "matched_rows": [_count_rows(price_alignment, pl.col("match_status") == "matched")],
            "error_rows": [_count_rows(price_alignment, (pl.col("match_status") == "matched") & (pl.col("adjusted_close_diff_bps").abs() > threshold_bps))],
            "eodhd_only_rows": [_count_rows(price_alignment, pl.col("match_status") == "eodhd_only")],
            "open_only_rows": [_count_rows(price_alignment, pl.col("match_status") == "yahoo_only")],
            "max_abs_diff_pct": [(_safe_max(price_alignment, "adjusted_close_diff_bps") or 0.0) / 100.0],
        }
    ).with_columns(
        _error_rate_expr("matched_rows", "error_rows").alias("error_rate_pct")
    )

    statement_summary = _aggregate_errors(
        financial_alignment,
        by=["open_source", "statement"],
        diff_col="value_diff_bps",
        threshold_bps=threshold_bps,
    ).rename({"open_source": "source"})
    metric_summary = _aggregate_errors(
        financial_alignment,
        by=["open_source", "statement", "metric"],
        diff_col="value_diff_bps",
        threshold_bps=threshold_bps,
    ).rename({"open_source": "source"})
    return price_summary, statement_summary, metric_summary


def build_coverage_audit(
    *,
    sp500_tickers: tuple[str, ...],
    benchmark_tickers: tuple[str, ...],
    sec_mapping: pl.DataFrame,
    yahoo_availability: pl.DataFrame,
) -> pl.DataFrame:
    base = pl.DataFrame({"ticker_root": list(sp500_tickers)}).with_columns(
        [
            (pl.col("ticker_root") + pl.lit(".US")).alias("ticker"),
            pl.col("ticker_root").is_in(list(benchmark_tickers)).alias("selected_for_benchmark"),
        ]
    )
    sec_rows = sec_mapping.select(
        [
            pl.col("ticker").alias("ticker_root"),
            pl.lit(True).alias("sec_filing_available"),
            pl.col("name").alias("sec_name"),
            pl.col("exchange").alias("sec_exchange"),
            pl.col("cik").cast(pl.Utf8).str.zfill(10).alias("sec_cik"),
        ]
    )
    return (
        base.join(sec_rows, on="ticker_root", how="left", coalesce=True)
        .join(yahoo_availability, on=["ticker", "ticker_root"], how="left", coalesce=True)
        .with_columns(
            [
                pl.col("sec_filing_available").fill_null(False),
                pl.col("yahoo_price_available").fill_null(False),
            ]
        )
        .with_columns(
            (pl.col("sec_filing_available") | pl.col("yahoo_price_available")).alias("available_in_yahoo_or_sec")
        )
        .sort("ticker_root")
    )


def write_html_report(
    *,
    output_path: Path,
    year: int,
    threshold_pct: float,
    benchmark_tickers: tuple[str, ...],
    coverage: pl.DataFrame,
    price_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
) -> None:
    coverage_totals = coverage.select(
        [
            pl.len().alias("sp500_2025_tickers"),
            pl.col("yahoo_price_available").sum().alias("yahoo_price_available"),
            pl.col("sec_filing_available").sum().alias("sec_filing_available"),
            pl.col("available_in_yahoo_or_sec").sum().alias("available_in_yahoo_or_sec"),
        ]
    ).to_dicts()[0]
    missing = coverage.filter(~pl.col("available_in_yahoo_or_sec")).select(["ticker_root"]).to_series().to_list()
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Open-source cadrage {year}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #1f2937; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .meta {{ margin-bottom: 24px; color: #475569; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: white; border: 1px solid #dbeafe; border-radius: 10px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #e2e8f0; padding: 8px 10px; text-align: left; font-size: 13px; }}
    th {{ background: #eff6ff; }}
    code {{ background: #e2e8f0; padding: 1px 4px; border-radius: 4px; }}
    .muted {{ color: #64748b; }}
  </style>
</head>
<body>
  <h1>Open-source cadrage {year}</h1>
  <div class="meta">
    Benchmark tickers: <code>{", ".join(benchmark_tickers)}</code><br>
    Error threshold: <code>{threshold_pct:.3f}%</code>
  </div>
  <h2>Coverage Audit</h2>
  <div class="grid">
    <div class="card"><strong>S&amp;P 500 tickers in {year}</strong><br>{coverage_totals["sp500_2025_tickers"]}</div>
    <div class="card"><strong>Yahoo price available</strong><br>{coverage_totals["yahoo_price_available"]}</div>
    <div class="card"><strong>SEC filing available</strong><br>{coverage_totals["sec_filing_available"]}</div>
    <div class="card"><strong>Available in Yahoo or SEC</strong><br>{coverage_totals["available_in_yahoo_or_sec"]}</div>
  </div>
  <p class="muted">Missing from both sources: {", ".join(missing) if missing else "none"}</p>
  <h2>Price Errors</h2>
  {_frame_to_html(price_summary)}
  <h2>Statement Errors</h2>
  {_frame_to_html(statement_summary)}
  <h2>Metric Errors</h2>
  {_frame_to_html(metric_summary)}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


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


def _aggregate_errors(df: pl.DataFrame, *, by: list[str], diff_col: str, threshold_bps: float) -> pl.DataFrame:
    return df.group_by(by).agg(
        [
            _count_expr(pl.col("match_status") == "matched").alias("matched_rows"),
            _count_expr((pl.col("match_status") == "matched") & (pl.col(diff_col).abs() > threshold_bps)).alias("error_rows"),
            _count_expr(pl.col("match_status") == "eodhd_only").alias("eodhd_only_rows"),
            _count_expr(pl.col("match_status") == "open_only").alias("open_only_rows"),
            (pl.col(diff_col).abs().max() / 100.0).alias("max_abs_diff_pct"),
        ]
    ).with_columns(_error_rate_expr("matched_rows", "error_rows").alias("error_rate_pct")).sort(by)


def _count_expr(predicate: pl.Expr) -> pl.Expr:
    return pl.when(predicate).then(pl.lit(1)).otherwise(pl.lit(0)).sum()


def _count_rows(df: pl.DataFrame, predicate: pl.Expr) -> int:
    return int(df.select(_count_expr(predicate).alias("count")).item())


def _safe_max(df: pl.DataFrame, column: str) -> float | None:
    if column not in df.columns or df.is_empty():
        return None
    value = df.select(pl.col(column).abs().max()).item()
    return None if value is None else float(value)


def _error_rate_expr(matched_col: str, error_col: str) -> pl.Expr:
    return (
        pl.when(pl.col(matched_col) > 0)
        .then((pl.col(error_col) / pl.col(matched_col)) * 100.0)
        .otherwise(0.0)
    )


def _frame_to_html(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p class='muted'>No rows.</p>"
    pdf = df.to_pandas()
    return pdf.to_html(index=False, border=0)
