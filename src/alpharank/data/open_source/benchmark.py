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
        .with_columns(pl.col("Ticker").str.replace_all(r"\\.", "-").alias("Ticker"))
        .select("Ticker")
        .unique()
        .sort("Ticker")
        .to_series()
        .to_list()
    )


def normalize_eodhd_financials(data_dir: Path, tickers: Iterable[str], year: int) -> pl.DataFrame:
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    frames: list[pl.DataFrame] = []
    for path in sorted({spec.eodhd_path for spec in METRIC_SPECS if spec.statement not in {"earnings"}}):
        df = pl.read_parquet(data_dir / path).filter(pl.col("ticker").is_in(ticker_set))
        date_col = "dateFormatted" if path == "US_share.parquet" else "date"
        if date_col not in df.columns:
            continue
        df = df.filter(pl.col(date_col).cast(pl.Utf8, strict=False).str.starts_with(f"{year}"))
        for spec in [spec for spec in METRIC_SPECS if spec.eodhd_path == path and spec.statement != "earnings"]:
            if spec.eodhd_column not in df.columns:
                continue
            filing_col = "dateFormatted" if path == "US_share.parquet" else "filing_date"
            frames.append(
                df.select(
                    [
                        pl.col("ticker"),
                        pl.lit(spec.statement).alias("statement"),
                        pl.lit(spec.metric).alias("metric"),
                        pl.col(date_col).cast(pl.Utf8, strict=False).alias("date"),
                        pl.col(filing_col).cast(pl.Utf8, strict=False).alias("filing_date"),
                        pl.col(spec.eodhd_column).cast(pl.Float64, strict=False).alias("value"),
                        pl.lit("eodhd").alias("source"),
                        pl.lit(spec.eodhd_column).alias("source_label"),
                    ]
                ).filter(pl.col("value").is_not_null())
            )
    return pl.concat(frames, how="vertical").sort(["ticker", "statement", "metric", "date"]) if frames else _empty_financial_frame()


def normalize_eodhd_earnings(data_dir: Path, tickers: Iterable[str], year: int) -> pl.DataFrame:
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    df = (
        pl.read_parquet(data_dir / "US_Earnings.parquet")
        .filter(pl.col("ticker").is_in(ticker_set))
        .filter(pl.col("reportDate").str.starts_with(str(year)) | pl.col("date").str.starts_with(str(year)))
    )
    frames: list[pl.DataFrame] = []
    metric_map = {
        "eps_actual": "epsActual",
        "eps_estimate": "epsEstimate",
        "surprise_percent": "surprisePercent",
    }
    for metric, column in metric_map.items():
        if column not in df.columns:
            continue
        frames.append(
            df.select(
                [
                    pl.col("ticker"),
                    pl.lit("earnings").alias("statement"),
                    pl.lit(metric).alias("metric"),
                    pl.col("reportDate").cast(pl.Utf8, strict=False).alias("date"),
                    pl.col("reportDate").cast(pl.Utf8, strict=False).alias("filing_date"),
                    pl.col(column).cast(pl.Float64, strict=False).alias("value"),
                    pl.lit("eodhd").alias("source"),
                    pl.lit(column).alias("source_label"),
                ]
            ).filter(pl.col("value").is_not_null())
        )
    return pl.concat(frames, how="vertical").sort(["ticker", "metric", "date"]) if frames else _empty_financial_frame()


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
                pl.lit("price").alias("statement"),
                pl.lit("adjusted_close").alias("metric"),
                pl.when(pl.col("eodhd_adjusted_close").is_not_null() & pl.col("yahoo_adjusted_close").is_not_null())
                .then(pl.lit("matched"))
                .when(pl.col("eodhd_adjusted_close").is_not_null())
                .then(pl.lit("eodhd_only"))
                .otherwise(pl.lit("open_only"))
                .alias("match_status"),
                (pl.col("yahoo_adjusted_close") - pl.col("eodhd_adjusted_close")).alias("value_diff"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("eodhd_adjusted_close").abs() > 0)
                .then((pl.col("value_diff") / pl.col("eodhd_adjusted_close")) * 100.0)
                .otherwise(None)
                .alias("diff_pct"),
                pl.lit("yfinance").alias("source"),
                pl.lit(0).alias("date_diff_days"),
            ]
        )
        .sort(["ticker", "date"])
    )


def build_financial_alignment(
    eodhd_financials: pl.DataFrame,
    open_financials: pl.DataFrame,
    open_source: str,
    tolerance_days: int = 10,
) -> pl.DataFrame:
    return _build_nearest_alignment(
        eodhd_frame=eodhd_financials,
        open_frame=open_financials.filter(pl.col("source") == open_source),
        open_source=open_source,
        key_cols=["ticker", "statement", "metric"],
        tolerance_days=tolerance_days,
    )


def build_earnings_alignment(
    eodhd_earnings: pl.DataFrame,
    yahoo_earnings: pl.DataFrame,
    tolerance_days: int = 7,
) -> pl.DataFrame:
    return _build_nearest_alignment(
        eodhd_frame=eodhd_earnings,
        open_frame=yahoo_earnings.filter(pl.col("statement") == "earnings"),
        open_source="yfinance_earnings",
        key_cols=["ticker", "statement", "metric"],
        tolerance_days=tolerance_days,
    )


def summarize_alignment(*, tickers: Iterable[str], price_alignment: pl.DataFrame, financial_alignment: pl.DataFrame, output_path: Path) -> None:
    price_summary = price_alignment.group_by(["ticker", "match_status"]).agg(
        [
            pl.len().alias("rows"),
            pl.col("value_diff").abs().max().alias("max_abs_diff"),
            pl.col("diff_pct").abs().max().alias("max_abs_diff_pct"),
        ]
    )
    financial_summary = financial_alignment.group_by(["source", "statement", "metric", "match_status"]).agg(
        [
            pl.len().alias("rows"),
            pl.col("value_diff").abs().max().alias("max_abs_diff"),
            pl.col("diff_pct").abs().max().alias("max_abs_diff_pct"),
            pl.col("date_diff_days").abs().max().alias("max_abs_date_diff_days"),
        ]
    )
    payload = {
        "tickers": list(tickers),
        "price_summary": price_summary.to_dicts(),
        "financial_summary": financial_summary.to_dicts(),
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_error_summary_tables(*, price_alignment: pl.DataFrame, financial_alignment: pl.DataFrame, threshold_pct: float) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    price_summary = _aggregate_errors(price_alignment, by=["source", "statement", "metric"], threshold_pct=threshold_pct)
    statement_summary = _aggregate_errors(financial_alignment, by=["source", "statement"], threshold_pct=threshold_pct)
    metric_summary = _aggregate_errors(financial_alignment, by=["source", "statement", "metric"], threshold_pct=threshold_pct)
    ticker_summary = _aggregate_errors(financial_alignment, by=["source", "ticker", "statement"], threshold_pct=threshold_pct)
    ticker_metric_summary = _aggregate_errors(financial_alignment, by=["source", "ticker", "statement", "metric"], threshold_pct=threshold_pct)
    return price_summary, statement_summary, metric_summary, ticker_summary, ticker_metric_summary


def build_error_detail_tables(*, price_alignment: pl.DataFrame, financial_alignment: pl.DataFrame, threshold_pct: float) -> tuple[pl.DataFrame, pl.DataFrame]:
    price_errors = price_alignment.filter(
        ((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct))
        | (pl.col("match_status") != "matched")
    ).sort(["ticker", "date"])
    financial_errors = financial_alignment.filter(
        ((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct))
        | (pl.col("match_status") != "matched")
    ).sort(["source", "ticker", "statement", "metric", "date"])
    return price_errors, financial_errors


def build_coverage_audit(*, sp500_tickers: tuple[str, ...], benchmark_tickers: tuple[str, ...], sec_mapping: pl.DataFrame, yahoo_availability: pl.DataFrame) -> pl.DataFrame:
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
        .with_columns((pl.col("sec_filing_available") | pl.col("yahoo_price_available")).alias("available_in_yahoo_or_sec"))
        .sort("ticker_root")
    )


def write_html_report(*, output_path: Path, year: int, threshold_pct: float, benchmark_tickers: tuple[str, ...], coverage: pl.DataFrame, price_summary: pl.DataFrame, statement_summary: pl.DataFrame, metric_summary: pl.DataFrame, ticker_summary: pl.DataFrame, ticker_metric_summary: pl.DataFrame) -> None:
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
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>Open-source audit {year}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 16px 0 24px; }}
    .card {{ background: white; border: 1px solid #dbeafe; border-radius: 10px; padding: 14px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 12px; }}
    th {{ background: #eff6ff; position: sticky; top: 0; }}
    .muted {{ color: #64748b; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>Open-source audit {year}</h1>
  <div class=\"muted\">Threshold: <strong>{threshold_pct:.3f}%</strong><br>Detailed benchmark tickers: <strong>{len(benchmark_tickers)}</strong></div>
  <div class=\"grid\">
    <div class=\"card\"><strong>S&P 500 tickers</strong><br>{coverage_totals['sp500_2025_tickers']}</div>
    <div class=\"card\"><strong>Yahoo price available</strong><br>{coverage_totals['yahoo_price_available']}</div>
    <div class=\"card\"><strong>SEC filing available</strong><br>{coverage_totals['sec_filing_available']}</div>
    <div class=\"card\"><strong>Available in Yahoo or SEC</strong><br>{coverage_totals['available_in_yahoo_or_sec']}</div>
  </div>
  <p class=\"muted\">Missing from both: {', '.join(missing) if missing else 'none'}</p>
  <div class=\"section\"><h2>Price Summary</h2>{_frame_to_html(price_summary)}</div>
  <div class=\"section\"><h2>By Statement</h2>{_frame_to_html(statement_summary)}</div>
  <div class=\"section\"><h2>By KPI</h2>{_frame_to_html(metric_summary)}</div>
  <div class=\"section\"><h2>By Ticker</h2>{_frame_to_html(ticker_summary)}</div>
  <div class=\"section\"><h2>By Ticker and KPI</h2>{_frame_to_html(ticker_metric_summary)}</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _aggregate_errors(df: pl.DataFrame, *, by: list[str], threshold_pct: float) -> pl.DataFrame:
    return df.group_by(by).agg(
        [
            _count_expr(pl.col("match_status") == "matched").alias("matched_rows"),
            _count_expr((pl.col("match_status") == "matched") & (pl.col("diff_pct").abs() > threshold_pct)).alias("error_rows"),
            _count_expr(pl.col("match_status") == "eodhd_only").alias("eodhd_only_rows"),
            _count_expr(pl.col("match_status") == "open_only").alias("open_only_rows"),
            pl.col("diff_pct").abs().max().alias("max_abs_diff_pct"),
            pl.col("date_diff_days").abs().max().alias("max_abs_date_diff_days"),
        ]
    ).with_columns(_error_rate_expr("matched_rows", "error_rows").alias("error_rate_pct")).sort(by)


def _build_nearest_alignment(*, eodhd_frame: pl.DataFrame, open_frame: pl.DataFrame, open_source: str, key_cols: list[str], tolerance_days: int) -> pl.DataFrame:
    eod = (
        eodhd_frame.select(key_cols + ["date", "filing_date", "value", "source_label"]).rename(
            {"date": "eodhd_date", "filing_date": "eodhd_filing_date", "value": "eodhd_value", "source_label": "eodhd_source_label"}
        )
        .with_row_index("eodhd_row_id")
        .with_columns(_parsed_date_expr("eodhd_date").alias("eodhd_date_dt"))
    )
    opn = (
        open_frame.select(key_cols + ["date", "filing_date", "value", "source_label"]).rename(
            {"date": "open_date", "filing_date": "open_filing_date", "value": "open_value", "source_label": "open_source_label"}
        )
        .with_row_index("open_row_id")
        .with_columns(_parsed_date_expr("open_date").alias("open_date_dt"))
    )

    candidate_matches = (
        eod.join(opn, on=key_cols, how="inner")
        .with_columns((pl.col("open_date_dt") - pl.col("eodhd_date_dt")).dt.total_days().alias("date_diff_days"))
        .with_columns(pl.col("date_diff_days").abs().alias("abs_date_diff_days"))
        .filter(pl.col("abs_date_diff_days") <= tolerance_days)
        .sort(["abs_date_diff_days", "eodhd_row_id", "open_row_id"])
    )

    matches = []
    used_eodhd: set[int] = set()
    used_open: set[int] = set()
    for row in candidate_matches.iter_rows(named=True):
        eodhd_row_id = int(row["eodhd_row_id"])
        open_row_id = int(row["open_row_id"])
        if eodhd_row_id in used_eodhd or open_row_id in used_open:
            continue
        used_eodhd.add(eodhd_row_id)
        used_open.add(open_row_id)
        matches.append(row)

    matched = pl.DataFrame(matches) if matches else pl.DataFrame(schema={
        "eodhd_row_id": pl.UInt32,
        "open_row_id": pl.UInt32,
        **{col: pl.String for col in key_cols},
        "eodhd_date": pl.String,
        "open_date": pl.String,
        "eodhd_filing_date": pl.String,
        "open_filing_date": pl.String,
        "eodhd_value": pl.Float64,
        "open_value": pl.Float64,
        "eodhd_source_label": pl.String,
        "open_source_label": pl.String,
        "date_diff_days": pl.Int64,
        "abs_date_diff_days": pl.Int64,
    })

    matched_rows = matched.select(
        key_cols
        + [
            pl.col("eodhd_date").alias("date"),
            pl.col("eodhd_filing_date"),
            pl.col("open_filing_date"),
            pl.col("eodhd_value"),
            pl.col("open_value"),
            pl.col("eodhd_source_label"),
            pl.col("open_source_label"),
            pl.col("date_diff_days"),
        ]
    ) if not matched.is_empty() else _empty_alignment_frame()

    matched_rows = matched_rows.with_columns(
        [
            pl.lit(open_source).alias("source"),
            pl.lit("matched").alias("match_status"),
            (pl.col("open_value") - pl.col("eodhd_value")).alias("value_diff"),
        ]
    ).with_columns(
        pl.when(pl.col("eodhd_value").abs() > 0)
        .then((pl.col("value_diff") / pl.col("eodhd_value")) * 100.0)
        .otherwise(None)
        .alias("diff_pct")
    )

    matched_rows = _standardize_alignment_output(matched_rows)

    eodhd_only = eod.filter(~pl.col("eodhd_row_id").is_in(list(used_eodhd))).select(
        key_cols
        + [
            pl.col("eodhd_date").alias("date"),
            pl.col("eodhd_filing_date"),
            pl.lit(None).cast(pl.Utf8).alias("open_filing_date"),
            pl.col("eodhd_value"),
            pl.lit(None).cast(pl.Float64).alias("open_value"),
            pl.col("eodhd_source_label"),
            pl.lit(None).cast(pl.Utf8).alias("open_source_label"),
            pl.lit(None).cast(pl.Int64).alias("date_diff_days"),
        ]
    ).with_columns(
        [
            pl.lit(open_source).alias("source"),
            pl.lit("eodhd_only").alias("match_status"),
            pl.lit(None).cast(pl.Float64).alias("value_diff"),
            pl.lit(None).cast(pl.Float64).alias("diff_pct"),
        ]
    )
    eodhd_only = _standardize_alignment_output(eodhd_only)

    open_only = opn.filter(~pl.col("open_row_id").is_in(list(used_open))).select(
        key_cols
        + [
            pl.col("open_date").alias("date"),
            pl.lit(None).cast(pl.Utf8).alias("eodhd_filing_date"),
            pl.col("open_filing_date"),
            pl.lit(None).cast(pl.Float64).alias("eodhd_value"),
            pl.col("open_value"),
            pl.lit(None).cast(pl.Utf8).alias("eodhd_source_label"),
            pl.col("open_source_label"),
            pl.lit(None).cast(pl.Int64).alias("date_diff_days"),
        ]
    ).with_columns(
        [
            pl.lit(open_source).alias("source"),
            pl.lit("open_only").alias("match_status"),
            pl.lit(None).cast(pl.Float64).alias("value_diff"),
            pl.lit(None).cast(pl.Float64).alias("diff_pct"),
        ]
    )
    open_only = _standardize_alignment_output(open_only)

    return pl.concat([matched_rows, eodhd_only, open_only], how="vertical").sort(key_cols + ["date"])


def _parsed_date_expr(column: str) -> pl.Expr:
    return pl.coalesce(
        pl.col(column).str.strptime(pl.Date, "%Y-%m-%d", strict=False),
        pl.col(column).str.strptime(pl.Date, "%Y-%m", strict=False),
        pl.col(column).str.strptime(pl.Date, "%Y-Q%q", strict=False),
    )


def _count_expr(predicate: pl.Expr) -> pl.Expr:
    return pl.when(predicate).then(pl.lit(1)).otherwise(pl.lit(0)).sum()


def _error_rate_expr(matched_col: str, error_col: str) -> pl.Expr:
    return pl.when(pl.col(matched_col) > 0).then((pl.col(error_col) / pl.col(matched_col)) * 100.0).otherwise(0.0)


def _frame_to_html(df: pl.DataFrame) -> str:
    if df.is_empty():
        return "<p>No rows.</p>"
    return df.to_pandas().to_html(index=False, border=0)


def _empty_financial_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ticker": pl.String,
        "statement": pl.String,
        "metric": pl.String,
        "date": pl.String,
        "filing_date": pl.String,
        "value": pl.Float64,
        "source": pl.String,
        "source_label": pl.String,
    })


def _empty_alignment_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={
        "ticker": pl.String,
        "statement": pl.String,
        "metric": pl.String,
        "date": pl.String,
        "eodhd_filing_date": pl.String,
        "open_filing_date": pl.String,
        "eodhd_value": pl.Float64,
        "open_value": pl.Float64,
        "eodhd_source_label": pl.String,
        "open_source_label": pl.String,
        "date_diff_days": pl.Int64,
    })


def _standardize_alignment_output(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        [
            pl.col("ticker").cast(pl.Utf8, strict=False),
            pl.col("statement").cast(pl.Utf8, strict=False),
            pl.col("metric").cast(pl.Utf8, strict=False),
            pl.col("date").cast(pl.Utf8, strict=False),
            pl.col("eodhd_filing_date").cast(pl.Utf8, strict=False),
            pl.col("open_filing_date").cast(pl.Utf8, strict=False),
            pl.col("eodhd_value").cast(pl.Float64, strict=False),
            pl.col("open_value").cast(pl.Float64, strict=False),
            pl.col("eodhd_source_label").cast(pl.Utf8, strict=False),
            pl.col("open_source_label").cast(pl.Utf8, strict=False),
            pl.col("date_diff_days").cast(pl.Int64, strict=False),
            pl.col("source").cast(pl.Utf8, strict=False),
            pl.col("match_status").cast(pl.Utf8, strict=False),
            pl.col("value_diff").cast(pl.Float64, strict=False),
            pl.col("diff_pct").cast(pl.Float64, strict=False),
        ]
    )
