from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Iterable

import polars as pl

from alpharank.data.open_source.config import METRIC_SPECS


def resolve_eodhd_output_dir(data_dir: Path) -> Path:
    direct = data_dir
    nested = data_dir / "eodhd" / "output"
    if (direct / "US_Finalprice.parquet").exists() and (direct / "SP500_Constituents.csv").exists():
        return direct
    if (nested / "US_Finalprice.parquet").exists() and (nested / "SP500_Constituents.csv").exists():
        return nested
    return direct


def load_eodhd_prices_between(
    data_dir: Path,
    tickers: Iterable[str],
    *,
    start_date: str,
    end_date: str,
) -> pl.DataFrame:
    resolved_dir = resolve_eodhd_output_dir(data_dir)
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    return (
        pl.read_parquet(resolved_dir / "US_Finalprice.parquet")
        .filter(pl.col("ticker").is_in(ticker_set))
        .filter((pl.col("date") >= pl.lit(start_date)) & (pl.col("date") <= pl.lit(end_date)))
        .select(["ticker", "date", "adjusted_close", "close", "open", "high", "low", "volume"])
        .sort(["ticker", "date"])
    )


def load_eodhd_prices(data_dir: Path, tickers: Iterable[str], year: int) -> pl.DataFrame:
    return load_eodhd_prices_between(
        data_dir,
        tickers,
        start_date=f"{year}-01-01",
        end_date=f"{year}-12-31",
    )


def load_sp500_tickers_for_year(data_dir: Path, year: int) -> tuple[str, ...]:
    resolved_dir = resolve_eodhd_output_dir(data_dir)
    return tuple(
        pl.read_csv(resolved_dir / "SP500_Constituents.csv", try_parse_dates=True)
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
    resolved_dir = resolve_eodhd_output_dir(data_dir)
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    frames: list[pl.DataFrame] = []
    for path in sorted({spec.eodhd_path for spec in METRIC_SPECS if spec.statement not in {"earnings"}}):
        df = pl.read_parquet(resolved_dir / path).filter(pl.col("ticker").is_in(ticker_set))
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
    resolved_dir = resolve_eodhd_output_dir(data_dir)
    ticker_set = [f"{ticker}.US" for ticker in tickers]
    df = (
        pl.read_parquet(resolved_dir / "US_Earnings.parquet")
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


def build_error_summary_tables(
    *,
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    threshold_pct: float,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    price_summary = _aggregate_errors(price_alignment, by=["source", "statement", "metric"], threshold_pct=threshold_pct)
    statement_summary = _aggregate_errors(financial_alignment, by=["source", "statement"], threshold_pct=threshold_pct)
    metric_summary = _aggregate_errors(financial_alignment, by=["source", "statement", "metric"], threshold_pct=threshold_pct)
    ticker_summary = _aggregate_errors(financial_alignment, by=["source", "ticker", "statement"], threshold_pct=threshold_pct)
    ticker_metric_summary = _aggregate_errors(financial_alignment, by=["source", "ticker", "statement", "metric"], threshold_pct=threshold_pct)
    price_ticker_summary = _aggregate_errors(price_alignment, by=["source", "ticker", "statement"], threshold_pct=threshold_pct)
    price_ticker_metric_summary = _aggregate_errors(
        price_alignment,
        by=["source", "ticker", "statement", "metric"],
        threshold_pct=threshold_pct,
    )
    return (
        price_summary,
        statement_summary,
        metric_summary,
        ticker_summary,
        ticker_metric_summary,
        price_ticker_summary,
        price_ticker_metric_summary,
    )


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


def build_audited_metric_catalog(
    *,
    include_yfinance_financials: bool,
    include_yfinance_earnings: bool,
    include_sec_filing_financials: bool,
    include_simfin_financials: bool,
    include_open_source_consolidated: bool,
) -> pl.DataFrame:
    rows: list[dict[str, str]] = [
        {
            "source": "yfinance",
            "statement": "price",
            "metric": "adjusted_close",
            "reference_field": "adjusted_close",
            "open_source_field": "adjusted_close",
        }
    ]
    for spec in METRIC_SPECS:
        if spec.statement == "earnings":
            if include_yfinance_earnings:
                rows.append(
                    {
                        "source": "yfinance_earnings",
                        "statement": spec.statement,
                        "metric": spec.metric,
                        "reference_field": spec.eodhd_column,
                        "open_source_field": spec.metric,
                    }
                )
            continue
        rows.append(
            {
                "source": "sec_companyfacts",
                "statement": spec.statement,
                "metric": spec.metric,
                "reference_field": spec.eodhd_column,
                "open_source_field": ",".join(spec.sec_tags) if spec.sec_tags else "derived",
            }
        )
        if include_sec_filing_financials:
            rows.append(
                {
                    "source": "sec_filing",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "reference_field": spec.eodhd_column,
                    "open_source_field": ",".join(spec.sec_tags) if spec.sec_tags else "derived_from_filing",
                }
            )
        if include_yfinance_financials and spec.yfinance_rows:
            rows.append(
                {
                    "source": "yfinance",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "reference_field": spec.eodhd_column,
                    "open_source_field": ",".join(spec.yfinance_rows),
                }
            )
        if include_simfin_financials and spec.simfin_columns:
            rows.append(
                {
                    "source": "simfin",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "reference_field": spec.eodhd_column,
                    "open_source_field": ",".join(spec.simfin_columns),
                }
            )
        if include_open_source_consolidated and spec.statement != "earnings":
            rows.append(
                {
                    "source": "open_source_consolidated",
                    "statement": spec.statement,
                    "metric": spec.metric,
                    "reference_field": spec.eodhd_column,
                    "open_source_field": "sec_companyfacts -> sec_filing -> simfin -> yfinance fallback",
                }
            )
    return pl.DataFrame(rows).sort(["source", "statement", "metric"])


def write_html_report(
    *,
    output_path: Path,
    year: int,
    threshold_pct: float,
    benchmark_tickers: tuple[str, ...],
    coverage: pl.DataFrame,
    audited_metric_catalog: pl.DataFrame,
    consolidation_source_summary: pl.DataFrame,
    price_summary: pl.DataFrame,
    statement_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    ticker_metric_summary: pl.DataFrame,
    price_ticker_summary: pl.DataFrame,
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
    ticker_overview = _build_ticker_overview(coverage=coverage, price_ticker_summary=price_ticker_summary, ticker_summary=ticker_summary)
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
    a {{ color: #1d4ed8; text-decoration: none; }}
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
  <p><a href=\"tickers/index.html\">Ticker reports</a> | <a href=\"kpis/index.html\">KPI reports</a></p>
  <div class=\"section\"><h2>Audited KPI Catalog</h2>{_frame_to_html(audited_metric_catalog)}</div>
  <div class=\"section\"><h2>Price Summary</h2>{_frame_to_html(_sort_for_display(price_summary))}</div>
  <div class=\"section\"><h2>Open-source Consolidation Lineage</h2>{_frame_to_html(_sort_for_display(consolidation_source_summary))}</div>
  <div class=\"section\"><h2>By Statement</h2>{_frame_to_html(_sort_for_display(statement_summary))}</div>
  <div class=\"section\"><h2>By KPI</h2>{_frame_to_html(_sort_for_display(metric_summary))}</div>
  <div class=\"section\"><h2>By Ticker</h2>{_frame_to_html(_sort_for_display(ticker_overview))}</div>
  <div class=\"section\"><h2>By Ticker and KPI</h2>{_frame_to_html(_sort_for_display(ticker_metric_summary))}</div>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def write_detail_reports(
    *,
    output_dir: Path,
    year: int,
    threshold_pct: float,
    coverage: pl.DataFrame,
    audited_metric_catalog: pl.DataFrame,
    consolidated_financials: pl.DataFrame,
    consolidated_lineage: pl.DataFrame,
    price_alignment: pl.DataFrame,
    financial_alignment: pl.DataFrame,
    price_error_details: pl.DataFrame,
    financial_error_details: pl.DataFrame,
    price_summary: pl.DataFrame,
    metric_summary: pl.DataFrame,
    ticker_summary: pl.DataFrame,
    ticker_metric_summary: pl.DataFrame,
    price_ticker_summary: pl.DataFrame,
    price_ticker_metric_summary: pl.DataFrame,
) -> None:
    ticker_dir = output_dir / "tickers"
    metric_dir = output_dir / "kpis"
    ticker_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)

    ticker_index_rows: list[dict[str, object]] = []
    statement_catalog = (
        audited_metric_catalog.select(["source", "statement"])
        .unique()
        .sort(["source", "statement"])
    )
    for coverage_row in coverage.sort("ticker").iter_rows(named=True):
        ticker = str(coverage_row["ticker"])
        ticker_root = str(coverage_row["ticker_root"])
        file_name = f"{_slugify(ticker)}.html"
        page_path = ticker_dir / file_name

        local_price_summary = price_ticker_metric_summary.filter(pl.col("ticker") == ticker)
        local_metric_summary = ticker_metric_summary.filter(pl.col("ticker") == ticker)
        local_kpi_summary = _catalog_join(
            audited_metric_catalog,
            pl.concat([local_price_summary, local_metric_summary], how="diagonal_relaxed"),
            keys=["source", "statement", "metric"],
        )
        local_statement_summary = _catalog_join(
            statement_catalog,
            pl.concat(
                [
                    price_ticker_summary.filter(pl.col("ticker") == ticker),
                    ticker_summary.filter(pl.col("ticker") == ticker),
                ],
                how="diagonal_relaxed",
            ),
            keys=["source", "statement"],
        )
        local_price_comparison = _build_price_comparison_table(
            price_alignment.filter(pl.col("ticker") == ticker),
            threshold_pct=threshold_pct,
        )
        local_consolidated_financials = _build_consolidated_financial_table(
            consolidated_financials.filter(pl.col("ticker") == ticker)
        )
        local_consolidated_lineage = _build_consolidated_lineage_table(
            consolidated_lineage.filter(pl.col("ticker") == ticker)
        )
        local_financial_comparison = _build_financial_comparison_table(
            financial_alignment.filter(pl.col("ticker") == ticker),
            threshold_pct=threshold_pct,
        )
        local_price_errors = price_error_details.filter(pl.col("ticker") == ticker)
        local_financial_errors = financial_error_details.filter(pl.col("ticker") == ticker)

        _write_html_page(
            output_path=page_path,
            title=f"{ticker} audit {year}",
            subtitle=(
                f"Threshold {threshold_pct:.3f}% | Yahoo price available: {coverage_row['yahoo_price_available']} | "
                f"SEC filing available: {coverage_row['sec_filing_available']}"
            ),
            sections=[
                ("Statement coverage", _sort_for_display(local_statement_summary)),
                ("KPI coverage", _sort_for_display(local_kpi_summary)),
                ("Price comparison", local_price_comparison),
                ("Open-source consolidated financials", local_consolidated_financials),
                ("Open-source source lineage", local_consolidated_lineage),
                ("Financial comparison", local_financial_comparison),
                ("Price errors", local_price_errors),
                ("Financial errors", local_financial_errors),
            ],
            navigation='<p><a href="../report.html">Global report</a> | <a href="index.html">Ticker index</a> | <a href="../kpis/index.html">KPI index</a></p>',
        )

        price_overview = _aggregate_overview(price_ticker_summary.filter(pl.col("ticker") == ticker), "price")
        financial_overview = _aggregate_overview(ticker_summary.filter(pl.col("ticker") == ticker), "financial")
        ticker_index_rows.append(
            {
                "ticker": ticker,
                "ticker_root": ticker_root,
                "yahoo_price_available": bool(coverage_row["yahoo_price_available"]),
                "sec_filing_available": bool(coverage_row["sec_filing_available"]),
                "price_matched_rows": price_overview["matched_rows"],
                "price_error_rows": price_overview["error_rows"],
                "price_error_rate_pct": price_overview["error_rate_pct"],
                "financial_matched_rows": financial_overview["matched_rows"],
                "financial_error_rows": financial_overview["error_rows"],
                "financial_error_rate_pct": financial_overview["error_rate_pct"],
                "report": file_name,
            }
        )

    ticker_index = pl.DataFrame(ticker_index_rows).sort(["financial_error_rate_pct", "price_error_rate_pct", "ticker"], descending=[True, True, False])
    _write_html_page(
        output_path=ticker_dir / "index.html",
        title=f"Ticker audit index {year}",
        subtitle=f"Threshold {threshold_pct:.3f}%",
        sections=[("Tickers", _with_links(ticker_index, "report", label_col="ticker"))],
        navigation='<p><a href="../report.html">Global report</a> | <a href="../kpis/index.html">KPI index</a></p>',
    )

    metric_index_rows: list[dict[str, object]] = []
    metric_catalog = audited_metric_catalog.select(["source", "statement", "metric"]).iter_rows(named=True)
    for metric_row in metric_catalog:
        source = str(metric_row["source"])
        statement = str(metric_row["statement"])
        metric = str(metric_row["metric"])
        file_name = f"{_slugify(source)}__{_slugify(statement)}__{_slugify(metric)}.html"
        page_path = metric_dir / file_name

        if statement == "price":
            global_summary = price_summary.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )
            ticker_rows = price_ticker_metric_summary.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )
            error_rows = price_error_details.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )
        else:
            global_summary = metric_summary.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )
            ticker_rows = ticker_metric_summary.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )
            error_rows = financial_error_details.filter(
                (pl.col("source") == source) & (pl.col("statement") == statement) & (pl.col("metric") == metric)
            )

        _write_html_page(
            output_path=page_path,
            title=f"{source} / {statement} / {metric}",
            subtitle=f"Threshold {threshold_pct:.3f}%",
            sections=[
                ("Global summary", global_summary),
                ("By ticker", _sort_for_display(ticker_rows)),
                ("Error details", error_rows),
            ],
            navigation='<p><a href="../report.html">Global report</a> | <a href="../tickers/index.html">Ticker index</a> | <a href="index.html">KPI index</a></p>',
        )
        summary_row = global_summary.to_dicts()[0] if not global_summary.is_empty() else {}
        metric_index_rows.append(
            {
                "source": source,
                "statement": statement,
                "metric": metric,
                "matched_rows": int(summary_row.get("matched_rows", 0)),
                "error_rows": int(summary_row.get("error_rows", 0)),
                "error_rate_pct": float(summary_row.get("error_rate_pct", 0.0)),
                "eodhd_rows": int(summary_row.get("eodhd_rows", 0)),
                "open_rows": int(summary_row.get("open_rows", 0)),
                "report": file_name,
            }
        )

    metric_index = pl.DataFrame(metric_index_rows).sort(["error_rate_pct", "matched_rows", "metric"], descending=[True, True, False])
    _write_html_page(
        output_path=metric_dir / "index.html",
        title=f"KPI audit index {year}",
        subtitle=f"Threshold {threshold_pct:.3f}%",
        sections=[("KPIs", _with_links(metric_index, "report", label_cols=["source", "statement", "metric"]))],
        navigation='<p><a href="../report.html">Global report</a> | <a href="../tickers/index.html">Ticker index</a></p>',
    )


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
    ).with_columns(
        [
            (pl.col("matched_rows") - pl.col("error_rows")).alias("ok_rows"),
            (pl.col("matched_rows") + pl.col("eodhd_only_rows")).alias("eodhd_rows"),
            (pl.col("matched_rows") + pl.col("open_only_rows")).alias("open_rows"),
            (pl.col("matched_rows") + pl.col("eodhd_only_rows") + pl.col("open_only_rows")).alias("total_rows"),
            (pl.col("eodhd_only_rows") + pl.col("open_only_rows")).alias("coverage_gap_rows"),
            _error_rate_expr("matched_rows", "error_rows").alias("error_rate_pct"),
        ]
    ).sort(by)


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
    headers = "".join(f"<th>{escape(str(column))}</th>" for column in df.columns)
    body_rows = []
    for row in df.iter_rows(named=True):
        cells = "".join(f"<td>{_format_html_value(row[column])}</td>" for column in df.columns)
        body_rows.append(f"<tr>{cells}</tr>")
    body = "".join(body_rows)
    return f"<table><thead><tr>{headers}</tr></thead><tbody>{body}</tbody></table>"


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


def _format_html_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, str) and value.startswith("<a href=") and value.endswith("</a>"):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        if value.is_integer():
            return f"{int(value):,}"
        return f"{value:,.6f}".rstrip("0").rstrip(".")
    return escape(str(value))


def _catalog_join(catalog: pl.DataFrame, actual: pl.DataFrame, *, keys: list[str]) -> pl.DataFrame:
    numeric_columns = [
        "matched_rows",
        "ok_rows",
        "error_rows",
        "eodhd_rows",
        "open_rows",
        "total_rows",
        "coverage_gap_rows",
        "eodhd_only_rows",
        "open_only_rows",
    ]
    value_columns = ["max_abs_diff_pct", "max_abs_date_diff_days", "error_rate_pct"]
    return (
        catalog.join(actual, on=keys, how="left", coalesce=True)
        .with_columns(
            [pl.col(column).fill_null(0) for column in numeric_columns if column in actual.columns]
            + [pl.col(column).fill_null(0.0) for column in value_columns if column in actual.columns]
        )
        .sort(keys)
    )


def _sort_for_display(df: pl.DataFrame) -> pl.DataFrame:
    sort_columns: list[str] = []
    descending: list[bool] = []
    for column in ("error_rate_pct", "error_rows", "matched_rows"):
        if column in df.columns:
            sort_columns.append(column)
            descending.append(True)
    if "ticker" in df.columns:
        sort_columns.append("ticker")
        descending.append(False)
    elif "metric" in df.columns:
        sort_columns.append("metric")
        descending.append(False)
    elif "statement" in df.columns:
        sort_columns.append("statement")
        descending.append(False)
    return df.sort(sort_columns, descending=descending) if sort_columns else df


def _build_ticker_overview(*, coverage: pl.DataFrame, price_ticker_summary: pl.DataFrame, ticker_summary: pl.DataFrame) -> pl.DataFrame:
    price_overview = _build_group_overview(price_ticker_summary, "ticker", "price")
    financial_overview = _build_group_overview(ticker_summary, "ticker", "financial")
    return (
        coverage.select(["ticker", "ticker_root", "yahoo_price_available", "sec_filing_available"])
        .join(price_overview, on="ticker", how="left", coalesce=True)
        .join(financial_overview, on="ticker", how="left", coalesce=True)
        .with_columns(
            [
                pl.col("price_matched_rows").fill_null(0),
                pl.col("price_error_rows").fill_null(0),
                pl.col("price_error_rate_pct").fill_null(0.0),
                pl.col("financial_matched_rows").fill_null(0),
                pl.col("financial_error_rows").fill_null(0),
                pl.col("financial_error_rate_pct").fill_null(0.0),
            ]
        )
    )


def _build_group_overview(df: pl.DataFrame, group_col: str, prefix: str) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                group_col: pl.String,
                f"{prefix}_matched_rows": pl.Int64,
                f"{prefix}_error_rows": pl.Int64,
                f"{prefix}_error_rate_pct": pl.Float64,
            }
        )
    return df.group_by(group_col).agg(
        [
            pl.col("matched_rows").sum().alias(f"{prefix}_matched_rows"),
            pl.col("error_rows").sum().alias(f"{prefix}_error_rows"),
        ]
    ).with_columns(_error_rate_expr(f"{prefix}_matched_rows", f"{prefix}_error_rows").alias(f"{prefix}_error_rate_pct"))


def _aggregate_overview(df: pl.DataFrame, prefix: str) -> dict[str, float]:
    if df.is_empty():
        return {"matched_rows": 0, "error_rows": 0, "error_rate_pct": 0.0}
    row = df.select(
        [
            pl.col("matched_rows").sum().alias("matched_rows"),
            pl.col("error_rows").sum().alias("error_rows"),
        ]
    ).to_dicts()[0]
    matched_rows = int(row["matched_rows"])
    error_rows = int(row["error_rows"])
    error_rate_pct = (error_rows / matched_rows * 100.0) if matched_rows else 0.0
    return {"matched_rows": matched_rows, "error_rows": error_rows, "error_rate_pct": error_rate_pct}


def _build_price_comparison_table(df: pl.DataFrame, *, threshold_pct: float) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "date": pl.String,
                "match_status": pl.String,
                "comparison_status": pl.String,
                "eodhd_adjusted_close": pl.Float64,
                "yfinance_adjusted_close": pl.Float64,
                "value_diff": pl.Float64,
                "diff_pct": pl.Float64,
            }
        )
    return (
        df.with_columns(_comparison_status_expr(threshold_pct).alias("comparison_status"))
        .select(
            [
                pl.col("date"),
                pl.col("match_status"),
                pl.col("comparison_status"),
                pl.col("eodhd_adjusted_close"),
                pl.col("yahoo_adjusted_close").alias("yfinance_adjusted_close"),
                pl.col("value_diff"),
                pl.col("diff_pct"),
            ]
        )
        .sort("date")
    )


def _build_financial_comparison_table(df: pl.DataFrame, *, threshold_pct: float) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "source": pl.String,
                "statement": pl.String,
                "metric": pl.String,
                "date": pl.String,
                "match_status": pl.String,
                "comparison_status": pl.String,
                "eodhd_value": pl.Float64,
                "open_value": pl.Float64,
                "value_diff": pl.Float64,
                "diff_pct": pl.Float64,
                "eodhd_filing_date": pl.String,
                "open_filing_date": pl.String,
                "date_diff_days": pl.Int64,
                "eodhd_source_label": pl.String,
                "open_source_label": pl.String,
            }
        )
    return (
        df.with_columns(_comparison_status_expr(threshold_pct).alias("comparison_status"))
        .select(
            [
                pl.col("source"),
                pl.col("statement"),
                pl.col("metric"),
                pl.col("date"),
                pl.col("match_status"),
                pl.col("comparison_status"),
                pl.col("eodhd_value"),
                pl.col("open_value"),
                pl.col("value_diff"),
                pl.col("diff_pct"),
                pl.col("eodhd_filing_date"),
                pl.col("open_filing_date"),
                pl.col("date_diff_days"),
                pl.col("eodhd_source_label"),
                pl.col("open_source_label"),
            ]
        )
        .sort(["source", "statement", "metric", "date"])
    )


def _build_consolidated_financial_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "statement": pl.String,
                "metric": pl.String,
                "date": pl.String,
                "value": pl.Float64,
                "filing_date": pl.String,
                "selected_source": pl.String,
                "selected_source_label": pl.String,
                "fallback_used": pl.Boolean,
                "candidate_source_count": pl.Int64,
                "candidate_sources": pl.String,
            }
        )
    return df.select(
        [
            pl.col("statement"),
            pl.col("metric"),
            pl.col("date"),
            pl.col("value"),
            pl.col("filing_date"),
            pl.col("selected_source"),
            pl.col("selected_source_label"),
            pl.col("fallback_used"),
            pl.col("candidate_source_count"),
            pl.col("candidate_sources"),
        ]
    ).sort(["statement", "metric", "date"])


def _build_consolidated_lineage_table(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return pl.DataFrame(
            schema={
                "statement": pl.String,
                "metric": pl.String,
                "date": pl.String,
                "source": pl.String,
                "source_priority": pl.Int64,
                "value": pl.Float64,
                "filing_date": pl.String,
                "source_label": pl.String,
                "selected_form": pl.String,
                "selected_fiscal_period": pl.String,
                "selected_fiscal_year": pl.Int64,
            }
        )
    return df.select(
        [
            pl.col("statement"),
            pl.col("metric"),
            pl.col("date"),
            pl.col("source"),
            pl.col("source_priority"),
            pl.col("value"),
            pl.col("filing_date"),
            pl.col("source_label"),
            pl.col("selected_form"),
            pl.col("selected_fiscal_period"),
            pl.col("selected_fiscal_year"),
        ]
    ).sort(["statement", "metric", "date", "source_priority"])


def _comparison_status_expr(threshold_pct: float) -> pl.Expr:
    return (
        pl.when(pl.col("match_status") == "matched")
        .then(
            pl.when(pl.col("diff_pct").abs() > threshold_pct)
            .then(pl.lit("threshold_breach"))
            .otherwise(pl.lit("within_threshold"))
        )
        .when(pl.col("match_status") == "eodhd_only")
        .then(pl.lit("missing_in_open_source"))
        .otherwise(pl.lit("missing_in_eodhd"))
    )


def _slugify(value: str) -> str:
    return value.lower().replace(".", "_").replace("/", "_").replace(" ", "_")


def _write_html_page(*, output_path: Path, title: str, subtitle: str, sections: list[tuple[str, pl.DataFrame]], navigation: str) -> None:
    section_html = "".join(f"<div class=\"section\"><h2>{escape(name)}</h2>{_frame_to_html(frame)}</div>" for name, frame in sections)
    html = f"""<!doctype html>
<html lang=\"en\">
<head>
  <meta charset=\"utf-8\">
  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 24px; color: #111827; background: #f8fafc; }}
    h1, h2 {{ margin-bottom: 8px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; background: white; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 8px 10px; text-align: left; font-size: 12px; vertical-align: top; }}
    th {{ background: #eff6ff; position: sticky; top: 0; }}
    a {{ color: #1d4ed8; text-decoration: none; }}
    .muted {{ color: #64748b; }}
    .section {{ margin-top: 28px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <div class=\"muted\">{escape(subtitle)}</div>
  {navigation}
  {section_html}
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _with_links(df: pl.DataFrame, link_col: str, *, label_col: str | None = None, label_cols: list[str] | None = None) -> pl.DataFrame:
    rows = []
    for row in df.iter_rows(named=True):
        label = row[label_col] if label_col is not None else " / ".join(str(row[col]) for col in (label_cols or []))
        link = row.get(link_col)
        rows.append(
            {
                **{key: value for key, value in row.items() if key != link_col},
                "report_link": f'<a href="{escape(str(link))}">{escape(str(label))}</a>' if link else "",
            }
        )
    ordered_columns = ["report_link"] + [column for column in df.columns if column != link_col]
    return pl.DataFrame(rows).select([column for column in ordered_columns if column in rows[0]] if rows else [])
