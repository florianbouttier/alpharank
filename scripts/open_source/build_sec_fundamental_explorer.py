#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import polars as pl


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sec_output_dir = args.sec_output_dir.resolve()
    price_source_dir = args.price_source_dir.resolve()
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sec_fundamental_explorer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = output_dir / "tickers"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    general = pl.read_parquet(sec_output_dir / "US_General.parquet")
    income = pl.read_parquet(sec_output_dir / "US_Income_statement.parquet")
    shares = pl.read_parquet(sec_output_dir / "US_share.parquet")
    earnings = pl.read_parquet(sec_output_dir / "US_Earnings.parquet")
    prices = pl.read_parquet(price_source_dir / "US_Finalprice.parquet")

    share_anomalies = _build_share_anomalies(shares=shares, ratio_threshold=args.share_ratio_threshold)
    price_anomalies = _build_price_anomalies(prices=prices, ratio_threshold=args.price_factor_ratio_threshold)

    ticker_rows = _build_ticker_summary(
        general=general,
        income=income,
        shares=shares,
        earnings=earnings,
        prices=prices,
        share_anomalies=share_anomalies,
        price_anomalies=price_anomalies,
    )
    ticker_rows.write_parquet(output_dir / "ticker_summary.parquet")
    ticker_rows.write_csv(output_dir / "ticker_summary.csv")
    share_anomalies.write_parquet(output_dir / "share_anomalies.parquet")
    price_anomalies.write_parquet(output_dir / "price_anomalies.parquet")

    tickers = ticker_rows.get_column("ticker").to_list()
    for ticker in tickers:
        (ticker_dir / f"{ticker}.html").write_text(
            _render_ticker_page(
                ticker=ticker,
                general=general.filter(pl.col("ticker") == ticker),
                income=income.filter(pl.col("ticker") == ticker),
                shares=shares.filter(pl.col("ticker") == ticker),
                earnings=earnings.filter(pl.col("ticker") == ticker),
                prices=prices.filter(pl.col("ticker") == ticker),
                share_anomalies=share_anomalies.filter(pl.col("ticker") == ticker),
                price_anomalies=price_anomalies.filter(pl.col("ticker") == ticker),
                max_price_years=args.max_price_years,
                max_quarters=args.max_quarters,
            ),
            encoding="utf-8",
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sec_output_dir": str(sec_output_dir),
        "price_source_dir": str(price_source_dir),
        "output_dir": str(output_dir),
        "max_quarters": args.max_quarters,
        "max_price_years": args.max_price_years,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_index_page(ticker_rows=ticker_rows, manifest=manifest),
        encoding="utf-8",
    )
    print(output_dir)
    print(output_dir / "index.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-friendly SEC fundamentals explorer by ticker.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--sec-output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument("--price-source-dir", type=Path, default=project_root / "data" / "eodhd" / "output")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--share-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--price-factor-ratio-threshold", type=float, default=1.2)
    parser.add_argument("--max-price-years", type=int, default=10)
    parser.add_argument("--max-quarters", type=int, default=24)
    return parser.parse_args()


def _build_ticker_summary(
    *,
    general: pl.DataFrame,
    income: pl.DataFrame,
    shares: pl.DataFrame,
    earnings: pl.DataFrame,
    prices: pl.DataFrame,
    share_anomalies: pl.DataFrame,
    price_anomalies: pl.DataFrame,
) -> pl.DataFrame:
    income_latest = (
        income.select(
            [
                "ticker",
                pl.col("date").alias("fundamental_date"),
                pl.col("totalRevenue").cast(pl.Float64, strict=False).alias("latest_revenue"),
                pl.col("netIncome").cast(pl.Float64, strict=False).alias("latest_net_income"),
            ]
        )
        .sort(["ticker", "fundamental_date"])
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    shares_latest = (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").alias("shares_date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("latest_shares"),
            ]
        )
        .sort(["ticker", "shares_date"])
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    earnings_latest = (
        earnings.select(
            [
                "ticker",
                pl.col("reportDate").alias("earnings_report_date"),
                pl.col("date").alias("earnings_period_end"),
                pl.col("epsActual").cast(pl.Float64, strict=False).alias("latest_eps_actual"),
            ]
        )
        .sort(["ticker", "earnings_report_date"])
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    prices_latest = (
        prices.select(
            [
                "ticker",
                pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("price_date"),
                pl.col("close").cast(pl.Float64, strict=False).alias("latest_close"),
                pl.col("adjusted_close").cast(pl.Float64, strict=False).alias("latest_adjusted_close"),
            ]
        )
        .sort(["ticker", "price_date"])
        .unique(subset=["ticker"], keep="last", maintain_order=True)
    )
    share_flags = (
        share_anomalies.group_by("ticker").agg(pl.len().alias("share_anomaly_count"))
        if not share_anomalies.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "share_anomaly_count": pl.Int64})
    )
    price_flags = (
        price_anomalies.group_by("ticker").agg(pl.len().alias("price_anomaly_count"))
        if not price_anomalies.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "price_anomaly_count": pl.Int64})
    )
    return (
        general.select(
            [
                "ticker",
                "Code",
                "Name",
                "Sector",
                "Industry",
                "Exchange",
            ]
        )
        .join(income_latest, on="ticker", how="left")
        .join(shares_latest, on="ticker", how="left")
        .join(earnings_latest, on="ticker", how="left")
        .join(prices_latest, on="ticker", how="left")
        .join(share_flags, on="ticker", how="left")
        .join(price_flags, on="ticker", how="left")
        .with_columns(
            [
                pl.col("share_anomaly_count").fill_null(0),
                pl.col("price_anomaly_count").fill_null(0),
            ]
        )
        .sort("Code")
    )


def _build_share_anomalies(*, shares: pl.DataFrame, ratio_threshold: float) -> pl.DataFrame:
    return (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .sort(["ticker", "date"])
        .with_columns(pl.col("shares").shift(1).over("ticker").alias("prev_shares"))
        .with_columns((pl.col("shares") / pl.col("prev_shares")).alias("share_ratio"))
        .with_columns(
            pl.when(pl.col("share_ratio") >= ratio_threshold)
            .then(pl.lit("share_increase"))
            .when(pl.col("share_ratio") <= (1.0 / ratio_threshold))
            .then(pl.lit("share_decrease"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("candidate_kind")
        )
        .filter(pl.col("candidate_kind").is_not_null())
        .sort(["ticker", "date"])
    )


def _build_price_anomalies(*, prices: pl.DataFrame, ratio_threshold: float) -> pl.DataFrame:
    return (
        prices.select(
            [
                "ticker",
                pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("date"),
                pl.col("close").cast(pl.Float64, strict=False).alias("close"),
                pl.col("adjusted_close").cast(pl.Float64, strict=False).alias("adjusted_close"),
            ]
        )
        .filter(pl.col("close").is_not_null() & pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") != 0))
        .with_columns((pl.col("close") / pl.col("adjusted_close")).alias("adjustment_factor"))
        .sort(["ticker", "date"])
        .with_columns(pl.col("adjustment_factor").shift(1).over("ticker").alias("prev_adjustment_factor"))
        .with_columns((pl.col("adjustment_factor") / pl.col("prev_adjustment_factor")).alias("factor_ratio"))
        .with_columns(
            pl.when(pl.col("factor_ratio") >= ratio_threshold)
            .then(pl.lit("factor_jump_up"))
            .when(pl.col("factor_ratio") <= (1.0 / ratio_threshold))
            .then(pl.lit("factor_jump_down"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("candidate_kind")
        )
        .filter(pl.col("candidate_kind").is_not_null())
        .sort(["ticker", "date"])
    )


def _render_index_page(*, ticker_rows: pl.DataFrame, manifest: dict[str, object]) -> str:
    rows_html = "".join(
        _ticker_row_html(row)
        for row in ticker_rows.select(
            [
                "ticker",
                "Code",
                "Name",
                "Sector",
                "latest_close",
                "latest_revenue",
                "latest_net_income",
                "latest_eps_actual",
                "share_anomaly_count",
                "price_anomaly_count",
            ]
        ).to_dicts()
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SEC Fundamental Explorer</title>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; color: #1c1e21; background: linear-gradient(180deg, #f1efe8 0%, #fcfbf8 40%, #ffffff 100%); }}
    .hero {{ padding: 36px 44px 24px 44px; border-bottom: 1px solid rgba(0,0,0,0.08); background: radial-gradient(circle at top right, rgba(182,145,89,0.18), transparent 32%), linear-gradient(135deg, #f8f3e8, #ffffff); }}
    .hero h1 {{ margin: 0; font-size: 42px; letter-spacing: -0.03em; }}
    .hero p {{ max-width: 840px; color: #5d5f63; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .toolbar {{ display: flex; gap: 12px; padding: 18px 44px; align-items: center; position: sticky; top: 0; background: rgba(252,251,248,0.9); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(0,0,0,0.06); z-index: 5; }}
    .toolbar input {{ width: 320px; padding: 11px 14px; border: 1px solid #d9d3c5; border-radius: 12px; font-size: 14px; }}
    .toolbar .meta {{ color: #6b6e73; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .table-wrap {{ padding: 20px 44px 40px 44px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 34px rgba(0,0,0,0.06); }}
    th, td {{ padding: 14px 12px; border-bottom: 1px solid #efede7; text-align: left; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #6e6558; background: #faf7f1; }}
    tr:hover {{ background: #fcfbf7; }}
    a {{ color: #7b4f24; text-decoration: none; font-weight: 600; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #f0e2c7; color: #7b4f24; font-size: 11px; font-weight: 700; }}
    .muted {{ color: #8a8d93; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>SEC Fundamental Explorer</h1>
    <p>Explorer par ticker, quarter par quarter, les fondamentaux SEC utiles: <strong>revenue</strong>, <strong>net income</strong>, <strong>EPS actual</strong>, <strong>shares outstanding</strong>, avec les prix en ligne et des drapeaux split/share pour repérer les incohérences rapidement.</p>
  </div>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Search ticker, company or sector">
    <div class="meta">Prix source: <code>{escape(str(manifest['price_source_dir']))}</code></div>
  </div>
  <div class="table-wrap">
    <table id="ticker-table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Name</th>
          <th>Sector</th>
          <th>Last Price</th>
          <th>Revenue</th>
          <th>Net Income</th>
          <th>EPS</th>
          <th>Flags</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
  <script>
    const search = document.getElementById('search');
    const rows = Array.from(document.querySelectorAll('#ticker-table tbody tr'));
    search.addEventListener('input', () => {{
      const q = search.value.toLowerCase().trim();
      rows.forEach((row) => {{
        const hay = row.dataset.search;
        row.style.display = hay.includes(q) ? '' : 'none';
      }});
    }});
  </script>
</body>
</html>
"""


def _ticker_row_html(row: dict[str, object]) -> str:
    search = " ".join(
        str(row.get(key) or "").lower()
        for key in ("Code", "ticker", "Name", "Sector")
    )
    flags = int(row.get("share_anomaly_count") or 0) + int(row.get("price_anomaly_count") or 0)
    flag_html = f"<span class='pill'>{flags} flags</span>" if flags else "<span class='muted'>clean</span>"
    return (
        f"<tr data-search='{escape(search)}'>"
        f"<td><a href='tickers/{escape(str(row['ticker']))}.html'>{escape(str(row['Code']))}</a></td>"
        f"<td>{escape(str(row.get('Name') or ''))}</td>"
        f"<td>{escape(str(row.get('Sector') or ''))}</td>"
        f"<td>{_fmt_number(row.get('latest_close'))}</td>"
        f"<td>{_fmt_large_number(row.get('latest_revenue'))}</td>"
        f"<td>{_fmt_large_number(row.get('latest_net_income'))}</td>"
        f"<td>{_fmt_number(row.get('latest_eps_actual'))}</td>"
        f"<td>{flag_html}</td>"
        f"</tr>"
    )


def _render_ticker_page(
    *,
    ticker: str,
    general: pl.DataFrame,
    income: pl.DataFrame,
    shares: pl.DataFrame,
    earnings: pl.DataFrame,
    prices: pl.DataFrame,
    share_anomalies: pl.DataFrame,
    price_anomalies: pl.DataFrame,
    max_price_years: int,
    max_quarters: int,
) -> str:
    import plotly.graph_objects as go
    import plotly.io as pio
    from plotly.subplots import make_subplots

    info = general.row(0, named=True) if not general.is_empty() else {}
    income_clean = (
        income.select(
            [
                "date",
                "filing_date",
                pl.col("totalRevenue").cast(pl.Float64, strict=False).alias("revenue"),
                pl.col("netIncome").cast(pl.Float64, strict=False).alias("net_income"),
            ]
        )
        .sort("date")
        .tail(max_quarters)
    )
    shares_clean = (
        shares.select(
            [
                pl.col("dateFormatted").alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .sort("date")
        .tail(max_quarters)
    )
    earnings_clean = (
        earnings.select(
            [
                "date",
                "reportDate",
                pl.col("epsActual").cast(pl.Float64, strict=False).alias("eps_actual"),
            ]
        )
        .sort("date")
        .tail(max_quarters)
    )
    price_cutoff = prices.get_column("date").max() if not prices.is_empty() else None
    prices_clean = (
        prices.filter(pl.col("date") >= (price_cutoff.replace(year=price_cutoff.year - max_price_years) if price_cutoff is not None else pl.datetime(2000, 1, 1)))
        if not prices.is_empty() and price_cutoff is not None
        else prices
    )
    price_table = (
        prices_clean.select(
            [
                pl.col("date").cast(pl.Utf8).str.slice(0, 10).alias("date"),
                pl.col("close").cast(pl.Float64, strict=False).alias("close"),
                pl.col("adjusted_close").cast(pl.Float64, strict=False).alias("adjusted_close"),
            ]
        )
        .sort("date")
    )

    quarter_table = _build_quarter_table(
        income=income_clean,
        shares=shares_clean,
        earnings=earnings_clean,
    )

    plots: list[str] = []
    if not price_table.is_empty():
        price_fig = go.Figure()
        price_fig.add_trace(go.Scatter(x=price_table["date"], y=price_table["close"], mode="lines", name="Close", line=dict(color="#2c5f8a", width=2)))
        price_fig.add_trace(go.Scatter(x=price_table["date"], y=price_table["adjusted_close"], mode="lines", name="Adjusted", line=dict(color="#bb6b34", width=2)))
        price_fig.update_layout(template="plotly_white", height=420, title="Price history", legend=dict(orientation="h"))
        plots.append(pio.to_html(price_fig, full_html=False, include_plotlyjs="cdn"))

    if not income_clean.is_empty():
        fundamental_fig = make_subplots(specs=[[{"secondary_y": True}]])
        fundamental_fig.add_trace(
            go.Bar(x=income_clean["date"], y=income_clean["revenue"], name="Revenue", marker_color="#c9a45c"),
            secondary_y=False,
        )
        fundamental_fig.add_trace(
            go.Scatter(x=income_clean["date"], y=income_clean["net_income"], name="Net income", mode="lines+markers", line=dict(color="#214761", width=3)),
            secondary_y=True,
        )
        fundamental_fig.update_layout(template="plotly_white", height=440, title="Quarterly revenue and net income")
        fundamental_fig.update_yaxes(title_text="Revenue", secondary_y=False)
        fundamental_fig.update_yaxes(title_text="Net income", secondary_y=True)
        plots.append(pio.to_html(fundamental_fig, full_html=False, include_plotlyjs=False))

    if not earnings_clean.is_empty() or not shares_clean.is_empty():
        combo_fig = make_subplots(specs=[[{"secondary_y": True}]])
        if not earnings_clean.is_empty():
            combo_fig.add_trace(
                go.Bar(x=earnings_clean["date"], y=earnings_clean["eps_actual"], name="EPS actual", marker_color="#8a3d3a"),
                secondary_y=False,
            )
        if not shares_clean.is_empty():
            combo_fig.add_trace(
                go.Scatter(x=shares_clean["date"], y=shares_clean["shares"], name="Shares", mode="lines+markers", line=dict(color="#357266", width=3)),
                secondary_y=True,
            )
        combo_fig.update_layout(template="plotly_white", height=440, title="Quarterly EPS and shares outstanding")
        combo_fig.update_yaxes(title_text="EPS actual", secondary_y=False)
        combo_fig.update_yaxes(title_text="Shares outstanding", secondary_y=True)
        plots.append(pio.to_html(combo_fig, full_html=False, include_plotlyjs=False))

    latest = _latest_snapshot(
        info=info,
        income=income_clean,
        shares=shares_clean,
        earnings=earnings_clean,
        prices=price_table,
        share_anomalies=share_anomalies,
        price_anomalies=price_anomalies,
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(info.get('Code') or ticker))} Fundamental Explorer</title>
  <style>
    body {{ margin: 0; font-family: Georgia, 'Times New Roman', serif; color: #1c1e21; background: #fbfaf6; }}
    .wrap {{ padding: 28px 34px 40px 34px; }}
    .topbar {{ display: flex; align-items: baseline; justify-content: space-between; gap: 20px; }}
    .topbar h1 {{ margin: 0; font-size: 40px; letter-spacing: -0.03em; }}
    .muted {{ color: #6d7075; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 14px; margin: 22px 0; }}
    .card {{ background: white; border-radius: 16px; padding: 16px 18px; box-shadow: 0 10px 28px rgba(0,0,0,0.06); }}
    .card .label {{ font-size: 12px; letter-spacing: 0.06em; text-transform: uppercase; color: #7a6e61; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .card .value {{ margin-top: 10px; font-size: 28px; font-weight: 700; }}
    .section {{ background: white; border-radius: 18px; padding: 18px 20px; box-shadow: 0 12px 34px rgba(0,0,0,0.06); margin: 18px 0; }}
    h2 {{ margin: 0 0 14px 0; }}
    table {{ width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }}
    th, td {{ padding: 9px 8px; border-bottom: 1px solid #efede7; text-align: left; }}
    th {{ background: #faf7f1; color: #6e6558; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    a {{ color: #7b4f24; text-decoration: none; font-weight: 600; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #f3e2cf; color: #7b4f24; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 12px; margin-right: 8px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="topbar">
      <div>
        <p class="muted"><a href="../index.html">Back to explorer</a></p>
        <h1>{escape(str(info.get('Code') or ticker))}</h1>
        <p class="muted">{escape(str(info.get('Name') or ticker))} · {escape(str(info.get('Sector') or ''))} · {escape(str(info.get('Industry') or ''))}</p>
      </div>
      <div class="muted">{escape(str(info.get('Exchange') or ''))}</div>
    </div>
    <div class="cards">
      {_cards_html(latest)}
    </div>
    <div class="section">
      <h2>Charts</h2>
      {''.join(plots) if plots else '<p class="muted">No charts available.</p>'}
    </div>
    <div class="section">
      <h2>Quarter table</h2>
      {_table_html(quarter_table)}
    </div>
    <div class="section">
      <h2>Split / share flags</h2>
      <p>{_flag_badges(share_anomalies.height, price_anomalies.height)}</p>
      <h3>Share count jumps</h3>
      {_table_html(share_anomalies)}
      <h3>Price adjustment jumps</h3>
      {_table_html(price_anomalies)}
    </div>
  </div>
</body>
</html>
"""


def _build_quarter_table(*, income: pl.DataFrame, shares: pl.DataFrame, earnings: pl.DataFrame) -> pl.DataFrame:
    quarter = income.join(shares, on="date", how="full", coalesce=True)
    quarter = quarter.join(earnings.rename({"reportDate": "earnings_reportDate"}), on="date", how="full", coalesce=True)
    preferred = ["date", "filing_date", "earnings_reportDate", "revenue", "net_income", "eps_actual", "shares"]
    existing = [column for column in preferred if column in quarter.columns]
    return quarter.select(existing).sort("date", descending=True)


def _latest_snapshot(
    *,
    info: dict[str, object],
    income: pl.DataFrame,
    shares: pl.DataFrame,
    earnings: pl.DataFrame,
    prices: pl.DataFrame,
    share_anomalies: pl.DataFrame,
    price_anomalies: pl.DataFrame,
) -> list[tuple[str, str]]:
    return [
        ("Last price", _fmt_number(prices.get_column("close").tail(1).item() if not prices.is_empty() else None)),
        ("Revenue", _fmt_large_number(income.get_column("revenue").tail(1).item() if not income.is_empty() else None)),
        ("Net income", _fmt_large_number(income.get_column("net_income").tail(1).item() if not income.is_empty() else None)),
        ("EPS actual", _fmt_number(earnings.get_column("eps_actual").tail(1).item() if not earnings.is_empty() else None)),
        ("Shares", _fmt_large_number(shares.get_column("shares").tail(1).item() if not shares.is_empty() else None)),
        ("Flags", str(share_anomalies.height + price_anomalies.height)),
    ]


def _cards_html(cards: list[tuple[str, str]]) -> str:
    return "".join(
        f"<div class='card'><div class='label'>{escape(label)}</div><div class='value'>{escape(value)}</div></div>"
        for label, value in cards
    )


def _flag_badges(share_count: int, price_count: int) -> str:
    return (
        f"<span class='pill'>{share_count} share flags</span>"
        f"<span class='pill'>{price_count} price flags</span>"
    )


def _table_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "<p class='muted'>No rows.</p>"
    columns = frame.columns
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    rows = []
    for row in frame.to_dicts():
        cells = "".join(f"<td>{escape('' if value is None else str(value))}</td>" for value in row.values())
        rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def _fmt_number(value: object) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return str(value)


def _fmt_large_number(value: object) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except Exception:
        return str(value)
    abs_value = abs(numeric)
    if abs_value >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"
    return f"{numeric:,.0f}"


if __name__ == "__main__":
    main()
