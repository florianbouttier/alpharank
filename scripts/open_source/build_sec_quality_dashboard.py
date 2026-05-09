#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import polars as pl


CORE_METRICS: tuple[str, ...] = ("revenue", "net_income", "outstanding_shares")


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sec_output_dir = args.sec_output_dir.resolve()
    price_source_dir = args.price_source_dir.resolve()
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sec_quality_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    ticker_dir = output_dir / "tickers"
    ticker_dir.mkdir(parents=True, exist_ok=True)

    financials = pl.read_parquet(sec_output_dir / "lineage" / "financials_sec_consolidated.parquet")
    earnings = pl.read_parquet(sec_output_dir / "lineage" / "earnings_sec_consolidated.parquet")
    shares = pl.read_parquet(sec_output_dir / "US_share.parquet")
    general = pl.read_parquet(sec_output_dir / "US_General.parquet")
    prices = pl.read_parquet(price_source_dir / "US_Finalprice.parquet")

    coverage = _build_coverage_summary(financials=financials, earnings=earnings, general=general)
    missing = _build_missing_ticker_table(financials=financials, earnings=earnings, general=general)
    share_candidates = _build_share_split_candidates(shares=shares, ratio_threshold=args.share_ratio_threshold)
    price_candidates = _build_price_adjustment_candidates(prices=prices, ratio_threshold=args.price_factor_ratio_threshold)
    ticker_anomalies = _build_ticker_anomaly_summary(
        share_candidates=share_candidates,
        price_candidates=price_candidates,
    ).join(general.select(["Code", "Sector"]).rename({"Code": "ticker_code"}), on="ticker_code", how="left")

    coverage.write_parquet(output_dir / "coverage_summary.parquet")
    coverage.write_csv(output_dir / "coverage_summary.csv")
    missing.write_parquet(output_dir / "missing_tickers.parquet")
    missing.write_csv(output_dir / "missing_tickers.csv")
    share_candidates.write_parquet(output_dir / "share_split_candidates.parquet")
    share_candidates.write_csv(output_dir / "share_split_candidates.csv")
    price_candidates.write_parquet(output_dir / "price_adjustment_candidates.parquet")
    price_candidates.write_csv(output_dir / "price_adjustment_candidates.csv")
    ticker_anomalies.write_parquet(output_dir / "ticker_anomaly_summary.parquet")
    ticker_anomalies.write_csv(output_dir / "ticker_anomaly_summary.csv")

    deep_dive_tickers = _select_deep_dive_tickers(
        share_candidates=share_candidates,
        price_candidates=price_candidates,
        limit=args.deep_dive_limit,
    )
    for ticker in deep_dive_tickers:
        (ticker_dir / f"{ticker}.html").write_text(
            _render_ticker_page(
                ticker=ticker,
                shares=shares.filter(pl.col("ticker") == ticker),
                prices=prices.filter(pl.col("ticker") == ticker),
                financials=financials.filter(
                    (pl.col("ticker") == ticker) & pl.col("metric").is_in(["revenue", "net_income", "outstanding_shares"])
                ),
                earnings=earnings.filter(pl.col("ticker") == ticker),
                share_candidates=share_candidates.filter(pl.col("ticker") == ticker),
                price_candidates=price_candidates.filter(pl.col("ticker") == ticker),
            ),
            encoding="utf-8",
        )

    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sec_output_dir": str(sec_output_dir),
        "price_source_dir": str(price_source_dir),
        "output_dir": str(output_dir),
        "share_ratio_threshold": args.share_ratio_threshold,
        "price_factor_ratio_threshold": args.price_factor_ratio_threshold,
        "deep_dive_tickers": deep_dive_tickers,
        "notes": {
            "fundamentals_source": "SEC-only package",
            "price_source": str(price_source_dir),
            "price_stitched_package": None,
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.html").write_text(
        _render_dashboard_html(
            coverage=coverage,
            missing=missing,
            share_candidates=share_candidates,
            price_candidates=price_candidates,
            ticker_anomalies=ticker_anomalies,
            deep_dive_tickers=deep_dive_tickers,
            price_source_dir=price_source_dir,
        ),
        encoding="utf-8",
    )
    print(output_dir)
    print(output_dir / "report.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SEC-only fundamentals quality dashboard with split/share anomaly checks.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--sec-output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument("--price-source-dir", type=Path, default=project_root / "data" / "eodhd" / "output")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--share-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--price-factor-ratio-threshold", type=float, default=1.2)
    parser.add_argument("--deep-dive-limit", type=int, default=30)
    return parser.parse_args()


def _build_coverage_summary(*, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame) -> pl.DataFrame:
    all_tickers = general.select(pl.col("Code").alias("ticker")).with_columns(pl.col("ticker") + pl.lit(".US")).get_column("ticker").to_list()
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        sub = financials.filter(pl.col("metric") == metric)
        rows.append(
            {
                "dataset": "financials",
                "metric": metric,
                "tickers_with_data": sub.get_column("ticker").n_unique() if not sub.is_empty() else 0,
                "total_tickers": len(all_tickers),
                "coverage_pct": _pct(sub.get_column("ticker").n_unique() if not sub.is_empty() else 0, len(all_tickers)),
                "rows": sub.height,
                "min_date": sub.get_column("date").min() if not sub.is_empty() else None,
                "max_date": sub.get_column("date").max() if not sub.is_empty() else None,
            }
        )
    eps = earnings.filter(pl.col("epsActual").is_not_null())
    rows.append(
        {
            "dataset": "earnings",
            "metric": "epsActual",
            "tickers_with_data": eps.get_column("ticker").n_unique() if not eps.is_empty() else 0,
            "total_tickers": len(all_tickers),
            "coverage_pct": _pct(eps.get_column("ticker").n_unique() if not eps.is_empty() else 0, len(all_tickers)),
            "rows": eps.height,
            "min_date": eps.get_column("period_end").min() if not eps.is_empty() else None,
            "max_date": eps.get_column("period_end").max() if not eps.is_empty() else None,
        }
    )
    return pl.DataFrame(rows)


def _build_missing_ticker_table(*, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame) -> pl.DataFrame:
    ticker_frame = general.select(
        [
            (pl.col("Code") + pl.lit(".US")).alias("ticker"),
            pl.col("Sector").alias("sector"),
            pl.col("Industry").alias("industry"),
        ]
    )
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        present = financials.filter(pl.col("metric") == metric).select("ticker").unique()
        missing = ticker_frame.join(present, on="ticker", how="anti")
        rows.extend(missing.with_columns(pl.lit(metric).alias("metric")).to_dicts())
    eps_present = earnings.filter(pl.col("epsActual").is_not_null()).select("ticker").unique()
    eps_missing = ticker_frame.join(eps_present, on="ticker", how="anti")
    rows.extend(eps_missing.with_columns(pl.lit("epsActual").alias("metric")).to_dicts())
    return pl.DataFrame(rows).sort(["metric", "ticker"]) if rows else pl.DataFrame(
        schema={"ticker": pl.String, "sector": pl.String, "industry": pl.String, "metric": pl.String}
    )


def _build_share_split_candidates(*, shares: pl.DataFrame, ratio_threshold: float) -> pl.DataFrame:
    if shares.is_empty():
        return pl.DataFrame(
            schema={"ticker": pl.String, "date": pl.String, "shares": pl.Float64, "prev_shares": pl.Float64, "share_ratio": pl.Float64}
        )
    return (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").cast(pl.Utf8).alias("date"),
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


def _build_price_adjustment_candidates(*, prices: pl.DataFrame, ratio_threshold: float) -> pl.DataFrame:
    if prices.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "date": pl.String,
                "close": pl.Float64,
                "adjusted_close": pl.Float64,
                "adjustment_factor": pl.Float64,
                "prev_adjustment_factor": pl.Float64,
                "factor_ratio": pl.Float64,
                "candidate_kind": pl.String,
            }
        )
    return (
        prices.select(
            [
                "ticker",
                pl.col("date").cast(pl.Utf8),
                pl.col("close").cast(pl.Float64, strict=False),
                pl.col("adjusted_close").cast(pl.Float64, strict=False),
            ]
        )
        .filter(pl.col("close").is_not_null() & pl.col("adjusted_close").is_not_null() & (pl.col("adjusted_close") != 0))
        .with_columns((pl.col("close") / pl.col("adjusted_close")).alias("adjustment_factor"))
        .sort(["ticker", "date"])
        .with_columns(pl.col("adjustment_factor").shift(1).over("ticker").alias("prev_adjustment_factor"))
        .with_columns((pl.col("adjustment_factor") / pl.col("prev_adjustment_factor")).alias("factor_ratio"))
        .with_columns(
            pl.when(pl.col("factor_ratio") >= ratio_threshold)
            .then(pl.lit("adjustment_factor_jump_up"))
            .when(pl.col("factor_ratio") <= (1.0 / ratio_threshold))
            .then(pl.lit("adjustment_factor_jump_down"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("candidate_kind")
        )
        .filter(pl.col("candidate_kind").is_not_null())
        .sort(["ticker", "date"])
    )


def _build_ticker_anomaly_summary(*, share_candidates: pl.DataFrame, price_candidates: pl.DataFrame) -> pl.DataFrame:
    share_summary = (
        share_candidates.group_by("ticker")
        .agg(
            [
                pl.len().alias("share_candidate_count"),
                pl.col("share_ratio").max().alias("max_share_ratio"),
                pl.col("share_ratio").min().alias("min_share_ratio"),
            ]
        )
        if not share_candidates.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "share_candidate_count": pl.Int64, "max_share_ratio": pl.Float64, "min_share_ratio": pl.Float64})
    )
    price_summary = (
        price_candidates.group_by("ticker")
        .agg(
            [
                pl.len().alias("price_candidate_count"),
                pl.col("factor_ratio").max().alias("max_factor_ratio"),
                pl.col("factor_ratio").min().alias("min_factor_ratio"),
            ]
        )
        if not price_candidates.is_empty()
        else pl.DataFrame(schema={"ticker": pl.String, "price_candidate_count": pl.Int64, "max_factor_ratio": pl.Float64, "min_factor_ratio": pl.Float64})
    )
    return (
        share_summary.join(price_summary, on="ticker", how="full", coalesce=True)
        .with_columns(
            [
                pl.col("ticker").str.replace(r"\.US$", "").alias("ticker_code"),
                (pl.coalesce([pl.col("share_candidate_count"), pl.lit(0)]) + pl.coalesce([pl.col("price_candidate_count"), pl.lit(0)])).alias("total_candidates"),
            ]
        )
        .sort(["total_candidates", "ticker"], descending=[True, False])
    )


def _select_deep_dive_tickers(*, share_candidates: pl.DataFrame, price_candidates: pl.DataFrame, limit: int) -> list[str]:
    tickers = (
        pl.concat(
            [
                share_candidates.select("ticker") if not share_candidates.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
                price_candidates.select("ticker") if not price_candidates.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
            ],
            how="vertical",
        )
        .unique()
        .get_column("ticker")
        .to_list()
    )
    return sorted(tickers)[:limit]


def _render_dashboard_html(
    *,
    coverage: pl.DataFrame,
    missing: pl.DataFrame,
    share_candidates: pl.DataFrame,
    price_candidates: pl.DataFrame,
    ticker_anomalies: pl.DataFrame,
    deep_dive_tickers: list[str],
    price_source_dir: Path,
) -> str:
    import plotly.express as px

    coverage_pdf = coverage.to_pandas()
    coverage_fig = px.bar(
        coverage_pdf,
        x="metric",
        y="coverage_pct",
        color="dataset",
        title="Coverage by core metric",
        text="tickers_with_data",
    )
    coverage_fig.update_layout(template="plotly_white", height=420, yaxis_title="Coverage %")

    share_pdf = share_candidates.head(5000).to_pandas() if not share_candidates.is_empty() else None
    share_fig_html = "<p>No share split candidates.</p>"
    if share_pdf is not None and not share_pdf.empty:
        share_fig = px.scatter(
            share_pdf,
            x="date",
            y="share_ratio",
            color="candidate_kind",
            hover_data=["ticker", "shares", "prev_shares"],
            title="Quarterly share-count ratio anomalies",
        )
        share_fig.update_layout(template="plotly_white", height=500)
        share_fig_html = share_fig.to_html(full_html=False, include_plotlyjs=False)

    price_pdf = price_candidates.head(10000).to_pandas() if not price_candidates.is_empty() else None
    price_fig_html = "<p>No price adjustment-factor candidates.</p>"
    if price_pdf is not None and not price_pdf.empty:
        price_fig = px.scatter(
            price_pdf,
            x="date",
            y="factor_ratio",
            color="candidate_kind",
            hover_data=["ticker", "adjustment_factor", "prev_adjustment_factor", "close", "adjusted_close"],
            title="Daily price adjustment-factor jumps",
        )
        price_fig.update_layout(template="plotly_white", height=500)
        price_fig_html = price_fig.to_html(full_html=False, include_plotlyjs=False)

    anomaly_pdf = ticker_anomalies.head(50).to_pandas() if not ticker_anomalies.is_empty() else None
    anomaly_fig_html = "<p>No ticker-level anomalies.</p>"
    if anomaly_pdf is not None and not anomaly_pdf.empty:
        anomaly_fig = px.bar(
            anomaly_pdf,
            x="ticker",
            y=["share_candidate_count", "price_candidate_count"],
            title="Top anomaly tickers",
            barmode="stack",
            hover_data=["Sector", "max_share_ratio", "max_factor_ratio"],
        )
        anomaly_fig.update_layout(template="plotly_white", height=500)
        anomaly_fig_html = anomaly_fig.to_html(full_html=False, include_plotlyjs=False)

    links = "".join(
        f"<li><a href='tickers/{escape(ticker)}.html'>{escape(ticker)}</a></li>"
        for ticker in deep_dive_tickers
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SEC Quality Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1a1a1a; background: #f6f7f9; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .muted {{ color: #5f6b7a; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 18px 0 26px 0; }}
    .card {{ background: white; border-radius: 14px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
    .section {{ background: white; border-radius: 16px; padding: 18px 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); margin: 18px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6e8eb; text-align: left; padding: 8px; }}
    th {{ background: #fafbfc; }}
    ul {{ columns: 3; }}
    code {{ background: #eef2f7; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <h1>SEC Fundamentals Quality Dashboard</h1>
  <p class="muted">Fundamentals source: <code>data/sec/output</code>. Price source for split diagnostics: <code>{escape(str(price_source_dir))}</code>.</p>
  <div class="card-grid">
    {_coverage_cards_html(coverage)}
  </div>
  <div class="section">
    <h2>Coverage</h2>
    {coverage_fig.to_html(full_html=False, include_plotlyjs=False)}
    {_table_html(coverage)}
  </div>
  <div class="section">
    <h2>Ticker anomaly summary</h2>
    {anomaly_fig_html}
    {_table_html(ticker_anomalies.head(50))}
  </div>
  <div class="section">
    <h2>Share-count anomaly candidates</h2>
    {share_fig_html}
    {_table_html(share_candidates.head(100))}
  </div>
  <div class="section">
    <h2>Price adjustment anomaly candidates</h2>
    {price_fig_html}
    {_table_html(price_candidates.head(100))}
  </div>
  <div class="section">
    <h2>Missing core metrics</h2>
    {_table_html(missing.head(200))}
  </div>
  <div class="section">
    <h2>Ticker deep dives</h2>
    <ul>{links}</ul>
  </div>
</body>
</html>
"""


def _coverage_cards_html(coverage: pl.DataFrame) -> str:
    cards: list[str] = []
    for row in coverage.to_dicts():
        cards.append(
            f"<div class='card'><div class='muted'>{escape(str(row['dataset']))} / {escape(str(row['metric']))}</div>"
            f"<div style='font-size:32px;font-weight:700;margin-top:8px'>{row['coverage_pct']:.1f}%</div>"
            f"<div class='muted'>{row['tickers_with_data']} / {row['total_tickers']} tickers</div></div>"
        )
    return "".join(cards)


def _render_ticker_page(
    *,
    ticker: str,
    shares: pl.DataFrame,
    prices: pl.DataFrame,
    financials: pl.DataFrame,
    earnings: pl.DataFrame,
    share_candidates: pl.DataFrame,
    price_candidates: pl.DataFrame,
) -> str:
    import plotly.express as px

    plots: list[str] = []
    if not shares.is_empty():
        share_pdf = shares.select(
            [pl.col("dateFormatted").alias("date"), pl.col("shares").cast(pl.Float64, strict=False)]
        ).sort("date").to_pandas()
        fig = px.line(share_pdf, x="date", y="shares", title=f"{ticker} shares outstanding")
        fig.update_layout(template="plotly_white", height=420)
        plots.append(fig.to_html(full_html=False, include_plotlyjs="cdn"))
    if not prices.is_empty():
        price_pdf = prices.select(["date", "close", "adjusted_close"]).sort("date").to_pandas()
        fig = px.line(price_pdf, x="date", y=["close", "adjusted_close"], title=f"{ticker} price vs adjusted price")
        fig.update_layout(template="plotly_white", height=420)
        plots.append(fig.to_html(full_html=False, include_plotlyjs=False))
    if not earnings.is_empty():
        earn_pdf = earnings.select(["period_end", "epsActual"]).sort("period_end").to_pandas()
        fig = px.line(earn_pdf, x="period_end", y="epsActual", title=f"{ticker} EPS actual")
        fig.update_layout(template="plotly_white", height=360)
        plots.append(fig.to_html(full_html=False, include_plotlyjs=False))

    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(ticker)} deep dive</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1a1a1a; }}
    .section {{ margin: 22px 0; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6e8eb; text-align: left; padding: 8px; }}
    th {{ background: #fafbfc; }}
  </style>
</head>
<body>
  <h1>{escape(ticker)}</h1>
  <p><a href="../report.html">Back to dashboard</a></p>
  <div class="section">{''.join(plots) if plots else '<p>No charts available.</p>'}</div>
  <div class="section"><h2>Core fundamentals</h2>{_table_html(financials.sort(['metric','date'], descending=[False, False]).tail(40))}</div>
  <div class="section"><h2>Earnings</h2>{_table_html(earnings.sort('period_end').tail(20))}</div>
  <div class="section"><h2>Share anomaly candidates</h2>{_table_html(share_candidates)}</div>
  <div class="section"><h2>Price adjustment candidates</h2>{_table_html(price_candidates)}</div>
</body>
</html>
"""


def _table_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "<p>No rows.</p>"
    columns = frame.columns
    header = "".join(f"<th>{escape(column)}</th>" for column in columns)
    body_rows: list[str] = []
    for row in frame.to_dicts():
        cells = "".join(f"<td>{escape('' if value is None else str(value))}</td>" for value in row.values())
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return (numerator / denominator) * 100.0


if __name__ == "__main__":
    main()
