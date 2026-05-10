#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import polars as pl


CORE_METRICS: tuple[str, ...] = ("revenue", "net_income", "outstanding_shares")
AUDIT_METRICS: tuple[str, ...] = CORE_METRICS + ("epsActual",)


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
    quarterly_presence = _build_quarterly_presence(financials=financials, earnings=earnings, general=general)
    ticker_metric_holes = _build_ticker_metric_holes(quarterly_presence=quarterly_presence)
    quarterly_holes = quarterly_presence.filter(~pl.col("present")).sort(["metric", "ticker", "date"])
    kpi_hole_summary = _build_kpi_hole_summary(quarterly_presence=quarterly_presence)
    ticker_gap_summary = _build_ticker_gap_summary(ticker_metric_holes=ticker_metric_holes)
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
    quarterly_presence.write_parquet(output_dir / "quarterly_presence.parquet")
    quarterly_presence.write_csv(output_dir / "quarterly_presence.csv")
    quarterly_holes.write_parquet(output_dir / "quarterly_holes.parquet")
    quarterly_holes.write_csv(output_dir / "quarterly_holes.csv")
    ticker_metric_holes.write_parquet(output_dir / "ticker_metric_holes.parquet")
    ticker_metric_holes.write_csv(output_dir / "ticker_metric_holes.csv")
    kpi_hole_summary.write_parquet(output_dir / "kpi_hole_summary.parquet")
    kpi_hole_summary.write_csv(output_dir / "kpi_hole_summary.csv")
    ticker_gap_summary.write_parquet(output_dir / "ticker_gap_summary.parquet")
    ticker_gap_summary.write_csv(output_dir / "ticker_gap_summary.csv")
    share_candidates.write_parquet(output_dir / "share_split_candidates.parquet")
    share_candidates.write_csv(output_dir / "share_split_candidates.csv")
    price_candidates.write_parquet(output_dir / "price_adjustment_candidates.parquet")
    price_candidates.write_csv(output_dir / "price_adjustment_candidates.csv")
    ticker_anomalies.write_parquet(output_dir / "ticker_anomaly_summary.parquet")
    ticker_anomalies.write_csv(output_dir / "ticker_anomaly_summary.csv")

    deep_dive_tickers = _select_deep_dive_tickers(
        share_candidates=share_candidates,
        price_candidates=price_candidates,
        ticker_gap_summary=ticker_gap_summary,
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
                ticker_metric_holes=ticker_metric_holes.filter(pl.col("ticker") == ticker),
                quarterly_holes=quarterly_holes.filter(pl.col("ticker") == ticker),
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
            kpi_hole_summary=kpi_hole_summary,
            ticker_gap_summary=ticker_gap_summary,
            ticker_metric_holes=ticker_metric_holes,
            quarterly_holes=quarterly_holes,
            share_candidates=share_candidates,
            price_candidates=price_candidates,
            ticker_anomalies=ticker_anomalies,
            deep_dive_tickers=deep_dive_tickers,
            price_source_dir=price_source_dir,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_dashboard_markdown(
            coverage=coverage,
            kpi_hole_summary=kpi_hole_summary,
            ticker_gap_summary=ticker_gap_summary,
            missing=missing,
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


def _build_quarterly_presence(*, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame) -> pl.DataFrame:
    ticker_info = general.select(
        [
            (pl.col("Code") + pl.lit(".US")).alias("ticker"),
            pl.col("Code").alias("ticker_code"),
            pl.col("Sector").alias("sector"),
            pl.col("Industry").alias("industry"),
        ]
    ).unique(subset=["ticker"])
    financial_presence = (
        financials.filter(pl.col("metric").is_in(list(CORE_METRICS)))
        .select(["ticker", "date", "metric"])
        .unique()
    )
    earnings_presence = (
        earnings.filter(pl.col("epsActual").is_not_null())
        .select(
            [
                "ticker",
                pl.col("period_end").alias("date"),
                pl.lit("epsActual").alias("metric"),
            ]
        )
        .unique()
    )
    observed = pl.concat([financial_presence, earnings_presence], how="diagonal_relaxed").unique()
    if observed.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "date": pl.String,
                "metric": pl.String,
                "present": pl.Boolean,
            }
        )

    quarter_grid = observed.select(["ticker", "date"]).unique()
    metric_grid = pl.DataFrame({"metric": list(AUDIT_METRICS)})
    expected = quarter_grid.join(metric_grid, how="cross")
    presence = (
        expected.join(
            observed.with_columns(pl.lit(True).alias("present")),
            on=["ticker", "date", "metric"],
            how="left",
        )
        .with_columns(pl.col("present").fill_null(False))
        .join(ticker_info, on="ticker", how="left")
        .select(["ticker", "ticker_code", "sector", "industry", "date", "metric", "present"])
        .sort(["ticker", "date", "metric"])
    )
    return presence


def _build_ticker_metric_holes(*, quarterly_presence: pl.DataFrame) -> pl.DataFrame:
    if quarterly_presence.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "metric": pl.String,
                "expected_quarters": pl.Int64,
                "present_quarters": pl.Int64,
                "hole_count": pl.Int64,
                "hole_pct": pl.Float64,
                "first_quarter": pl.String,
                "last_quarter": pl.String,
                "sample_missing_dates": pl.String,
            }
        )
    return (
        quarterly_presence.group_by(["ticker", "ticker_code", "sector", "industry", "metric"])
        .agg(
            [
                pl.len().alias("expected_quarters"),
                pl.col("present").cast(pl.Int64).sum().alias("present_quarters"),
                pl.col("date").min().alias("first_quarter"),
                pl.col("date").max().alias("last_quarter"),
                pl.col("date").filter(~pl.col("present")).sort().head(8).alias("missing_dates"),
            ]
        )
        .with_columns(
            [
                (pl.col("expected_quarters") - pl.col("present_quarters")).alias("hole_count"),
                (
                    (pl.col("expected_quarters") - pl.col("present_quarters")) * 100.0
                    / pl.col("expected_quarters").clip(lower_bound=1)
                ).alias("hole_pct"),
                pl.col("missing_dates").list.join(", ").alias("sample_missing_dates"),
            ]
        )
        .drop("missing_dates")
        .sort(["hole_count", "ticker", "metric"], descending=[True, False, False])
    )


def _build_kpi_hole_summary(*, quarterly_presence: pl.DataFrame) -> pl.DataFrame:
    if quarterly_presence.is_empty():
        return pl.DataFrame(
            schema={
                "metric": pl.String,
                "expected_quarters": pl.Int64,
                "present_quarters": pl.Int64,
                "hole_count": pl.Int64,
                "hole_pct": pl.Float64,
                "tickers_with_holes": pl.Int64,
            }
        )
    return (
        quarterly_presence.group_by("metric")
        .agg(
            [
                pl.len().alias("expected_quarters"),
                pl.col("present").cast(pl.Int64).sum().alias("present_quarters"),
                pl.col("ticker").filter(~pl.col("present")).n_unique().alias("tickers_with_holes"),
            ]
        )
        .with_columns(
            [
                (pl.col("expected_quarters") - pl.col("present_quarters")).alias("hole_count"),
                (
                    (pl.col("expected_quarters") - pl.col("present_quarters")) * 100.0
                    / pl.col("expected_quarters").clip(lower_bound=1)
                ).alias("hole_pct"),
            ]
        )
        .sort("hole_count", descending=True)
    )


def _build_ticker_gap_summary(*, ticker_metric_holes: pl.DataFrame) -> pl.DataFrame:
    if ticker_metric_holes.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "total_holes": pl.Int64,
                "worst_metric": pl.String,
                "worst_metric_holes": pl.Int64,
                "revenue_holes": pl.Int64,
                "net_income_holes": pl.Int64,
                "outstanding_shares_holes": pl.Int64,
                "epsActual_holes": pl.Int64,
            }
        )
    pivot = (
        ticker_metric_holes.select(["ticker", "ticker_code", "sector", "industry", "metric", "hole_count"])
        .pivot(index=["ticker", "ticker_code", "sector", "industry"], on="metric", values="hole_count", aggregate_function="first")
    )
    for metric in AUDIT_METRICS:
        if metric not in pivot.columns:
            pivot = pivot.with_columns(pl.lit(0).cast(pl.Int64).alias(metric))
    pivot = pivot.with_columns([pl.col(metric).fill_null(0).cast(pl.Int64).alias(metric) for metric in AUDIT_METRICS])
    summary = pivot.with_columns(
        [
            (
                pl.col("revenue")
                + pl.col("net_income")
                + pl.col("outstanding_shares")
                + pl.col("epsActual")
            ).alias("total_holes"),
        ]
    )
    rows: list[dict[str, object]] = []
    for row in summary.to_dicts():
        metric_pairs = [(metric, int(row.get(metric) or 0)) for metric in AUDIT_METRICS]
        worst_metric, worst_metric_holes = max(metric_pairs, key=lambda item: item[1])
        rows.append(
            {
                "ticker": row["ticker"],
                "ticker_code": row["ticker_code"],
                "sector": row["sector"],
                "industry": row["industry"],
                "total_holes": int(row["total_holes"] or 0),
                "worst_metric": worst_metric,
                "worst_metric_holes": worst_metric_holes,
                "revenue_holes": int(row.get("revenue") or 0),
                "net_income_holes": int(row.get("net_income") or 0),
                "outstanding_shares_holes": int(row.get("outstanding_shares") or 0),
                "epsActual_holes": int(row.get("epsActual") or 0),
            }
        )
    return pl.DataFrame(rows).sort(["total_holes", "ticker"], descending=[True, False])


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


def _select_deep_dive_tickers(
    *,
    share_candidates: pl.DataFrame,
    price_candidates: pl.DataFrame,
    ticker_gap_summary: pl.DataFrame,
    limit: int,
) -> list[str]:
    frames = [
        share_candidates.select("ticker") if not share_candidates.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
        price_candidates.select("ticker") if not price_candidates.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
        ticker_gap_summary.head(limit).select("ticker") if not ticker_gap_summary.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
    ]
    tickers = pl.concat(frames, how="vertical").unique().get_column("ticker").to_list()
    return sorted(tickers)[:limit]


def _render_dashboard_html(
    *,
    coverage: pl.DataFrame,
    missing: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    ticker_gap_summary: pl.DataFrame,
    ticker_metric_holes: pl.DataFrame,
    quarterly_holes: pl.DataFrame,
    share_candidates: pl.DataFrame,
    price_candidates: pl.DataFrame,
    ticker_anomalies: pl.DataFrame,
    deep_dive_tickers: list[str],
    price_source_dir: Path,
) -> str:
    links = "".join(
        f"<li><a href='tickers/{escape(ticker)}.html'>{escape(ticker)}</a></li>"
        for ticker in deep_dive_tickers
    )
    payload = {
        "coverage": coverage.to_dicts(),
        "kpi_holes": kpi_hole_summary.to_dicts(),
        "ticker_gaps": ticker_gap_summary.head(40).to_dicts(),
        "ticker_metric_holes": ticker_metric_holes.to_dicts(),
        "quarterly_holes": quarterly_holes.head(5000).to_dicts(),
        "share_candidates": share_candidates.head(5000).to_dicts(),
        "price_candidates": price_candidates.head(10000).to_dicts(),
        "ticker_anomalies": ticker_anomalies.head(50).to_dicts(),
    }
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>SEC Quality Dashboard</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 24px; color: #1a1a1a; background: #f6f7f9; }}
    h1, h2 {{ margin: 0 0 12px 0; }}
    .muted {{ color: #5f6b7a; }}
    .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 16px; margin: 18px 0 26px 0; }}
    .card {{ background: white; border-radius: 14px; padding: 16px 18px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); }}
    .section {{ background: white; border-radius: 16px; padding: 18px 20px; box-shadow: 0 8px 24px rgba(0,0,0,0.06); margin: 18px 0; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .chart-box {{ width: 100%; height: 420px; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 12px 0 16px 0; }}
    .controls input, .controls select {{ padding: 9px 10px; border: 1px solid #d7dde5; border-radius: 10px; background: #fff; }}
    .pill {{ display: inline-block; padding: 4px 10px; border-radius: 999px; background: #eef2f7; color: #415467; font-size: 12px; margin-right: 8px; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e6e8eb; text-align: left; padding: 8px; }}
    th {{ background: #fafbfc; }}
    ul {{ columns: 3; }}
    code {{ background: #eef2f7; padding: 2px 6px; border-radius: 6px; }}
    .table-wrap {{ max-height: 560px; overflow: auto; border: 1px solid #e6e8eb; border-radius: 12px; }}
    @media (max-width: 1100px) {{ .chart-grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>SEC Fundamentals Quality Dashboard</h1>
  <p class="muted">Fundamentals source: <code>data/sec/output</code>. Price source for split diagnostics: <code>{escape(str(price_source_dir))}</code>.</p>
  <div class="card-grid">
    {_coverage_cards_html(coverage)}
  </div>
  <div class="section">
    <h2>Coverage and KPI holes</h2>
    <div class="chart-grid">
      <div id="coverage-chart" class="chart-box"></div>
      <div id="kpi-holes-chart" class="chart-box"></div>
    </div>
    {_table_html(coverage)}
  </div>
  <div class="section">
    <h2>Where are the holes?</h2>
    <p class="muted">This view counts missing quarters inside each ticker's observed quarterly grid. It is the clearest view of holes by ticker and KPI.</p>
    <div class="chart-grid">
      <div id="ticker-holes-chart" class="chart-box"></div>
      <div id="ticker-heatmap-chart" class="chart-box"></div>
    </div>
    <div class="controls">
      <input id="ticker-filter" placeholder="Filter ticker or sector">
      <select id="metric-filter">
        <option value="">All KPIs</option>
        <option value="revenue">revenue</option>
        <option value="net_income">net_income</option>
        <option value="outstanding_shares">outstanding_shares</option>
        <option value="epsActual">epsActual</option>
      </select>
      <select id="hole-filter">
        <option value="1">At least 1 hole</option>
        <option value="4">At least 4 holes</option>
        <option value="8">At least 8 holes</option>
        <option value="12">At least 12 holes</option>
      </select>
    </div>
    <div class="table-wrap" id="ticker-hole-table"></div>
  </div>
  <div class="section">
    <h2>Quarter-level missing rows</h2>
    <p class="muted">Raw missing quarter rows, useful when you want to know exactly which quarter is absent for which KPI.</p>
    <div class="table-wrap" id="quarter-hole-table"></div>
  </div>
  <div class="section">
    <h2>Split and adjustment diagnostics</h2>
    <div class="chart-grid">
      <div id="anomaly-chart" class="chart-box"></div>
      <div id="share-price-chart" class="chart-box"></div>
    </div>
    <div class="controls">
      <span class="pill">{share_candidates.height} share anomaly rows</span>
      <span class="pill">{price_candidates.height} price adjustment anomaly rows</span>
    </div>
    {_table_html(ticker_anomalies.head(50))}
  </div>
  <div class="section">
    <h2>Tickers with zero coverage on a KPI</h2>
    {_table_html(missing.head(200))}
  </div>
  <div class="section">
    <h2>Ticker deep dives</h2>
    <ul>{links}</ul>
  </div>
  <script>
    const payload = {json.dumps(payload)};

    function fmtPct(value) {{
      return `${{Number(value || 0).toFixed(1)}}%`;
    }}

    function renderTable(targetId, rows, columns) {{
      const target = document.getElementById(targetId);
      if (!rows.length) {{
        target.innerHTML = '<p>No rows.</p>';
        return;
      }}
      const head = `<thead><tr>${{columns.map((column) => `<th>${{column}}</th>`).join('')}}</tr></thead>`;
      const body = rows.map((row) => `<tr>${{columns.map((column) => `<td>${{row[column] ?? ''}}</td>`).join('')}}</tr>`).join('');
      target.innerHTML = `<table>${{head}}<tbody>${{body}}</tbody></table>`;
    }}

    const coverageChart = echarts.init(document.getElementById('coverage-chart'));
    coverageChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis' }},
      grid: {{ left: 60, right: 20, top: 40, bottom: 40 }},
      xAxis: {{ type: 'category', data: payload.coverage.map((row) => row.metric) }},
      yAxis: {{ type: 'value', name: 'Coverage %', max: 100 }},
      series: [{{
        type: 'bar',
        data: payload.coverage.map((row) => row.coverage_pct),
        itemStyle: {{ color: '#3f6db5' }},
        label: {{ show: true, formatter: (params) => `${{params.value.toFixed(1)}}%` }}
      }}]
    }});

    const kpiHoleChart = echarts.init(document.getElementById('kpi-holes-chart'));
    kpiHoleChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis' }},
      grid: {{ left: 110, right: 20, top: 30, bottom: 30 }},
      xAxis: {{ type: 'value', name: 'Missing quarters' }},
      yAxis: {{ type: 'category', data: payload.kpi_holes.map((row) => row.metric) }},
      series: [{{
        type: 'bar',
        data: payload.kpi_holes.map((row) => row.hole_count),
        itemStyle: {{ color: '#b85c38' }},
        label: {{ show: true, position: 'right' }}
      }}]
    }});

    const tickerHoleRows = payload.ticker_gaps.filter((row) => Number(row.total_holes || 0) > 0);
    const tickerHolesChart = echarts.init(document.getElementById('ticker-holes-chart'));
    tickerHolesChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      legend: {{ top: 0 }},
      grid: {{ left: 60, right: 20, top: 40, bottom: 80 }},
      xAxis: {{ type: 'category', data: tickerHoleRows.map((row) => row.ticker_code), axisLabel: {{ rotate: 45 }} }},
      yAxis: {{ type: 'value', name: 'Holes' }},
      series: [
        {{ name: 'revenue', type: 'bar', stack: 'holes', data: tickerHoleRows.map((row) => row.revenue_holes), itemStyle: {{ color: '#d18f53' }} }},
        {{ name: 'net_income', type: 'bar', stack: 'holes', data: tickerHoleRows.map((row) => row.net_income_holes), itemStyle: {{ color: '#c55b5b' }} }},
        {{ name: 'outstanding_shares', type: 'bar', stack: 'holes', data: tickerHoleRows.map((row) => row.outstanding_shares_holes), itemStyle: {{ color: '#5a8bb0' }} }},
        {{ name: 'epsActual', type: 'bar', stack: 'holes', data: tickerHoleRows.map((row) => row.epsActual_holes), itemStyle: {{ color: '#6a5ea8' }} }},
      ]
    }});

    const heatmapSource = payload.ticker_metric_holes
      .filter((row) => Number(row.hole_count || 0) > 0)
      .sort((left, right) => Number(right.hole_count || 0) - Number(left.hole_count || 0))
      .slice(0, 120);
    const heatmapTickers = Array.from(new Set(heatmapSource.map((row) => row.ticker_code)));
    const heatmapMetrics = ['revenue', 'net_income', 'outstanding_shares', 'epsActual'];
    const heatmapData = heatmapSource.map((row) => [heatmapMetrics.indexOf(row.metric), heatmapTickers.indexOf(row.ticker_code), Number(row.hole_count || 0)]);
    const heatmapChart = echarts.init(document.getElementById('ticker-heatmap-chart'));
    heatmapChart.setOption({{
      animation: false,
      tooltip: {{
        formatter: (params) => {{
          const metric = heatmapMetrics[params.value[0]];
          const ticker = heatmapTickers[params.value[1]];
          return `${{ticker}}<br>${{metric}}<br>holes: ${{params.value[2]}}`;
        }}
      }},
      grid: {{ left: 100, right: 30, top: 30, bottom: 50 }},
      xAxis: {{ type: 'category', data: heatmapMetrics }},
      yAxis: {{ type: 'category', data: heatmapTickers }},
      visualMap: {{
        min: 0,
        max: Math.max(...heatmapData.map((row) => row[2]), 1),
        calculable: true,
        orient: 'horizontal',
        left: 'center',
        bottom: 0,
        inRange: {{ color: ['#eef4fb', '#f5cb8f', '#d1654e'] }}
      }},
      series: [{{
        type: 'heatmap',
        data: heatmapData,
        label: {{ show: true }}
      }}]
    }});

    const anomalyChart = echarts.init(document.getElementById('anomaly-chart'));
    anomalyChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      legend: {{ top: 0 }},
      grid: {{ left: 60, right: 20, top: 40, bottom: 80 }},
      xAxis: {{ type: 'category', data: payload.ticker_anomalies.map((row) => row.ticker_code), axisLabel: {{ rotate: 45 }} }},
      yAxis: {{ type: 'value', name: 'Anomaly rows' }},
      series: [
        {{ name: 'share anomalies', type: 'bar', stack: 'anomaly', data: payload.ticker_anomalies.map((row) => row.share_candidate_count || 0), itemStyle: {{ color: '#3b7a57' }} }},
        {{ name: 'price anomalies', type: 'bar', stack: 'anomaly', data: payload.ticker_anomalies.map((row) => row.price_candidate_count || 0), itemStyle: {{ color: '#a64b2a' }} }},
      ]
    }});

    const sharePriceChart = echarts.init(document.getElementById('share-price-chart'));
    sharePriceChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis' }},
      legend: {{ top: 0 }},
      grid: {{ left: 60, right: 20, top: 40, bottom: 50 }},
      xAxis: {{ type: 'category', data: ['share_ratio', 'factor_ratio'] }},
      yAxis: {{ type: 'value', name: 'Max observed ratio' }},
      series: [
        {{ name: 'max share ratio', type: 'bar', data: [Math.max(...payload.share_candidates.map((row) => Number(row.share_ratio || 0)), 0), null], itemStyle: {{ color: '#3b7a57' }} }},
        {{ name: 'max factor ratio', type: 'bar', data: [null, Math.max(...payload.price_candidates.map((row) => Number(row.factor_ratio || 0)), 0)], itemStyle: {{ color: '#a64b2a' }} }},
      ]
    }});

    function updateHoleTables() {{
      const tickerFilter = document.getElementById('ticker-filter').value.toLowerCase();
      const metricFilter = document.getElementById('metric-filter').value;
      const holeFilter = Number(document.getElementById('hole-filter').value || '1');
      const tickerRows = payload.ticker_metric_holes.filter((row) => {{
        const text = `${{row.ticker}} ${{row.sector || ''}} ${{row.industry || ''}}`.toLowerCase();
        const metricOk = !metricFilter || row.metric === metricFilter;
        return Number(row.hole_count || 0) >= holeFilter && metricOk && text.includes(tickerFilter);
      }});
      renderTable('ticker-hole-table', tickerRows.slice(0, 500), ['ticker', 'metric', 'hole_count', 'hole_pct', 'expected_quarters', 'present_quarters', 'first_quarter', 'last_quarter', 'sample_missing_dates']);

      const quarterRows = payload.quarterly_holes.filter((row) => {{
        const text = `${{row.ticker}} ${{row.sector || ''}} ${{row.industry || ''}}`.toLowerCase();
        const metricOk = !metricFilter || row.metric === metricFilter;
        return metricOk && text.includes(tickerFilter);
      }});
      renderTable('quarter-hole-table', quarterRows.slice(0, 1000), ['ticker', 'metric', 'date', 'sector', 'industry']);
    }}

    document.getElementById('ticker-filter').addEventListener('input', updateHoleTables);
    document.getElementById('metric-filter').addEventListener('change', updateHoleTables);
    document.getElementById('hole-filter').addEventListener('change', updateHoleTables);
    updateHoleTables();

    window.addEventListener('resize', () => {{
      coverageChart.resize();
      kpiHoleChart.resize();
      tickerHolesChart.resize();
      heatmapChart.resize();
      anomalyChart.resize();
      sharePriceChart.resize();
    }});
  </script>
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
    ticker_metric_holes: pl.DataFrame,
    quarterly_holes: pl.DataFrame,
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
  <div class="section"><h2>KPI hole summary</h2>{_table_html(ticker_metric_holes)}</div>
  <div class="section"><h2>Missing quarter rows</h2>{_table_html(quarterly_holes.head(120))}</div>
  <div class="section"><h2>Earnings</h2>{_table_html(earnings.sort('period_end').tail(20))}</div>
  <div class="section"><h2>Share anomaly candidates</h2>{_table_html(share_candidates)}</div>
  <div class="section"><h2>Price adjustment candidates</h2>{_table_html(price_candidates)}</div>
</body>
</html>
"""


def _render_dashboard_markdown(
    *,
    coverage: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    ticker_gap_summary: pl.DataFrame,
    missing: pl.DataFrame,
) -> str:
    lines = [
        "# SEC Audit Report",
        "",
        f"Generated at: {datetime.utcnow().isoformat()}Z",
        "",
        "## Coverage",
        "",
    ]
    for row in coverage.to_dicts():
        lines.append(
            f"- `{row['metric']}`: {row['tickers_with_data']} / {row['total_tickers']} tickers "
            f"({row['coverage_pct']:.2f}%), rows `{row['rows']}`, range `{row['min_date']}` -> `{row['max_date']}`"
        )
    lines.extend(["", "## KPI Holes", ""])
    for row in kpi_hole_summary.to_dicts():
        lines.append(
            f"- `{row['metric']}`: `{row['hole_count']}` missing quarters across `{row['tickers_with_holes']}` tickers "
            f"({row['hole_pct']:.2f}% of expected quarters)"
        )
    lines.extend(["", "## Worst Tickers", ""])
    for row in ticker_gap_summary.head(15).to_dicts():
        if int(row["total_holes"] or 0) <= 0:
            continue
        lines.append(
            f"- `{row['ticker']}`: `{row['total_holes']}` holes; worst KPI `{row['worst_metric']}` "
            f"(`{row['worst_metric_holes']}`)"
        )
    lines.extend(["", "## Zero-Coverage Tickers", ""])
    for metric in AUDIT_METRICS:
        subset = missing.filter(pl.col("metric") == metric)
        sample = ", ".join(subset.get_column("ticker").head(15).to_list()) if not subset.is_empty() else "none"
        lines.append(f"- `{metric}`: {subset.height} tickers missing entirely. Sample: {sample}")
    lines.append("")
    return "\n".join(lines)


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
