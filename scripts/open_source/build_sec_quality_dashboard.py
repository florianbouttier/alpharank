#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

import polars as pl


CORE_METRICS: tuple[str, ...] = ("revenue", "net_income", "outstanding_shares")
AUDIT_METRICS: tuple[str, ...] = CORE_METRICS + ("epsActual",)
METRIC_LABELS: dict[str, str] = {
    "revenue": "Chiffre d'affaires",
    "net_income": "Resultat net",
    "outstanding_shares": "Actions en circulation",
    "epsActual": "EPS publie",
}


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sec_output_dir = args.sec_output_dir.resolve()
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

    coverage = _build_coverage_summary(financials=financials, earnings=earnings, general=general)
    missing = _build_missing_ticker_table(financials=financials, earnings=earnings, general=general)
    zero_coverage = _build_zero_coverage_summary(missing=missing)
    quarterly_presence = _build_quarterly_presence(financials=financials, earnings=earnings, general=general)
    ticker_metric_holes = _build_ticker_metric_holes(quarterly_presence=quarterly_presence)
    quarterly_holes = quarterly_presence.filter(~pl.col("present")).sort(["metric", "ticker", "date"])
    kpi_hole_summary = _build_kpi_hole_summary(quarterly_presence=quarterly_presence)
    sector_gap_summary = _build_sector_gap_summary(ticker_metric_holes=ticker_metric_holes)
    ticker_gap_summary = _build_ticker_gap_summary(ticker_metric_holes=ticker_metric_holes)
    share_candidates = _build_share_split_candidates(shares=shares, ratio_threshold=args.share_ratio_threshold)
    share_anomaly_summary = _build_share_anomaly_summary(share_candidates=share_candidates)
    overview = _build_overview_frame(
        coverage=coverage,
        kpi_hole_summary=kpi_hole_summary,
        zero_coverage=zero_coverage,
    )

    overview.write_parquet(output_dir / "overview.parquet")
    overview.write_csv(output_dir / "overview.csv")
    coverage.write_parquet(output_dir / "coverage_summary.parquet")
    coverage.write_csv(output_dir / "coverage_summary.csv")
    zero_coverage.write_parquet(output_dir / "zero_coverage_summary.parquet")
    zero_coverage.write_csv(output_dir / "zero_coverage_summary.csv")
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
    sector_gap_summary.write_parquet(output_dir / "sector_gap_summary.parquet")
    sector_gap_summary.write_csv(output_dir / "sector_gap_summary.csv")
    ticker_gap_summary.write_parquet(output_dir / "ticker_gap_summary.parquet")
    ticker_gap_summary.write_csv(output_dir / "ticker_gap_summary.csv")
    share_candidates.write_parquet(output_dir / "share_split_candidates.parquet")
    share_candidates.write_csv(output_dir / "share_split_candidates.csv")
    share_anomaly_summary.write_parquet(output_dir / "share_anomaly_summary.parquet")
    share_anomaly_summary.write_csv(output_dir / "share_anomaly_summary.csv")

    deep_dive_tickers = _select_deep_dive_tickers(
        ticker_gap_summary=ticker_gap_summary,
        share_anomaly_summary=share_anomaly_summary,
        limit=args.deep_dive_limit,
    )
    for ticker in deep_dive_tickers:
        (ticker_dir / f"{ticker}.html").write_text(
            _render_ticker_page(
                ticker=ticker,
                shares=shares.filter(pl.col("ticker") == ticker),
                financials=financials.filter(
                    (pl.col("ticker") == ticker) & pl.col("metric").is_in(["revenue", "net_income", "outstanding_shares"])
                ),
                earnings=earnings.filter(pl.col("ticker") == ticker),
                ticker_metric_holes=ticker_metric_holes.filter(pl.col("ticker") == ticker),
                quarterly_holes=quarterly_holes.filter(pl.col("ticker") == ticker),
                share_candidates=share_candidates.filter(pl.col("ticker") == ticker),
            ),
            encoding="utf-8",
        )

    manifest = {
        "generated_at": _utc_now_iso(),
        "sec_output_dir": str(sec_output_dir),
        "output_dir": str(output_dir),
        "report_scope": "sec_only_fundamentals",
        "share_ratio_threshold": args.share_ratio_threshold,
        "deep_dive_tickers": deep_dive_tickers,
        "notes": {
            "source_of_truth": "SEC only",
            "uses_eodhd": False,
            "uses_yahoo": False,
            "uses_simfin": False,
            "hole_definition": (
                "Pour un ticker, on prend tous les trimestres ou au moins un des 4 KPI SEC existe. "
                "Si un KPI manque sur un de ces trimestres, on compte 1 trou."
            ),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "report.html").write_text(
        _render_dashboard_html(
            overview=overview,
            kpi_hole_summary=kpi_hole_summary,
            sector_gap_summary=sector_gap_summary,
            ticker_gap_summary=ticker_gap_summary,
            ticker_metric_holes=ticker_metric_holes,
            quarterly_holes=quarterly_holes,
            share_anomaly_summary=share_anomaly_summary,
            missing=missing,
            deep_dive_tickers=deep_dive_tickers,
        ),
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(
        _render_dashboard_markdown(
            overview=overview,
            kpi_hole_summary=kpi_hole_summary,
            ticker_gap_summary=ticker_gap_summary,
            missing=missing,
        ),
        encoding="utf-8",
    )
    print(output_dir)
    print(output_dir / "report.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SEC-only fundamentals audit dashboard in French.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--sec-output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--share-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--deep-dive-limit", type=int, default=30)
    return parser.parse_args()


def _build_coverage_summary(*, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame) -> pl.DataFrame:
    total_tickers = general.height
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        subset = financials.filter(pl.col("metric") == metric)
        tickers_with_data = subset.get_column("ticker").n_unique() if not subset.is_empty() else 0
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "tickers_with_data": tickers_with_data,
                "total_tickers": total_tickers,
                "coverage_pct": _pct(tickers_with_data, total_tickers),
                "first_date": subset.get_column("date").min() if not subset.is_empty() else None,
                "last_date": subset.get_column("date").max() if not subset.is_empty() else None,
            }
        )
    eps = earnings.filter(pl.col("epsActual").is_not_null())
    eps_tickers = eps.get_column("ticker").n_unique() if not eps.is_empty() else 0
    rows.append(
        {
            "metric": "epsActual",
            "metric_label": METRIC_LABELS["epsActual"],
            "tickers_with_data": eps_tickers,
            "total_tickers": total_tickers,
            "coverage_pct": _pct(eps_tickers, total_tickers),
            "first_date": eps.get_column("period_end").min() if not eps.is_empty() else None,
            "last_date": eps.get_column("period_end").max() if not eps.is_empty() else None,
        }
    )
    return pl.DataFrame(rows)


def _build_missing_ticker_table(*, financials: pl.DataFrame, earnings: pl.DataFrame, general: pl.DataFrame) -> pl.DataFrame:
    ticker_frame = general.select(
        [
            (pl.col("Code") + pl.lit(".US")).alias("ticker"),
            pl.col("Code").alias("ticker_code"),
            pl.col("Sector").alias("sector"),
            pl.col("Industry").alias("industry"),
        ]
    )
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        present = financials.filter(pl.col("metric") == metric).select("ticker").unique()
        rows.extend(
            ticker_frame.join(present, on="ticker", how="anti")
            .with_columns(pl.lit(metric).alias("metric"), pl.lit(METRIC_LABELS[metric]).alias("metric_label"))
            .to_dicts()
        )
    eps_present = earnings.filter(pl.col("epsActual").is_not_null()).select("ticker").unique()
    rows.extend(
        ticker_frame.join(eps_present, on="ticker", how="anti")
        .with_columns(pl.lit("epsActual").alias("metric"), pl.lit(METRIC_LABELS["epsActual"]).alias("metric_label"))
        .to_dicts()
    )
    if not rows:
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
            }
        )
    return pl.DataFrame(rows).sort(["metric", "ticker"])


def _build_zero_coverage_summary(*, missing: pl.DataFrame) -> pl.DataFrame:
    if missing.is_empty():
        return pl.DataFrame(schema={"metric": pl.String, "metric_label": pl.String, "zero_coverage_tickers": pl.Int64})
    return (
        missing.group_by(["metric", "metric_label"])
        .agg(pl.len().alias("zero_coverage_tickers"))
        .sort("metric")
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
        .select(["ticker", pl.col("period_end").alias("date"), pl.lit("epsActual").alias("metric")])
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
                "metric_label": pl.String,
                "present": pl.Boolean,
            }
        )

    expected = observed.select(["ticker", "date"]).unique().join(pl.DataFrame({"metric": list(AUDIT_METRICS)}), how="cross")
    return (
        expected.join(
            observed.with_columns(pl.lit(True).alias("present")),
            on=["ticker", "date", "metric"],
            how="left",
        )
        .with_columns(
            [
                pl.col("present").fill_null(False),
                pl.col("metric").replace_strict(METRIC_LABELS, default=pl.col("metric")).alias("metric_label"),
            ]
        )
        .join(ticker_info, on="ticker", how="left")
        .select(["ticker", "ticker_code", "sector", "industry", "date", "metric", "metric_label", "present"])
        .sort(["ticker", "date", "metric"])
    )


def _build_ticker_metric_holes(*, quarterly_presence: pl.DataFrame) -> pl.DataFrame:
    if quarterly_presence.is_empty():
        return pl.DataFrame(
            schema={
                "ticker": pl.String,
                "ticker_code": pl.String,
                "sector": pl.String,
                "industry": pl.String,
                "metric": pl.String,
                "metric_label": pl.String,
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
        quarterly_presence.group_by(["ticker", "ticker_code", "sector", "industry", "metric", "metric_label"])
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
                "metric_label": pl.String,
                "expected_quarters": pl.Int64,
                "present_quarters": pl.Int64,
                "tickers_with_holes": pl.Int64,
                "hole_count": pl.Int64,
                "hole_pct": pl.Float64,
            }
        )
    return (
        quarterly_presence.group_by(["metric", "metric_label"])
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


def _build_sector_gap_summary(*, ticker_metric_holes: pl.DataFrame) -> pl.DataFrame:
    if ticker_metric_holes.is_empty():
        return pl.DataFrame(
            schema={"sector": pl.String, "metric": pl.String, "metric_label": pl.String, "hole_count": pl.Int64}
        )
    return (
        ticker_metric_holes.group_by(["sector", "metric", "metric_label"])
        .agg(pl.col("hole_count").sum())
        .sort(["hole_count", "sector"], descending=[True, False])
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
                "worst_metric_label": pl.String,
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
    pivot = pivot.with_columns(
        (
            pl.col("revenue") + pl.col("net_income") + pl.col("outstanding_shares") + pl.col("epsActual")
        ).alias("total_holes")
    )

    rows: list[dict[str, object]] = []
    for row in pivot.to_dicts():
        metric_pairs = [(metric, int(row.get(metric) or 0)) for metric in AUDIT_METRICS]
        worst_metric, worst_value = max(metric_pairs, key=lambda item: item[1])
        rows.append(
            {
                "ticker": row["ticker"],
                "ticker_code": row["ticker_code"],
                "sector": row["sector"],
                "industry": row["industry"],
                "total_holes": int(row["total_holes"] or 0),
                "worst_metric": worst_metric,
                "worst_metric_label": METRIC_LABELS[worst_metric],
                "worst_metric_holes": worst_value,
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
            schema={
                "ticker": pl.String,
                "date": pl.String,
                "shares": pl.Float64,
                "prev_shares": pl.Float64,
                "share_ratio": pl.Float64,
                "candidate_kind": pl.String,
            }
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
            .then(pl.lit("hausse forte"))
            .when(pl.col("share_ratio") <= (1.0 / ratio_threshold))
            .then(pl.lit("baisse forte"))
            .otherwise(pl.lit(None).cast(pl.Utf8))
            .alias("candidate_kind")
        )
        .filter(pl.col("candidate_kind").is_not_null())
        .sort(["ticker", "date"])
    )


def _build_share_anomaly_summary(*, share_candidates: pl.DataFrame) -> pl.DataFrame:
    if share_candidates.is_empty():
        return pl.DataFrame(
            schema={"ticker": pl.String, "candidate_count": pl.Int64, "max_ratio": pl.Float64, "min_ratio": pl.Float64}
        )
    return (
        share_candidates.group_by("ticker")
        .agg(
            [
                pl.len().alias("candidate_count"),
                pl.col("share_ratio").max().alias("max_ratio"),
                pl.col("share_ratio").min().alias("min_ratio"),
            ]
        )
        .sort(["candidate_count", "ticker"], descending=[True, False])
    )


def _build_overview_frame(
    *,
    coverage: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    zero_coverage: pl.DataFrame,
) -> pl.DataFrame:
    return (
        coverage.join(
            kpi_hole_summary.select(["metric", "tickers_with_holes", "hole_count", "hole_pct"]),
            on="metric",
            how="left",
        )
        .join(zero_coverage, on=["metric", "metric_label"], how="left")
        .with_columns(
            [
                pl.col("tickers_with_holes").fill_null(0).cast(pl.Int64),
                pl.col("hole_count").fill_null(0).cast(pl.Int64),
                pl.col("hole_pct").fill_null(0.0).cast(pl.Float64),
                pl.col("zero_coverage_tickers").fill_null(0).cast(pl.Int64),
            ]
        )
        .sort("metric")
    )


def _select_deep_dive_tickers(
    *,
    ticker_gap_summary: pl.DataFrame,
    share_anomaly_summary: pl.DataFrame,
    limit: int,
) -> list[str]:
    frames = [
        ticker_gap_summary.head(limit).select("ticker") if not ticker_gap_summary.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
        share_anomaly_summary.head(limit).select("ticker") if not share_anomaly_summary.is_empty() else pl.DataFrame(schema={"ticker": pl.String}),
    ]
    return sorted(pl.concat(frames, how="vertical").unique().get_column("ticker").to_list())[:limit]


def _render_dashboard_html(
    *,
    overview: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    sector_gap_summary: pl.DataFrame,
    ticker_gap_summary: pl.DataFrame,
    ticker_metric_holes: pl.DataFrame,
    quarterly_holes: pl.DataFrame,
    share_anomaly_summary: pl.DataFrame,
    missing: pl.DataFrame,
    deep_dive_tickers: list[str],
) -> str:
    links = "".join(
        f"<li><a href='tickers/{escape(ticker)}.html'>{escape(ticker)}</a></li>"
        for ticker in deep_dive_tickers
    )
    payload = {
        "overview": overview.to_dicts(),
        "kpi_holes": kpi_hole_summary.to_dicts(),
        "sector_holes": sector_gap_summary.to_dicts(),
        "ticker_gaps": ticker_gap_summary.head(12).to_dicts(),
        "ticker_metric_holes": ticker_metric_holes.to_dicts(),
        "quarterly_holes": quarterly_holes.head(6000).to_dicts(),
        "share_anomalies": share_anomaly_summary.head(30).to_dicts(),
        "missing": missing.to_dicts(),
    }
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Audit SEC des fondamentaux</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #12202f; background: #f4f6f8; }}
    .wrap {{ padding: 28px 32px 40px 32px; }}
    h1 {{ margin: 0 0 8px 0; font-size: 38px; letter-spacing: -0.03em; }}
    h2 {{ margin: 0 0 12px 0; font-size: 28px; letter-spacing: -0.03em; }}
    h3 {{ margin: 0 0 10px 0; font-size: 18px; }}
    .muted {{ color: #607081; }}
    .section {{ background: white; border-radius: 20px; padding: 22px 24px; box-shadow: 0 10px 30px rgba(10, 20, 30, 0.06); margin-top: 18px; }}
    .intro {{ display: grid; grid-template-columns: 1.2fr 1fr; gap: 18px; }}
    .info-box {{ background: #f7f9fb; border: 1px solid #e5ebf1; border-radius: 16px; padding: 16px 18px; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 16px; margin-top: 18px; }}
    .metric-card {{ background: linear-gradient(180deg, #ffffff 0%, #fafbfc 100%); border: 1px solid #e7ecf2; border-radius: 18px; padding: 18px; }}
    .metric-card .metric-title {{ font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em; color: #66788b; }}
    .metric-card .big {{ margin-top: 12px; font-size: 30px; font-weight: 700; }}
    .metric-card .line {{ margin-top: 6px; color: #304253; line-height: 1.45; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .chart-box {{ width: 100%; height: 420px; }}
    .controls {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 14px 0; }}
    .controls input, .controls select {{ padding: 10px 12px; border: 1px solid #d6dde5; border-radius: 12px; background: white; min-width: 180px; }}
    .table-wrap {{ border: 1px solid #e5ebf1; border-radius: 16px; overflow: auto; max-height: 620px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 10px 10px; border-bottom: 1px solid #edf1f5; text-align: left; vertical-align: top; }}
    th {{ position: sticky; top: 0; background: #f8fafc; z-index: 1; }}
    .pill {{ display: inline-block; padding: 5px 10px; border-radius: 999px; background: #eef3f8; color: #486074; font-size: 12px; margin-right: 8px; margin-bottom: 8px; }}
    ul {{ columns: 3; margin: 0; padding-left: 20px; }}
    @media (max-width: 1100px) {{
      .intro, .chart-grid {{ grid-template-columns: 1fr; }}
      ul {{ columns: 1; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Audit SEC des fondamentaux</h1>
    <div class="intro">
      <div class="info-box">
        <h3>Ce rapport est 100% SEC</h3>
        <p class="muted">Il n'utilise ni EODHD, ni Yahoo, ni SimFin. Toutes les vues ci-dessous partent uniquement de <code>data/sec/output</code>.</p>
      </div>
      <div class="info-box">
        <h3>Definition d'un trou</h3>
        <p class="muted">Pour un ticker, on regarde tous les trimestres ou au moins un des 4 KPI SEC existe. Si un KPI manque sur un de ces trimestres, on compte 1 trou pour ce KPI.</p>
      </div>
    </div>

    <div class="section">
      <h2>Lecture directe</h2>
      <p class="muted">Les grands nombres ci-dessous comptent les trous. Un trou = un trimestre manquant pour un KPI donne. Le but du rapport est de montrer ou ca manque, pas de t'afficher un taux de couverture abstrait.</p>
    </div>

    <div class="metric-grid">
      {_metric_cards_html(overview)}
    </div>

    <div class="section">
      <h2>Ce qui manque vraiment</h2>
      <p class="muted">A gauche: combien de tickers sont touches par au moins un trou. A droite: combien de trimestres manquent au total. C'est la vue la plus importante du rapport.</p>
      <div class="chart-grid">
        <div id="hole-ticker-chart" class="chart-box"></div>
        <div id="kpi-holes-chart" class="chart-box"></div>
      </div>
    </div>

    <div class="section">
      <h2>Les pires zones du package</h2>
      <p class="muted">A gauche: les 12 tickers qui concentrent le plus de trous. A droite: les secteurs les plus touches. Ici, on regarde ou il faut enqueter en premier.</p>
      <div class="chart-grid">
        <div id="ticker-chart" class="chart-box"></div>
        <div id="sector-chart" class="chart-box"></div>
      </div>
    </div>

    <div class="section">
      <h2>Recherche par ticker et par KPI</h2>
      <p class="muted">Cette table sert a repondre a des questions concretes: "Combien de trous sur ce ticker ?", "Sur quel KPI ?", "Sur quelle periode ?", "Quels trimestres manquent ?"</p>
      <div class="controls">
        <input id="ticker-filter" placeholder="Filtrer ticker, secteur ou industrie">
        <select id="metric-filter">
          <option value="">Tous les KPI</option>
          <option value="revenue">Chiffre d'affaires</option>
          <option value="net_income">Resultat net</option>
          <option value="outstanding_shares">Actions en circulation</option>
          <option value="epsActual">EPS publie</option>
        </select>
        <select id="hole-filter">
          <option value="1">Au moins 1 trou</option>
          <option value="4">Au moins 4 trous</option>
          <option value="8">Au moins 8 trous</option>
          <option value="12">Au moins 12 trous</option>
        </select>
      </div>
      <div class="table-wrap" id="ticker-hole-table"></div>
    </div>

    <div class="section">
      <h2>Liste des trimestres manquants</h2>
      <p class="muted">Vue brute des quarts manquants, utile quand tu veux descendre au niveau exact du trimestre.</p>
      <div class="table-wrap" id="quarter-hole-table"></div>
    </div>

    <div class="section">
      <h2>Tickers sans aucune donnee pour un KPI</h2>
      <p class="muted">Ici, on ne parle plus de trous partiels. On parle des tickers pour lesquels le KPI est totalement absent du package SEC.</p>
      <div id="zero-coverage-pills">{_zero_coverage_html(missing)}</div>
      <div class="table-wrap">
        {_table_html(
            _rename_columns(
                missing.head(200),
                {
                    "ticker": "ticker",
                    "metric_label": "kpi",
                    "sector": "secteur",
                    "industry": "industrie",
                },
            )
        )}
      </div>
    </div>

    <div class="section">
      <h2>Anomalies SEC sur les actions en circulation</h2>
      <p class="muted">Cette vue reste 100% SEC. Elle signale les forts sauts de la serie <code>US_share</code>. Cela peut correspondre a un split, a un changement de base, ou a une incoherence SEC a investiguer.</p>
      <div class="chart-grid">
        <div id="share-anomaly-chart" class="chart-box"></div>
        <div class="info-box">
          <h3>Interpretation</h3>
          <p class="muted">Un saut important sur les actions en circulation n'est pas forcement une erreur. Mais c'est un bon point d'entree pour verifier les splits et les changements de semantique par action.</p>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Deep dives</h2>
      <ul>{links}</ul>
    </div>
  </div>

  <script>
    const payload = {json.dumps(payload)};
    const KPI_ORDER = ['revenue', 'net_income', 'outstanding_shares', 'epsActual'];
    const KPI_LABELS = {{
      revenue: "Chiffre d'affaires",
      net_income: "Resultat net",
      outstanding_shares: "Actions en circulation",
      epsActual: "EPS publie",
    }};

    function renderTable(targetId, rows, columns, labels) {{
      const target = document.getElementById(targetId);
      if (!rows.length) {{
        target.innerHTML = '<p class="muted">Aucune ligne.</p>';
        return;
      }}
      const header = `<thead><tr>${{columns.map((column) => `<th>${{labels[column] || column}}</th>`).join('')}}</tr></thead>`;
      const body = rows.map((row) => `<tr>${{columns.map((column) => `<td>${{row[column] ?? ''}}</td>`).join('')}}</tr>`).join('');
      target.innerHTML = `<table>${{header}}<tbody>${{body}}</tbody></table>`;
    }}

    const holeTickerChart = echarts.init(document.getElementById('hole-ticker-chart'));
    holeTickerChart.setOption({{
      animation: false,
      tooltip: {{
        trigger: 'axis',
        formatter: (params) => {{
          const row = payload.overview[params[0].dataIndex];
          return `${{row.metric_label}}<br>${{row.tickers_with_holes}} tickers sur ${{row.total_tickers}} ont au moins un trou<br>${{row.zero_coverage_tickers}} tickers sont totalement vides`;
        }}
      }},
      grid: {{ left: 70, right: 20, top: 40, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'Tickers touches' }},
      yAxis: {{ type: 'category', data: payload.overview.map((row) => row.metric_label) }},
      series: [
        {{
          type: 'bar',
          data: payload.overview.map((row) => row.tickers_with_holes),
          itemStyle: {{ color: '#8c4a2f' }},
          label: {{ show: true, position: 'right' }}
        }}
      ]
    }});

    const kpiHolesChart = echarts.init(document.getElementById('kpi-holes-chart'));
    kpiHolesChart.setOption({{
      animation: false,
      tooltip: {{
        trigger: 'axis',
        formatter: (params) => {{
          const row = payload.kpi_holes[params[0].dataIndex];
          return `${{row.metric_label}}<br>${{row.hole_count}} trimestres manquants au total<br>${{row.tickers_with_holes}} tickers touches`;
        }}
      }},
      grid: {{ left: 90, right: 20, top: 40, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'Trimestres manquants' }},
      yAxis: {{ type: 'category', data: payload.kpi_holes.map((row) => row.metric_label) }},
      series: [
        {{
          type: 'bar',
          data: payload.kpi_holes.map((row) => row.hole_count),
          itemStyle: {{ color: '#b85c38' }},
          label: {{ show: true, position: 'right' }}
        }}
      ]
    }});

    const tickerChart = echarts.init(document.getElementById('ticker-chart'));
    tickerChart.setOption({{
      animation: false,
      tooltip: {{
        trigger: 'axis',
        axisPointer: {{ type: 'shadow' }},
      }},
      legend: {{ top: 0 }},
      grid: {{ left: 90, right: 20, top: 40, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'Trimestres manquants' }},
      yAxis: {{ type: 'category', data: payload.ticker_gaps.map((row) => row.ticker_code) }},
      series: KPI_ORDER.map((metric) => ({{
        name: KPI_LABELS[metric],
        type: 'bar',
        stack: 'holes',
        data: payload.ticker_gaps.map((row) => row[`${{metric}}_holes`]),
      }})),
    }});

    const sectorRows = payload.sector_holes
      .reduce((acc, row) => {{
        if (!row.sector) return acc;
        const current = acc.get(row.sector) || {{ sector: row.sector, revenue: 0, net_income: 0, outstanding_shares: 0, epsActual: 0, total: 0 }};
        current[row.metric] = Number(row.hole_count || 0);
        current.total += Number(row.hole_count || 0);
        acc.set(row.sector, current);
        return acc;
      }}, new Map());
    const topSectors = Array.from(sectorRows.values()).sort((a, b) => b.total - a.total).slice(0, 12);
    const sectorChart = echarts.init(document.getElementById('sector-chart'));
    sectorChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      legend: {{ top: 0 }},
      grid: {{ left: 120, right: 20, top: 40, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'Trimestres manquants' }},
      yAxis: {{ type: 'category', data: topSectors.map((row) => row.sector) }},
      series: KPI_ORDER.map((metric) => ({{
        name: KPI_LABELS[metric],
        type: 'bar',
        stack: 'holes',
        data: topSectors.map((row) => row[metric] || 0),
      }})),
    }});

    const shareAnomalyChart = echarts.init(document.getElementById('share-anomaly-chart'));
    shareAnomalyChart.setOption({{
      animation: false,
      tooltip: {{ trigger: 'axis', axisPointer: {{ type: 'shadow' }} }},
      grid: {{ left: 90, right: 20, top: 30, bottom: 40 }},
      xAxis: {{ type: 'value', name: 'Nombre de sauts SEC' }},
      yAxis: {{ type: 'category', data: payload.share_anomalies.map((row) => row.ticker.replace('.US','')) }},
      series: [
        {{
          type: 'bar',
          data: payload.share_anomalies.map((row) => row.candidate_count),
          itemStyle: {{ color: '#356e63' }},
          label: {{ show: true, position: 'right' }}
        }}
      ]
    }});

    function updateHoleTables() {{
      const tickerFilter = document.getElementById('ticker-filter').value.toLowerCase();
      const metricFilter = document.getElementById('metric-filter').value;
      const holeFilter = Number(document.getElementById('hole-filter').value || '1');

      const tickerRows = payload.ticker_metric_holes.filter((row) => {{
        const haystack = `${{row.ticker}} ${{row.sector || ''}} ${{row.industry || ''}}`.toLowerCase();
        const metricOk = !metricFilter || row.metric === metricFilter;
        return metricOk && Number(row.hole_count || 0) >= holeFilter && haystack.includes(tickerFilter);
      }});
      renderTable(
        'ticker-hole-table',
        tickerRows.slice(0, 400).map((row) => ({{
          ticker: row.ticker,
          kpi: row.metric_label,
          trous: row.hole_count,
          pct_trous: `${{Number(row.hole_pct || 0).toFixed(1)}}%`,
          trimestres_attendus: row.expected_quarters,
          trimestres_presents: row.present_quarters,
          premier_trimestre: row.first_quarter,
          dernier_trimestre: row.last_quarter,
          dates_manquantes: row.sample_missing_dates,
        }})),
        ['ticker', 'kpi', 'trous', 'pct_trous', 'trimestres_attendus', 'trimestres_presents', 'premier_trimestre', 'dernier_trimestre', 'dates_manquantes'],
        {{
          ticker: 'Ticker',
          kpi: 'KPI',
          trous: 'Trous',
          pct_trous: '% de trous',
          trimestres_attendus: 'Trimestres attendus',
          trimestres_presents: 'Trimestres presents',
          premier_trimestre: 'Premier trimestre',
          dernier_trimestre: 'Dernier trimestre',
          dates_manquantes: 'Exemples de dates manquantes',
        }},
      );

      const quarterRows = payload.quarterly_holes.filter((row) => {{
        const haystack = `${{row.ticker}} ${{row.sector || ''}} ${{row.industry || ''}}`.toLowerCase();
        const metricOk = !metricFilter || row.metric === metricFilter;
        return metricOk && haystack.includes(tickerFilter);
      }});
      renderTable(
        'quarter-hole-table',
        quarterRows.slice(0, 1200).map((row) => ({{
          ticker: row.ticker,
          kpi: row.metric_label,
          trimestre: row.date,
          secteur: row.sector,
          industrie: row.industry,
        }})),
        ['ticker', 'kpi', 'trimestre', 'secteur', 'industrie'],
        {{
          ticker: 'Ticker',
          kpi: 'KPI',
          trimestre: 'Trimestre manquant',
          secteur: 'Secteur',
          industrie: 'Industrie',
        }},
      );
    }}

    document.getElementById('ticker-filter').addEventListener('input', updateHoleTables);
    document.getElementById('metric-filter').addEventListener('change', updateHoleTables);
    document.getElementById('hole-filter').addEventListener('change', updateHoleTables);
    updateHoleTables();

    window.addEventListener('resize', () => {{
      holeTickerChart.resize();
      kpiHolesChart.resize();
      tickerChart.resize();
      sectorChart.resize();
      shareAnomalyChart.resize();
    }});
  </script>
</body>
</html>
"""


def _metric_cards_html(overview: pl.DataFrame) -> str:
    cards: list[str] = []
    for row in overview.sort("metric").to_dicts():
        headline = _format_int(int(row["hole_count"] or 0))
        headline_label = "trimestres manquants"
        touched_phrase = (
            f"{_format_int(int(row['tickers_with_holes'] or 0))} tickers sur "
            f"{_format_int(int(row['total_tickers'] or 0))} ont au moins un trou sur ce KPI."
        )
        zero_phrase = (
            f"{_format_int(int(row['zero_coverage_tickers'] or 0))} tickers sont totalement vides "
            "pour ce KPI."
        )
        period_phrase = f"Periode observee: {row['first_date']} -> {row['last_date']}."
        cards.append(
            "<div class='metric-card'>"
            f"<div class='metric-title'>{escape(str(row['metric_label']))}</div>"
            f"<div class='big'>{headline}</div>"
            f"<div class='line'><strong>{headline_label}</strong></div>"
            f"<div class='line'>{escape(touched_phrase)}</div>"
            f"<div class='line'>{escape(zero_phrase)}</div>"
            f"<div class='line muted'>{escape(period_phrase)}</div>"
            "</div>"
        )
    return "".join(cards)


def _zero_coverage_html(missing: pl.DataFrame) -> str:
    if missing.is_empty():
        return "<span class='pill'>Aucun ticker totalement vide</span>"
    rows = (
        missing.group_by(["metric", "metric_label"])
        .agg(pl.len().alias("count"))
        .sort("metric")
        .to_dicts()
    )
    return "".join(
        f"<span class='pill'>{escape(str(row['metric_label']))}: {row['count']} tickers totalement vides</span>"
        for row in rows
    )


def _render_ticker_page(
    *,
    ticker: str,
    shares: pl.DataFrame,
    financials: pl.DataFrame,
    earnings: pl.DataFrame,
    ticker_metric_holes: pl.DataFrame,
    quarterly_holes: pl.DataFrame,
    share_candidates: pl.DataFrame,
) -> str:
    revenue_rows = (
        financials.filter(pl.col("metric") == "revenue")
        .select(["date", pl.col("value").alias("revenue")])
        .sort("date")
        .to_dicts()
    )
    net_income_rows = (
        financials.filter(pl.col("metric") == "net_income")
        .select(["date", pl.col("value").alias("net_income")])
        .sort("date")
        .to_dicts()
    )
    share_rows = (
        shares.select([pl.col("dateFormatted").alias("date"), pl.col("shares").cast(pl.Float64, strict=False).alias("shares")])
        .sort("date")
        .to_dicts()
    )
    earnings_rows = (
        earnings.select([pl.col("period_end").alias("date"), pl.col("epsActual").cast(pl.Float64, strict=False).alias("eps_actual")])
        .sort("date")
        .to_dicts()
    )
    payload = {
        "revenue": revenue_rows,
        "net_income": net_income_rows,
        "shares": share_rows,
        "earnings": earnings_rows,
    }
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>{escape(ticker)} - Audit SEC</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #12202f; background: #f4f6f8; }}
    .wrap {{ padding: 28px 32px 40px 32px; }}
    .section {{ background: white; border-radius: 18px; padding: 20px 22px; box-shadow: 0 10px 30px rgba(10,20,30,0.06); margin-top: 18px; }}
    .chart-box {{ width: 100%; height: 360px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #edf1f5; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; position: sticky; top: 0; }}
    .table-wrap {{ max-height: 520px; overflow: auto; border: 1px solid #e5ebf1; border-radius: 14px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>{escape(ticker)}</h1>
    <p><a href="../report.html">Retour au rapport d'audit</a></p>

    <div class="section">
      <h2>Courbes SEC</h2>
      <div id="income-chart" class="chart-box"></div>
      <div id="shares-chart" class="chart-box"></div>
      <div id="eps-chart" class="chart-box"></div>
    </div>

    <div class="section">
      <h2>Resume des trous</h2>
      <div class="table-wrap">{_table_html(_rename_columns(ticker_metric_holes, {'metric_label': 'kpi', 'hole_count': 'trous', 'hole_pct': 'pct_trous', 'expected_quarters': 'trimestres_attendus', 'present_quarters': 'trimestres_presents', 'first_quarter': 'premier_trimestre', 'last_quarter': 'dernier_trimestre', 'sample_missing_dates': 'dates_manquantes'}))}</div>
    </div>

    <div class="section">
      <h2>Trimestres manquants</h2>
      <div class="table-wrap">{_table_html(_rename_columns(quarterly_holes, {'metric_label': 'kpi', 'date': 'trimestre'}).head(200))}</div>
    </div>

    <div class="section">
      <h2>Anomalies sur les actions en circulation</h2>
      <div class="table-wrap">{_table_html(share_candidates.head(100))}</div>
    </div>
  </div>
  <script>
    const payload = {json.dumps(payload)};

    function lineChart(id, title, rows, series) {{
      const chart = echarts.init(document.getElementById(id));
      chart.setOption({{
        animation: false,
        title: {{ text: title, left: 0, top: 0, textStyle: {{ fontSize: 18, fontWeight: 600 }} }},
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 26 }},
        grid: {{ left: 60, right: 20, top: 60, bottom: 40 }},
        xAxis: {{ type: 'category', data: rows.map((row) => row.date) }},
        yAxis: {{ type: 'value', scale: true }},
        series,
      }});
      window.addEventListener('resize', () => chart.resize());
    }}

    const incomeRows = payload.revenue.map((row, index) => ({{
      date: row.date,
      revenue: row.revenue,
      net_income: payload.net_income[index] ? payload.net_income[index].net_income : null,
    }}));
    lineChart('income-chart', 'Chiffre d\\'affaires et resultat net', incomeRows, [
      {{ name: 'Chiffre d\\'affaires', type: 'bar', data: incomeRows.map((row) => row.revenue), itemStyle: {{ color: '#c7934a' }} }},
      {{ name: 'Resultat net', type: 'line', data: incomeRows.map((row) => row.net_income), smooth: true, itemStyle: {{ color: '#b85c38' }} }},
    ]);
    lineChart('shares-chart', 'Actions en circulation', payload.shares, [
      {{ name: 'Actions en circulation', type: 'line', data: payload.shares.map((row) => row.shares), smooth: true, itemStyle: {{ color: '#356e63' }} }},
    ]);
    lineChart('eps-chart', 'EPS publie', payload.earnings, [
      {{ name: 'EPS publie', type: 'bar', data: payload.earnings.map((row) => row.eps_actual), itemStyle: {{ color: '#6a5ea8' }} }},
    ]);
  </script>
</body>
</html>
"""


def _render_dashboard_markdown(
    *,
    overview: pl.DataFrame,
    kpi_hole_summary: pl.DataFrame,
    ticker_gap_summary: pl.DataFrame,
    missing: pl.DataFrame,
) -> str:
    lines = [
        "# Audit SEC des fondamentaux",
        "",
        f"Genere le: {_utc_now_iso()}",
        "",
        "Ce rapport est 100% SEC. Il n'utilise ni EODHD, ni Yahoo, ni SimFin.",
        "",
        "## Resume par KPI",
        "",
    ]
    for row in overview.sort("metric").to_dicts():
        lines.append(
            f"- **{row['metric_label']}**: {row['hole_count']} trimestres manquants au total. "
            f"{row['tickers_with_holes']} tickers sur {row['total_tickers']} ont au moins un trou. "
            f"{row['zero_coverage_tickers']} tickers sont totalement vides."
        )
    lines.extend(["", "## Tickers les plus problematiques", ""])
    for row in ticker_gap_summary.head(20).to_dicts():
        if int(row["total_holes"] or 0) <= 0:
            continue
        lines.append(
            f"- **{row['ticker']}**: {row['total_holes']} trous au total. "
            f"KPI le plus touche: {row['worst_metric_label']} ({row['worst_metric_holes']} trous)."
        )
    lines.extend(["", "## Tickers totalement vides par KPI", ""])
    for metric in AUDIT_METRICS:
        metric_label = METRIC_LABELS[metric]
        subset = missing.filter(pl.col("metric") == metric)
        sample = ", ".join(subset.get_column("ticker").head(12).to_list()) if not subset.is_empty() else "aucun"
        lines.append(f"- **{metric_label}**: {subset.height} tickers totalement vides. Exemple: {sample}")
    lines.append("")
    return "\n".join(lines)


def _rename_columns(frame: pl.DataFrame, mapping: dict[str, str]) -> pl.DataFrame:
    existing = {key: value for key, value in mapping.items() if key in frame.columns}
    return frame.rename(existing) if existing else frame


def _table_html(frame: pl.DataFrame) -> str:
    if frame.is_empty():
        return "<p class='muted'>Aucune ligne.</p>"
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


def _format_int(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


if __name__ == "__main__":
    main()
