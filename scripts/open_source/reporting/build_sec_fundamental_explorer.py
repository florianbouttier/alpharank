#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import polars as pl

ONE_METRIC_KEY = "__one__"
INDEX_METRIC_ORDER = [
    "totalRevenue",
    "grossProfit",
    "operatingIncome",
    "netIncome",
    "shares",
    "epsActual",
]
METRIC_LABELS = {
    "totalRevenue": "Chiffre d'affaires",
    "grossProfit": "Marge brute",
    "operatingIncome": "Résultat opérationnel",
    "netIncome": "Résultat net",
    "shares": "Actions en circulation",
    "epsActual": "EPS publié",
    ONE_METRIC_KEY: "1 (valeur brute)",
}


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[3]
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

    plot_payload = _build_index_plot_payload(
        income=income,
        shares=shares,
        earnings=earnings,
    )
    ticker_rows = _build_ticker_summary(
        general=general,
        income=income,
        shares=shares,
        earnings=earnings,
    )
    ticker_rows.write_parquet(output_dir / "ticker_summary.parquet")
    ticker_rows.write_csv(output_dir / "ticker_summary.csv")
    share_anomalies.write_parquet(output_dir / "share_anomalies.parquet")
    price_anomalies.write_parquet(output_dir / "price_anomalies.parquet")
    (output_dir / "plot_payload.json").write_text(
        json.dumps(plot_payload, separators=(",", ":")),
        encoding="utf-8",
    )

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
        "ui_policy": {
            "default_view": "all_available_history",
            "chart_stack": "echarts",
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(
        _render_index_page(ticker_rows=ticker_rows, manifest=manifest, plot_payload=plot_payload),
        encoding="utf-8",
    )
    print(output_dir)
    print(output_dir / "index.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build user-friendly SEC fundamentals explorer by ticker.")
    project_root = Path(__file__).resolve().parents[3]
    parser.add_argument("--sec-output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument("--price-source-dir", type=Path, default=project_root / "data" / "eodhd" / "output")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--share-ratio-threshold", type=float, default=1.5)
    parser.add_argument("--price-factor-ratio-threshold", type=float, default=1.2)
    parser.add_argument("--max-price-years", type=int, default=50)
    parser.add_argument("--max-quarters", type=int, default=200)
    return parser.parse_args()


def _build_ticker_summary(
    *,
    general: pl.DataFrame,
    income: pl.DataFrame,
    shares: pl.DataFrame,
    earnings: pl.DataFrame,
) -> pl.DataFrame:
    quarter_matrix = _build_quarter_metric_matrix(
        income=income,
        shares=shares,
        earnings=earnings,
    )
    metric_keys = [metric for metric in INDEX_METRIC_ORDER if metric in quarter_matrix.columns]
    coverage_rows: list[dict[str, object]] = []
    for ticker, frame in quarter_matrix.group_by("ticker", maintain_order=True):
        ticker_key = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        series_rows = frame.to_dicts()
        total_quarters = frame.height
        coverage_row: dict[str, object] = {"ticker": ticker_key, "observed_quarters": total_quarters}
        for metric in metric_keys:
            present = sum(1 for row in series_rows if row.get(metric) is not None)
            missing = max(total_quarters - present, 0)
            pct = (present / total_quarters * 100.0) if total_quarters else None
            coverage_row[f"{metric}_present"] = present
            coverage_row[f"{metric}_missing"] = missing
            coverage_row[f"{metric}_fill_pct"] = pct
        available_pcts = [coverage_row[f"{metric}_fill_pct"] for metric in metric_keys if coverage_row[f"{metric}_fill_pct"] is not None]
        coverage_row["global_fill_pct"] = sum(available_pcts) / len(available_pcts) if available_pcts else None
        coverage_rows.append(coverage_row)

    coverage = pl.DataFrame(coverage_rows) if coverage_rows else pl.DataFrame(schema={"ticker": pl.String})
    return (
        general.select(
            [
                "ticker",
                "Code",
                "Name",
            ]
        )
        .join(coverage, on="ticker", how="left")
        .sort("Code")
    )


def _build_quarter_metric_matrix(
    *,
    income: pl.DataFrame,
    shares: pl.DataFrame,
    earnings: pl.DataFrame,
) -> pl.DataFrame:
    income_metrics = [
        metric
        for metric in INDEX_METRIC_ORDER
        if metric in income.columns and metric not in {"shares", "epsActual"}
    ]
    income_q = (
        income.select(
            [
                "ticker",
                pl.col("date").str.strptime(pl.Date, strict=False).alias("_dt"),
                *[pl.col(metric).cast(pl.Float64, strict=False).alias(metric) for metric in income_metrics],
            ]
        )
        .filter(pl.col("_dt").is_not_null())
        .with_columns(
            [
                pl.col("_dt").dt.year().alias("_year"),
                _calendar_period_expr(pl.col("_dt")).alias("_period"),
            ]
        )
        .group_by(["ticker", "_year", "_period"])
        .agg(
            [
                pl.col("_dt").max().alias("_quarter_dt"),
                *[pl.col(metric).drop_nulls().last().alias(metric) for metric in income_metrics],
            ]
        )
    )
    shares_q = (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").str.strptime(pl.Date, strict=False).alias("_dt"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .filter(pl.col("_dt").is_not_null())
        .with_columns(
            [
                pl.col("_dt").dt.year().alias("_year"),
                _calendar_period_expr(pl.col("_dt")).alias("_period"),
            ]
        )
        .group_by(["ticker", "_year", "_period"])
        .agg(
            [
                pl.col("_dt").max().alias("_shares_dt"),
                pl.col("shares").drop_nulls().last().alias("shares"),
            ]
        )
    )
    earnings_q = (
        earnings.select(
            [
                "ticker",
                pl.col("date").str.strptime(pl.Date, strict=False).alias("_dt"),
                pl.col("epsActual").cast(pl.Float64, strict=False).alias("epsActual"),
            ]
        )
        .filter(pl.col("_dt").is_not_null())
        .with_columns(
            [
                pl.col("_dt").dt.year().alias("_year"),
                _calendar_period_expr(pl.col("_dt")).alias("_period"),
            ]
        )
        .group_by(["ticker", "_year", "_period"])
        .agg(
            [
                pl.col("_dt").max().alias("_earnings_dt"),
                pl.col("epsActual").drop_nulls().last().alias("epsActual"),
            ]
        )
    )

    merged = income_q.join(shares_q, on=["ticker", "_year", "_period"], how="full", coalesce=True)
    merged = merged.join(earnings_q, on=["ticker", "_year", "_period"], how="full", coalesce=True)
    merged = merged.with_columns(
        pl.coalesce(["_quarter_dt", "_shares_dt", "_earnings_dt"]).alias("_plot_dt")
    )
    metric_cols = [metric for metric in INDEX_METRIC_ORDER if metric in merged.columns]
    return (
        merged.filter(pl.any_horizontal([pl.col(metric).is_not_null() for metric in metric_cols]))
        .sort(["ticker", "_year", "_period"])
        .select(["ticker", "_plot_dt", "_year", "_period", *metric_cols])
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


def _render_index_page(
    *,
    ticker_rows: pl.DataFrame,
    manifest: dict[str, object],
    plot_payload: dict[str, object],
) -> str:
    metric_keys = [metric for metric, _label in plot_payload["metric_options"]]
    ticker_options = "".join(
        (
            f"<option value='{escape(str(row['ticker']))}'>"
            f"{escape(str(row.get('Code') or row['ticker']))} · {escape(str(row.get('Name') or ''))}"
            f"</option>"
        )
        for row in ticker_rows.select(["ticker", "Code", "Name"]).to_dicts()
    )
    metric_options = "".join(
        f"<option value='{escape(metric_key)}'>{escape(metric_label)}</option>"
        for metric_key, metric_label in plot_payload["metric_options"]
    )
    heatmap_headers = "".join(
        f"<th>{escape(METRIC_LABELS.get(metric, metric))}</th>"
        for metric in metric_keys
    )
    rows_html = "".join(
        _ticker_row_html(row, metric_keys=metric_keys)
        for row in ticker_rows.to_dicts()
    )
    plot_payload_json = json.dumps(plot_payload, separators=(",", ":"))
    return f"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>SEC Fundamental Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ font-family: Georgia, 'Times New Roman', serif; margin: 0; color: #1c1e21; background: linear-gradient(180deg, #f1efe8 0%, #fcfbf8 40%, #ffffff 100%); }}
    .hero {{ padding: 36px 44px 24px 44px; border-bottom: 1px solid rgba(0,0,0,0.08); background: radial-gradient(circle at top right, rgba(182,145,89,0.18), transparent 32%), linear-gradient(135deg, #f8f3e8, #ffffff); }}
    .hero h1 {{ margin: 0; font-size: 42px; letter-spacing: -0.03em; }}
    .hero p {{ max-width: 840px; color: #5d5f63; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .toolbar {{ display: flex; gap: 12px; padding: 18px 44px; align-items: center; position: sticky; top: 0; background: rgba(252,251,248,0.9); backdrop-filter: blur(10px); border-bottom: 1px solid rgba(0,0,0,0.06); z-index: 5; }}
    .toolbar input {{ width: 320px; padding: 11px 14px; border: 1px solid #d9d3c5; border-radius: 12px; font-size: 14px; }}
    .toolbar .meta {{ color: #6b6e73; font-size: 13px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .plotter-wrap {{ padding: 28px 44px 8px 44px; }}
    .plotter {{ background: white; border-radius: 20px; padding: 22px 22px 18px 22px; box-shadow: 0 14px 36px rgba(0,0,0,0.06); }}
    .plotter-head {{ display: flex; gap: 18px; align-items: baseline; justify-content: space-between; flex-wrap: wrap; }}
    .plotter-head h2 {{ margin: 0; font-size: 28px; letter-spacing: -0.03em; }}
    .plotter-head p {{ margin: 6px 0 0 0; color: #666a70; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .plotter-controls {{ display: grid; grid-template-columns: 1.4fr 1fr 1fr 0.9fr; gap: 12px; margin: 18px 0 10px 0; }}
    .field {{ display: flex; flex-direction: column; gap: 6px; }}
    .field label {{ font-size: 12px; letter-spacing: 0.05em; text-transform: uppercase; color: #7a6e61; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .field select {{ padding: 11px 12px; border: 1px solid #d9d3c5; border-radius: 12px; background: #fff; font-size: 14px; }}
    .plotter-meta {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .formula {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: #f6efe3; color: #7b4f24; font-size: 13px; font-weight: 600; }}
    .hint {{ color: #71757b; font-size: 13px; }}
    .toggle {{ display: inline-flex; align-items: center; gap: 8px; padding: 8px 12px; border-radius: 999px; background: #faf7f1; border: 1px solid #e3d7c5; }}
    .chart-box {{ width: 100%; height: 470px; }}
    .table-wrap {{ padding: 20px 44px 40px 44px; }}
    .table-head {{ display: flex; align-items: end; justify-content: space-between; gap: 18px; margin-bottom: 14px; }}
    .table-head h2 {{ margin: 0; font-size: 28px; letter-spacing: -0.03em; }}
    .table-head p {{ margin: 6px 0 0 0; color: #666a70; font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 860px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 16px; overflow: hidden; box-shadow: 0 12px 34px rgba(0,0,0,0.06); }}
    th, td {{ padding: 14px 12px; border-bottom: 1px solid #efede7; text-align: left; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }}
    th {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; color: #6e6558; background: #faf7f1; }}
    tr:hover {{ background: #fcfbf7; }}
    a {{ color: #7b4f24; text-decoration: none; font-weight: 600; }}
    .pill {{ display: inline-block; padding: 3px 8px; border-radius: 999px; background: #f0e2c7; color: #7b4f24; font-size: 11px; font-weight: 700; }}
    .muted {{ color: #8a8d93; }}
    .heat-cell {{ min-width: 118px; }}
    .overall-col {{ min-width: 108px; }}
    .heat-box {{ border-radius: 12px; padding: 10px 10px 9px 10px; }}
    .heat-main {{ font-size: 18px; font-weight: 700; line-height: 1; }}
    .heat-sub {{ margin-top: 6px; font-size: 11px; color: #655c53; }}
    @media (max-width: 1100px) {{
      .plotter-controls {{ grid-template-columns: 1fr 1fr; }}
    }}
    @media (max-width: 760px) {{
      .plotter-wrap, .table-wrap, .toolbar, .hero {{ padding-left: 20px; padding-right: 20px; }}
      .toolbar {{ flex-direction: column; align-items: stretch; }}
      .toolbar input {{ width: auto; }}
      .plotter-controls {{ grid-template-columns: 1fr; }}
      .chart-box {{ height: 420px; }}
    }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>SEC Fundamental Explorer</h1>
    <p>Explorer par ticker, quarter par quarter, les fondamentaux SEC utiles. Tu peux maintenant choisir un indicateur brut ou construire une formule <strong>numérateur / dénominateur</strong> pour visualiser une marge, un rendement, un indicateur par action ou n'importe quel ratio simple.</p>
  </div>
  <div class="toolbar">
    <input id="search" type="search" placeholder="Rechercher ticker, société ou secteur">
    <div class="meta">Prix source: <code>{escape(str(manifest['price_source_dir']))}</code></div>
  </div>
  <div class="plotter-wrap">
    <div class="plotter">
      <div class="plotter-head">
        <div>
          <h2>Ploteur d'indicateurs</h2>
          <p>Choisis un ticker, un numérateur et un dénominateur. Si tu mets <strong>1</strong> au dénominateur, tu affiches simplement la série brute. Si tu coches <strong>%</strong>, le ratio est multiplié par 100 pour lire directement une marge.</p>
        </div>
      </div>
      <div class="plotter-controls">
        <div class="field">
          <label for="ticker-select">Ticker</label>
          <select id="ticker-select">{ticker_options}</select>
        </div>
        <div class="field">
          <label for="numerator-select">Numérateur</label>
          <select id="numerator-select">{metric_options}</select>
        </div>
        <div class="field">
          <label for="denominator-select">Dénominateur</label>
          <select id="denominator-select"><option value="{ONE_METRIC_KEY}">1 (valeur brute)</option>{metric_options}</select>
        </div>
        <div class="field">
          <label for="value-mode">Lecture</label>
          <select id="value-mode">
            <option value="auto">Auto</option>
            <option value="percent">En %</option>
            <option value="multiple">Ratio simple</option>
          </select>
        </div>
      </div>
      <div class="plotter-meta">
        <div class="formula" id="formula-chip">Chiffre d'affaires / 1</div>
        <label class="toggle"><input type="checkbox" id="show-points"> Afficher les points</label>
        <div class="hint" id="plot-hint">Série trimestrielle SEC.</div>
      </div>
      <div id="index-plot-chart" class="chart-box"></div>
    </div>
  </div>
  <div class="table-wrap">
    <div class="table-head">
      <div>
        <h2>Couverture par ticker</h2>
        <p>Chaque case montre le pourcentage de trimestres observés qui ont une vraie valeur SEC pour le KPI. Le sous-texte indique le nombre de trimestres manquants. Donc si Alcoa a des trous sur le chiffre d'affaires, le résultat net, les actions en circulation ou l'EPS, tu le vois immédiatement ici.</p>
      </div>
    </div>
    <table id="ticker-table">
      <thead>
        <tr>
          <th>Ticker</th>
          <th>Name</th>
          <th>Quarters</th>
          <th class="overall-col">Global</th>
          {heatmap_headers}
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
  <script>
    const plotPayload = {plot_payload_json};
    const search = document.getElementById('search');
    const rows = Array.from(document.querySelectorAll('#ticker-table tbody tr'));
    const tickerSelect = document.getElementById('ticker-select');
    const numeratorSelect = document.getElementById('numerator-select');
    const denominatorSelect = document.getElementById('denominator-select');
    const valueModeSelect = document.getElementById('value-mode');
    const showPointsCheckbox = document.getElementById('show-points');
    const formulaChip = document.getElementById('formula-chip');
    const plotHint = document.getElementById('plot-hint');
    const chart = echarts.init(document.getElementById('index-plot-chart'));

    const metricLabels = Object.fromEntries(plotPayload.metric_options);

    search.addEventListener('input', () => {{
      const q = search.value.toLowerCase().trim();
      rows.forEach((row) => {{
        const hay = row.dataset.search;
        row.style.display = hay.includes(q) ? '' : 'none';
      }});
    }});

    function getSeriesRows(ticker) {{
      return plotPayload.ticker_series[ticker] || [];
    }}

    function resolveMode(denominatorKey, modeKey) {{
      if (modeKey !== 'auto') return modeKey;
      return denominatorKey === '{ONE_METRIC_KEY}' ? 'raw' : 'percent';
    }}

    function buildComputedSeries(rows, numeratorKey, denominatorKey, modeKey) {{
      const resolvedMode = resolveMode(denominatorKey, modeKey);
      return rows.map((row) => {{
        const numerator = row[numeratorKey];
        const denominator = denominatorKey === '{ONE_METRIC_KEY}' ? 1 : row[denominatorKey];
        if (numerator == null || denominator == null || denominator === 0) {{
          return null;
        }}
        const raw = numerator / denominator;
        return resolvedMode === 'percent' ? raw * 100 : raw;
      }});
    }}

    function formatValue(value, denominatorKey, modeKey) {{
      if (value == null || Number.isNaN(value)) return 'NA';
      const resolvedMode = resolveMode(denominatorKey, modeKey);
      if (resolvedMode === 'percent') return value.toFixed(2) + ' %';
      if (denominatorKey === '{ONE_METRIC_KEY}') {{
        const absValue = Math.abs(value);
        if (absValue >= 1_000_000_000) return (value / 1_000_000_000).toFixed(2) + 'B';
        if (absValue >= 1_000_000) return (value / 1_000_000).toFixed(2) + 'M';
        return value.toLocaleString('en-US', {{ maximumFractionDigits: 2 }});
      }}
      return value.toFixed(4);
    }}

    function renderIndexPlot() {{
      const ticker = tickerSelect.value;
      const numeratorKey = numeratorSelect.value;
      const denominatorKey = denominatorSelect.value;
      const modeKey = valueModeSelect.value;
      const rows = getSeriesRows(ticker);
      const values = buildComputedSeries(rows, numeratorKey, denominatorKey, modeKey);
      const numeratorLabel = metricLabels[numeratorKey] || numeratorKey;
      const denominatorLabel = denominatorKey === '{ONE_METRIC_KEY}' ? '1' : (metricLabels[denominatorKey] || denominatorKey);
      const resolvedMode = resolveMode(denominatorKey, modeKey);
      const seriesName = denominatorKey === '{ONE_METRIC_KEY}'
        ? numeratorLabel
        : `${{numeratorLabel}} / ${{denominatorLabel}}`;

      formulaChip.textContent = `${{numeratorLabel}} / ${{denominatorLabel}}`;
      plotHint.textContent = denominatorKey === '{ONE_METRIC_KEY}'
        ? `Affichage brut de ${{numeratorLabel.toLowerCase()}} sur toute l'historique disponible.`
        : `Ratio trimestriel calculé sur la base SEC. Lecture: ${{resolvedMode === 'percent' ? 'pourcentage' : 'ratio simple'}}.`;

      chart.setOption({{
        animation: false,
        tooltip: {{
          trigger: 'axis',
          valueFormatter: (value) => formatValue(value, denominatorKey, modeKey),
        }},
        legend: {{
          top: 0,
          data: [seriesName],
        }},
        grid: {{ left: 70, right: 30, top: 50, bottom: 56 }},
        xAxis: {{
          type: 'category',
          data: rows.map((row) => row.date),
        }},
        yAxis: {{
          type: 'value',
          scale: true,
          axisLabel: {{
            formatter: (value) => formatValue(value, denominatorKey, modeKey),
          }},
        }},
        dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 18, bottom: 10 }}],
        series: [
          {{
            name: seriesName,
            type: 'line',
            smooth: true,
            showSymbol: showPointsCheckbox.checked,
            symbolSize: 7,
            connectNulls: false,
            data: values,
            lineStyle: {{ width: 3, color: '#1f4762' }},
            itemStyle: {{ color: '#b86a35' }},
            areaStyle: denominatorKey === '{ONE_METRIC_KEY}'
              ? {{ color: 'rgba(184,106,53,0.10)' }}
              : {{ color: 'rgba(31,71,98,0.08)' }},
          }},
        ],
      }});
    }}

    [tickerSelect, numeratorSelect, denominatorSelect, valueModeSelect, showPointsCheckbox].forEach((node) => {{
      node.addEventListener('change', renderIndexPlot);
    }});

    numeratorSelect.value = plotPayload.default_numerator;
    denominatorSelect.value = '{ONE_METRIC_KEY}';
    renderIndexPlot();
    window.addEventListener('resize', () => chart.resize());
  </script>
</body>
</html>
"""


def _build_index_plot_payload(*, income: pl.DataFrame, shares: pl.DataFrame, earnings: pl.DataFrame) -> dict[str, object]:
    income_metrics = [
        metric
        for metric in INDEX_METRIC_ORDER
        if metric in income.columns and metric not in {"shares", "epsActual"}
    ]
    income_selected = (
        income.select(
            [
                "ticker",
                "date",
                *[
                    pl.col(metric).cast(pl.Float64, strict=False).alias(metric)
                    for metric in income_metrics
                ],
            ]
        )
        .sort(["ticker", "date"])
    )
    shares_selected = (
        shares.select(
            [
                "ticker",
                pl.col("dateFormatted").alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .sort(["ticker", "date"])
    )
    earnings_selected = (
        earnings.select(
            [
                "ticker",
                "date",
                pl.col("epsActual").cast(pl.Float64, strict=False).alias("epsActual"),
            ]
        )
        .sort(["ticker", "date"])
    )
    merged = income_selected.join(shares_selected, on=["ticker", "date"], how="full", coalesce=True)
    merged = merged.join(earnings_selected, on=["ticker", "date"], how="full", coalesce=True)
    available_metrics = [
        metric
        for metric in INDEX_METRIC_ORDER
        if metric in merged.columns and merged.select(pl.col(metric).is_not_null().any()).item()
    ]
    ticker_series: dict[str, list[dict[str, object]]] = {}
    for ticker, frame in merged.group_by("ticker", maintain_order=True):
        ticker_key = str(ticker[0] if isinstance(ticker, tuple) else ticker)
        ticker_series[ticker_key] = (
            frame.sort("date")
            .filter(pl.any_horizontal([pl.col(metric).is_not_null() for metric in available_metrics]))
            .select(["date", *available_metrics])
            .to_dicts()
        )
    return {
        "metric_options": [
            [metric, METRIC_LABELS.get(metric, metric)]
            for metric in available_metrics
        ],
        "default_numerator": "totalRevenue" if "totalRevenue" in available_metrics else available_metrics[0],
        "ticker_series": ticker_series,
    }


def _calendar_period_expr(expr: pl.Expr) -> pl.Expr:
    return (
        pl.when(expr.dt.month().is_in([1, 2, 3]))
        .then(pl.lit("Q1"))
        .when(expr.dt.month().is_in([4, 5, 6]))
        .then(pl.lit("Q2"))
        .when(expr.dt.month().is_in([7, 8, 9]))
        .then(pl.lit("Q3"))
        .otherwise(pl.lit("Q4"))
    )


def _heatmap_cell_html(
    *,
    fill_pct: object,
    missing: object,
    observed_quarters: object,
    overall: bool,
) -> str:
    numeric_pct = float(fill_pct) if fill_pct is not None else None
    if numeric_pct is None:
        background = "linear-gradient(135deg, #f2f2f2, #ececec)"
        main_text = "NA"
        sub_text = "pas de base"
    else:
        alpha = max(0.12, min(numeric_pct / 100.0, 1.0))
        background = (
            f"linear-gradient(135deg, rgba(40, 147, 90, {alpha:.3f}), "
            f"rgba(201, 93, 56, {max(0.08, 1.0 - numeric_pct / 100.0):.3f}))"
        )
        main_text = f"{numeric_pct:.0f}%"
        if overall:
            sub_text = f"{int(observed_quarters or 0)} trimestres"
        else:
            sub_text = f"{int(missing or 0)} trou{'s' if int(missing or 0) != 1 else ''}"
    cell_class = "overall-col" if overall else "heat-cell"
    return (
        f"<td class='{cell_class}'>"
        f"<div class='heat-box' style='background:{background}'>"
        f"<div class='heat-main'>{escape(main_text)}</div>"
        f"<div class='heat-sub'>{escape(sub_text)}</div>"
        f"</div>"
        f"</td>"
    )


def _ticker_row_html(row: dict[str, object], *, metric_keys: list[str]) -> str:
    search = " ".join(
        str(row.get(key) or "").lower()
        for key in ("Code", "ticker", "Name")
    )
    heat_cells = "".join(
        _heatmap_cell_html(
            fill_pct=row.get(f"{metric}_fill_pct"),
            missing=row.get(f"{metric}_missing"),
            observed_quarters=row.get("observed_quarters"),
            overall=False,
        )
        for metric in metric_keys
    )
    return (
        f"<tr data-search='{escape(search)}'>"
        f"<td><a href='tickers/{escape(str(row['ticker']))}.html'>{escape(str(row['Code']))}</a></td>"
        f"<td>{escape(str(row.get('Name') or ''))}</td>"
        f"<td>{escape(str(row.get('observed_quarters') or 0))}</td>"
        f"{_heatmap_cell_html(fill_pct=row.get('global_fill_pct'), missing=None, observed_quarters=row.get('observed_quarters'), overall=True)}"
        f"{heat_cells}"
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
    )
    shares_clean = (
        shares.select(
            [
                pl.col("dateFormatted").alias("date"),
                pl.col("shares").cast(pl.Float64, strict=False).alias("shares"),
            ]
        )
        .sort("date")
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
    )
    price_table = (
        prices.select(
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

    latest = _latest_snapshot(
        info=info,
        income=income_clean,
        shares=shares_clean,
        earnings=earnings_clean,
        prices=price_table,
        share_anomalies=share_anomalies,
        price_anomalies=price_anomalies,
    )
    chart_payload = json.dumps(
        {
            "price": price_table.to_dicts(),
            "income": income_clean.to_dicts(),
            "shares": shares_clean.to_dicts(),
            "earnings": earnings_clean.to_dicts(),
        }
    )
    return f"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(str(info.get('Code') or ticker))} Fundamental Explorer</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
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
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .chart-toolbar {{ display: flex; gap: 8px; flex-wrap: wrap; margin: 10px 0 14px 0; }}
    .chart-toolbar button {{ border: 1px solid #dccfb8; background: #faf7f1; border-radius: 999px; padding: 7px 11px; cursor: pointer; font-size: 12px; }}
    .chart-toolbar button.active {{ background: #7b4f24; border-color: #7b4f24; color: white; }}
    .chart-box {{ width: 100%; height: 420px; }}
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
      <div class="chart-toolbar" id="price-range">
        <button data-range="5y">5Y</button>
        <button data-range="10y">10Y</button>
        <button data-range="20y">20Y</button>
        <button data-range="max" class="active">Max</button>
      </div>
      <div class="chart-grid">
        <div id="price-chart" class="chart-box"></div>
        <div class="chart-toolbar" id="quarter-range">
          <button data-range="8q">8Q</button>
          <button data-range="16q">16Q</button>
          <button data-range="24q">24Q</button>
          <button data-range="max" class="active">Max</button>
        </div>
        <div id="fundamentals-chart" class="chart-box"></div>
        <div id="eps-shares-chart" class="chart-box"></div>
      </div>
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
  <script>
    const payload = {chart_payload};
    const priceChart = echarts.init(document.getElementById('price-chart'));
    const fundamentalsChart = echarts.init(document.getElementById('fundamentals-chart'));
    const epsSharesChart = echarts.init(document.getElementById('eps-shares-chart'));

    function sliceByYears(rows, years) {{
      if (!rows.length || years === 'max') return rows;
      const last = new Date(rows[rows.length - 1].date);
      const cutoff = new Date(last);
      cutoff.setFullYear(last.getFullYear() - years);
      return rows.filter((row) => new Date(row.date) >= cutoff);
    }}

    function sliceByQuarters(rows, quarters) {{
      if (!rows.length || quarters === 'max') return rows;
      return rows.slice(Math.max(0, rows.length - quarters));
    }}

    function renderPrice(rangeKey = 'max') {{
      const years = rangeKey === 'max' ? 'max' : parseInt(rangeKey, 10);
      const rows = sliceByYears(payload.price, years);
      priceChart.setOption({{
        animation: false,
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 0 }},
        grid: {{ left: 60, right: 30, top: 45, bottom: 50 }},
        xAxis: {{ type: 'category', data: rows.map((row) => row.date) }},
        yAxis: {{ type: 'value', scale: true }},
        dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 18, bottom: 10 }}],
        series: [
          {{ name: 'Close', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.close), lineStyle: {{ width: 2, color: '#305f86' }} }},
          {{ name: 'Adjusted', type: 'line', smooth: true, showSymbol: false, data: rows.map((row) => row.adjusted_close), lineStyle: {{ width: 2, color: '#b86a35' }} }},
        ],
      }});
    }}

    function renderQuarterly(rangeKey = 'max') {{
      const quarters = rangeKey === 'max' ? 'max' : parseInt(rangeKey, 10);
      const incomeRows = sliceByQuarters(payload.income, quarters);
      const earningsRows = sliceByQuarters(payload.earnings, quarters);
      const sharesRows = sliceByQuarters(payload.shares, quarters);
      fundamentalsChart.setOption({{
        animation: false,
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 0 }},
        grid: {{ left: 70, right: 70, top: 45, bottom: 50 }},
        xAxis: {{ type: 'category', data: incomeRows.map((row) => row.date) }},
        yAxis: [
          {{ type: 'value', name: 'Revenue', scale: true }},
          {{ type: 'value', name: 'Net income', scale: true }},
        ],
        dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 18, bottom: 10 }}],
        series: [
          {{ name: 'Revenue', type: 'bar', data: incomeRows.map((row) => row.revenue), itemStyle: {{ color: '#c8a05e' }} }},
          {{ name: 'Net income', type: 'line', yAxisIndex: 1, smooth: true, data: incomeRows.map((row) => row.net_income), lineStyle: {{ width: 3, color: '#1f4762' }} }},
        ],
      }});

      epsSharesChart.setOption({{
        animation: false,
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 0 }},
        grid: {{ left: 70, right: 90, top: 45, bottom: 50 }},
        xAxis: {{ type: 'category', data: sharesRows.length ? sharesRows.map((row) => row.date) : earningsRows.map((row) => row.date) }},
        yAxis: [
          {{ type: 'value', name: 'EPS', scale: true }},
          {{ type: 'value', name: 'Shares', scale: true }},
        ],
        dataZoom: [{{ type: 'inside' }}, {{ type: 'slider', height: 18, bottom: 10 }}],
        series: [
          {{ name: 'EPS actual', type: 'bar', data: earningsRows.map((row) => row.eps_actual), itemStyle: {{ color: '#8c403d' }} }},
          {{ name: 'Shares', type: 'line', yAxisIndex: 1, smooth: true, data: sharesRows.map((row) => row.shares), lineStyle: {{ width: 3, color: '#356e63' }} }},
        ],
      }});
    }}

    function bindToolbar(containerId, callback) {{
      const container = document.getElementById(containerId);
      const buttons = Array.from(container.querySelectorAll('button'));
      buttons.forEach((button) => {{
        button.addEventListener('click', () => {{
          buttons.forEach((item) => item.classList.remove('active'));
          button.classList.add('active');
          callback(button.dataset.range);
        }});
      }});
    }}

    bindToolbar('price-range', renderPrice);
    bindToolbar('quarter-range', renderQuarterly);
    renderPrice('max');
    renderQuarterly('max');
    window.addEventListener('resize', () => {{
      priceChart.resize();
      fundamentalsChart.resize();
      epsSharesChart.resize();
    }});
  </script>
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
    except (TypeError, ValueError):
        return str(value)


def _fmt_large_number(value: object) -> str:
    if value is None:
        return "NA"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    abs_value = abs(numeric)
    if abs_value >= 1_000_000_000:
        return f"{numeric / 1_000_000_000:,.2f}B"
    if abs_value >= 1_000_000:
        return f"{numeric / 1_000_000:,.2f}M"
    return f"{numeric:,.0f}"


if __name__ == "__main__":
    main()
