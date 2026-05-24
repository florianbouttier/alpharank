#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime
from html import escape
from pathlib import Path

import polars as pl


CORE_METRICS: tuple[str, ...] = ("revenue", "net_income", "epsActual")
METRIC_LABELS: dict[str, str] = {
    "revenue": "Chiffre d'affaires",
    "net_income": "Résultat net",
    "epsActual": "EPS publié",
}
METRIC_COLORS: dict[str, str] = {
    "revenue": "#b86a35",
    "net_income": "#1f4762",
    "epsActual": "#8c403d",
}


def main() -> None:
    args = _parse_args()
    project_root = Path(__file__).resolve().parents[2]
    sec_output_dir = args.sec_output_dir.resolve()
    quality_dir = args.quality_dir.resolve()
    output_dir = args.output_dir or (
        project_root / "outputs" / f"sec_core_kpi_yearly_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    quarterly_presence = pl.read_csv(quality_dir / "quarterly_presence.csv")
    full_year_end = args.end_year or (datetime.now().year - 1)
    filtered = quarterly_presence.filter(
        pl.col("metric").is_in(CORE_METRICS)
        & pl.col("fiscal_year").is_between(args.start_year, full_year_end, closed="both")
    )

    yearly_summary = (
        filtered.group_by(["metric", "metric_label", "fiscal_year"])
        .agg(
            [
                pl.len().alias("expected_quarters"),
                pl.col("present").sum().alias("present_quarters"),
                (~pl.col("present")).sum().alias("missing_quarters"),
                pl.col("ticker").n_unique().alias("ticker_count"),
            ]
        )
        .with_columns(
            [
                (pl.col("missing_quarters") / pl.col("expected_quarters") * 100.0).alias("missing_pct"),
                (pl.col("present_quarters") / pl.col("expected_quarters") * 100.0).alias("fill_pct"),
            ]
        )
        .sort(["metric", "fiscal_year"])
    )

    worst_years = (
        yearly_summary.sort(["metric", "missing_quarters"], descending=[False, True])
        .group_by("metric", maintain_order=True)
        .head(5)
        .sort(["metric", "missing_quarters"], descending=[False, True])
    )

    top_tickers = _build_top_tickers_by_worst_year(filtered=filtered, yearly_summary=yearly_summary)
    worst_year_brief = _build_worst_year_brief(yearly_summary=yearly_summary, top_tickers=top_tickers)
    recommendation_rows = _build_recommendations(top_tickers=top_tickers, yearly_summary=yearly_summary)

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "sec_output_dir": str(sec_output_dir),
        "quality_dir": str(quality_dir),
        "year_start": args.start_year,
        "year_end": full_year_end,
        "metric_order": list(CORE_METRICS),
        "metric_labels": METRIC_LABELS,
        "metric_colors": METRIC_COLORS,
        "yearly_summary": yearly_summary.to_dicts(),
        "worst_years": worst_years.to_dicts(),
        "worst_year_brief": worst_year_brief.to_dicts(),
        "top_tickers": top_tickers.to_dicts(),
        "recommendations": recommendation_rows,
    }

    yearly_summary.write_csv(output_dir / "yearly_summary.csv")
    yearly_summary.write_parquet(output_dir / "yearly_summary.parquet")
    worst_years.write_csv(output_dir / "worst_years.csv")
    worst_year_brief.write_csv(output_dir / "worst_year_brief.csv")
    top_tickers.write_csv(output_dir / "top_tickers_by_worst_year.csv")
    (output_dir / "payload.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (output_dir / "worst_year_brief.md").write_text(_render_worst_year_brief_markdown(worst_year_brief), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_markdown(payload), encoding="utf-8")
    (output_dir / "report.html").write_text(_render_html(payload), encoding="utf-8")
    print(output_dir)
    print(output_dir / "report.html")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build yearly missing report for SEC core KPIs.")
    project_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--sec-output-dir", type=Path, default=project_root / "data" / "sec" / "output")
    parser.add_argument(
        "--quality-dir",
        type=Path,
        default=project_root / "outputs" / "sec_quality_dashboard_latest",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=None)
    return parser.parse_args()


def _build_top_tickers_by_worst_year(*, filtered: pl.DataFrame, yearly_summary: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        metric_years = yearly_summary.filter(pl.col("metric") == metric).sort("missing_quarters", descending=True)
        if metric_years.is_empty():
            continue
        worst_year = int(metric_years.row(0, named=True)["fiscal_year"])
        top = (
            filtered.filter((pl.col("metric") == metric) & (pl.col("fiscal_year") == worst_year) & (~pl.col("present")))
            .group_by(["metric", "metric_label", "fiscal_year", "ticker", "ticker_code"])
            .agg(pl.len().alias("missing_quarters"))
            .sort(["missing_quarters", "ticker"], descending=[True, False])
            .head(20)
        )
        rows.extend(top.to_dicts())
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "metric": pl.String,
            "metric_label": pl.String,
            "fiscal_year": pl.Int64,
            "ticker": pl.String,
            "ticker_code": pl.String,
            "missing_quarters": pl.Int64,
        }
    )


def _build_worst_year_brief(*, yearly_summary: pl.DataFrame, top_tickers: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in CORE_METRICS:
        metric_years = yearly_summary.filter(pl.col("metric") == metric).sort(
            ["missing_quarters", "missing_pct"],
            descending=[True, True],
        )
        if metric_years.is_empty():
            continue
        worst = metric_years.row(0, named=True)
        fiscal_year = int(worst["fiscal_year"])
        top_codes = (
            top_tickers.filter((pl.col("metric") == metric) & (pl.col("fiscal_year") == fiscal_year))
            .get_column("ticker_code")
            .head(6)
            .to_list()
        )
        rows.append(
            {
                "metric": metric,
                "metric_label": METRIC_LABELS[metric],
                "fiscal_year": fiscal_year,
                "missing_quarters": int(worst["missing_quarters"]),
                "missing_pct": float(worst["missing_pct"]),
                "fill_pct": float(worst["fill_pct"]),
                "top_tickers": ", ".join(top_codes),
            }
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame(
        schema={
            "metric": pl.String,
            "metric_label": pl.String,
            "fiscal_year": pl.Int64,
            "missing_quarters": pl.Int64,
            "missing_pct": pl.Float64,
            "fill_pct": pl.Float64,
            "top_tickers": pl.String,
        }
    )


def _build_recommendations(*, top_tickers: pl.DataFrame, yearly_summary: pl.DataFrame) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for metric in CORE_METRICS:
        metric_label = METRIC_LABELS[metric]
        metric_years = yearly_summary.filter(pl.col("metric") == metric).sort("missing_quarters", descending=True)
        if metric_years.is_empty():
            continue
        worst = metric_years.row(0, named=True)
        worst_year = int(worst["fiscal_year"])
        pct = float(worst["missing_pct"])
        missing = int(worst["missing_quarters"])
        top_names = ", ".join(
            top_tickers.filter((pl.col("metric") == metric) & (pl.col("fiscal_year") == worst_year))
            .get_column("ticker_code")
            .head(6)
            .to_list()
        )
        if metric == "revenue":
            action = (
                "Priorité 1: filing-level historique sur les trous 2017, puis mapping métier des tags revenus "
                "(énergie, financières, telecom, holdings)."
            )
        elif metric == "net_income":
            action = (
                "Priorité 1: filing-level historique 2009-2010 et contrôle des quarters issus de fusions / changements de FY."
            )
        else:
            action = (
                "Priorité 1: fallback filing-level per-share, plus revue ciblée des quelques tickers récurrents."
            )
        rows.append(
            {
                "metric": metric,
                "title": f"{metric_label}: année la plus trouée = {worst_year}",
                "summary": (
                    f"{missing} trimestres manquants ({pct:.1f}%) sur l'année {worst_year}. "
                    f"Tickers les plus visibles: {top_names or 'aucun'}."
                ),
                "action": action,
            }
        )
    return rows


def _render_worst_year_brief_markdown(worst_year_brief: pl.DataFrame) -> str:
    lines = [
        "# Brief KPI trous SEC",
        "",
        "Ce fichier donne la synthese la plus courte a publier apres un run:",
        "- par KPI",
        "- pire annee observee",
        "- nombre de trimestres manquants",
        "- pourcentage de trous",
        "- principaux tickers contributeurs",
        "",
    ]
    for row in worst_year_brief.to_dicts():
        lines.append(
            f"- {row['metric_label']}: pire année = {row['fiscal_year']}, "
            f"{row['missing_quarters']} trous, {row['missing_pct']:.2f}% manquants. "
            f"Tickers principaux: {row['top_tickers'] or 'aucun'}."
        )
    return "\n".join(lines) + "\n"


def _render_markdown(payload: dict[str, object]) -> str:
    lines = [
        "# Rapport annuel des trous SEC",
        "",
        f"Période analysée: `{payload['year_start']}` -> `{payload['year_end']}`",
        "",
        "Ce rapport se concentre uniquement sur les KPI coeur:",
        "- chiffre d'affaires",
        "- résultat net",
        "- EPS publié",
        "",
        "Les années incomplètes après la fin de période choisie sont exclues pour éviter de mélanger les vrais trous avec des quarters encore en cours de publication.",
        "",
        "## Brief KPI",
    ]
    for row in payload["worst_year_brief"]:
        lines.append(
            f"- {row['metric_label']}: pire année = {row['fiscal_year']}, "
            f"{row['missing_quarters']} trous, {row['missing_pct']:.2f}% manquants. "
            f"Tickers principaux: {row['top_tickers'] or 'aucun'}."
        )
    lines.extend([
        "",
        "## Pires années",
    ])
    for row in payload["worst_years"]:
        lines.append(
            f"- {row['metric_label']} {row['fiscal_year']}: {row['missing_quarters']} trous, {row['missing_pct']:.1f}% manquants"
        )
    lines.extend(["", "## Recommandations"])
    for row in payload["recommendations"]:
        lines.append(f"- **{row['title']}**: {row['summary']} {row['action']}")
    return "\n".join(lines) + "\n"


def _render_html(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, separators=(",", ":"))
    cards_html = "".join(
        f"""
        <div class="card">
          <div class="card-label">{escape(item['title'])}</div>
          <div class="card-text">{escape(item['summary'])}</div>
          <div class="card-action">{escape(item['action'])}</div>
        </div>
        """
        for item in payload["recommendations"]
    )
    return f"""
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Rapport annuel des trous SEC</title>
  <script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
  <style>
    body {{ margin: 0; color: #1c1e21; background: linear-gradient(180deg, #f6f2e9 0%, #fbfaf7 45%, #ffffff 100%); font-family: Georgia, 'Times New Roman', serif; }}
    .hero {{ padding: 34px 42px 18px 42px; border-bottom: 1px solid rgba(0,0,0,0.08); }}
    .hero h1 {{ margin: 0; font-size: 40px; letter-spacing: -0.03em; }}
    .hero p {{ max-width: 980px; color: #5c6167; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .section {{ padding: 20px 42px 12px 42px; }}
    .section h2 {{ margin: 0 0 10px 0; font-size: 28px; letter-spacing: -0.03em; }}
    .section p {{ color: #61656b; font-family: -apple-system, BlinkMacSystemFont, sans-serif; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; }}
    .card {{ background: white; border-radius: 18px; padding: 18px; box-shadow: 0 10px 28px rgba(0,0,0,0.06); }}
    .card-label {{ font-size: 18px; font-weight: 700; }}
    .card-text {{ margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #4a4e55; }}
    .card-action {{ margin-top: 10px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; color: #7b4f24; font-weight: 600; }}
    .chart-grid {{ display: grid; grid-template-columns: 1fr; gap: 18px; }}
    .chart-box {{ height: 360px; background: white; border-radius: 18px; box-shadow: 0 10px 28px rgba(0,0,0,0.06); }}
    .table-wrap {{ background: white; border-radius: 18px; box-shadow: 0 10px 28px rgba(0,0,0,0.06); overflow: hidden; }}
    table {{ width: 100%; border-collapse: collapse; font-family: -apple-system, BlinkMacSystemFont, sans-serif; font-size: 13px; }}
    th, td {{ padding: 11px 10px; border-bottom: 1px solid #efece6; text-align: left; }}
    th {{ background: #faf7f1; color: #6e6558; font-size: 12px; text-transform: uppercase; letter-spacing: 0.06em; }}
    .muted {{ color: #7a7e84; }}
  </style>
</head>
<body>
  <div class="hero">
    <h1>Trous SEC par année</h1>
    <p>Focus exclusif sur les KPI coeur de ton problème: <strong>chiffre d'affaires</strong>, <strong>résultat net</strong> et <strong>EPS publié</strong>. Le scope affiché ici est limité aux années complètes <strong>{payload['year_start']} - {payload['year_end']}</strong> pour éviter de polluer l'analyse avec des quarters futurs ou encore en cours de publication.</p>
  </div>
  <div class="section">
    <h2>Comment viser 100%</h2>
    <div class="cards">{cards_html}</div>
  </div>
  <div class="section">
    <h2>Nombre de trimestres manquants par année</h2>
    <p>Chaque graphe montre le nombre de trous réels par année, ainsi que le pourcentage de trous cette année-là. C’est la bonne vue pour décider où concentrer le backfill filing-level.</p>
    <div class="chart-grid">
      <div id="revenue-chart" class="chart-box"></div>
      <div id="net-chart" class="chart-box"></div>
      <div id="eps-chart" class="chart-box"></div>
    </div>
  </div>
  <div class="section">
    <h2>Détail annuel</h2>
    <div class="table-wrap" id="yearly-table"></div>
  </div>
  <div class="section">
    <h2>Tickers les plus responsables sur l'année la plus trouée</h2>
    <div class="table-wrap" id="top-table"></div>
  </div>
  <script>
    const payload = {payload_json};
    const yearly = payload.yearly_summary;
    const topTickers = payload.top_tickers;
    const metricLabels = payload.metric_labels;
    const metricColors = payload.metric_colors;

    function tableHtml(rows, columns) {{
      const header = columns.map((col) => `<th>${{col.label}}</th>`).join('');
      const body = rows.map((row) => {{
        const cells = columns.map((col) => `<td>${{row[col.key] ?? ''}}</td>`).join('');
        return `<tr>${{cells}}</tr>`;
      }}).join('');
      return `<table><thead><tr>${{header}}</tr></thead><tbody>${{body}}</tbody></table>`;
    }}

    function renderChart(metric, elementId) {{
      const rows = yearly.filter((row) => row.metric === metric);
      const chart = echarts.init(document.getElementById(elementId));
      chart.setOption({{
        animation: false,
        tooltip: {{ trigger: 'axis' }},
        legend: {{ top: 8 }},
        grid: {{ left: 65, right: 60, top: 48, bottom: 44 }},
        xAxis: {{ type: 'category', data: rows.map((row) => String(row.fiscal_year)) }},
        yAxis: [
          {{ type: 'value', name: 'Trous' }},
          {{ type: 'value', name: '% manquant', axisLabel: {{ formatter: '{{value}}%' }} }},
        ],
        series: [
          {{
            name: 'Trous',
            type: 'bar',
            data: rows.map((row) => row.missing_quarters),
            itemStyle: {{ color: metricColors[metric] }},
          }},
          {{
            name: '% manquant',
            type: 'line',
            yAxisIndex: 1,
            smooth: true,
            data: rows.map((row) => Number(row.missing_pct.toFixed(2))),
            lineStyle: {{ width: 3, color: '#24445c' }},
            itemStyle: {{ color: '#24445c' }},
          }},
        ],
        title: {{
          text: metricLabels[metric],
          left: 16,
          top: 12,
          textStyle: {{ fontFamily: "Georgia, 'Times New Roman', serif", fontSize: 22, fontWeight: 700 }},
        }},
      }});
      window.addEventListener('resize', () => chart.resize());
    }}

    document.getElementById('yearly-table').innerHTML = tableHtml(
      yearly.map((row) => ({{
        kpi: row.metric_label,
        annee: row.fiscal_year,
        trimestres_attendus: row.expected_quarters,
        trous: row.missing_quarters,
        pct_manquant: row.missing_pct.toFixed(2) + '%',
      }})),
      [
        {{ key: 'kpi', label: 'KPI' }},
        {{ key: 'annee', label: 'Année' }},
        {{ key: 'trimestres_attendus', label: 'Trimestres attendus' }},
        {{ key: 'trous', label: 'Trous' }},
        {{ key: 'pct_manquant', label: '% manquant' }},
      ],
    );

    document.getElementById('top-table').innerHTML = tableHtml(
      topTickers.map((row) => ({{
        kpi: row.metric_label,
        annee: row.fiscal_year,
        ticker: row.ticker_code,
        trous: row.missing_quarters,
      }})),
      [
        {{ key: 'kpi', label: 'KPI' }},
        {{ key: 'annee', label: 'Année la plus trouée' }},
        {{ key: 'ticker', label: 'Ticker' }},
        {{ key: 'trous', label: 'Trous cette année' }},
      ],
    );

    renderChart('revenue', 'revenue-chart');
    renderChart('net_income', 'net-chart');
    renderChart('epsActual', 'eps-chart');
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
