#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import polars as pl


COLORS = {
    "portfolio": "#111D55",
    "benchmark": "#9B8816",
    "risk": "#25A18E",
    "sector": "#802331",
    "muted": "#64748B",
    "grid": "#D7E0EA",
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _number(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    return (
        '<div class="table-wrap"><table><thead><tr>'
        + "".join(f"<th>{html.escape(value)}</th>" for value in headers)
        + "</tr></thead><tbody>"
        + "".join(
            "<tr>"
            + "".join(f"<td>{value}</td>" for value in row)
            + "</tr>"
            for row in rows
        )
        + "</tbody></table></div>"
    )


def _line_svg(
    series: dict[str, tuple[np.ndarray, str]],
    *,
    logarithmic: bool,
    percent: bool,
) -> str:
    width, height = 920, 330
    left, right, top, bottom = 64, 18, 28, 42
    plot_width = width - left - right
    plot_height = height - top - bottom
    transformed = {
        name: np.log(np.maximum(values, 1e-8)) if logarithmic else values
        for name, (values, _) in series.items()
    }
    all_values = np.concatenate(list(transformed.values()))
    low = float(np.nanmin(all_values))
    high = float(np.nanmax(all_values))
    padding = max((high - low) * 0.08, 0.02)
    low -= padding
    high += padding

    def x(index: int, count: int) -> float:
        return left + index / max(1, count - 1) * plot_width

    def y(value: float) -> float:
        return top + (high - value) / max(1e-12, high - low) * plot_height

    grid: list[str] = []
    labels: list[str] = []
    for tick in np.linspace(low, high, 5):
        raw_tick = float(np.exp(tick)) if logarithmic else float(tick)
        label = (
            f"{raw_tick:.1f}×"
            if logarithmic
            else f"{raw_tick * 100:.0f}%"
            if percent
            else f"{raw_tick:.2f}"
        )
        grid.append(
            f'<line x1="{left}" y1="{y(tick):.1f}" x2="{width-right}" '
            f'y2="{y(tick):.1f}" stroke="{COLORS["grid"]}" '
            'stroke-dasharray="3 5"/>'
        )
        labels.append(
            f'<text x="{left-10}" y="{y(tick)+4:.1f}" text-anchor="end">'
            f"{html.escape(label)}</text>"
        )
    lines: list[str] = []
    legend: list[str] = []
    count = len(next(iter(transformed.values())))
    for legend_index, (name, values) in enumerate(transformed.items()):
        color = series[name][1]
        points = " ".join(
            f"{x(index, count):.1f},{y(float(value)):.1f}"
            for index, value in enumerate(values)
        )
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{color}" '
            'stroke-width="2.4" vector-effect="non-scaling-stroke"/>'
        )
        legend_x = left + legend_index * 205
        legend.extend(
            [
                f'<line x1="{legend_x}" y1="13" x2="{legend_x+24}" y2="13" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x+31}" y="17">{html.escape(name)}</text>',
            ]
        )
    x_labels = "".join(
        f'<text x="{x(index, count):.1f}" y="{height-12}" '
        f'text-anchor="{anchor}">{label}</text>'
        for index, label, anchor in (
            (0, "2011", "start"),
            (count // 3, "2016", "middle"),
            (2 * count // 3, "2021", "middle"),
            (count - 1, "2025", "end"),
        )
    )
    return (
        f'<svg class="line-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="courbes de performance">'
        + "".join(grid + labels + legend + lines)
        + x_labels
        + "</svg>"
    )


def _bar_svg(rows: list[dict], *, color: str) -> str:
    rows = rows[:10]
    width = 840
    row_height = 30
    height = 28 + row_height * len(rows)
    label_width = 305
    maximum = max(float(row["mean_abs_shap"]) for row in rows)
    elements: list[str] = []
    for index, row in enumerate(rows):
        y = 18 + index * row_height
        feature = str(row["feature"]).replace("relative_ema_ratio_", "EMA ")
        bar_width = 470 * float(row["mean_abs_shap"]) / maximum
        elements.append(
            f'<text x="{label_width-10}" y="{y+12}" text-anchor="end">'
            f"{html.escape(feature)}</text>"
        )
        elements.append(
            f'<rect x="{label_width}" y="{y}" width="{bar_width:.1f}" '
            f'height="17" rx="2" fill="{color}"/>'
        )
    return (
        f'<svg class="bar-chart" viewBox="0 0 {width} {height}" '
        'role="img" aria-label="importance SHAP">'
        + "".join(elements)
        + "</svg>"
    )


def _shell(
    *,
    title: str,
    subtitle: str,
    status: str,
    body: str,
    current: str,
) -> str:
    navigation = [
        ("index.html", "Synthèse"),
        ("risk_results_paper.html", "Résultats"),
        ("methodology_paper.html", "Méthode"),
    ]
    nav = "".join(
        f'<a class="nav-item {"active" if label == current else ""}" '
        f'href="{href}">{label}</a>'
        for href, label in navigation
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --primary:#0F172A; --secondary:#334155; --accent:#0369A1;
      --bg:#F8FAFC; --panel:#FFFFFF; --surface:#F1F5F9;
      --border:#D7E0EA; --text:#020617; --muted:#475569;
      --success:#265511; --warning:#D97706; --error:#802331;
      --portfolio:#111D55; --benchmark:#9B8816;
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0; background:var(--bg); color:var(--text);
      font-family:"IBM Plex Sans",sans-serif; line-height:1.55;
    }}
    .workspace {{
      max-width:1440px; margin:0 auto; padding:28px 24px 64px;
      display:grid; grid-template-columns:248px minmax(0,1fr); gap:24px;
    }}
    main {{ min-width:0; }}
    aside {{ position:sticky; top:28px; height:calc(100vh - 56px); }}
    .brand {{ padding:6px 8px 22px; border-bottom:1px solid var(--border); }}
    .brand b {{ display:block; font-size:19px; letter-spacing:-.02em; }}
    .mono,.eyebrow,.nav-item,.badge,.kpi-label,th,svg text {{
      font-family:"IBM Plex Mono",monospace;
    }}
    .brand span,.meta {{ color:var(--muted); font-size:12px; }}
    nav {{ display:grid; gap:6px; padding:22px 0; }}
    .nav-item {{
      color:var(--muted); text-decoration:none; padding:9px 11px;
      border:1px solid transparent; border-radius:10px; font-size:13px;
    }}
    .nav-item:hover,.nav-item.active {{
      color:var(--text); background:var(--panel); border-color:var(--border);
    }}
    .focus {{
      border-top:1px solid var(--border); padding:20px 8px;
      color:var(--muted); font-size:13px;
    }}
    .focus strong {{ color:var(--text); display:block; margin-top:5px; }}
    header {{
      display:flex; justify-content:space-between; gap:24px;
      padding:2px 0 24px; border-bottom:1px solid var(--border);
    }}
    .eyebrow {{
      margin:0 0 7px; color:var(--accent); text-transform:uppercase;
      letter-spacing:.10em; font-weight:600; font-size:11px;
    }}
    h1 {{ margin:0; font-size:34px; line-height:1.12; letter-spacing:-.035em; }}
    header p {{ margin:9px 0 0; color:var(--muted); max-width:760px; }}
    .badges {{ display:flex; align-items:flex-start; gap:8px; flex-wrap:wrap; }}
    .badge {{
      border:1px solid var(--border); border-radius:999px; background:var(--panel);
      padding:6px 9px; font-size:10px; white-space:nowrap;
    }}
    .badge.no-go {{ color:var(--error); border-color:#E9C4CA; background:#FFF7F8; }}
    section {{ margin-top:28px; }}
    h2 {{ margin:0 0 6px; font-size:22px; letter-spacing:-.025em; }}
    h3 {{ margin:0 0 6px; font-size:16px; }}
    .section-lede {{ margin:0 0 16px; color:var(--muted); max-width:900px; }}
    .strip {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:12px; }}
    .panel {{
      background:var(--panel); border:1px solid var(--border); border-radius:12px;
      padding:18px; box-shadow:0 1px 2px rgba(2,6,23,.04);
    }}
    .kpi-label {{
      color:var(--muted); text-transform:uppercase; letter-spacing:.06em;
      font-size:10px;
    }}
    .kpi-value {{ font-size:26px; font-weight:650; letter-spacing:-.035em; margin:5px 0 1px; }}
    .kpi-note {{ color:var(--muted); font-size:12px; }}
    .grid-2 {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
    .callout {{
      background:var(--surface); border:1px solid var(--border);
      border-left:4px solid var(--warning); border-radius:10px;
      padding:15px 17px; margin:16px 0;
    }}
    .callout.good {{ border-left-color:var(--success); }}
    .callout.bad {{ border-left-color:var(--error); }}
    .callout p {{ margin:4px 0 0; color:var(--muted); }}
    .table-wrap {{
      overflow:auto; background:var(--panel); border:1px solid var(--border);
      border-radius:12px;
    }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th {{
      background:var(--surface); color:var(--muted); text-align:left;
      padding:10px 12px; font-size:10px; letter-spacing:.04em; white-space:nowrap;
    }}
    td {{ border-top:1px solid #E8EDF3; padding:10px 12px; white-space:nowrap; }}
    tr.primary td {{ font-weight:600; background:#F8FBFF; }}
    .yes {{ color:var(--success); }} .no {{ color:var(--error); }}
    .line-chart,.bar-chart {{ display:block; width:100%; height:auto; }}
    svg text {{ fill:var(--muted); font-size:10px; }}
    .chart-meta {{ margin-top:8px; color:var(--muted); font-size:11px; }}
    ul {{ padding-left:20px; }} li {{ margin:8px 0; }}
    code {{ font-family:"IBM Plex Mono"; background:var(--surface); padding:2px 4px; border-radius:4px; }}
    footer {{
      margin-top:34px; padding-top:18px; border-top:1px solid var(--border);
      color:var(--muted); font-size:11px; font-family:"IBM Plex Mono";
    }}
    @media(max-width:900px) {{
      .workspace {{ grid-template-columns:1fr; padding:18px 14px 48px; }}
      aside {{ position:static; height:auto; }}
      nav {{ display:flex; overflow:auto; padding:14px 0; }}
      .focus {{ display:none; }}
      header {{ display:block; }} .badges {{ margin-top:14px; }}
      .strip,.grid-2 {{ grid-template-columns:1fr 1fr; }}
    }}
    @media(max-width:560px) {{
      .strip,.grid-2 {{ grid-template-columns:1fr; }}
      h1 {{ font-size:28px; }}
    }}
    @media print {{
      .workspace {{ display:block; max-width:none; padding:0; }}
      aside {{ display:none; }} .panel,.table-wrap {{ break-inside:avoid; }}
    }}
  </style>
</head>
<body>
<div class="workspace">
  <aside>
    <div class="brand"><b>AlphaRank</b><span>Research monitor</span></div>
    <nav>{nav}</nav>
    <div class="focus"><span class="mono">FOCUS</span><strong>Exact EMA · Risk heads v1</strong><span>172 mois OOS · 15 folds</span></div>
  </aside>
  <main>
    <header>
      <div>
        <p class="eyebrow">Recherche boosting · 25 juillet 2026</p>
        <h1>{html.escape(title)}</h1>
        <p>{html.escape(subtitle)}</p>
      </div>
      <div class="badges"><span class="badge no-go">{html.escape(status)}</span><span class="badge">2011-07 → 2025-10</span><span class="badge">S&amp;P 500</span></div>
    </header>
    {body}
    <footer>legacy_ema_risk_overlay_long_history_v1 · snapshot 20260719_194418 · rapport reproductible</footer>
  </main>
</div>
</body>
</html>"""


def _results_page(output_dir: Path) -> str:
    metrics = pl.read_csv(output_dir / "risk_model_metrics.csv")
    performance = pl.read_csv(
        output_dir / "allocation_performance.csv",
        try_parse_dates=True,
    )
    monthly = pl.read_csv(
        output_dir / "allocation_monthly.csv",
        try_parse_dates=True,
    )
    gates = pl.read_csv(output_dir / "allocation_acceptance_gates.csv")
    bootstrap = pl.read_csv(output_dir / "allocation_paired_bootstrap.csv")
    vol3 = metrics.filter(
        (pl.col("head") == "realized_volatility")
        & (pl.col("horizon") == 3)
    ).row(0, named=True)
    high3 = metrics.filter(
        (pl.col("head") == "high_volatility")
        & (pl.col("horizon") == 3)
    ).row(0, named=True)
    baseline = performance.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).row(0, named=True)
    overlay = performance.filter(
        pl.col("strategy") == "alpha_top5_inverse_vol_h3"
    ).row(0, named=True)

    risk_rows: list[list[str]] = []
    for horizon in (1, 3, 6):
        vol = metrics.filter(
            (pl.col("head") == "realized_volatility")
            & (pl.col("horizon") == horizon)
        ).row(0, named=True)
        downside = metrics.filter(
            (pl.col("head") == "daily_downside")
            & (pl.col("horizon") == horizon)
        ).row(0, named=True)
        high = metrics.filter(
            (pl.col("head") == "high_volatility")
            & (pl.col("horizon") == horizon)
        ).row(0, named=True)
        risk_rows.append(
            [
                f"{horizon} mois",
                _number(vol["monthly_spearman"]),
                _number(vol["r2"]),
                _number(downside["monthly_spearman"]),
                _number(downside["r2"]),
                _number(high["roc_auc"]),
                _number(high["pr_auc_average_precision"]),
            ]
        )
    performance_names = [
        "alpha_top5_equal",
        "alpha_top5_inverse_vol_h1",
        "alpha_top5_inverse_vol_h3",
        "alpha_top5_inverse_vol_h6",
        "alpha_top5_inverse_downside_h6",
        "alpha_top5_inverse_vol_h3_sector2",
    ]
    performance_rows: list[list[str]] = []
    labels = {
        "alpha_top5_equal": "Alpha top 5 · égal",
        "alpha_top5_inverse_vol_h1": "Inverse vol · 1m",
        "alpha_top5_inverse_vol_h3": "Inverse vol · 3m (primaire)",
        "alpha_top5_inverse_vol_h6": "Inverse vol · 6m",
        "alpha_top5_inverse_downside_h6": "Inverse downside · 6m",
        "alpha_top5_inverse_vol_h3_sector2": "Vol 3m + secteur",
    }
    for name in performance_names:
        row = performance.filter(pl.col("strategy") == name).row(0, named=True)
        performance_rows.append(
            [
                labels[name],
                _pct(row["model_cagr"]),
                _number(row["model_sharpe"]),
                _pct(row["model_annualized_volatility"]),
                _pct(row["model_max_drawdown"]),
                _pct(row["average_turnover"]),
                _pct(row["maximum_sector_weight"]),
            ]
        )

    equal_monthly = monthly.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).sort("decision_month")
    risk_monthly = monthly.filter(
        pl.col("strategy") == "alpha_top5_inverse_vol_h3"
    ).sort("decision_month")
    sector_monthly = monthly.filter(
        pl.col("strategy") == "alpha_top5_inverse_vol_h3_sector2"
    ).sort("decision_month")
    wealth_series = {
        "Alpha égal": (
            np.cumprod(1.0 + equal_monthly["net_return"].to_numpy()),
            COLORS["portfolio"],
        ),
        "Inverse vol 3m": (
            np.cumprod(1.0 + risk_monthly["net_return"].to_numpy()),
            COLORS["risk"],
        ),
        "Legacy": (
            np.cumprod(1.0 + equal_monthly["legacy_return"].to_numpy()),
            COLORS["benchmark"],
        ),
        "S&P 500": (
            np.cumprod(1.0 + equal_monthly["benchmark_return"].to_numpy()),
            COLORS["muted"],
        ),
    }

    def drawdown(values: np.ndarray) -> np.ndarray:
        wealth = np.cumprod(1.0 + values)
        return wealth / np.maximum.accumulate(wealth) - 1.0

    drawdown_series = {
        "Alpha égal": (
            drawdown(equal_monthly["net_return"].to_numpy()),
            COLORS["portfolio"],
        ),
        "Inverse vol 3m": (
            drawdown(risk_monthly["net_return"].to_numpy()),
            COLORS["risk"],
        ),
        "Secteur contraint": (
            drawdown(sector_monthly["net_return"].to_numpy()),
            COLORS["sector"],
        ),
        "Legacy": (
            drawdown(equal_monthly["legacy_return"].to_numpy()),
            COLORS["benchmark"],
        ),
    }
    shap_sections: list[str] = []
    shap_configs = [
        (
            "Volatilité réalisée à 3 mois",
            "predicted_realized_volatility_3m",
            COLORS["risk"],
        ),
        (
            "Probabilité de forte volatilité à 3 mois",
            "predicted_high_volatility_3m_score",
            COLORS["portfolio"],
        ),
        (
            "Downside journalier à 3 mois",
            "predicted_daily_downside_3m",
            COLORS["benchmark"],
        ),
    ]
    for title, directory, color in shap_configs:
        importance = pl.read_csv(
            output_dir / "shap" / directory / "shap_importance.csv"
        ).head(10)
        shap_sections.append(
            f'<div class="panel"><h3>{html.escape(title)}</h3>'
            + _bar_svg(importance.to_dicts(), color=color)
            + '<p class="chart-meta">SHAP moyen absolu. Pour les régressions, la contribution est exprimée dans l’espace log-risque appris par XGBoost.</p></div>'
        )

    overlay_bootstrap = bootstrap.filter(
        (pl.col("strategy") == "alpha_top5_inverse_vol_h3")
        & (pl.col("comparator") == "alpha_top5_equal")
    ).row(0, named=True)
    gate_rows = [
        [
            row["strategy"].replace("alpha_top5_", ""),
            '<span class="yes">oui</span>'
            if row["sharpe_higher"]
            else '<span class="no">non</span>',
            '<span class="yes">oui</span>'
            if row["drawdown_improves_5pp"]
            else '<span class="no">non</span>',
            '<span class="yes">oui</span>'
            if row["cagr_loss_within_3pp"]
            else '<span class="no">non</span>',
            '<span class="yes">oui</span>'
            if row["sector_weight_at_most_40pct"]
            else '<span class="no">non</span>',
            '<span class="yes">oui</span>'
            if row["sharpe_higher_at_50bps"]
            else '<span class="no">non</span>',
            '<span class="yes">PASS</span>'
            if row["all_gates_pass"]
            else '<span class="no">NO-GO</span>',
        ]
        for row in gates.to_dicts()
    ]
    body = f"""
<section>
  <div class="strip">
    <div class="panel"><div class="kpi-label">Historique OOS</div><div class="kpi-value">172 mois</div><div class="kpi-note">15 folds · juillet 2011 à octobre 2025</div></div>
    <div class="panel"><div class="kpi-label">Vol 3m · R²</div><div class="kpi-value">{_number(vol3["r2"])}</div><div class="kpi-note">Spearman mensuel {_number(vol3["monthly_spearman"])}</div></div>
    <div class="panel"><div class="kpi-label">Risque élevé 3m</div><div class="kpi-value">{_number(high3["roc_auc"])}</div><div class="kpi-note">ROC-AUC · PR-AUC {_number(high3["pr_auc_average_precision"])}</div></div>
    <div class="panel"><div class="kpi-label">Overlays acceptés</div><div class="kpi-value">0 / 2</div><div class="kpi-note">Aucune promotion allocation</div></div>
  </div>
  <div class="callout bad"><strong>Verdict : modèles de risque utiles, allocation non validée.</strong><p>Le booster apprend un signal de volatilité stable, mais le sizing inverse-vol ne prouve pas une amélioration économique face au top 5 équipondéré. Le filtre sectoriel coûte trop de rendement.</p></div>
</section>
<section>
  <h2>Qualité des têtes de risque</h2>
  <p class="section-lede">Toutes les métriques sont calculées sur les mêmes 172 mois hors échantillon. La classification vise le quintile de volatilité future le plus élevé.</p>
  {_table(["Horizon","Vol Spearman","Vol R²","Downside Spearman","Downside R²","High-vol ROC","High-vol PR"], risk_rows)}
</section>
<section>
  <h2>Résultat portefeuille</h2>
  <p class="section-lede">Le classement alpha reste intact. Le risque ne sert qu’au poids, sauf dans la variante sectorielle explicitement contrainte.</p>
  {_table(["Stratégie","CAGR net","Sharpe","Vol. ann.","Drawdown","Turnover","Secteur max"], performance_rows)}
  <div class="callout"><strong>Lecture.</strong><p>L’inverse-vol 3 mois améliore le drawdown de 3,63 points et le Sharpe de seulement 0,003, au prix de 1,91 point de CAGR. Le bootstrap apparié estime la différence de Sharpe à {_number(overlay_bootstrap["observed_sharpe_difference"])} avec un IC 95 % [{_number(overlay_bootstrap["sharpe_difference_ci_low"])}, {_number(overlay_bootstrap["sharpe_difference_ci_high"])}].</p></div>
</section>
<section>
  <h2>Trajectoire de capital</h2>
  <div class="panel">{_line_svg(wealth_series, logarithmic=True, percent=False)}<p class="chart-meta">Multiple de capital, échelle logarithmique. 10 pb × turnover inclus pour les stratégies Alpha.</p></div>
</section>
<section>
  <h2>Drawdown</h2>
  <div class="panel">{_line_svg(drawdown_series, logarithmic=False, percent=True)}<p class="chart-meta">Le contrôle sectoriel ne réduit pas le pire drawdown ; il ne passe donc pas le rôle attendu.</p></div>
</section>
<section>
  <h2>Garde-fous pré-enregistrés</h2>
  {_table(["Overlay","Sharpe ↑","DD +5pp","CAGR ≤−3pp","Secteur ≤40%","Sharpe ↑ @50pb","Décision"], gate_rows)}
</section>
<section>
  <h2>Ce que les EMA racontent sur le risque</h2>
  <p class="section-lede">Les mêmes rapports EMA Legacy portent un signal de risque mesurable. Les variables dominantes diffèrent entre niveau de volatilité, probabilité de régime risqué et downside.</p>
  <div class="grid-2">{''.join(shap_sections)}</div>
</section>
<section>
  <h2>Décision recommandée</h2>
  <div class="grid-2">
    <div class="panel"><h3>À conserver</h3><ul><li>Les têtes boosting de volatilité et de probabilité high-vol.</li><li>Leur sortie par titre et leur SHAP pour l’explication.</li><li>Le top 5 alpha équipondéré comme référence de recherche.</li></ul></div>
    <div class="panel"><h3>À ne pas activer</h3><ul><li>Le sizing inverse-vol 3 mois en production.</li><li>Le filtre deux titres par secteur dans sa forme actuelle.</li><li>Toute sélection post-hoc de l’horizon 1 ou 6 sur ce même historique.</li></ul></div>
  </div>
</section>
"""
    return _shell(
        title="Têtes de risque EMA : résultat long historique",
        subtitle=(
            "Volatilité, downside et probabilité de régime risqué à 1, 3 et 6 "
            "mois, puis impact réel sur l’allocation."
        ),
        status="NO-GO OVERLAY",
        body=body,
        current="Résultats",
    )


def _methodology_page(specification: dict, manifest: dict) -> str:
    body = f"""
<section>
  <h2>Question testée</h2>
  <p class="section-lede">Peut-on conserver le classement alpha boosting exact-EMA à six mois et utiliser des modèles boosting séparés pour rationaliser le risque futur, sans fuite de données ni optimisation implicite sur le backtest ?</p>
  <div class="callout good"><strong>Séparation des rôles.</strong><p>Le score alpha ordonne les titres. Les têtes de risque estiment volatilité, downside et probabilité high-vol. Elles ne sont jamais additionnées au score alpha.</p></div>
</section>
<section>
  <h2>Historique maximal défendable</h2>
  <div class="strip">
    <div class="panel"><div class="kpi-label">Prix disponibles</div><div class="kpi-value">2005-01</div><div class="kpi-note">Début du snapshot quotidien</div></div>
    <div class="panel"><div class="kpi-label">1re EMA observable</div><div class="kpi-value">2010-02</div><div class="kpi-note">Première gagnante Legacy connue</div></div>
    <div class="panel"><div class="kpi-label">1er test possible</div><div class="kpi-value">2011-07</div><div class="kpi-note">Après validation et purge</div></div>
    <div class="panel"><div class="kpi-label">Dernier test mûr</div><div class="kpi-value">2025-10</div><div class="kpi-note">Cible alpha 6 mois disponible</div></div>
  </div>
  <div class="callout"><strong>Pourquoi pas avant ?</strong><p>Choisir en 2005 une paire EMA qui ne devient gagnante Legacy qu’en 2010 serait une fuite de sélection. Le train peut utiliser les prix 2005–2010, mais la première fenêtre test doit attendre qu’au moins une paire gagnante soit connue au cutoff train.</p></div>
</section>
<section>
  <h2>Géométrie temporelle</h2>
  <div class="grid-2">
    <div class="panel"><h3>Outer walk-forward</h3><ul><li>62 mois de train minimum.</li><li>6 mois de validation.</li><li>Purge conservatrice de 6 mois pour toutes les têtes.</li><li>12 mois de test, avec dernier bloc partiel conservé.</li><li>15 modèles fixes par tête.</li></ul></div>
    <div class="panel"><h3>Point-in-time</h3><ul><li>Paires EMA gagnantes arrêtées au cutoff train.</li><li>Prétraitement ajusté dans chaque fold.</li><li>Early stopping sur la validation antérieure au test.</li><li>Calibration isotone réservée aux probabilités.</li><li>Score brut utilisé pour toute hiérarchie.</li></ul></div>
  </div>
</section>
<section>
  <h2>Définition des cibles</h2>
  <div class="grid-2">
    <div class="panel"><h3>Volatilité réalisée</h3><p>Écart-type annualisé des rendements journaliers strictement situés dans les mois t+1 à t+h. Chaque mois futur doit avoir au moins 10 observations valides.</p></div>
    <div class="panel"><h3>Downside journalier</h3><p>Racine annualisée de la moyenne des rendements journaliers négatifs élevés au carré, sur la même fenêtre future stricte.</p></div>
    <div class="panel"><h3>Risque élevé</h3><p>Classification du quintile supérieur de volatilité réalisée future, recalculé transversalement chaque mois.</p></div>
    <div class="panel"><h3>Rendement trading</h3><p>Rendement du mois t+1 après chaque décision, avec 10 points de base multipliés par le turnover mensuel.</p></div>
  </div>
</section>
<section>
  <h2>Allocations figées avant résultat</h2>
  <div class="grid-2">
    <div class="panel"><h3>A · Référence</h3><p>Top 5 du score alpha brut, équipondéré.</p></div>
    <div class="panel"><h3>B · Risque primaire</h3><p>Mêmes cinq titres, poids inverse de la volatilité prédite à trois mois, 30 % maximum par titre.</p></div>
    <div class="panel"><h3>C · Risque + secteur</h3><p>Classement alpha avec deux titres maximum par secteur, puis poids inverse-vol et 40 % maximum par secteur.</p></div>
    <div class="panel"><h3>Sensibilités</h3><p>Horizons 1 et 6 mois et downside sont publiés comme diagnostics ; ils ne peuvent pas remplacer rétroactivement l’horizon 3 mois primaire.</p></div>
  </div>
</section>
<section>
  <h2>Traçabilité</h2>
  {_table(["Élément","Valeur"], [
      ["Identifiant", html.escape(specification["research_id"])],
      ["Snapshot", html.escape(specification["data"]["input_snapshot"])],
      ["Alpha run", html.escape(specification["alpha"]["prediction_run"])],
      ["Rows test", f'{manifest["test_rows"]:,}'.replace(",", " ")],
      ["Folds", str(manifest["outer_folds"])],
      ["Hash spec", html.escape(manifest["spec_sha256"][:16] + "…")],
      ["Commit au run", html.escape(manifest["repository_head"][:12])],
  ])}
</section>
"""
    return _shell(
        title="Méthodologie sans fuite des têtes de risque",
        subtitle=(
            "Cibles journalières futures, sélection EMA point-in-time et "
            "allocation évaluée sur le même calendrier hors échantillon."
        ),
        status="PROTOCOLE AUDITÉ",
        body=body,
        current="Méthode",
    )


def _index_page(output_dir: Path) -> str:
    performance = pl.read_csv(output_dir / "allocation_performance.csv")
    baseline = performance.filter(
        pl.col("strategy") == "alpha_top5_equal"
    ).row(0, named=True)
    risk = performance.filter(
        pl.col("strategy") == "alpha_top5_inverse_vol_h3"
    ).row(0, named=True)
    body = f"""
<section>
  <div class="strip">
    <div class="panel"><div class="kpi-label">Signal de risque</div><div class="kpi-value">Oui</div><div class="kpi-note">ROC-AUC high-vol 3m : 0,783</div></div>
    <div class="panel"><div class="kpi-label">Overlay validé</div><div class="kpi-value">Non</div><div class="kpi-note">0 garde-fou complet</div></div>
    <div class="panel"><div class="kpi-label">Sharpe égal</div><div class="kpi-value">{_number(baseline["model_sharpe"])}</div><div class="kpi-note">Top 5 équipondéré</div></div>
    <div class="panel"><div class="kpi-label">Sharpe inverse vol</div><div class="kpi-value">{_number(risk["model_sharpe"])}</div><div class="kpi-note">Gain non significatif</div></div>
  </div>
</section>
<section>
  <h2>Conclusion en une phrase</h2>
  <div class="callout bad"><strong>On sait désormais expliquer et quantifier le risque avec du boosting exact-EMA, mais pas encore l’utiliser pour améliorer de façon robuste le portefeuille.</strong><p>Le bon résultat de cette phase est la séparation propre alpha/risque et la preuve que l’overlay naïf ne mérite pas d’être activé.</p></div>
</section>
<section>
  <h2>Deux papiers</h2>
  <div class="grid-2">
    <a class="panel" href="risk_results_paper.html" style="color:inherit;text-decoration:none"><span class="kpi-label">Papier 01</span><h3 style="margin-top:8px">Résultats et SHAP</h3><p class="meta">Métriques modèles, backtests, coûts, bootstrap, garde-fous et interprétation des EMA.</p></a>
    <a class="panel" href="methodology_paper.html" style="color:inherit;text-decoration:none"><span class="kpi-label">Papier 02</span><h3 style="margin-top:8px">Méthodologie</h3><p class="meta">Couverture maximale, cibles journalières, purges, allocation pré-enregistrée et traçabilité.</p></a>
  </div>
</section>
<section>
  <h2>Décision</h2>
  <div class="grid-2">
    <div class="panel"><h3>Livrable réutilisable</h3><p>Conserver par titre la volatilité attendue, le downside, la probabilité high-vol et leurs contributions SHAP.</p></div>
    <div class="panel"><h3>Règle de prudence</h3><p>Ne pas modifier les poids réels avec cet overlay avant une nouvelle variante pré-enregistrée et un holdout réellement neuf.</p></div>
  </div>
</section>
"""
    return _shell(
        title="EMA exactes : alpha fort, risque explicable",
        subtitle=(
            "Synthèse de la phase long historique et des têtes de risque "
            "boosting à 1, 3 et 6 mois."
        ),
        status="RECHERCHE · NO-GO",
        body=body,
        current="Synthèse",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AlphaRank risk papers.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    specification = json.loads(args.spec.read_text())
    manifest = json.loads((args.output_dir / "manifest.json").read_text())
    html_dir = args.output_dir / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    (html_dir / "index.html").write_text(
        _index_page(args.output_dir),
        encoding="utf-8",
    )
    (html_dir / "risk_results_paper.html").write_text(
        _results_page(args.output_dir),
        encoding="utf-8",
    )
    (html_dir / "methodology_paper.html").write_text(
        _methodology_page(specification, manifest),
        encoding="utf-8",
    )
    print(html_dir)


if __name__ == "__main__":
    main()
