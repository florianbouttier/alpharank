#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl


PALETTE = {
    "ink": "#14213d",
    "muted": "#64748b",
    "paper": "#f7f5ef",
    "card": "#ffffff",
    "line": "#d9dee8",
    "blue": "#1d4ed8",
    "green": "#047857",
    "orange": "#c2410c",
    "red": "#b91c1c",
    "purple": "#7e22ce",
}


def _pct(value: float, digits: int = 1) -> str:
    return f"{value * 100:.{digits}f}%"


def _num(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def _wealth(returns: pl.Series) -> np.ndarray:
    return np.cumprod(1.0 + returns.to_numpy())


def _drawdown(returns: pl.Series) -> np.ndarray:
    curve = _wealth(returns)
    return curve / np.maximum.accumulate(curve) - 1.0


def _chart(fig: go.Figure) -> str:
    fig.update_layout(
        template="plotly_white",
        font={"family": "Inter, ui-sans-serif, system-ui", "color": PALETTE["ink"]},
        margin={"l": 48, "r": 24, "t": 62, "b": 46},
        paper_bgcolor=PALETTE["card"],
        plot_bgcolor=PALETTE["card"],
        legend={"orientation": "h", "y": 1.12, "x": 0},
        hovermode="x unified",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False)


def _table(
    headers: list[str],
    rows: list[list[str]],
    *,
    note: str | None = None,
) -> str:
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    note_html = f'<p class="table-note">{html.escape(note)}</p>' if note else ""
    return (
        f'<div class="table-wrap"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>{note_html}"
    )


def _document(
    *,
    title: str,
    eyebrow: str,
    verdict: str,
    verdict_tone: str,
    body: str,
) -> str:
    return f"""<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    :root {{
      --ink:{PALETTE["ink"]}; --muted:{PALETTE["muted"]};
      --paper:{PALETTE["paper"]}; --card:{PALETTE["card"]};
      --line:{PALETTE["line"]}; --blue:{PALETTE["blue"]};
      --green:{PALETTE["green"]}; --orange:{PALETTE["orange"]};
      --red:{PALETTE["red"]}; --purple:{PALETTE["purple"]};
    }}
    * {{ box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
      margin:0; background:var(--paper); color:var(--ink);
      font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
      line-height:1.62;
    }}
    .page {{ max-width:1180px; margin:0 auto; padding:56px 28px 96px; }}
    .hero {{ display:grid; grid-template-columns:minmax(0,1.6fr) minmax(280px,.7fr);
      gap:32px; padding:42px; background:var(--ink); color:white; border-radius:24px;
      box-shadow:0 18px 60px rgba(20,33,61,.18); }}
    .eyebrow {{ text-transform:uppercase; letter-spacing:.16em; font-size:12px;
      font-weight:800; color:#93c5fd; margin:0 0 14px; }}
    h1 {{ font-size:clamp(38px,6vw,70px); line-height:1.02; letter-spacing:-.045em;
      margin:0 0 20px; max-width:850px; }}
    .dek {{ font-size:19px; color:#dbeafe; max-width:800px; margin:0; }}
    .verdict {{ align-self:end; border:1px solid rgba(255,255,255,.2);
      background:rgba(255,255,255,.08); padding:22px; border-radius:18px; }}
    .verdict strong {{ display:block; text-transform:uppercase; letter-spacing:.12em;
      font-size:11px; margin-bottom:8px; color:{verdict_tone}; }}
    .nav {{ display:flex; flex-wrap:wrap; gap:10px; margin:22px 0 0; }}
    .nav a {{ color:var(--ink); text-decoration:none; background:white; border:1px solid var(--line);
      border-radius:999px; padding:8px 13px; font-size:13px; font-weight:700; }}
    section {{ margin-top:54px; scroll-margin-top:24px; }}
    h2 {{ font-size:32px; letter-spacing:-.03em; margin:0 0 14px; }}
    h3 {{ font-size:21px; margin:30px 0 10px; }}
    p {{ max-width:880px; }}
    .lede {{ font-size:18px; color:#334155; }}
    .grid {{ display:grid; grid-template-columns:repeat(12,1fr); gap:18px; margin-top:22px; }}
    .card {{ grid-column:span 4; background:var(--card); border:1px solid var(--line);
      border-radius:18px; padding:22px; box-shadow:0 8px 30px rgba(20,33,61,.05); }}
    .card.wide {{ grid-column:span 6; }}
    .card.full {{ grid-column:1/-1; }}
    .kpi {{ font-size:34px; line-height:1; letter-spacing:-.04em; font-weight:850; margin:9px 0 8px; }}
    .label {{ text-transform:uppercase; letter-spacing:.1em; color:var(--muted);
      font-size:11px; font-weight:800; }}
    .meta {{ color:var(--muted); font-size:13px; }}
    .callout {{ border-left:5px solid var(--orange); background:#fff7ed;
      padding:18px 22px; border-radius:0 14px 14px 0; margin:22px 0; max-width:920px; }}
    .callout.good {{ border-color:var(--green); background:#ecfdf5; }}
    .callout.bad {{ border-color:var(--red); background:#fef2f2; }}
    .chart {{ background:var(--card); border:1px solid var(--line); border-radius:18px;
      overflow:hidden; padding:8px; min-height:390px; }}
    .table-wrap {{ overflow:auto; background:white; border:1px solid var(--line);
      border-radius:16px; margin-top:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:14px; }}
    th {{ background:#eef2f7; color:#334155; text-align:left; text-transform:uppercase;
      letter-spacing:.06em; font-size:11px; padding:13px 14px; white-space:nowrap; }}
    td {{ padding:12px 14px; border-top:1px solid #edf0f4; white-space:nowrap; }}
    tr:hover td {{ background:#f8fafc; }}
    .table-note {{ color:var(--muted); font-size:12px; margin-top:8px; }}
    .flow {{ display:grid; grid-template-columns:repeat(5,1fr); gap:12px; margin-top:24px; }}
    .flow div {{ position:relative; background:white; border:1px solid var(--line);
      padding:18px; border-radius:15px; min-height:130px; }}
    .flow b {{ display:block; color:var(--blue); margin-bottom:8px; }}
    code {{ background:#e9edf3; padding:2px 5px; border-radius:5px; font-size:.92em; }}
    .footer {{ margin-top:70px; padding-top:24px; border-top:1px solid var(--line);
      color:var(--muted); font-size:13px; }}
    @media(max-width:850px) {{
      .hero {{ grid-template-columns:1fr; padding:28px; }}
      .card,.card.wide {{ grid-column:1/-1; }}
      .flow {{ grid-template-columns:1fr; }}
      .page {{ padding:24px 14px 64px; }}
    }}
    @media print {{
      body {{ background:white; }}
      .page {{ max-width:none; padding:0; }}
      .hero {{ box-shadow:none; break-inside:avoid; }}
      .nav {{ display:none; }}
      .card,.chart,.table-wrap {{ break-inside:avoid; box-shadow:none; }}
    }}
  </style>
</head>
<body>
<main class="page">
  <header class="hero">
    <div>
      <p class="eyebrow">{html.escape(eyebrow)}</p>
      <h1>{html.escape(title)}</h1>
      <p class="dek">Validation confirmatoire du booster exact‑EMA, avec séparation stricte entre résultat exploratoire, correction des essais multiples et preuve prospective.</p>
    </div>
    <aside class="verdict"><strong>Verdict</strong>{html.escape(verdict)}</aside>
  </header>
  {body}
  <footer class="footer">AlphaRank · paquet figé <code>legacy_ema_pit_classification_h06_top05_v1</code> · génération reproductible</footer>
</main>
</body>
</html>"""


def _methodology_paper(spec: dict, audit: dict) -> str:
    body = f"""
<nav class="nav">
  <a href="#question">Question</a><a href="#lock">Verrou</a>
  <a href="#protocol">Protocole</a><a href="#bias">Biais</a>
  <a href="#decision">Décision</a>
</nav>
<section id="question">
  <h2>La question confirmatoire</h2>
  <p class="lede">Le screening a désigné une classification à six mois, fondée uniquement sur les couples EMA déjà gagnants dans Legacy, puis un portefeuille mensuel des cinq meilleurs scores. Le test confirmatoire ne cherche plus un meilleur modèle : il mesure ce que cette règle vaut une fois figée.</p>
  <div class="callout"><b>Divulgation essentielle.</b> Le walk-forward 2013–2025 a servi à choisir la variante. Il ne constitue donc pas un holdout intact, même si chaque prédiction individuelle est hors échantillon.</div>
</section>
<section id="lock">
  <h2>Ce qui est verrouillé</h2>
  <div class="grid">
    <div class="card"><div class="label">Variables</div><div class="kpi">EMA PIT</div><p>Uniquement les couples gagnants observables avant la coupure train du fold.</p></div>
    <div class="card"><div class="label">Cible</div><div class="kpi">Top 10%</div><p>Classement de surperformance future face au S&amp;P 500 à six mois.</p></div>
    <div class="card"><div class="label">Allocation</div><div class="kpi">Top 5</div><p>Équipondéré, rebalancé mensuellement, rendement réalisé le mois suivant.</p></div>
    <div class="card wide"><div class="label">Score de classement</div><div class="kpi">Brut</div><p>La probabilité XGBoost brute ordonne les titres. La calibration isotone ne peut pas modifier le rang.</p></div>
    <div class="card wide"><div class="label">Code figé</div><div class="kpi">1b32a95</div><p class="meta">{html.escape(spec["code_commit"])}</p></div>
  </div>
</section>
<section id="protocol">
  <h2>Chaîne de validation</h2>
  <div class="flow">
    <div><b>1 · Données</b>Snapshot figé, hashes Legacy vérifiés, univers historique mensuel.</div>
    <div><b>2 · Walk-forward</b>72 mois train, 24 validation, purge six mois, 12 test.</div>
    <div><b>3 · Portefeuille</b>Top 5 du score brut, 10 pb × turnover, comparaison appariée.</div>
    <div><b>4 · Anti-biais</b>Bootstrap en blocs de 12 mois et Deflated Sharpe sur 162 essais.</div>
    <div><b>5 · Prospective</b>Aucun changement autorisé sous cet identifiant après observation future.</div>
  </div>
  <h3>Replay de sélection temporelle</h3>
  <p>Un second diagnostic choisit chaque janvier, uniquement sur les 36 mois hors échantillon antérieurs, le meilleur Sharpe parmi 108 configurations autonomes. Il évalue ensuite ce choix pendant l’année suivante. Ce replay teste la procédure de sélection, pas seulement le champion choisi a posteriori.</p>
</section>
<section id="bias">
  <h2>Ce que les contrôles peuvent — et ne peuvent pas — prouver</h2>
  <div class="grid">
    <div class="card wide"><div class="label">Contrôlé</div><h3>Fuite temporelle</h3><p>Cibles mûres, purges par horizon, prétraitement train-only, dictionnaire EMA point-in-time, modèle fixe par bloc test.</p></div>
    <div class="card wide"><div class="label">Contrôlé</div><h3>Erreur de calibration</h3><p>Les anciens plateaux isotones sont archivés comme invalides. Rang brut et probabilité calibrée sont deux sorties distinctes.</p></div>
    <div class="card wide"><div class="label">Quantifié</div><h3>Data snooping</h3><p>Le Deflated Sharpe utilise 162 combinaisons autonomes. Le bootstrap conserve des blocs annuels pour respecter l’autocorrélation.</p></div>
    <div class="card wide"><div class="label">Non résolu</div><h3>Holdout prospectif</h3><p>{html.escape(audit["partial_holdout"]["reason"])}</p></div>
  </div>
</section>
<section id="decision">
  <h2>Règle de décision</h2>
  <p class="lede">Le challenger peut passer à une observation prospective, mais pas encore à la production. Les têtes volatilité et downside seront entraînées séparément et ne pourront modifier le classement alpha tant que le challenger n’aura pas accumulé un holdout réellement nouveau.</p>
  <div class="callout good"><b>Suite autorisée :</b> enregistrer les scores mensuels, probabilités, SHAP et résultats réalisés sans retuner. Toute nouvelle version reçoit un nouvel identifiant et recommence son propre holdout.</div>
</section>
"""
    return _document(
        title="Protocole du challenger verrouillé",
        eyebrow="Papier méthodologique · 25 juillet 2026",
        verdict="Protocole propre et auditable ; preuve prospective encore absente.",
        verdict_tone="#fbbf24",
        body=body,
    )


def _results_paper(
    spec: dict,
    audit: dict,
    confirmation_dir: Path,
    champion_run: Path,
) -> str:
    monthly = pl.read_csv(
        champion_run / "classification_h06" / "trading_monthly.csv",
        try_parse_dates=True,
    ).filter(pl.col("top_n") == 5)
    yearly = pl.read_csv(confirmation_dir / "yearly_stability.csv")
    bootstrap = pl.read_csv(confirmation_dir / "paired_block_bootstrap.csv")
    dsr = pl.read_csv(confirmation_dir / "deflated_sharpe.csv").row(0, named=True)
    costs = pl.read_csv(confirmation_dir / "cost_sensitivity.csv")
    folds = pl.read_csv(confirmation_dir / "fold_trading_stability.csv")
    concentration = pl.read_csv(confirmation_dir / "portfolio_concentration.csv").row(
        0, named=True
    )
    sectors = pl.read_csv(confirmation_dir / "sector_concentration.csv")
    tickers = pl.read_csv(confirmation_dir / "ticker_concentration.csv")
    calibration = pl.read_csv(confirmation_dir / "probability_calibration.csv")
    meta = pl.read_csv(
        confirmation_dir / "meta_selection_summary.csv"
    ).row(0, named=True)
    meta_monthly = pl.read_csv(
        confirmation_dir / "meta_selection_monthly.csv",
        try_parse_dates=True,
    )
    choices = pl.read_csv(confirmation_dir / "meta_selection_choices.csv")
    shap = pl.read_csv(
        champion_run / "classification_h06" / "shap_importance.csv"
    ).head(12)

    cumulative = go.Figure()
    for label, column, color in (
        ("Challenger", "net_return", PALETTE["blue"]),
        ("Legacy", "legacy_return", PALETTE["green"]),
        ("S&P 500", "benchmark_return", PALETTE["muted"]),
    ):
        cumulative.add_trace(
            go.Scatter(
                x=monthly["decision_month"],
                y=_wealth(monthly[column]),
                name=label,
                line={"width": 2.8, "color": color},
            )
        )
    cumulative.update_yaxes(type="log", title="Capital, base 1 · échelle log")
    cumulative.update_layout(title="Croissance hors échantillon du capital")

    drawdown = go.Figure()
    for label, column, color in (
        ("Challenger", "net_return", PALETTE["blue"]),
        ("Legacy", "legacy_return", PALETTE["green"]),
        ("S&P 500", "benchmark_return", PALETTE["muted"]),
    ):
        drawdown.add_trace(
            go.Scatter(
                x=monthly["decision_month"],
                y=_drawdown(monthly[column]),
                name=label,
                line={"width": 2.4, "color": color},
            )
        )
    drawdown.update_yaxes(tickformat=".0%", title="Drawdown")
    drawdown.update_layout(title="Risque de parcours")

    yearly_fig = go.Figure()
    for label, column, color in (
        ("Challenger", "model_return", PALETTE["blue"]),
        ("Legacy", "legacy_return", PALETTE["green"]),
        ("S&P 500", "benchmark_return", PALETTE["muted"]),
    ):
        yearly_fig.add_trace(
            go.Bar(x=yearly["year"], y=yearly[column], name=label, marker_color=color)
        )
    yearly_fig.update_yaxes(tickformat=".0%", title="Rendement")
    yearly_fig.update_layout(title="Stabilité annuelle", barmode="group")

    fold_fig = go.Figure()
    fold_fig.add_trace(
        go.Bar(
            x=folds["fold"],
            y=folds["model_total_return"],
            name="Challenger",
            marker_color=[
                PALETTE["green"] if value >= 0 else PALETTE["red"]
                for value in folds["model_total_return"]
            ],
        )
    )
    fold_fig.add_trace(
        go.Scatter(
            x=folds["fold"],
            y=folds["legacy_total_return"],
            name="Legacy",
            mode="lines+markers",
            line={"color": PALETTE["ink"], "width": 2},
        )
    )
    fold_fig.update_yaxes(tickformat=".0%", title="Rendement du bloc")
    fold_fig.update_layout(title="Douze folds annuels : dispersion réelle")

    meta_fig = go.Figure()
    for label, column, color in (
        ("Méta-sélecteur", "net_return", PALETTE["purple"]),
        ("Legacy", "legacy_return", PALETTE["green"]),
        ("S&P 500", "benchmark_return", PALETTE["muted"]),
    ):
        meta_fig.add_trace(
            go.Scatter(
                x=meta_monthly["decision_month"],
                y=_wealth(meta_monthly[column]),
                name=label,
                line={"width": 2.5, "color": color},
            )
        )
    meta_fig.update_yaxes(title="Capital, base 1")
    meta_fig.update_layout(title="Replay du choix annuel sur 36 mois passés")

    calibration_fig = go.Figure()
    calibration_fig.add_trace(
        go.Scatter(
            x=calibration["mean_probability"],
            y=calibration["observed_positive_rate"],
            mode="markers+lines",
            name="Observé",
            marker={
                "size": np.clip(
                    np.sqrt(calibration["observations"].to_numpy()) * 1.2,
                    8,
                    36,
                ),
                "color": PALETTE["blue"],
            },
        )
    )
    calibration_fig.add_trace(
        go.Scatter(
            x=[0, 0.75],
            y=[0, 0.75],
            mode="lines",
            name="Calibration parfaite",
            line={"dash": "dash", "color": PALETTE["muted"]},
        )
    )
    calibration_fig.update_xaxes(title="Probabilité moyenne")
    calibration_fig.update_yaxes(title="Fréquence observée")
    calibration_fig.update_layout(title="Probabilités : bonnes en masse, fragiles aux extrêmes")

    perf = audit["exploratory_performance"]
    bootstrap_rows = [
        [
            html.escape(row["comparator"]),
            _pct(row["observed_annualized_mean_difference"]),
            f'{_pct(row["annualized_mean_ci_low"])} à {_pct(row["annualized_mean_ci_high"])}',
            _num(row["observed_sharpe_difference"]),
            f'{_num(row["sharpe_difference_ci_low"])} à {_num(row["sharpe_difference_ci_high"])}',
        ]
        for row in bootstrap.iter_rows(named=True)
    ]
    cost_rows = [
        [
            f'{row["cost_bps"]:.0f} pb',
            _pct(row["model_total_return"]),
            _pct(row["model_cagr"]),
            _num(row["model_sharpe"]),
            _pct(row["model_max_drawdown"]),
        ]
        for row in costs.iter_rows(named=True)
    ]
    sector_rows = [
        [
            html.escape(row["sector"]),
            _pct(row["average_monthly_weight"]),
            _pct(row["maximum_monthly_weight"]),
            _pct(row["active_month_rate"]),
        ]
        for row in sectors.head(8).iter_rows(named=True)
    ]
    ticker_rows = [
        [
            html.escape(row["ticker"]),
            str(row["selected_months"]),
            _pct(row["month_selection_rate"]),
            html.escape(row["sector"]),
        ]
        for row in tickers.head(10).iter_rows(named=True)
    ]
    choice_rows = [
        [
            html.escape(row["selection_date"][:4]),
            html.escape(row["selected_feature_mode"]),
            html.escape(row["selected_method"]),
            str(row["selected_horizon"]),
            str(row["selected_top_n"]),
            _num(row["historical_net_sharpe"]),
        ]
        for row in choices.iter_rows(named=True)
    ]
    shap_rows = [
        [html.escape(row["feature"]), _num(row["mean_abs_shap"], 4)]
        for row in shap.iter_rows(named=True)
    ]

    body = f"""
<nav class="nav">
  <a href="#headline">Résultat</a><a href="#statistics">Statistiques</a>
  <a href="#stability">Stabilité</a><a href="#selection">Méta-replay</a>
  <a href="#risk">Concentration</a><a href="#signals">Signaux</a>
  <a href="#verdict">Verdict</a>
</nav>
<section id="headline">
  <h2>Un rendement très fort, une preuve encore incomplète</h2>
  <div class="grid">
    <div class="card"><div class="label">Rendement net</div><div class="kpi">{_pct(perf["model_total_return"])}</div><div class="meta">144 mois · top 5</div></div>
    <div class="card"><div class="label">CAGR</div><div class="kpi">{_pct(perf["model_cagr"])}</div><div class="meta">Legacy {_pct(perf["legacy_cagr"])}</div></div>
    <div class="card"><div class="label">Sharpe</div><div class="kpi">{_num(perf["model_sharpe"])}</div><div class="meta">Legacy {_num(perf["legacy_sharpe"])}</div></div>
    <div class="card"><div class="label">Max drawdown</div><div class="kpi">{_pct(perf["model_max_drawdown"])}</div><div class="meta">Legacy {_pct(perf["legacy_max_drawdown"])}</div></div>
    <div class="card"><div class="label">Deflated Sharpe</div><div class="kpi">{_pct(dsr["deflated_sharpe_probability"])}</div><div class="meta">162 variantes autonomes</div></div>
    <div class="card"><div class="label">Holdout nouveau</div><div class="kpi">0 mois</div><div class="meta">aucune cible mûre après octobre 2025</div></div>
  </div>
  <div class="callout"><b>Lecture honnête :</b> le challenger crée un surplus de rendement convaincant dans l’historique, mais son avantage de Sharpe n’est pas significatif dans le bootstrap et la correction des essais multiples ne dépasse pas 95 %.</div>
  <div class="grid"><div class="card full chart">{_chart(cumulative)}</div><div class="card full chart">{_chart(drawdown)}</div></div>
</section>
<section id="statistics">
  <h2>Tests appariés et essais multiples</h2>
  <p class="lede">Le bootstrap rééchantillonne des blocs circulaires de douze mois. Il conserve mieux les régimes qu’un bootstrap mensuel indépendant.</p>
  {_table(["Comparateur","Écart moyen annualisé","IC 95%","Écart Sharpe","IC 95%"], bootstrap_rows)}
  <div class="callout good"><b>Rendement actif :</b> l’intervalle de l’écart moyen reste positif contre le S&amp;P 500 et Legacy.</div>
  <div class="callout bad"><b>Risque-ajusté :</b> les deux intervalles de différence de Sharpe contiennent largement zéro. Le Deflated Sharpe vaut {_pct(dsr["deflated_sharpe_probability"])} après 162 essais, insuffisant pour une confirmation forte.</div>
  <h3>Sensibilité aux coûts</h3>
  {_table(["Coût × turnover","Rendement net","CAGR","Sharpe","Max DD"], cost_rows)}
</section>
<section id="stability">
  <h2>Stabilité dans le temps</h2>
  <div class="grid"><div class="card full chart">{_chart(yearly_fig)}</div><div class="card full chart">{_chart(fold_fig)}</div></div>
  <p>Deux folds sur douze sont négatifs et plusieurs années sont dominées par Legacy ou le S&amp;P 500. La performance n’est donc pas un artefact d’une seule année, mais elle reste irrégulière et plus volatile.</p>
</section>
<section id="selection">
  <h2>Le replay de la procédure de sélection est plus faible</h2>
  <div class="grid">
    <div class="card"><div class="label">Méta CAGR</div><div class="kpi">{_pct(meta["model_cagr"])}</div><div class="meta">Legacy {_pct(meta["legacy_cagr"])}</div></div>
    <div class="card"><div class="label">Méta Sharpe</div><div class="kpi">{_num(meta["model_sharpe"])}</div><div class="meta">Legacy {_num(meta["legacy_sharpe"])} · S&amp;P {_num(meta["benchmark_sharpe"])}</div></div>
    <div class="card"><div class="label">Méta drawdown</div><div class="kpi">{_pct(meta["model_max_drawdown"])}</div><div class="meta">108 candidats · 36 mois passés</div></div>
  </div>
  <div class="grid"><div class="card full chart">{_chart(meta_fig)}</div></div>
  {_table(["Année","Features","Objectif","Horizon","Top N","Sharpe passé"], choice_rows)}
  <div class="callout bad"><b>Signal de prudence :</b> la procédure annuelle bat légèrement Legacy en rendement brut, mais pas en Sharpe ni en drawdown. Elle ne sélectionne jamais le champion 6m/top5 exact tel quel.</div>
</section>
<section id="risk">
  <h2>Concentration du top 5</h2>
  <div class="grid">
    <div class="card"><div class="label">Titres distincts</div><div class="kpi">{concentration["unique_tickers"]}</div><div class="meta">sur {concentration["portfolio_slots"]} positions-mois</div></div>
    <div class="card"><div class="label">Top 10 des titres</div><div class="kpi">{_pct(concentration["top_10_ticker_slot_share"])}</div><div class="meta">part de tous les slots</div></div>
    <div class="card"><div class="label">Secteur max moyen</div><div class="kpi">{_pct(concentration["average_monthly_max_sector_weight"])}</div><div class="meta">maximum ponctuel {_pct(concentration["maximum_sector_weight_any_month"])}</div></div>
  </div>
  <h3>Expositions sectorielles</h3>
  {_table(["Secteur","Poids moyen","Maximum","Mois actifs"], sector_rows)}
  <h3>Titres les plus souvent sélectionnés</h3>
  {_table(["Ticker","Mois","Fréquence","Secteur"], ticker_rows)}
  <div class="callout"><b>Risque structurel :</b> un secteur représente en moyenne 44,3 % du portefeuille et atteint parfois 100 %. La future tête de volatilité doit traiter ce risque sans réoptimiser le score alpha sur le même historique.</div>
</section>
<section id="signals">
  <h2>Probabilités et SHAP</h2>
  <div class="grid"><div class="card full chart">{_chart(calibration_fig)}</div></div>
  <p>La calibration globale est correcte — ECE historique inférieur à 1 % — mais les probabilités supérieures à 50 % concernent très peu d’observations et sont instables. Elles ne doivent pas encore servir à dimensionner fortement les positions.</p>
  {_table(["Variable","SHAP absolu moyen"], shap_rows)}
</section>
<section id="verdict">
  <h2>Décision finale de cette étape</h2>
  <div class="callout good"><b>Conserver :</b> le challenger 6 mois/top 5 mérite un suivi prospectif verrouillé. Son rendement actif, sa résistance aux coûts et sa persistance sur plusieurs folds sont réels.</div>
  <div class="callout bad"><b>Ne pas promouvoir :</b> aucune preuve prospective n’est encore disponible ; le Deflated Sharpe est seulement de {_pct(dsr["deflated_sharpe_probability"])} et le méta-replay reste inférieur à Legacy en Sharpe.</div>
  <p class="lede">Prochaine étape : produire mensuellement le score alpha brut, sa probabilité calibrée, SHAP et les expositions. Les têtes volatilité/downside seront une version séparée et ne modifieront pas le classement de ce challenger.</p>
</section>
"""
    return _document(
        title="Résultats confirmatoires du booster exact‑EMA",
        eyebrow="Papier de résultats · 25 juillet 2026",
        verdict="Challenger prometteur à suivre en paper ; pas encore production-ready.",
        verdict_tone="#fbbf24",
        body=body,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the locked challenger HTML papers.")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--confirmation-dir", type=Path, required=True)
    parser.add_argument("--champion-run", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.spec.read_text())
    audit = json.loads((args.confirmation_dir / "lock_audit.json").read_text())
    methodology = _methodology_paper(spec, audit)
    results = _results_paper(spec, audit, args.confirmation_dir, args.champion_run)
    methodology_path = args.output_dir / "methodology_paper.html"
    results_path = args.output_dir / "results_paper.html"
    methodology_path.write_text(methodology)
    results_path.write_text(results)
    index = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Locked challenger papers</title><style>
body{{font-family:Inter,system-ui;background:#f7f5ef;color:#14213d;margin:0;padding:8vw}}
h1{{font-size:clamp(44px,8vw,92px);letter-spacing:-.05em;line-height:.95;max-width:1000px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}
a{{display:block;background:white;color:#14213d;text-decoration:none;border:1px solid #d9dee8;
border-radius:20px;padding:28px;font-size:24px;font-weight:800}}a span{{display:block;color:#64748b;
font-size:14px;font-weight:500;margin-top:12px}}</style></head><body>
<p>ALPHARANK · CONFIRMATORY RESEARCH</p><h1>Challenger exact‑EMA verrouillé</h1>
<div class="grid"><a href="{methodology_path.name}">Papier méthodologique
<span>Verrou, protocole temporel, anti-fuite et règles prospectives.</span></a>
<a href="{results_path.name}">Papier de résultats
<span>Performance, bootstrap, Deflated Sharpe, stabilité, coûts, concentration et SHAP.</span></a></div>
</body></html>"""
    (args.output_dir / "index.html").write_text(index)
    print(args.output_dir)


if __name__ == "__main__":
    main()
