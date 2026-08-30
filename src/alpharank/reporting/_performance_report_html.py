from __future__ import annotations

import base64
import gzip
import json
from typing import Any

from alpharank.reporting._performance_report_composer_script import (
    PERFORMANCE_REPORT_COMPOSER_SCRIPT,
)
from alpharank.reporting._performance_report_script import PERFORMANCE_REPORT_SCRIPT
from alpharank.reporting._performance_report_styles import PERFORMANCE_REPORT_STYLES


def render_performance_report_html(payload: dict[str, Any]) -> str:
    """Render one autonomous browser report with an inline compressed payload."""

    packed = gzip.compress(
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8"),
        compresslevel=9,
        mtime=0,
    )
    encoded = base64.b64encode(packed).decode("ascii")
    return (
        '<!doctype html>\n<html lang="fr">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        "<title>AlphaRank · Rapport de backtest complet</title>\n"
        f"<style>{PERFORMANCE_REPORT_STYLES}</style>\n"
        "</head>\n<body>\n"
        '<div id="loading" class="loading">Chargement du rapport canonique…</div>\n'
        '<div id="app" class="shell" hidden>\n' + _sidebar() + _main() + "</div>\n"
        f"<script>const PAYLOAD_GZIP_BASE64={json.dumps(encoded)};\n"
        f"{PERFORMANCE_REPORT_COMPOSER_SCRIPT}\n"
        f"{PERFORMANCE_REPORT_SCRIPT}</script>\n"
        "</body>\n</html>\n"
    )


def _sidebar() -> str:
    return """
<aside class="sidebar">
  <div class="brand">
    <div class="brand-mark">AR</div>
    <div><strong>AlphaRank</strong><small>Backtest reporting</small></div>
  </div>
  <nav>
    <div class="nav-label">Performance</div>
    <a class="nav-link is-active" href="#overview">Vue d'ensemble</a>
    <a class="nav-link" href="#kpis">Tous les KPI</a>
    <a class="nav-link" href="#matrix">Model cards</a>
    <a class="nav-link" href="#composer">Composer un portefeuille</a>
    <div class="nav-label">Audit</div>
    <a class="nav-link" href="#portfolios">Portefeuilles historiques</a>
    <a class="nav-link" href="#methodologies">Méthodologies</a>
    <a class="nav-link" href="#lineage">Lignée et contrats</a>
  </nav>
  <div class="sidebar-meta">
    <div id="report-calendar">—</div>
    <div id="report-generated">—</div>
  </div>
</aside>
"""


def _main() -> str:
    return (
        """
<main><div class="content">
  <header class="hero">
    <div>
      <span class="eyebrow">Standard de performance · REPORT-006</span>
      <h1>Rapport de backtest complet</h1>
      <p>Legacy, Boosting natif, variantes filtrées par tendance et SPY sur un même
      calendrier. Les KPI de chaque fenêtre sont pré-calculés par le moteur commun ;
      cette page ne possède aucune formule financière parallèle.</p>
    </div>
    <div class="status-badge"><strong>Statut de preuve</strong><br><span id="status-message">—</span></div>
  </header>
  <div class="toolbar" aria-label="Filtres de performance">
    <label>Début de la fenêtre<select id="start-month"></select></label>
    <label>Fin de la fenêtre<select id="end-month"></select></label>
    <div class="curve-control">
      <span class="field-label">Courbes affichées</span>
      <details class="multi-select" id="curve-multiselect">
        <summary id="curve-select-label">Choisir les stratégies</summary>
        <div class="multi-select-menu">
          <div class="multi-select-actions">
            <button id="select-all-curves" type="button">Toutes</button>
            <button id="select-reference-curves" type="button">Legacy + SPY</button>
          </div>
          <div class="curve-options" id="curve-options"></div>
        </div>
      </details>
    </div>
    <button class="button secondary" id="reset-window" type="button">Toute la période</button>
  </div>
"""
        + _performance_sections()
        + _composer_section()
        + _audit_sections()
        + """
</div></main>
"""
    )


def _performance_sections() -> str:
    return """
  <section class="section" id="overview">
    <div class="section-head">
      <div><span class="section-kicker">01 · Vue d'ensemble</span><h2>Comparaison de la fenêtre</h2></div>
      <p id="window-label">—</p>
    </div>
    <div class="kpi-grid" id="kpi-grid"></div>
    <div class="chart-grid">
      <article class="panel"><h3>Croissance composée</h3><p class="panel-subtitle">Courbes rebasées à 1 au début de la fenêtre.</p><canvas id="wealth-chart"></canvas><div class="legend" id="wealth-legend"></div></article>
      <article class="panel"><h3>Drawdown</h3><p class="panel-subtitle">Écart à chaque plus-haut de richesse.</p><canvas id="drawdown-chart"></canvas><div class="legend" id="drawdown-legend"></div></article>
    </div>
  </section>
  <section class="section" id="kpis">
    <div class="section-head"><div><span class="section-kicker">02 · Mesure</span><h2>Tous les KPI des courbes affichées</h2></div><p>Le multiselect pilote aussi ces colonnes. Les cellules vertes surpassent SPY selon le sens économique du KPI ; les métriques descriptives restent neutres.</p></div>
    <div class="table-wrap"><table class="metric-table"><thead><tr id="metric-head"></tr></thead><tbody id="metric-body"></tbody></table></div>
  </section>
  <section class="section" id="matrix">
    <div class="section-head"><div><span class="section-kicker">03 · Model cards</span><h2>Performance cumulée et annuelle</h2></div><p>Les années et les stratégies affichées suivent strictement la fenêtre et les courbes actives.</p></div>
    <article class="panel">
      <div class="matrix-controls">
        <button class="is-active" type="button" data-matrix-metric="cagr">CAGR</button>
        <button type="button" data-matrix-metric="annualized_volatility">Volatilité</button>
        <button type="button" data-matrix-metric="max_drawdown">Max drawdown</button>
      </div>
      <div class="matrix-block">
        <h3>Depuis chaque année jusqu'à la fin sélectionnée</h3>
        <p class="panel-subtitle" id="cumulative-matrix-window">—</p>
        <div class="heatmap-wrap"><div class="heatmap" id="cumulative-heatmap"></div></div>
        <div class="viridis-legend"><span>Faible</span><i class="viridis-bar"></i><span>Élevé</span><strong id="cumulative-matrix-caption"></strong></div>
      </div>
      <div class="matrix-block incremental-block">
        <h3>Chaque année isolée · incrémental</h3>
        <p class="panel-subtitle">Chaque cellule utilise seulement les mois de l'année indiquée, sans capital antérieur.</p>
        <div class="heatmap-wrap"><div class="heatmap" id="incremental-heatmap"></div></div>
        <div class="viridis-legend"><span>Faible</span><i class="viridis-bar"></i><span>Élevé</span><strong id="incremental-matrix-caption"></strong></div>
      </div>
    </article>
  </section>
"""


def _composer_section() -> str:
    return """
  <section class="section" id="composer">
    <div class="section-head">
      <div><span class="section-kicker">04 · Portefeuille composé</span><h2>Combiner les stratégies et mesurer la diversification</h2></div>
      <p>Choisissez les poches : elles reçoivent le même poids chaque mois. Le résultat est comparé au SPY sur la fenêtre active.</p>
    </div>
    <div class="composer-layout">
      <article class="panel composer-selection">
        <div class="composer-selection-head">
          <div><h3>Stratégies incluses</h3><p class="panel-subtitle" id="composer-summary">—</p></div>
          <div class="composer-actions"><button id="composer-reference" type="button">Legacy + tendance Top 5</button><button id="composer-all" type="button">Toutes</button></div>
        </div>
        <div class="composer-options" id="composer-options"></div>
      </article>
      <aside class="composer-contract">
        <strong>Règle du laboratoire</strong>
        <p>Équipondération des stratégies cochées, rééquilibrée mensuellement. Les rendements de chaque poche sont déjà nets de leurs frais propres ; aucun coût supplémentaire entre poches n'est ajouté.</p>
        <small>Diagnostic post-hoc non promu. Un titre présent dans plusieurs stratégies reste exposé dans chacune de ces poches. Les 1 023 combinaisons et leurs KPI sont pré-calculés par le moteur commun ; le navigateur ne calcule que l'affichage des courbes.</small>
      </aside>
    </div>
    <div class="composer-kpis" id="composer-kpis"></div>
    <div class="composer-charts">
      <article class="panel"><h3>Croissance composée · portefeuille contre SPY</h3><p class="panel-subtitle">Courbes rebasées à 1 au début de la fenêtre.</p><canvas id="composer-wealth-chart"></canvas><div class="legend" id="composer-wealth-legend"></div></article>
      <article class="panel"><h3>Drawdown · portefeuille contre SPY</h3><p class="panel-subtitle">Écart à chaque plus-haut de richesse, sur un graphique pleine largeur.</p><canvas id="composer-drawdown-chart"></canvas><div class="legend" id="composer-drawdown-legend"></div></article>
    </div>
  </section>
"""


def _audit_sections() -> str:
    return """
  <section class="section" id="portfolios">
    <div class="section-head"><div><span class="section-kicker">05 · Positions</span><h2>Tous les portefeuilles historiques</h2></div><p>Poids décidés à t, rendement réalisé pendant t+1, score OOS lorsqu'il existe.</p></div>
    <div class="portfolio-controls">
      <label>Stratégie<select id="portfolio-strategy"></select></label>
      <label>Mois de détention<select id="portfolio-month"></select></label>
      <label>Filtrer un ticker<input id="ticker-search" type="search" placeholder="SATS, DELL…"></label>
      <button class="button secondary" id="export-holdings" type="button">Exporter CSV</button>
    </div>
    <p class="portfolio-summary" id="portfolio-summary"></p>
    <div class="table-wrap"><table><thead><tr><th>Ticker</th><th>Rang</th><th>Poids</th><th>Score</th><th>Secteur</th><th>Rendement réalisé</th><th>Votes</th></tr></thead><tbody id="holdings-body"></tbody></table></div>
    <div class="pager"><button id="page-prev" type="button">Précédent</button><span id="page-label">—</span><button id="page-next" type="button">Suivant</button></div>
  </section>
  <section class="section" id="methodologies">
    <div class="section-head"><div><span class="section-kicker">06 · Méthodes</span><h2>Règles et pseudo-codes</h2></div><p>Projection lisible des contrats canoniques ; aucun statut R&D n'est présenté comme une recommandation.</p></div>
    <div class="method-grid" id="method-grid"></div>
  </section>
  <section class="section" id="lineage">
    <div class="section-head"><div><span class="section-kicker">07 · Audit</span><h2>Lignée, hashes et conventions</h2></div><p>Le rapport cite ses entrées ; il ne résout jamais un artefact au nom « latest ».</p></div>
    <div class="lineage-grid">
      <article class="lineage-card"><h3>Contrats économiques</h3><dl class="definition" id="lineage-contracts"></dl></article>
      <article class="lineage-card"><h3>Snapshot et sources</h3><dl class="definition" id="lineage-data"></dl></article>
    </div>
  </section>
"""
